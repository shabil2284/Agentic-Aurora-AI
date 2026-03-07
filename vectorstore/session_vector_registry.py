"""
SessionVectorRegistry
─────────────────────
Singleton registry that maps session_id → ChromaStore.

All call sites (app.py, extractor_graph.py, tabular_viewer.py) must use
`SessionVectorRegistry.get_store(session_id)` instead of instantiating
ChromaStore directly.  This ensures:

  * One store per session (no orphaned instances)
  * Deterministic, audited cleanup on Clear Session
  * TTL-based admin purge for stale sessions
  * Feature-flag controlled (EPHEMERAL_VECTOR_ADAPTER_ENABLED)

Opt-in persistence is handled by ChromaStore.persist().
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Feature flag (imported lazily to avoid circular import at module load time)
def _adapter_enabled() -> bool:
    try:
        from config import settings
        return getattr(settings, "EPHEMERAL_VECTOR_ADAPTER_ENABLED", True)
    except Exception:
        return True


class SessionVectorRegistry:
    """Thread-safe singleton registry for session-scoped ChromaStore instances."""

    _lock = threading.Lock()
    _stores: Dict[str, dict] = {}   # {session_id: {"store": ChromaStore, "created_at": float}}

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def get_store(cls, session_id: str):
        """Return the ChromaStore for this session, creating it if necessary."""
        if not _adapter_enabled():
            # Adapter disabled: return a fresh store without tracking
            from vectorstore.chroma_store import ChromaStore
            logger.debug("[VectorRegistry] Adapter disabled — returning untracked store")
            return ChromaStore(session_id)

        with cls._lock:
            if session_id not in cls._stores:
                from vectorstore.chroma_store import ChromaStore
                store = ChromaStore(session_id)
                cls._stores[session_id] = {
                    "store": store,
                    "created_at": time.time(),
                }
                logger.info(
                    f"[VectorRegistry] Created session namespace: session={session_id}"
                )
            return cls._stores[session_id]["store"]

    @classmethod
    def clear_store(cls, session_id: str) -> dict:
        """
        Hard-delete all vectors for a session.

        Returns audit info: {session_id, timestamp, item_count, status}
        """
        audit = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "item_count": 0,
            "status": "not_found",
        }

        with cls._lock:
            entry = cls._stores.get(session_id)
            if entry is None:
                logger.warning(f"[VectorRegistry] clear_store: no store for session={session_id}")
                return audit

            store = entry["store"]
            try:
                item_count = store.count()
                audit["item_count"] = item_count
                store.clear_session()
                audit["status"] = "cleared"
            except Exception as e:
                audit["status"] = f"error: {e}"
                logger.error(f"[VectorRegistry] clear_store error for session={session_id}: {e}")
            finally:
                del cls._stores[session_id]

        logger.info(
            f"[VectorRegistry][AUDIT] SESSION_VECTOR_CLEARED "
            f"session={session_id} "
            f"item_count={audit['item_count']} "
            f"timestamp={audit['timestamp']} "
            f"status={audit['status']}"
        )
        return audit

    @classmethod
    def purge_stale(cls, ttl_seconds: Optional[int] = None) -> int:
        """
        Admin: purge all sessions older than ttl_seconds.
        Returns number of sessions purged.
        """
        if ttl_seconds is None:
            try:
                from config import settings
                ttl_seconds = getattr(settings, "VECTOR_SESSION_TTL_SECONDS", 3600)
            except Exception:
                ttl_seconds = 3600

        now = time.time()
        stale_ids = []
        with cls._lock:
            for sid, entry in cls._stores.items():
                age = now - entry["created_at"]
                if age > ttl_seconds:
                    stale_ids.append(sid)

        purged = 0
        for sid in stale_ids:
            cls.clear_store(sid)
            purged += 1

        if purged:
            logger.info(f"[VectorRegistry] Purged {purged} stale session(s) (TTL={ttl_seconds}s)")
        return purged

    @classmethod
    def active_sessions(cls) -> list:
        """Return list of currently tracked session IDs (for observability)."""
        with cls._lock:
            return list(cls._stores.keys())
