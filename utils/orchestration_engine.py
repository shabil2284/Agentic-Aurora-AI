"""
Aurora Orchestration Engine — utils/orchestration_engine.py
============================================================
Centralized orchestration layer for document processing.

Responsibilities:
- Full state reset on new file load (reset_tabular_state)
- Tabular-eligibility routing (compute_available_export_formats)
- MASTER_DATA_UPDATED_EVENT firing
- Structured audit trail logging

NON-INTERFERENCE RULE:
  This module does NOT modify extraction, table-creation, SQL execution,
  or schema generation internals. It only controls orchestration, binding,
  UI visibility state, and audit logging.
"""

import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ── State Reset ───────────────────────────────────────────────────────────────

def reset_tabular_state():
    """
    Full state reset on new file load.
    Clears: tabular_state, table_dataset, sql active_table, staged table,
            and all related UI references.

    Call this BEFORE running extraction on a new document.
    """
    # Core data models
    st.session_state.tabular_preview = None
    st.session_state.tabular_mode_on = False
    st.session_state.master_df = None
    st.session_state.df = None
    st.session_state.csv_df = None
    st.session_state.sql_result_df = None
    st.session_state.master_json = None
    st.session_state.json_data = None
    st.session_state.schema_list = []
    st.session_state.active_table = "document"
    st.session_state.orch_target = "document"
    st.session_state.canonical_data = None
    st.session_state.doc_title = None
    st.session_state.status_logs = []

    # Tabular viewer scoped keys
    st.session_state.tabular_staged_rows = []
    st.session_state.staged_csv_data = []
    st.session_state.csv_has_pending_master_commit = False
    st.session_state.pop("grid_key_counter", None)
    st.session_state.pop("_tabular_last_global_event", None)
    st.session_state.pop("_sql_tab_active_tbl", None)
    st.session_state.pop("tabular_table_ready", None)
    st.session_state.pop("tabular_autosync_on", None)
    st.session_state.pop("tabular_conflicts", None)

    # SQL mode reset — will be set to LIVE_TABLE if tabular detected
    st.session_state.sql_mode = "JSON"

    # Export formats — default to JSON only, upgraded when tabular is detected
    st.session_state.available_export_formats = ["JSON"]

    print("[OrchestrationEngine] Full state reset for new document.")


# ── Export Format Computation ─────────────────────────────────────────────────

def compute_available_export_formats(
    tabular_active: bool = True,
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Single source of truth for export options.
    Used by Export Data section, DB Push dropdown, and Cloud Push dropdown.

    SIMPLE RULE (per user spec):
      - If df exists and has rows → ["JSON", "CSV", "EXCEL"]
      - Otherwise → ["JSON"]

    Does NOT inspect cell content, document type, schema, or anything else.
    """
    if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
        formats = ["JSON", "CSV", "EXCEL"]
    else:
        formats = ["JSON"]

    st.session_state.available_export_formats = formats
    return formats


# ── Event Broadcasting ────────────────────────────────────────────────────────

def fire_master_data_updated_event(source: str = "unknown"):
    """
    Increments the global MASTER_DATA_UPDATED_EVENT counter.
    All UI components (Schema Editor, Table View, CSV Editor) listen to this.
    If it increments, they must drop caches and re-read from live MASTER_DATA_MODEL.
    """
    if "MASTER_DATA_UPDATED_EVENT" not in st.session_state:
        st.session_state.MASTER_DATA_UPDATED_EVENT = 0
    st.session_state.MASTER_DATA_UPDATED_EVENT += 1

    _emit_structured_log(
        actor="SYSTEM",
        action="MASTER_DATA_UPDATED_EVENT",
        details={"event_id": st.session_state.MASTER_DATA_UPDATED_EVENT, "source": source},
    )


# ── Audit Trail ───────────────────────────────────────────────────────────────

def append_audit_entry(
    actor: str,
    action: str,
    operations: Optional[List[Dict]] = None,
    status: str = "STAGED",
):
    """
    Appends a structured entry to the edit_history audit trail.

    Args:
        actor:      Who initiated (e.g. "CHATBOT", "CSV_EDITOR", "SQL_ENGINE", "USER")
        action:     What happened (e.g. "COMMIT", "REVERT", "STAGE")
        operations: List of atomic operation dicts
        status:     "STAGED", "COMMITTED", "REVERTED", "UNDONE"
    """
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = []

    entry = {
        "timestamp": datetime.datetime.now().isoformat()[:19],
        "actor": actor,
        "action": action,
        "status": status,
        "operations": operations or [],
    }

    st.session_state.edit_history.append(entry)

    _emit_structured_log(
        actor=actor,
        action=action,
        details={"status": status, "ops_count": len(operations or [])},
    )

    return entry


# ── Structured Logging ────────────────────────────────────────────────────────

def _emit_structured_log(actor: str, action: str, details: Optional[Dict] = None):
    """
    Emits a structured audit log line to stdout for observability.
    Format: [Aurora Audit] <timestamp> | <actor> | <action> | <details>
    """
    ts = datetime.datetime.now().isoformat()[:19]
    detail_str = " | ".join(f"{k}={v}" for k, v in (details or {}).items())
    print(f"[Aurora Audit] {ts} | {actor} | {action} | {detail_str}")
