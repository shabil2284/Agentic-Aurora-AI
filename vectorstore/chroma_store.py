import os
import sys

# Streamlit Cloud uses an older SQLite version. We must override it before importing chromadb.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from config import settings
import uuid
import shutil
import os
import logging
import time

logger = logging.getLogger(__name__)


class ChromaStore:
    """Session-scoped, ephemeral vector store.

    All data lives in memory and is destroyed when the session ends or
    ``clear_session()`` is called.  Nothing is ever written to disk.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._collection_name = f"session_{session_id.replace('-', '_')}"
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self.created_at = time.time()

        # ── Step 1: Create an in-memory ChromaDB client ──────────────
        self.client = self._create_ephemeral_client()

        # ── Step 2: Load embedding function ──────────────────────────
        try:
            from chromadb.utils import embedding_functions
            self.embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=settings.EMBEDDING_MODEL
                )
            )
        except Exception as e:
            logger.warning(f"SentenceTransformer embedding unavailable: {e}")
            self.embedding_fn = None  # ChromaDB will use its own default

        # ── Step 3: Create session collection ────────────────────────
        try:
            kwargs = {"name": self._collection_name, "metadata": {"hnsw:space": "cosine"}}
            if self.embedding_fn is not None:
                kwargs["embedding_function"] = self.embedding_fn
            self.collection = self.client.get_or_create_collection(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {e}")
            raise

    # ── Client factory — resilient initialization ──────────────────
    @staticmethod
    def _create_ephemeral_client():
        """Return an in-memory ChromaDB client."""
        try:
            # On newer versions, EphemeralClient is robust on Streamlit 
            import chromadb
            return chromadb.EphemeralClient()
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB EphemeralClient: {e}")
            raise RuntimeError(
                f"Could not initialise ChromaDB: {str(e)}"
            ) from e

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str] = None):
        if not documents:
            return
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.error(f"Error adding to Chroma: {e}")

    def query(self, query_text: str, n_results: int = 5):
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
        except Exception as e:
            logger.error(f"Chroma Query Error: {e}")
            return {"documents": [[]], "metadatas": [[]]}

    def count(self) -> int:
        """Return number of vectors in this session's collection."""
        try:
            return self.collection.count() if self.collection else 0
        except Exception:
            return 0

    def clear_session(self):
        """Hard-delete all in-memory vector data for this session."""
        try:
            item_count = self.count()
            self.client.delete_collection(self._collection_name)
            self.collection = None
            logger.info(
                f"[ChromaStore][AUDIT] SESSION_CLEARED "
                f"session={self.session_id} "
                f"collection={self._collection_name} "
                f"item_count={item_count} "
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
            )
        except Exception as e:
            logger.warning(f"ChromaStore clear_session error: {e}")

    def persist(self, target_namespace: str) -> bool:
        """
        Opt-in persistence: copy all documents from this session's ephemeral
        collection into a persistent-namespace collection.

        target_namespace — e.g. 'user:{user_id}:persistent'
        Returns True on success, False on failure.
        """
        try:
            if not self.collection or self.count() == 0:
                logger.info(f"[ChromaStore] persist: nothing to persist for session={self.session_id}")
                return False

            # Retrieve all items from the ephemeral collection
            results = self.collection.get(include=["documents", "metadatas", "embeddings"])
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            ids = results.get("ids") or []

            if not docs:
                return False

            # Create or get the persistent collection
            safe_ns = target_namespace.replace(":", "_").replace("-", "_")
            kwargs = {"name": safe_ns, "metadata": {"hnsw:space": "cosine"}}
            if self.embedding_fn is not None:
                kwargs["embedding_function"] = self.embedding_fn
            persistent_col = self.client.get_or_create_collection(**kwargs)

            # Copy documents
            persistent_col.upsert(documents=docs, metadatas=metas, ids=ids)

            logger.info(
                f"[ChromaStore][AUDIT] SESSION_PERSISTED "
                f"session={self.session_id} "
                f"target_namespace={target_namespace} "
                f"item_count={len(docs)} "
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
            )
            return True
        except Exception as e:
            logger.error(f"[ChromaStore] persist error: {e}")
            return False

class RAGAssistant:
    def __init__(self, store: ChromaStore):
        self.store = store
        from utils.strict_llm_client import StrictLLMClient
        self.llm_client = StrictLLMClient(require_json=False)

    def answer_question(self, question: str) -> dict:
        try:
            results = self.store.query(question)
            
            if not results['documents'][0]:
                return {
                    "answer": "I don't have enough information from the document.",
                    "sources": [],
                    "context_snippets": []
                }

            context = "\n\n".join(results['documents'][0])
            sources = results['metadatas'][0]
            
            prompt = f"""
            Answer the following question based ONLY on the provided document context.
            If the answer is not in the context, say "I don't have enough information from the document."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            response = self.llm_client.invoke(prompt, input_kwargs={})
            answer_text = response.get("data", "") if response.get("status") != "error" else "Fallback Error: I could not reach my reasoning engines."
            
            return {
                "answer": answer_text,
                "sources": sources,
                "context_snippets": results['documents'][0]
            }
        except Exception as e:
            logging.error(f"RAG Error: {e}")
            return {"answer": f"Retrieval Error: {str(e)}", "sources": [], "context_snippets": []}
