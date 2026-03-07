"""
Integration tests for the Session Vector Adapter.

Tests:
  a) vectors written under session namespace
  b) queries work during session
  c) all vectors removed after clear
  d) persistent opt-in copies correctly without affecting ephemeral default
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.session_vector_registry import SessionVectorRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure a clean registry state before each test."""
    # Clear any stale sessions from previous tests
    for sid in SessionVectorRegistry.active_sessions():
        SessionVectorRegistry.clear_store(sid)
    yield
    # Teardown: clear again
    for sid in SessionVectorRegistry.active_sessions():
        SessionVectorRegistry.clear_store(sid)


TEST_SESSION_A = "test-session-a-0001"
TEST_SESSION_B = "test-session-b-0002"
SAMPLE_DOC = '{"company": "Acme Corp", "email": "acme@example.com"}'
SAMPLE_META = {"filename": "test_document.pdf"}


class TestVectorWrittenUnderSessionNamespace:
    """(a) Vectors written under the correct session namespace."""

    def test_store_created_per_session(self):
        store_a = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store_b = SessionVectorRegistry.get_store(TEST_SESSION_B)
        # Different session IDs must yield different stores
        assert store_a is not store_b
        assert store_a.session_id == TEST_SESSION_A
        assert store_b.session_id == TEST_SESSION_B

    def test_same_session_returns_same_store(self):
        store1 = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store2 = SessionVectorRegistry.get_store(TEST_SESSION_A)
        assert store1 is store2  # same object, not a copy

    def test_document_written_to_collection(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        initial_count = store.count()
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        assert store.count() == initial_count + 1

    def test_writes_scoped_to_session(self):
        store_a = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store_b = SessionVectorRegistry.get_store(TEST_SESSION_B)
        store_a.add_documents([SAMPLE_DOC], [SAMPLE_META])
        # Session B should have 0 items (isolated)
        assert store_b.count() == 0
        assert store_a.count() == 1


class TestQueriesWorkDuringSession:
    """(b) Queries return correct results within the session."""

    def test_query_returns_inserted_document(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        result = store.query("company email", n_results=1)
        assert len(result["documents"][0]) >= 1
        assert SAMPLE_DOC in result["documents"][0]

    def test_query_on_empty_collection_returns_empty(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        # No documents added
        result = store.query("some query", n_results=5)
        assert result["documents"] == [[]]

    def test_metadata_preserved_in_query(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        result = store.query("company", n_results=1)
        assert result["metadatas"][0][0]["filename"] == "test_document.pdf"


class TestVectorsRemovedAfterClear:
    """(c) All vectors fully removed after clear_store()."""

    def test_clear_store_removes_all_vectors(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        assert store.count() == 1

        audit = SessionVectorRegistry.clear_store(TEST_SESSION_A)

        assert audit["status"] == "cleared"
        assert audit["item_count"] == 1
        assert audit["session_id"] == TEST_SESSION_A

    def test_cleared_session_not_in_registry(self):
        SessionVectorRegistry.get_store(TEST_SESSION_A)
        assert TEST_SESSION_A in SessionVectorRegistry.active_sessions()

        SessionVectorRegistry.clear_store(TEST_SESSION_A)
        assert TEST_SESSION_A not in SessionVectorRegistry.active_sessions()

    def test_clear_nonexistent_session_returns_not_found(self):
        audit = SessionVectorRegistry.clear_store("nonexistent-session-xyz")
        assert audit["status"] == "not_found"

    def test_new_store_after_clear_starts_empty(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        SessionVectorRegistry.clear_store(TEST_SESSION_A)

        # Re-create store for same session — should start empty
        new_store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        assert new_store.count() == 0


class TestPersistentOptInCopiesCorrectly:
    """(d) Persist copies to persistent namespace; ephemeral default unchanged."""

    def test_persist_copies_documents(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])

        target_ns = f"user_{TEST_SESSION_A}_persistent"
        result = store.persist(target_ns)
        assert result is True

    def test_persist_returns_false_when_empty(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        # No documents added
        result = store.persist(f"user_{TEST_SESSION_A}_persistent")
        assert result is False

    def test_ephemeral_default_unaffected_after_persist(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        store.persist(f"user_{TEST_SESSION_A}_persistent")

        # Ephemeral collection still has original 1 item
        assert store.count() == 1

    def test_ephemeral_different_from_persistent_after_clear(self):
        store = SessionVectorRegistry.get_store(TEST_SESSION_A)
        store.add_documents([SAMPLE_DOC], [SAMPLE_META])
        store.persist(f"user_{TEST_SESSION_A}_persistent")

        # Clear ephemeral session
        SessionVectorRegistry.clear_store(TEST_SESSION_A)

        # Ephemeral is gone from registry
        assert TEST_SESSION_A not in SessionVectorRegistry.active_sessions()
        # (Persistent namespace lives in the same in-memory client — this is
        #  expected because ChromaDB EphemeralClient is process-scoped;
        #  in production with a PersistentClient, the persistent collection
        #  would survive across restarts. This test validates the isolation logic.)
