"""
Unit tests for Aurora Orchestration Engine & Cloud Push Validator
Tests: reset, tabular routing, export format computation, audit trail,
       credential validation, error mapping, filename generation
"""
import os
import sys
import re
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Mock st.session_state for unit tests (no Streamlit runtime) ──────────────
class _MockSessionState(dict):
    """Dict subclass that allows attribute access, mimicking st.session_state."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


# Patch streamlit before importing modules that use it
import types
st_mock = types.ModuleType("streamlit")
st_mock.session_state = _MockSessionState()
sys.modules["streamlit"] = st_mock

# Now import the modules under test
from utils.orchestration_engine import (
    reset_tabular_state,
    compute_available_export_formats,
    fire_master_data_updated_event,
    append_audit_entry,
)
from utils.cloud_push_validator import (
    validate_credentials,
    map_push_error,
    generate_push_filename,
    build_verification_summary,
)


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION ENGINE TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_tabular_state():
    """After reset, all relevant keys should be cleared to defaults."""
    import streamlit as st
    # Set some state that should be cleared
    st.session_state.tabular_preview = {"rows": [1, 2]}
    st.session_state.tabular_mode_on = True
    st.session_state.master_df = "some_df"
    st.session_state.sql_mode = "LIVE_TABLE"
    st.session_state.active_table = "invoices"
    st.session_state.available_export_formats = ["JSON", "CSV", "EXCEL"]

    reset_tabular_state()

    assert st.session_state.tabular_preview is None
    assert st.session_state.tabular_mode_on == False
    assert st.session_state.master_df is None
    assert st.session_state.sql_mode == "JSON"
    assert st.session_state.active_table == "document"
    assert st.session_state.available_export_formats == ["JSON"]
    print("PASS test_reset_tabular_state")


def test_compute_export_formats_tabular():
    """When a DataFrame with rows is passed, formats should include CSV and EXCEL."""
    import streamlit as st
    import pandas as pd
    _df = pd.DataFrame([{"SNo": "1", "Name": "Alice"}, {"SNo": "2", "Name": "Bob"}])
    result = compute_available_export_formats(df=_df)
    assert result == ["JSON", "CSV", "EXCEL"]
    assert st.session_state.available_export_formats == ["JSON", "CSV", "EXCEL"]
    print("PASS test_compute_export_formats_tabular")


def test_compute_export_formats_hierarchical():
    """When not tabular, only JSON should be available."""
    import streamlit as st
    result = compute_available_export_formats(tabular_active=False)
    assert result == ["JSON"]
    assert st.session_state.available_export_formats == ["JSON"]
    print("PASS test_compute_export_formats_hierarchical")


def test_fire_master_data_updated_event():
    """Event counter should increment."""
    import streamlit as st
    st.session_state.MASTER_DATA_UPDATED_EVENT = 5
    fire_master_data_updated_event(source="TEST")
    assert st.session_state.MASTER_DATA_UPDATED_EVENT == 6
    print("PASS test_fire_master_data_updated_event")


def test_append_audit_entry():
    """Should append a structured entry to edit_history."""
    import streamlit as st
    st.session_state.edit_history = []

    entry = append_audit_entry(
        actor="CHATBOT",
        action="COMMIT",
        operations=[{"type": "UPDATE_CELL", "target": "col1"}],
        status="COMMITTED",
    )

    assert len(st.session_state.edit_history) == 1
    assert entry["actor"] == "CHATBOT"
    assert entry["action"] == "COMMIT"
    assert entry["status"] == "COMMITTED"
    assert len(entry["operations"]) == 1
    assert "timestamp" in entry
    print("PASS test_append_audit_entry")


# ──────────────────────────────────────────────────────────────────────────────
# CLOUD PUSH VALIDATOR TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_validate_credentials_aws_missing():
    """AWS S3 with empty creds should fail and list missing fields."""
    valid, missing = validate_credentials("AWS S3", {}, "")
    assert not valid
    assert "access_key" in missing
    assert "secret_key" in missing
    assert "region" in missing
    assert "bucket/container name" in missing
    print("PASS test_validate_credentials_aws_missing")


def test_validate_credentials_aws_valid():
    """AWS S3 with all creds should pass."""
    valid, missing = validate_credentials(
        "AWS S3",
        {"access_key": "AKIA...", "secret_key": "abc123", "region": "us-east-1"},
        "my-bucket",
    )
    assert valid
    assert missing == []
    print("PASS test_validate_credentials_aws_valid")


def test_validate_credentials_gcs():
    """GCS with empty service_account_json should fail."""
    valid, missing = validate_credentials("GCS", {}, "my-bucket")
    assert not valid
    assert "service_account_json" in missing
    print("PASS test_validate_credentials_gcs")


def test_validate_credentials_azure():
    """Azure with connection string should pass."""
    valid, missing = validate_credentials(
        "Azure Blob",
        {"connection_string": "DefaultEndpointsProtocol=..."},
        "my-container",
    )
    assert valid
    assert missing == []
    print("PASS test_validate_credentials_azure")


def test_map_push_error_aws():
    """AWS error mapping should return friendly messages."""
    msg = map_push_error("AWS S3", "An error occurred (InvalidAccessKeyId): The key is invalid")
    assert "Invalid Access Key" in msg
    msg2 = map_push_error("AWS S3", "NoSuchBucket: The specified bucket does not exist")
    assert "Bucket not found" in msg2
    print("PASS test_map_push_error_aws")


def test_map_push_error_gcs():
    """GCS error mapping should return friendly messages."""
    msg = map_push_error("GCS", "PERMISSION_DENIED: Caller lacks storage.objects.create")
    assert "permission" in msg.lower()
    print("PASS test_map_push_error_gcs")


def test_map_push_error_azure():
    """Azure error mapping should return friendly messages."""
    msg = map_push_error("Azure Blob", "AuthenticationFailed: Server failed to authenticate the request.")
    assert "Invalid Account Key" in msg
    print("PASS test_map_push_error_azure")


def test_map_push_error_unknown():
    """Unknown errors should return generic fallback."""
    msg = map_push_error("AWS S3", "Some completely unknown error")
    assert "Connection failed" in msg
    print("PASS test_map_push_error_unknown")


def test_generate_push_filename_json():
    """Should generate a timestamped JSON filename."""
    ts = datetime.datetime(2026, 3, 1, 22, 37)
    name = generate_push_filename("invoice_13442", "JSON", timestamp=ts)
    assert name == "invoice_13442__20260301T2237.json"
    print(f"PASS test_generate_push_filename_json → {name}")


def test_generate_push_filename_csv():
    """Should generate a timestamped CSV filename."""
    ts = datetime.datetime(2026, 3, 1, 22, 37)
    name = generate_push_filename("resume_john", "CSV", timestamp=ts)
    assert name == "resume_john__20260301T2237.csv"
    print(f"PASS test_generate_push_filename_csv → {name}")


def test_generate_push_filename_excel():
    """Should generate a timestamped Excel filename."""
    ts = datetime.datetime(2026, 3, 1, 22, 37)
    name = generate_push_filename("bank_statement", "EXCEL", timestamp=ts)
    assert name == "bank_statement__20260301T2237.xlsx"
    print(f"PASS test_generate_push_filename_excel → {name}")


def test_generate_push_filename_sanitize():
    """Unsafe characters in doc_id should be sanitized."""
    ts = datetime.datetime(2026, 3, 1, 22, 37)
    name = generate_push_filename("doc/with spaces&special!", "JSON", timestamp=ts)
    assert "/" not in name
    assert "&" not in name
    assert "!" not in name
    assert name.endswith(".json")
    print(f"PASS test_generate_push_filename_sanitize → {name}")


def test_build_verification_summary_success():
    """Successful verification should return verified=True."""
    verified, msg = build_verification_summary("AWS S3", "s3://bucket/file.json", 1024, 1024, True)
    assert verified
    assert "verified" in msg.lower()
    print("PASS test_build_verification_summary_success")


def test_build_verification_summary_size_zero():
    """Zero-size remote object should fail verification."""
    verified, msg = build_verification_summary("GCS", "gs://bucket/file.json", 1024, 0)
    assert not verified
    assert "size 0" in msg.lower()
    print("PASS test_build_verification_summary_size_zero")


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION: Tabular vs Hierarchical Routing
# ──────────────────────────────────────────────────────────────────────────────

def test_invoice_tabular_routing():
    """Invoice-like data with a real DataFrame → formats include CSV/EXCEL."""
    import streamlit as st
    import pandas as pd
    reset_tabular_state()

    # Simulate tabular detection success with a real DataFrame
    st.session_state.tabular_mode_on = True
    _df = pd.DataFrame([{"InvoiceNo": "INV-001", "Amount": 100}, {"InvoiceNo": "INV-002", "Amount": 200}])
    st.session_state.master_df = _df
    result = compute_available_export_formats(df=_df)

    assert result == ["JSON", "CSV", "EXCEL"]
    print("PASS test_invoice_tabular_routing")


def test_resume_hierarchical_routing():
    """Resume JSON (no arrays) should be hierarchical → formats = JSON only."""
    import streamlit as st
    reset_tabular_state()

    # Simulate hierarchical detection
    st.session_state.tabular_mode_on = False
    result = compute_available_export_formats(tabular_active=False)

    assert result == ["JSON"]
    print("PASS test_resume_hierarchical_routing")


# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback

    tests = [
        # Orchestration Engine
        test_reset_tabular_state,
        test_compute_export_formats_tabular,
        test_compute_export_formats_hierarchical,
        test_fire_master_data_updated_event,
        test_append_audit_entry,
        # Cloud Push Validator
        test_validate_credentials_aws_missing,
        test_validate_credentials_aws_valid,
        test_validate_credentials_gcs,
        test_validate_credentials_azure,
        test_map_push_error_aws,
        test_map_push_error_gcs,
        test_map_push_error_azure,
        test_map_push_error_unknown,
        test_generate_push_filename_json,
        test_generate_push_filename_csv,
        test_generate_push_filename_excel,
        test_generate_push_filename_sanitize,
        test_build_verification_summary_success,
        test_build_verification_summary_size_zero,
        # Integration routing
        test_invoice_tabular_routing,
        test_resume_hierarchical_routing,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'ALL TESTS PASSED' if failed == 0 else 'SOME TESTS FAILED'}")
