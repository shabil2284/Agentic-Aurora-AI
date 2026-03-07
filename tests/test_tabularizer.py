"""
Unit tests for Aurora Tabularization Module
Tests: TabularDetector (positive/negative) + Tabularizer (flattening + roundtrip)
"""
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tabularizer import TabularDetector, Tabularizer, _parse_currency, _parse_date

# ── Sample data ────────────────────────────────────────────────────────────────

INVOICE_JSON = {
    "Invoice No": "13442",
    "Date": "2025-07-19",
    "Vendor": "Stehr-Kutch",
    "Client": "Howell Club 941",
    "Address": "55499-5119 Lake Philipmouth",
    "Order Number": "Mayra Hahn",
    "Payment Details": "7 days",
    "Line Items": [
        {"Description": "Et qui repellat delectus.", "Quantity": "5 Pcs.", "Unit Price": "775,70", "Total": "3.878,50"},
        {"Description": "Doloribus dolore est.",    "Quantity": "83 Pcs.", "Unit Price": "565,69", "Total": "46.952,27"},
        {"Description": "Dolore incidunt et.",      "Quantity": "86 Pcs.", "Unit Price": "580,01", "Total": "49.880,86"},
    ],
    "Subtotal": "100.711,63",
    "Tax": "15% VAT",
    "Tax Amount": "15.106,74",
    "Grand Total": "115.818,37"
}

BANK_JSON = {
    "Account Number": "1234-5678",
    "Statement Period": "Jan 2025",
    "Transactions": [
        {"Date": "2025-01-02", "Description": "Opening Balance", "Debit": "",   "Credit": "", "Balance": "5000.00"},
        {"Date": "2025-01-05", "Description": "POS Amazon",       "Debit": "120.50", "Credit": "", "Balance": "4879.50"},
        {"Date": "2025-01-10", "Description": "Salary Credit",    "Debit": "",   "Credit": "3000.00", "Balance": "7879.50"},
    ]
}

HR_LIST_JSON = [
    {"Name": "Alice Smith",  "Email": "alice@corp.com",  "Phone": "9876543210", "Company": "Alpha Ltd"},
    {"Name": "Bob Johnson",  "Email": "bob@corp.com",    "Phone": "9123456789", "Company": "Beta Inc"},
    {"Name": "Carol White",  "Email": "carol@corp.com",  "Phone": "9988776655", "Company": "Gamma LLC"},
]

RESUME_JSON = {
    "Name": "John Doe",
    "Objective": "A passionate software engineer with 5 years of experience in backend systems and distributed architecture.",
    "Experience": "Worked at Google as a SWE from 2018 to 2022. Led a team of 6 engineers.",
    "Education": "B.Tech Computer Science, IIT Bombay, 2018",
    "Skills": "Python, Go, Kubernetes, PostgreSQL",
    "References": "Available on request."
}

SIMPLE_KV_JSON = {
    "Full Name": "Jane Doe",
    "Date of Birth": "12/05/1990",
    "Nationality": "UAE",
    "Passport No": "P1234567",
    "Email": "jane@email.com",
    "Phone": "+971 50 123 4567"
}

# ──────────────────────────────────────────────────────────────────────────────
# POSITIVE DETECTION TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_invoice_detection_positive():
    is_tab, conf, reason = TabularDetector.detect(INVOICE_JSON)
    assert is_tab, f"Expected tabular=True for invoice, got conf={conf}, reason={reason}"
    assert conf >= 0.85, f"Expected confidence >= 0.85, got {conf}"
    print(f"PASS test_invoice_detection_positive (conf={conf})")


def test_bank_statement_detection_positive():
    is_tab, conf, reason = TabularDetector.detect(BANK_JSON)
    assert is_tab, f"Expected tabular=True for bank statement, got conf={conf}, reason={reason}"
    print(f"PASS test_bank_statement_detection_positive (conf={conf})")


def test_hr_list_detection_positive():
    is_tab, conf, reason = TabularDetector.detect(HR_LIST_JSON)
    assert is_tab, f"Expected tabular=True for HR list, got conf={conf}, reason={reason}"
    print(f"PASS test_hr_list_detection_positive (conf={conf})")


# ──────────────────────────────────────────────────────────────────────────────
# NEGATIVE DETECTION TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_resume_detection_negative():
    is_tab, conf, reason = TabularDetector.detect(RESUME_JSON)
    assert not is_tab, f"Expected tabular=False for resume, got conf={conf}, reason={reason}"
    print(f"PASS test_resume_detection_negative (conf={conf})")


def test_simple_kv_detection_negative():
    is_tab, conf, reason = TabularDetector.detect(SIMPLE_KV_JSON)
    assert not is_tab, f"Expected tabular=False for simple KV form, got conf={conf}"
    print(f"PASS test_simple_kv_detection_negative (conf={conf})")


def test_empty_json_detection_negative():
    is_tab, conf, reason = TabularDetector.detect({})
    assert not is_tab
    print(f"PASS test_empty_json_detection_negative (conf={conf})")


# ──────────────────────────────────────────────────────────────────────────────
# TABULARIZATION CORRECTNESS
# ──────────────────────────────────────────────────────────────────────────────

def test_invoice_flattening():
    preview = Tabularizer.build_preview(INVOICE_JSON)
    rows = preview["rows"]
    assert len(rows) == 3, f"Expected 3 rows (1 per line item), got {len(rows)}"
    assert rows[0]["Invoice No"] == "13442"
    assert rows[0]["Description"] == "Et qui repellat delectus."
    assert rows[0]["__source_row_id"] == "row-0"
    assert rows[0]["__master_path"] == "/Line Items/0"
    assert rows[1]["__source_row_id"] == "row-1"
    assert rows[2]["__source_row_id"] == "row-2"
    print(f"PASS test_invoice_flattening ({len(rows)} rows)")


def test_currency_parsing():
    assert _parse_currency("775,70") == 775.70,  f"Got {_parse_currency('775,70')}"
    assert _parse_currency("3.878,50") == 3878.50, f"Got {_parse_currency('3.878,50')}"
    assert _parse_currency("$1,234.56") == 1234.56, f"Got {_parse_currency('$1,234.56')}"
    assert _parse_currency("100.711,63") == 100711.63, f"Got {_parse_currency('100.711,63')}"
    print("PASS test_currency_parsing")


def test_date_parsing():
    assert _parse_date("2025-07-19") == "2025-07-19"
    assert _parse_date("19/07/2025") == "2025-07-19"
    assert _parse_date("July 19, 2025") == "2025-07-19"
    assert _parse_date("gibberish") == "gibberish"  # fallback to raw
    print("PASS test_date_parsing")


def test_apply_edits_roundtrip():
    """Edit a line item cell and verify master_json updates at the correct pointer."""
    import copy
    original = copy.deepcopy(INVOICE_JSON)
    preview = Tabularizer.build_preview(original)
    rows = preview["rows"]
    # Edit row-1's Description
    staged = list(rows)
    staged[1] = dict(staged[1])
    staged[1]["Description"] = "EDITED DESCRIPTION"
    staged[1]["Unit Price"]  = "999,99"

    updated = Tabularizer.apply_edits(original, staged)
    assert updated["Line Items"][1]["Description"] == "EDITED DESCRIPTION", \
        f"Expected 'EDITED DESCRIPTION', got {updated['Line Items'][1]['Description']}"
    assert updated["Line Items"][1]["Unit Price"] == "999,99", \
        f"Expected '999,99', got {updated['Line Items'][1]['Unit Price']}"
    # Row 0 and 2 must be untouched
    assert updated["Line Items"][0]["Description"] == "Et qui repellat delectus."
    assert updated["Line Items"][2]["Description"] == "Dolore incidunt et."
    print("PASS test_apply_edits_roundtrip")


def test_csv_export_matches_rows():
    """CSV from tabular preview should match flattened row count."""
    import io
    import pandas as pd
    preview = Tabularizer.build_preview(INVOICE_JSON)
    rows = preview["rows"]
    # Strip hidden cols for export
    hidden = {"__source_row_id", "__master_path", "__created_at_confidence"}
    visible = [{k: v for k, v in r.items() if k not in hidden} for r in rows]
    df = pd.DataFrame(visible)
    csv_buf = io.StringIO(df.to_csv(index=False))
    loaded = pd.read_csv(csv_buf)
    assert len(loaded) == 3, f"Expected 3 rows in CSV, got {len(loaded)}"
    assert "Invoice No" in loaded.columns
    assert "Description" in loaded.columns
    print(f"PASS test_csv_export_matches_rows ({len(loaded)} rows, {len(loaded.columns)} cols)")


# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    tests = [
        test_invoice_detection_positive,
        test_bank_statement_detection_positive,
        test_hr_list_detection_positive,
        test_resume_detection_negative,
        test_simple_kv_detection_negative,
        test_empty_json_detection_negative,
        test_invoice_flattening,
        test_currency_parsing,
        test_date_parsing,
        test_apply_edits_roundtrip,
        test_csv_export_matches_rows,
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
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'ALL TESTS PASSED' if failed == 0 else 'SOME TESTS FAILED'}")
