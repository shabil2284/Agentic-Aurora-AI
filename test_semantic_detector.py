"""
Aurora Semantic Detector — Unit Tests
Run: uv run python test_semantic_detector.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.semantic_detector import (
    detect_field_type,
    detect_column_type,
    derive_schema_from_records,
    clean_markdown_email,
    clean_markdown_url,
)

PASS = 0
FAIL = 0

def check(description, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  ✅ PASS  {description}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {description}")
        print(f"         Expected: {expected!r}")
        print(f"         Got:      {got!r}")
        FAIL += 1


print("\n=== Email Detection ===")
check("plain email", detect_field_type("careers@tascoutsourcing.com"), "Email")
check("email with +", detect_field_type("user+alias@domain.co.uk"), "Email")
check("markdown mailto link", detect_field_type("[email](mailto:jobs@hays.com)"), "Email")

print("\n=== Phone Detection ===")
check("UAE phone", detect_field_type("+971 4 559 5800"), "Phone")
check("10-digit", detect_field_type("0501234567"), "Phone")
check("dash-separated", detect_field_type("+1-800-555-0199"), "Phone")

print("\n=== Website Detection ===")
check("https url", detect_field_type("https://www.tasc.com"), "Website")
check("www prefix", detect_field_type("www.hays.ae"), "Website")
check(".com domain", detect_field_type("tasc.com"), "Website")

print("\n=== Company Name Detection (explicit suffix only) ===")
check("Ltd suffix", detect_field_type("Acme Ltd"), "Company Name")
check("LLC suffix", detect_field_type("Smith & Jones LLC"), "Company Name")
check("Technologies suffix", detect_field_type("InfraCore Technologies"), "Company Name")
check("Solutions suffix", detect_field_type("Gulf Solutions"), "Company Name")
check("generic text (no suffix)", detect_field_type("Dubai Marina"), "Other")  # must NOT map to Company Name

print("\n=== SNo Detection ===")
check("pure number 26", detect_field_type("26"), "SNo")
check("pure number 1", detect_field_type("1"), "SNo")
check("pure number 100", detect_field_type("100"), "SNo")

print("\n=== Address Detection ===")
check("street keyword", detect_field_type("123 Sheikh Zayed Road, Dubai"), "Address")
check("nagar keyword", detect_field_type("Koramangala Nagar, Bangalore"), "Address")

print("\n=== Markdown Link Cleaning ===")
check("mailto link", clean_markdown_email("[careers@tasc.com](mailto:careers@tasc.com)"), "careers@tasc.com")
check("https link", clean_markdown_url("[Visit](https://tasc.com)"), "https://tasc.com")

print("\n=== Column Type Voting ===")
emails = ["a@b.com", "c@d.org", "not-email", "e@f.io"]
check("email column vote", detect_column_type(emails), "Email")

companies = ["TASC Solutions", "Hays Ltd", "Acme Inc", "some text"]
check("company column vote", detect_column_type(companies), "Company Name")

print("\n=== Schema Derivation ===")
records = [
    {"SNo": "26", "Company Name": "TASC Outsourcing", "Email": "careers@tascoutsourcing.com"},
    {"SNo": "27", "Company Name": "Hays UAE", "Email": "dubai@hays.com"},
]
schema = derive_schema_from_records(records)
check("schema fields count", len(schema), 3)
check("SNo type", schema[0]["type"], "number")
check("Company Name field", schema[1]["field"], "Company Name")
check("Email field", schema[2]["field"], "Email")

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED ✅")
    sys.exit(0)
else:
    print("SOME TESTS FAILED ❌")
    sys.exit(1)
