"""
Unit tests for Aurora Semantic Completer
Tests: _merge_non_destructive logic (dict and list handling)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.semantic_completer import SemanticCompleter

def test_merge_dict_adds_missing_keys():
    current = {"Invoice No": "123", "Total": "$500"}
    semantic = {"Document Title": "Invoice", "Issuer Organization": "Acme Corp"}
    
    merged = SemanticCompleter._merge_non_destructive(current, semantic)
    
    assert merged["Invoice No"] == "123"
    assert merged["Total"] == "$500"
    assert merged["Document Title"] == "Invoice"
    assert merged["Issuer Organization"] == "Acme Corp"
    print("PASS test_merge_dict_adds_missing_keys")

def test_merge_dict_preserves_existing_keys():
    # 'issuer organization' already exists in current data, should not be overwritten
    current = {"Invoice No": "123", "Issuer Organization": "Original Corp"}
    semantic = {"Document Title": "Invoice", "Issuer Organization": "Acme Corp", "issue Date": "2024-01-01"}
    
    merged = SemanticCompleter._merge_non_destructive(current, semantic)
    
    assert merged["Invoice No"] == "123"
    assert merged["Issuer Organization"] == "Original Corp" # Must be preserved!
    assert merged["Document Title"] == "Invoice"
    assert merged["issue Date"] == "2024-01-01"
    print("PASS test_merge_dict_preserves_existing_keys")

def test_merge_list_applies_to_all_rows():
    current = [
        {"Item": "A", "Price": "10"},
        {"Item": "B", "Price": "20"}
    ]
    semantic = {"Document Title": "Receipt", "Location": "NY"}
    
    merged = SemanticCompleter._merge_non_destructive(current, semantic)
    
    assert len(merged) == 2
    for row in merged:
        assert row["Document Title"] == "Receipt"
        assert row["Location"] == "NY"
        assert "Item" in row
        assert "Price" in row
    print("PASS test_merge_list_applies_to_all_rows")

def test_merge_empty_semantic():
    current = {"Invoice No": "123"}
    merged = SemanticCompleter._merge_non_destructive(current, {})
    assert merged == current
    print("PASS test_merge_empty_semantic")

def test_case_insensitive_key_preservation():
    current = {"Employee Name": "John Smith"}
    semantic = {"employee name": "Johnny", "Designation": "Dev"}
    
    merged = SemanticCompleter._merge_non_destructive(current, semantic)
    
    assert "Employee Name" in merged
    assert merged["Employee Name"] == "John Smith"
    assert "employee name" not in merged
    assert merged["Designation"] == "Dev"
    print("PASS test_case_insensitive_key_preservation")


if __name__ == "__main__":
    test_merge_dict_adds_missing_keys()
    test_merge_dict_preserves_existing_keys()
    test_merge_list_applies_to_all_rows()
    test_merge_empty_semantic()
    test_case_insensitive_key_preservation()
    print("ALL TESTS PASSED")
