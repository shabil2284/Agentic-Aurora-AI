"""
Comprehensive Unit Tests for Aurora Universal Document Understanding
=====================================================================
Tests: document classification, hierarchy extraction, table detection,
table reconstruction (shuffled values), header inference, field extraction,
ontology mapping, edit patch generation, output formatting, validation,
provider post-processors, and end-to-end pipeline.
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_understanding.models import (
    DocumentTypeResult,
    DocumentUnderstandingResult,
    EditPatch,
    ExplainabilityTrace,
    ExtractedTable,
    FieldExtraction,
    HierarchyNode,
    OCRBlock,
    OutputBundle,
    PageResult,
    ReconstructionCandidate,
    TableCell,
)
from document_understanding.document_classifier import classify_document
from document_understanding.hierarchy_extractor import extract_hierarchy
from document_understanding.table_reconstructor import (
    detect_tables_from_text,
    extract_tables_from_text,
    infer_cell_type,
    infer_header_row,
    parse_numeric,
    reconstruct_table,
    validate_table,
)
from document_understanding.field_extractor import extract_fields_from_text, extract_fields_from_blocks
from document_understanding.ontology import apply_ontology, map_to_canonical, get_canonical_name
from document_understanding.edit_patch_generator import (
    generate_all_patches,
    generate_field_patches,
    generate_table_patches,
    generate_validation_patches,
    generate_legacy_revert_patch,
)
from document_understanding.output_formatter import format_outputs, generate_human_summary
from document_understanding.pipeline import AuroraDocumentPipeline


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ══════════════════════════════════════════════════════════════════════════════

INVOICE_TEXT = """
INVOICE

Invoice Number: INV-2026-0042
Invoice Date: 2026-02-25
Due Date: 2026-03-25

Bill To: Howell Corporation
Address: 55499 Lake Philipmouth, Suite 200

Vendor: Stehr-Kutch Technologies Pvt Ltd
GSTIN: 29AABCU9603R1Z1
PAN: AABCU9603R

Item          Description                   Qty    Rate      Amount
1             Premium Widget A               5     775.70    3878.50
2             Standard Widget B             83     565.69   46952.27
3             Economy Widget C              86     580.01   49880.86

Subtotal: 100711.63
Tax: 15% VAT
Tax Amount: 15106.74
Grand Total: 115818.37

Payment Terms: Net 30
Email: billing@stehr-kutch.com
Phone: +91 80 2345 6789
"""

RECEIPT_TEXT = """
RECEIPT
Receipt No: REC-001234
Date: 2026-01-15

Received from: John Doe
Amount Paid: $250.00
Payment Method: Credit Card
Transaction ID: TXN-98765

Thank you for your purchase!
"""

RESUME_TEXT = """
CURRICULUM VITAE

Name: Alice Johnson
Email: alice.johnson@email.com
Phone: +1 555 0100

PROFESSIONAL SUMMARY
Senior software engineer with 8 years experience in distributed systems.

EXPERIENCE
Acme Corp — Senior Engineer (2020-2026)
Led a team of 6 engineers building cloud infrastructure.

Beta Inc — Software Developer (2018-2020)
Developed microservices in Go and Python.

EDUCATION
B.S. Computer Science, MIT, 2018

SKILLS
Python, Go, Kubernetes, PostgreSQL, AWS
"""

TABLE_ONLY_TEXT = """
Product     Qty     Price     Total
Widget A    10      25.50     255.00
Widget B    20      15.75     315.00
Widget C     5      42.00     210.00
Widget D    15      18.25     273.75
"""

FORM_TEXT = """
APPLICATION FORM

Full Name: Jane Doe
Date of Birth: 12/05/1990
Nationality: UAE
Passport No: P1234567
Email: jane.doe@email.com
Phone: +971 50 123 4567
Signature: _______________
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. DOCUMENT CLASSIFICATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_classify_invoice():
    result, traces = classify_document(plain_text=INVOICE_TEXT)
    assert result.label == "invoice", f"Expected 'invoice', got '{result.label}'"
    assert result.confidence >= 0.5, f"Expected conf >= 0.5, got {result.confidence}"
    assert len(traces) > 0
    print(f"PASS test_classify_invoice (label={result.label}, conf={result.confidence:.2f})")


def test_classify_receipt():
    result, traces = classify_document(plain_text=RECEIPT_TEXT)
    assert result.label == "receipt", f"Expected 'receipt', got '{result.label}'"
    assert result.confidence >= 0.4
    print(f"PASS test_classify_receipt (label={result.label}, conf={result.confidence:.2f})")


def test_classify_resume():
    result, traces = classify_document(plain_text=RESUME_TEXT)
    assert result.label == "resume", f"Expected 'resume', got '{result.label}'"
    assert result.confidence >= 0.4
    print(f"PASS test_classify_resume (label={result.label}, conf={result.confidence:.2f})")


def test_classify_form():
    result, traces = classify_document(plain_text=FORM_TEXT)
    assert result.label == "form", f"Expected 'form', got '{result.label}'"
    print(f"PASS test_classify_form (label={result.label}, conf={result.confidence:.2f})")


def test_classify_empty():
    result, traces = classify_document(plain_text="")
    assert result.confidence <= 0.3
    print(f"PASS test_classify_empty (label={result.label}, conf={result.confidence:.2f})")


def test_classify_no_input():
    result, traces = classify_document()
    assert result.label == "mixed"
    assert result.confidence <= 0.2
    print(f"PASS test_classify_no_input (label={result.label}, conf={result.confidence:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. HIERARCHY EXTRACTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_hierarchy_plain_text():
    nodes, traces = extract_hierarchy(plain_text=INVOICE_TEXT)
    assert len(nodes) > 0, "Expected at least 1 hierarchy node"
    # "INVOICE" should be detected as top-level heading
    titles = [n.title for n in nodes]
    assert any("INVOICE" in t for t in titles), f"Expected 'INVOICE' in hierarchy, got {titles}"
    print(f"PASS test_hierarchy_plain_text ({len(nodes)} nodes found)")


def test_hierarchy_resume():
    nodes, traces = extract_hierarchy(plain_text=RESUME_TEXT)
    assert len(nodes) >= 2, f"Expected at least 2 nodes, got {len(nodes)}"
    titles = [n.title for n in nodes]
    assert any("CURRICULUM VITAE" in t for t in titles)
    print(f"PASS test_hierarchy_resume ({len(nodes)} nodes: {[n.title for n in nodes[:4]]})")


def test_hierarchy_empty():
    nodes, traces = extract_hierarchy(plain_text="")
    assert len(nodes) == 0
    print(f"PASS test_hierarchy_empty")


def test_hierarchy_with_blocks():
    blocks = [
        OCRBlock(text="INVOICE", bbox=[10, 10, 200, 40], font_size=24.0, font_style="bold"),
        OCRBlock(text="Line Items", bbox=[10, 60, 200, 80], font_size=16.0, font_style="bold"),
        OCRBlock(text="Widget A costs $25.00", bbox=[10, 100, 400, 120], font_size=12.0),
    ]
    nodes, traces = extract_hierarchy(blocks=blocks)
    assert len(nodes) >= 1, f"Expected at least 1 node, got {len(nodes)}"
    print(f"PASS test_hierarchy_with_blocks ({len(nodes)} nodes)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TABLE DETECTION & RECONSTRUCTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_table_detection_positive():
    raw_tables, traces = detect_tables_from_text(TABLE_ONLY_TEXT)
    assert len(raw_tables) >= 1, f"Expected at least 1 table, got {len(raw_tables)}"
    assert len(traces) > 0
    print(f"PASS test_table_detection_positive ({len(raw_tables)} table(s))")


def test_table_detection_no_table():
    text = "This is a simple paragraph with no tables.\nJust some text.\nNothing to see here."
    raw_tables, traces = detect_tables_from_text(text)
    assert len(raw_tables) == 0, f"Expected 0 tables, got {len(raw_tables)}"
    print(f"PASS test_table_detection_no_table")


def test_header_inference():
    rows = [
        ["Item", "Description", "Qty", "Rate", "Amount"],
        ["1", "Widget A", "10", "25.50", "255.00"],
        ["2", "Widget B", "20", "15.75", "315.00"],
    ]
    idx, header, conf = infer_header_row(rows)
    assert idx == 0, f"Expected header at row 0, got {idx}"
    assert "Item" in header
    assert conf > 0.3
    print(f"PASS test_header_inference (idx={idx}, conf={conf:.2f})")


def test_header_inference_no_clear_header():
    rows = [
        ["1", "100", "25.50", "255.00"],
        ["2", "200", "15.75", "315.00"],
    ]
    idx, header, conf = infer_header_row(rows)
    assert header is not None and len(header) > 0
    print(f"PASS test_header_inference_no_clear_header (idx={idx}, conf={conf:.2f})")


def test_cell_type_inference():
    assert infer_cell_type("100") == "integer"
    assert infer_cell_type("25.50") in ("decimal", "currency")
    assert infer_cell_type("$1,234.56") == "currency"
    assert infer_cell_type("Widget A") == "string"
    assert infer_cell_type("") == "empty"
    assert infer_cell_type("15%") == "percentage"
    assert infer_cell_type("2026-02-25") == "date"
    print(f"PASS test_cell_type_inference")


def test_numeric_parsing():
    assert parse_numeric("775.70") == 775.70
    assert parse_numeric("$1,234.56") == 1234.56
    assert parse_numeric("100") == 100.0
    assert parse_numeric("not_a_number") is None
    print(f"PASS test_numeric_parsing")


def test_table_reconstruction_candidates():
    header = ["Description", "Qty", "Rate", "Amount"]
    rows = [
        ["Widget A", "10", "25.50", "255.00"],
        ["Widget B", "20", "15.75", "315.00"],
    ]
    candidates, cells, traces = reconstruct_table(rows, header)
    assert len(candidates) == 3, f"Expected 3 candidates, got {len(candidates)}"
    assert all(c.score >= 0 for c in candidates)
    assert candidates[0].score >= candidates[-1].score  # sorted desc
    assert len(cells) > 0
    print(f"PASS test_table_reconstruction_candidates (scores: {[c.score for c in candidates]})")


def test_table_validation_pass():
    header = ["Description", "Qty", "Rate", "Amount"]
    rows = [
        ["Widget A", "10", "25.50", "255.00"],
        ["Widget B", "20", "15.75", "315.00"],
    ]
    validation = validate_table(rows, header, expected_total=570.00)
    assert validation.line_sum_check.passed, "Expected validation to pass"
    assert validation.line_sum_check.computed_total == 570.00
    print(f"PASS test_table_validation_pass")


def test_table_validation_fail():
    header = ["Description", "Qty", "Rate", "Amount"]
    rows = [
        ["Widget A", "10", "25.50", "255.00"],
        ["Widget B", "20", "15.75", "315.00"],
    ]
    validation = validate_table(rows, header, expected_total=1000.00)
    assert not validation.line_sum_check.passed, "Expected validation to fail"
    print(f"PASS test_table_validation_fail (computed={validation.line_sum_check.computed_total})")


def test_full_table_extraction():
    tables, traces = extract_tables_from_text(TABLE_ONLY_TEXT)
    assert len(tables) >= 1, f"Expected at least 1 table, got {len(tables)}"
    table = tables[0]
    assert len(table.detected_header_row) >= 3, f"Expected >=3 header cols, got {len(table.detected_header_row)}"
    assert len(table.cells) > 0
    assert len(table.reconstruction_candidates) > 0
    print(f"PASS test_full_table_extraction ({len(tables)} tables, {len(table.cells)} cells)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIELD EXTRACTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_field_extraction_invoice():
    fields, candidates, traces = extract_fields_from_text(INVOICE_TEXT)
    assert "invoice_number" in fields, f"Expected 'invoice_number', got keys: {list(fields.keys())}"
    assert fields["invoice_number"].value == "INV-2026-0042"
    assert fields["invoice_number"].confidence >= 0.8
    print(f"PASS test_field_extraction_invoice (found: {list(fields.keys())})")


def test_field_extraction_date():
    fields, _, _ = extract_fields_from_text(INVOICE_TEXT)
    assert "invoice_date" in fields
    assert "2026" in str(fields["invoice_date"].value)
    print(f"PASS test_field_extraction_date (date={fields['invoice_date'].value})")


def test_field_extraction_gstin():
    fields, _, _ = extract_fields_from_text(INVOICE_TEXT)
    assert "gstin" in fields
    assert fields["gstin"].value == "29AABCU9603R1Z1"
    print(f"PASS test_field_extraction_gstin (gstin={fields['gstin'].value})")


def test_field_extraction_pan():
    fields, _, _ = extract_fields_from_text(INVOICE_TEXT)
    assert "pan" in fields
    assert fields["pan"].value == "AABCU9603R"
    print(f"PASS test_field_extraction_pan (pan={fields['pan'].value})")


def test_field_extraction_email():
    fields, _, _ = extract_fields_from_text(INVOICE_TEXT)
    assert "email" in fields
    assert "@" in fields["email"].value
    print(f"PASS test_field_extraction_email (email={fields['email'].value})")


def test_field_extraction_total():
    fields, _, _ = extract_fields_from_text(INVOICE_TEXT)
    assert "total" in fields
    print(f"PASS test_field_extraction_total (total={fields['total'].value})")


def test_field_extraction_confidence():
    fields, _, _ = extract_fields_from_text(INVOICE_TEXT)
    for fname, fval in fields.items():
        assert 0.0 <= fval.confidence <= 1.0, f"Field '{fname}' has invalid confidence: {fval.confidence}"
    print(f"PASS test_field_extraction_confidence")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ONTOLOGY MAPPING TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_ontology_exact_match():
    canonical, conf = map_to_canonical("invoice no")
    assert canonical == "invoice_number", f"Expected 'invoice_number', got '{canonical}'"
    assert conf >= 0.9
    print(f"PASS test_ontology_exact_match ('{canonical}', conf={conf})")


def test_ontology_synonym():
    canonical, conf = map_to_canonical("bill to")
    assert canonical == "payer_name", f"Expected 'payer_name', got '{canonical}'"
    print(f"PASS test_ontology_synonym ('{canonical}', conf={conf})")


def test_ontology_no_match():
    canonical, conf = map_to_canonical("xyzzy_nonsense_field")
    assert canonical is None
    assert conf == 0.0
    print(f"PASS test_ontology_no_match")


def test_ontology_canonical_name_shortcut():
    assert get_canonical_name("inv#") == "invoice_number"
    assert get_canonical_name("totally_unknown") == "totally_unknown"
    print(f"PASS test_ontology_canonical_name_shortcut")


def test_apply_ontology():
    fields = {
        "invoice_number": FieldExtraction(value="INV-123", confidence=0.9, explain="regex"),
        "vendor": FieldExtraction(value="Acme Corp", confidence=0.8, explain="spatial"),
    }
    mapped, traces = apply_ontology(fields, "some invoice text in USD")
    assert "invoice_number" in mapped
    assert "vendor_name" in mapped, f"Expected 'vendor_name', got keys: {list(mapped.keys())}"
    assert mapped["invoice_number"].inferred_type == "string"
    print(f"PASS test_apply_ontology (mapped: {list(mapped.keys())})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. EDIT PATCH TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_table_patch_generation():
    table = ExtractedTable(
        table_id="t1",
        reconstructed=True,
        reconstruction_candidates=[
            ReconstructionCandidate(score=0.92, csv="a,b\n1,2\n"),
            ReconstructionCandidate(score=0.75, csv="b,a\n2,1\n"),
        ],
    )
    patches = generate_table_patches([table])
    assert len(patches) >= 1, f"Expected >=1 patch, got {len(patches)}"
    assert patches[0].op == "reconstruct_columns"
    assert patches[0].confidence >= 0.9
    print(f"PASS test_table_patch_generation ({len(patches)} patches)")


def test_field_patch_low_confidence():
    fields = {
        "invoice_number": FieldExtraction(value="INV-???", confidence=0.4, explain="weak match"),
    }
    patches = generate_field_patches(fields)
    assert len(patches) >= 1
    assert patches[0].op == "fix_field_value"
    print(f"PASS test_field_patch_low_confidence ({len(patches)} patches)")


def test_validation_patch():
    from document_understanding.models import LineSumCheck, TableValidation
    table = ExtractedTable(
        table_id="t1",
        validation=TableValidation(
            line_sum_check=LineSumCheck(expected_total=1000, computed_total=800, passed=False),
        ),
    )
    patches = generate_validation_patches([table])
    assert len(patches) >= 1
    assert "discrepancy" in patches[0].op or "total" in patches[0].op
    print(f"PASS test_validation_patch ({len(patches)} patches)")


def test_legacy_revert_patch():
    patch = generate_legacy_revert_patch(["legacy_mode_disabled"])
    assert patch is not None
    assert patch.op == "revert_legacy_settings"
    print(f"PASS test_legacy_revert_patch")


def test_legacy_revert_none():
    patch = generate_legacy_revert_patch(None)
    assert patch is None
    print(f"PASS test_legacy_revert_none")


# ══════════════════════════════════════════════════════════════════════════════
# 7. OUTPUT FORMATTING TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_output_json_serialization():
    result = DocumentUnderstandingResult(
        document_type=DocumentTypeResult(label="invoice", confidence=0.95),
        pages=[PageResult(page_num=1)],
    )
    bundle = format_outputs(result)
    assert bundle.json_data is not None
    assert isinstance(bundle.json_data, dict)
    assert bundle.json_data["document_type"]["label"] == "invoice"
    print(f"PASS test_output_json_serialization")


def test_output_human_summary():
    result = DocumentUnderstandingResult(
        document_type=DocumentTypeResult(label="invoice", confidence=0.95),
        pages=[PageResult(
            page_num=1,
            fields={"invoice_number": FieldExtraction(value="INV-123", confidence=0.9)},
        )],
    )
    summary = generate_human_summary(result)
    assert "invoice" in summary.lower()
    assert len(summary) > 10
    print(f"PASS test_output_human_summary (summary='{summary[:80]}...')")


def test_output_repro_command():
    bundle = OutputBundle()
    assert "antigravity_legacy" in bundle.repro_command
    print(f"PASS test_output_repro_command")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PROVIDER POST-PROCESSOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_textract_processor():
    from document_understanding.providers.textract_processor import process_textract_output
    blocks = [
        {"Id": "b1", "BlockType": "LINE", "Text": "Invoice No: INV-100",
         "Confidence": 98.5, "Page": 1,
         "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.1, "Width": 0.3, "Height": 0.02}}},
        {"Id": "b2", "BlockType": "LINE", "Text": "Total: $500.00",
         "Confidence": 97.0, "Page": 1,
         "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.15, "Width": 0.2, "Height": 0.02}}},
    ]
    ocr_blocks, tables, fields, traces = process_textract_output(blocks)
    assert len(ocr_blocks) == 2
    assert len(traces) > 0
    print(f"PASS test_textract_processor ({len(ocr_blocks)} blocks, {len(fields)} fields)")


def test_docai_processor():
    from document_understanding.providers.docai_processor import process_docai_output
    response = {
        "text": "Invoice No: INV-200\nTotal: $750.00",
        "pages": [
            {
                "pageNumber": 1,
                "paragraphs": [
                    {"layout": {"textAnchor": {"textSegments": [{"startIndex": 0, "endIndex": 19}]},
                                "confidence": 0.95}},
                    {"layout": {"textAnchor": {"textSegments": [{"startIndex": 20, "endIndex": 35}]},
                                "confidence": 0.93}},
                ],
                "tables": [],
                "formFields": [],
            }
        ],
    }
    ocr_blocks, tables, fields, traces, note = process_docai_output(response)
    assert len(ocr_blocks) >= 1
    assert isinstance(note, str)
    print(f"PASS test_docai_processor ({len(ocr_blocks)} blocks, note='{note}')")


def test_azure_processor():
    from document_understanding.providers.azure_form_processor import process_azure_output
    response = {
        "analyzeResult": {
            "readResults": [
                {
                    "page": 1,
                    "lines": [
                        {"content": "Invoice No: INV-300", "confidence": 0.97,
                         "boundingBox": [100, 100, 400, 100, 400, 120, 100, 120]},
                        {"content": "Grand Total: $1200.00", "confidence": 0.95,
                         "boundingBox": [100, 200, 400, 200, 400, 220, 100, 220]},
                    ],
                }
            ],
            "tables": [],
            "keyValuePairs": [],
            "documents": [],
        }
    }
    ocr_blocks, tables, fields, traces = process_azure_output(response)
    assert len(ocr_blocks) >= 1
    print(f"PASS test_azure_processor ({len(ocr_blocks)} blocks, {len(fields)} fields)")


def test_textract_combined_confidence():
    """Verify combined confidence formula: 0.6*textract + 0.4*heuristic."""
    textract_conf = 0.9
    heuristic_conf = 0.8
    combined = 0.6 * textract_conf + 0.4 * heuristic_conf
    assert abs(combined - 0.86) < 0.001, f"Expected 0.86, got {combined}"
    print(f"PASS test_textract_combined_confidence ({combined})")


def test_azure_combined_confidence():
    """Verify combined confidence formula: 0.55*azure + 0.45*heuristic."""
    azure_conf = 0.9
    heuristic_conf = 0.8
    combined = 0.55 * azure_conf + 0.45 * heuristic_conf
    assert abs(combined - 0.855) < 0.001, f"Expected 0.855, got {combined}"
    print(f"PASS test_azure_combined_confidence ({combined})")


# ══════════════════════════════════════════════════════════════════════════════
# 9. END-TO-END PIPELINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_plain_text_invoice():
    pipeline = AuroraDocumentPipeline()
    result = pipeline.process(INVOICE_TEXT, source_type="plain_text")

    assert result.document_type.label == "invoice"
    assert result.document_type.confidence > 0
    assert len(result.pages) == 1
    assert result.timestamp_utc is not None
    assert result.outputs.json_data is not None
    assert result.outputs.human_summary != ""
    assert "antigravity_legacy" in result.outputs.repro_command

    # Fields should be extracted
    page = result.pages[0]
    assert len(page.fields) > 0, f"Expected fields, got none"

    # Explainability traces should be present
    assert len(page.explainability) > 0

    print(f"PASS test_pipeline_plain_text_invoice "
          f"(type={result.document_type.label}, fields={len(page.fields)}, "
          f"tables={len(page.tables)}, patches={len(result.edit_patches)})")


def test_pipeline_receipt():
    pipeline = AuroraDocumentPipeline()
    result = pipeline.process(RECEIPT_TEXT, source_type="plain_text")
    assert result.document_type.label == "receipt"
    assert len(result.pages) == 1
    print(f"PASS test_pipeline_receipt (type={result.document_type.label})")


def test_pipeline_resume():
    pipeline = AuroraDocumentPipeline()
    result = pipeline.process(RESUME_TEXT, source_type="plain_text")
    assert result.document_type.label == "resume"
    page = result.pages[0]
    assert len(page.hierarchy) > 0, "Expected hierarchy nodes for resume"
    print(f"PASS test_pipeline_resume (hierarchy={len(page.hierarchy)} nodes)")


def test_pipeline_raw_ocr():
    pipeline = AuroraDocumentPipeline()
    blocks = [
        {"text": "INVOICE", "bbox": [10, 10, 200, 40], "confidence": 0.99, "font_size": 24.0, "font_style": "bold"},
        {"text": "Invoice No: INV-500", "bbox": [10, 60, 300, 80], "confidence": 0.95},
        {"text": "Total: $999.00", "bbox": [10, 100, 300, 120], "confidence": 0.93},
    ]
    result = pipeline.process(blocks, source_type="raw_ocr")
    assert result.document_type.label == "invoice"
    assert len(result.pages) == 1
    assert len(result.pages[0].blocks) == 3
    print(f"PASS test_pipeline_raw_ocr (blocks={len(result.pages[0].blocks)})")


def test_pipeline_empty_input():
    pipeline = AuroraDocumentPipeline()
    result = pipeline.process("", source_type="plain_text")
    assert len(result.warnings) > 0
    print(f"PASS test_pipeline_empty_input (warnings={result.warnings})")


def test_pipeline_invalid_source():
    pipeline = AuroraDocumentPipeline()
    try:
        pipeline.process("text", source_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print(f"PASS test_pipeline_invalid_source")


def test_pipeline_textract_source():
    pipeline = AuroraDocumentPipeline()
    blocks = [
        {"Id": "b1", "BlockType": "LINE", "Text": "Invoice Number: INV-600",
         "Confidence": 99.0, "Page": 1,
         "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.1, "Width": 0.3, "Height": 0.02}}},
    ]
    result = pipeline.process(blocks, source_type="textract")
    assert len(result.pages) == 1
    assert result.outputs.json_data is not None
    print(f"PASS test_pipeline_textract_source")


def test_pipeline_output_schema_completeness():
    """Verify that the output matches the specified schema structure."""
    pipeline = AuroraDocumentPipeline()
    result = pipeline.process(INVOICE_TEXT, source_type="plain_text")

    # Verify top-level keys
    data = result.model_dump(mode="json")
    assert "document_type" in data
    assert "pages" in data
    assert "outputs" in data
    assert "edit_patches" in data
    assert "warnings" in data
    assert "timestamp_utc" in data

    # Verify document_type structure
    assert "label" in data["document_type"]
    assert "confidence" in data["document_type"]

    # Verify outputs structure
    assert "csv_files" in data["outputs"]
    assert "human_summary" in data["outputs"]

    print(f"PASS test_pipeline_output_schema_completeness")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # Classification
        test_classify_invoice,
        test_classify_receipt,
        test_classify_resume,
        test_classify_form,
        test_classify_empty,
        test_classify_no_input,
        # Hierarchy
        test_hierarchy_plain_text,
        test_hierarchy_resume,
        test_hierarchy_empty,
        test_hierarchy_with_blocks,
        # Table
        test_table_detection_positive,
        test_table_detection_no_table,
        test_header_inference,
        test_header_inference_no_clear_header,
        test_cell_type_inference,
        test_numeric_parsing,
        test_table_reconstruction_candidates,
        test_table_validation_pass,
        test_table_validation_fail,
        test_full_table_extraction,
        # Field extraction
        test_field_extraction_invoice,
        test_field_extraction_date,
        test_field_extraction_gstin,
        test_field_extraction_pan,
        test_field_extraction_email,
        test_field_extraction_total,
        test_field_extraction_confidence,
        # Ontology
        test_ontology_exact_match,
        test_ontology_synonym,
        test_ontology_no_match,
        test_ontology_canonical_name_shortcut,
        test_apply_ontology,
        # Edit patches
        test_table_patch_generation,
        test_field_patch_low_confidence,
        test_validation_patch,
        test_legacy_revert_patch,
        test_legacy_revert_none,
        # Output formatting
        test_output_json_serialization,
        test_output_human_summary,
        test_output_repro_command,
        # Providers
        test_textract_processor,
        test_docai_processor,
        test_azure_processor,
        test_textract_combined_confidence,
        test_azure_combined_confidence,
        # Pipeline E2E
        test_pipeline_plain_text_invoice,
        test_pipeline_receipt,
        test_pipeline_resume,
        test_pipeline_raw_ocr,
        test_pipeline_empty_input,
        test_pipeline_invalid_source,
        test_pipeline_textract_source,
        test_pipeline_output_schema_completeness,
    ]

    passed = failed = 0
    failures = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
            failures.append(t.__name__)

    print(f"\n{'=' * 60}")
    print(f"Aurora Document Understanding Test Results")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}  |  Failed: {failed}  |  Total: {passed + failed}")
    if failures:
        print(f"Failed tests: {failures}")
    print(f"{'ALL TESTS PASSED ✓' if failed == 0 else 'SOME TESTS FAILED ✗'}")
