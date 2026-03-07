"""
Aurora Document Understanding Pipeline
========================================
Single-pass orchestrator that chains all document understanding stages:

  Classification → Hierarchy → Table Reconstruction → Field Extraction
  → Ontology Mapping → Edit Patches → Output Formatting

Supports multiple input types:
  - raw_ocr: OCRBlock objects with bounding boxes
  - plain_text: Raw text (assumes top→bottom order)
  - textract: AWS Textract JSON blocks
  - docai: Google Document AI JSON response
  - azure: Azure Form Recognizer JSON response
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from document_understanding.models import (
    DocumentUnderstandingResult,
    EditPatch,
    ExplainabilityTrace,
    OCRBlock,
    OutputBundle,
    PageResult,
)
from document_understanding.document_classifier import classify_document
from document_understanding.hierarchy_extractor import extract_hierarchy
from document_understanding.table_reconstructor import (
    extract_tables_from_blocks,
    extract_tables_from_text,
    parse_numeric,
)
from document_understanding.field_extractor import (
    extract_fields_from_blocks,
    extract_fields_from_text,
)
from document_understanding.ontology import apply_ontology
from document_understanding.edit_patch_generator import generate_all_patches
from document_understanding.output_formatter import format_outputs


class AuroraDocumentPipeline:
    """
    Aurora Universal Document Understanding Pipeline.
    Single-pass, deterministic extraction with legacy antigravity mode enforced.
    """

    SUPPORTED_SOURCES = ("raw_ocr", "plain_text", "textract", "docai", "azure")

    def __init__(self, legacy_mode: bool = True):
        self.legacy_mode = legacy_mode
        self._legacy_rules_applied: List[str] = []

    def process(
        self,
        input_data: Any,
        source_type: str = "plain_text",
        expected_total: Optional[float] = None,
    ) -> DocumentUnderstandingResult:
        """
        Run the full document understanding pipeline.

        Args:
            input_data:
                - If source_type="raw_ocr": List[OCRBlock] or List[dict] with {text, bbox, ...}
                - If source_type="plain_text": str
                - If source_type="textract": List[dict] (Textract blocks)
                - If source_type="docai": dict (Document AI response)
                - If source_type="azure": dict (Azure Form Recognizer response)
            source_type: One of SUPPORTED_SOURCES
            expected_total: Optional known total for validation

        Returns:
            DocumentUnderstandingResult
        """
        if source_type not in self.SUPPORTED_SOURCES:
            raise ValueError(f"Unsupported source_type: {source_type}. Must be one of {self.SUPPORTED_SOURCES}")

        # Dispatch to appropriate handler
        if source_type == "textract":
            return self._process_textract(input_data, expected_total)
        elif source_type == "docai":
            return self._process_docai(input_data, expected_total)
        elif source_type == "azure":
            return self._process_azure(input_data, expected_total)
        elif source_type == "raw_ocr":
            return self._process_raw_ocr(input_data, expected_total)
        else:
            return self._process_plain_text(input_data, expected_total)

    def _process_plain_text(
        self,
        text: str,
        expected_total: Optional[float] = None,
    ) -> DocumentUnderstandingResult:
        """Process plain text input through all pipeline stages."""
        all_traces: List[ExplainabilityTrace] = []
        warnings: List[str] = []

        # ── Stage 1: Classification ───────────────────────────────────────
        doc_type, class_traces = classify_document(plain_text=text)
        all_traces.extend(class_traces)

        # ── Stage 2: Hierarchy ────────────────────────────────────────────
        hierarchy, hier_traces = extract_hierarchy(plain_text=text)
        all_traces.extend(hier_traces)

        # ── Stage 3: Table Extraction ─────────────────────────────────────
        tables, table_traces = extract_tables_from_text(text, expected_total)
        all_traces.extend(table_traces)

        # ── Stage 4: Field Extraction ─────────────────────────────────────
        fields, candidates, field_traces = extract_fields_from_text(text)
        all_traces.extend(field_traces)

        # ── Stage 5: Ontology Mapping ─────────────────────────────────────
        mapped_fields, onto_traces = apply_ontology(fields, text)
        all_traces.extend(onto_traces)

        # Auto-detect expected total from fields if not provided
        if expected_total is None and "total" in mapped_fields:
            total_val = parse_numeric(str(mapped_fields["total"].value))
            if total_val:
                expected_total = total_val

        # ── Stage 6: Edit Patches ─────────────────────────────────────────
        legacy_mismatches = self._check_legacy_compliance()
        patches = generate_all_patches(
            tables=tables,
            fields=mapped_fields,
            candidate_fields=candidates,
            legacy_mismatches=legacy_mismatches if legacy_mismatches else None,
        )

        # ── Stage 7: Warnings ─────────────────────────────────────────────
        for fname, fval in mapped_fields.items():
            if fval.confidence < 0.6:
                warnings.append(f"low_confidence_in_{fname}")

        if not text.strip():
            warnings.append("empty_input_text")

        # ── Assemble result ───────────────────────────────────────────────
        page = PageResult(
            page_num=1,
            hierarchy=hierarchy,
            fields=mapped_fields,
            candidate_fields=candidates,
            tables=tables,
            explainability=all_traces,
        )

        result = DocumentUnderstandingResult(
            document_type=doc_type,
            pages=[page],
            edit_patches=patches,
            warnings=warnings,
            reliability="medium" if not any(b for b in []) else "high",
        )

        # ── Stage 8: Output Formatting ────────────────────────────────────
        result.outputs = format_outputs(result)

        return result

    def _process_raw_ocr(
        self,
        blocks_input: Union[List[OCRBlock], List[Dict]],
        expected_total: Optional[float] = None,
    ) -> DocumentUnderstandingResult:
        """Process raw OCR blocks."""
        all_traces: List[ExplainabilityTrace] = []
        warnings: List[str] = []

        # Convert dicts to OCRBlock if needed
        blocks: List[OCRBlock] = []
        for b in blocks_input:
            if isinstance(b, OCRBlock):
                blocks.append(b)
            elif isinstance(b, dict):
                blocks.append(OCRBlock(**b))

        if not blocks:
            return DocumentUnderstandingResult(
                warnings=["empty_input_blocks"],
                reliability="low",
            )

        # Classification
        doc_type, class_traces = classify_document(blocks=blocks)
        all_traces.extend(class_traces)

        # Hierarchy
        hierarchy, hier_traces = extract_hierarchy(blocks=blocks)
        all_traces.extend(hier_traces)

        # Tables
        tables, table_traces = extract_tables_from_blocks(blocks, expected_total)
        all_traces.extend(table_traces)

        # Fields
        fields, candidates, field_traces = extract_fields_from_blocks(blocks)
        all_traces.extend(field_traces)

        # Ontology
        full_text = " ".join(b.text for b in blocks if b.text)
        mapped_fields, onto_traces = apply_ontology(fields, full_text)
        all_traces.extend(onto_traces)

        # Patches
        legacy_mismatches = self._check_legacy_compliance()
        patches = generate_all_patches(
            tables=tables,
            fields=mapped_fields,
            candidate_fields=candidates,
            legacy_mismatches=legacy_mismatches if legacy_mismatches else None,
        )

        # Warnings
        for fname, fval in mapped_fields.items():
            if fval.confidence < 0.6:
                warnings.append(f"low_confidence_in_{fname}")

        page = PageResult(
            page_num=1,
            blocks=blocks,
            hierarchy=hierarchy,
            fields=mapped_fields,
            candidate_fields=candidates,
            tables=tables,
            explainability=all_traces,
        )

        result = DocumentUnderstandingResult(
            document_type=doc_type,
            pages=[page],
            edit_patches=patches,
            warnings=warnings,
            reliability="high",
        )

        result.outputs = format_outputs(result)
        return result

    def _process_textract(
        self,
        textract_blocks: List[Dict],
        expected_total: Optional[float] = None,
    ) -> DocumentUnderstandingResult:
        """Process AWS Textract output."""
        from document_understanding.providers.textract_processor import process_textract_output

        ocr_blocks, tables, fields, traces = process_textract_output(
            textract_blocks, expected_total
        )

        # Classification
        doc_type, class_traces = classify_document(blocks=ocr_blocks)
        traces.extend(class_traces)

        # Hierarchy
        hierarchy, hier_traces = extract_hierarchy(blocks=ocr_blocks)
        traces.extend(hier_traces)

        # Ontology
        full_text = " ".join(b.text for b in ocr_blocks if b.text)
        mapped_fields, onto_traces = apply_ontology(fields, full_text)
        traces.extend(onto_traces)

        # Patches
        patches = generate_all_patches(tables=tables, fields=mapped_fields)

        # Warnings
        warnings = [f"low_confidence_in_{k}" for k, v in mapped_fields.items() if v.confidence < 0.6]

        page = PageResult(
            page_num=1,
            blocks=ocr_blocks,
            hierarchy=hierarchy,
            fields=mapped_fields,
            tables=tables,
            explainability=traces,
        )

        result = DocumentUnderstandingResult(
            document_type=doc_type,
            pages=[page],
            edit_patches=patches,
            warnings=warnings,
        )

        result.outputs = format_outputs(result)
        return result

    def _process_docai(
        self,
        docai_response: Dict,
        expected_total: Optional[float] = None,
    ) -> DocumentUnderstandingResult:
        """Process Google Document AI output."""
        from document_understanding.providers.docai_processor import process_docai_output

        ocr_blocks, tables, fields, traces, operator_note = process_docai_output(
            docai_response, expected_total
        )

        doc_type, class_traces = classify_document(blocks=ocr_blocks)
        traces.extend(class_traces)

        hierarchy, hier_traces = extract_hierarchy(blocks=ocr_blocks)
        traces.extend(hier_traces)

        full_text = " ".join(b.text for b in ocr_blocks if b.text)
        mapped_fields, onto_traces = apply_ontology(fields, full_text)
        traces.extend(onto_traces)

        patches = generate_all_patches(tables=tables, fields=mapped_fields)
        warnings = [f"low_confidence_in_{k}" for k, v in mapped_fields.items() if v.confidence < 0.6]
        if operator_note and operator_note != "No manual checks required.":
            warnings.append(f"operator_note: {operator_note}")

        page = PageResult(
            page_num=1,
            blocks=ocr_blocks,
            hierarchy=hierarchy,
            fields=mapped_fields,
            tables=tables,
            explainability=traces,
        )

        result = DocumentUnderstandingResult(
            document_type=doc_type,
            pages=[page],
            edit_patches=patches,
            warnings=warnings,
        )

        result.outputs = format_outputs(result)
        return result

    def _process_azure(
        self,
        azure_response: Dict,
        expected_total: Optional[float] = None,
    ) -> DocumentUnderstandingResult:
        """Process Azure Form Recognizer output."""
        from document_understanding.providers.azure_form_processor import process_azure_output

        ocr_blocks, tables, fields, traces = process_azure_output(
            azure_response, expected_total
        )

        doc_type, class_traces = classify_document(blocks=ocr_blocks)
        traces.extend(class_traces)

        hierarchy, hier_traces = extract_hierarchy(blocks=ocr_blocks)
        traces.extend(hier_traces)

        full_text = " ".join(b.text for b in ocr_blocks if b.text)
        mapped_fields, onto_traces = apply_ontology(fields, full_text)
        traces.extend(onto_traces)

        patches = generate_all_patches(tables=tables, fields=mapped_fields)
        warnings = [f"low_confidence_in_{k}" for k, v in mapped_fields.items() if v.confidence < 0.6]

        page = PageResult(
            page_num=1,
            blocks=ocr_blocks,
            hierarchy=hierarchy,
            fields=mapped_fields,
            tables=tables,
            explainability=traces,
        )

        result = DocumentUnderstandingResult(
            document_type=doc_type,
            pages=[page],
            edit_patches=patches,
            warnings=warnings,
        )

        result.outputs = format_outputs(result)
        return result

    def _check_legacy_compliance(self) -> List[str]:
        """
        Check if the pipeline configuration matches legacy antigravity rules.
        Returns list of rules that were re-enabled if any mismatches found.
        """
        mismatches = []

        if not self.legacy_mode:
            mismatches.append("legacy_mode_disabled")

        # Record which rules were applied
        expected_rules = [
            "top_to_bottom_reading_order",
            "left_to_right_band_sort",
            "3plus_alignment_table_detection",
            "header_synonym_inference",
            "type_heuristic_reconstruction",
            "description_gt_quantity_gt_price_priority",
        ]

        for rule in expected_rules:
            if rule not in self._legacy_rules_applied:
                self._legacy_rules_applied.append(rule)

        return mismatches
