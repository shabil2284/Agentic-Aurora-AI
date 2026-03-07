"""
Google Document AI Post-Processor
===================================
Uses Document AI formFields/tableRows as priors. Validates headers (>50% numeric
cells → likely data row), synthesizes headers from ontology when missing.
Includes docai_node_refs for traceability and operator_note with recommended checks.
"""

from typing import Any, Dict, List, Optional, Tuple

from document_understanding.models import (
    ExplainabilityTrace,
    ExtractedTable,
    FieldExtraction,
    OCRBlock,
)
from document_understanding.table_reconstructor import (
    infer_cell_type,
    infer_header_row,
    reconstruct_table,
    validate_table,
    HEADER_SYNONYMS,
)
from document_understanding.field_extractor import extract_fields_from_text
from document_understanding.ontology import map_to_canonical


def process_docai_output(
    docai_response: Dict[str, Any],
    expected_total: Optional[float] = None,
) -> Tuple[List[OCRBlock], List[ExtractedTable], Dict[str, FieldExtraction], List[ExplainabilityTrace], str]:
    """
    Process Google Document AI output into Aurora format.

    Args:
        docai_response: Document AI JSON response with pages, blocks, formFields, tableRows
        expected_total: Expected invoice total for validation

    Returns:
        (ocr_blocks, tables, fields, traces, operator_note)
    """
    traces: List[ExplainabilityTrace] = []
    ocr_blocks: List[OCRBlock] = []
    operator_notes: List[str] = []

    pages = docai_response.get("pages", [])

    # 1. Ingest blocks/paragraphs
    for page_data in pages:
        page_num = page_data.get("pageNumber", 1)
        paragraphs = page_data.get("paragraphs", [])
        blocks = page_data.get("blocks", [])

        source = paragraphs if paragraphs else blocks
        for item in source:
            text = _get_docai_text(item, docai_response)
            bbox = _layout_to_bbox(item.get("layout", {}))
            conf = item.get("layout", {}).get("confidence", 1.0)

            if text.strip():
                ocr_blocks.append(OCRBlock(
                    id=f"docai_{len(ocr_blocks)}",
                    text=text.strip(),
                    bbox=bbox,
                    confidence=conf,
                    page=page_num,
                ))

    traces.append(ExplainabilityTrace(
        step="docai_ingestion",
        reason=f"Ingested {len(ocr_blocks)} text blocks from {len(pages)} Document AI page(s)",
        confidence=0.95,
    ))

    # 2. Process tables from tableRows
    tables = _process_docai_tables(pages, docai_response, expected_total, traces, operator_notes)

    # 3. Process form fields
    fields = _process_docai_form_fields(pages, docai_response, ocr_blocks, traces)

    operator_note = " | ".join(operator_notes) if operator_notes else "No manual checks required."

    return ocr_blocks, tables, fields, traces, operator_note


def _get_docai_text(element: Dict, full_response: Dict) -> str:
    """Extract text from Document AI element using textAnchor."""
    layout = element.get("layout", {})
    text_anchor = layout.get("textAnchor", element.get("textAnchor", {}))
    text_segments = text_anchor.get("textSegments", [])

    full_text = full_response.get("text", "")

    if text_segments and full_text:
        parts = []
        for seg in text_segments:
            start = int(seg.get("startIndex", 0))
            end = int(seg.get("endIndex", start))
            parts.append(full_text[start:end])
        return " ".join(parts)

    # Fallback
    return element.get("mentionText", element.get("text", ""))


def _layout_to_bbox(layout: Dict) -> List[float]:
    """Convert Document AI layout.boundingPoly to [x1, y1, x2, y2]."""
    bounding_poly = layout.get("boundingPoly", {})
    vertices = bounding_poly.get("normalizedVertices", bounding_poly.get("vertices", []))

    if vertices and len(vertices) >= 4:
        xs = [v.get("x", 0) * 1000 for v in vertices]  # Scale to pixel-like coords
        ys = [v.get("y", 0) * 1000 for v in vertices]
        return [min(xs), min(ys), max(xs), max(ys)]

    return [0, 0, 0, 0]


def _process_docai_tables(
    pages: List[Dict],
    full_response: Dict,
    expected_total: Optional[float],
    traces: List[ExplainabilityTrace],
    operator_notes: List[str],
) -> List[ExtractedTable]:
    """Process Document AI tables from tableRows."""
    tables: List[ExtractedTable] = []

    for page_data in pages:
        page_tables = page_data.get("tables", [])

        for tbl_idx, tbl in enumerate(page_tables):
            header_rows = tbl.get("headerRows", [])
            body_rows = tbl.get("bodyRows", [])

            # Convert to text rows
            text_header_rows = []
            for row in header_rows:
                cells = row.get("cells", [])
                text_row = [_get_docai_text(c, full_response).strip() for c in cells]
                text_header_rows.append(text_row)

            text_body_rows = []
            for row in body_rows:
                cells = row.get("cells", [])
                text_row = [_get_docai_text(c, full_response).strip() for c in cells]
                text_body_rows.append(text_row)

            # Validate headers: if >50% numeric cells, it's likely a data row
            all_rows = text_header_rows + text_body_rows
            header = text_header_rows[0] if text_header_rows else []

            if header:
                numeric_count = sum(1 for c in header if infer_cell_type(c) in ("integer", "decimal", "currency"))
                if numeric_count > len(header) * 0.5:
                    # Header looks like a data row — search upward for real header
                    traces.append(ExplainabilityTrace(
                        step="docai_header_validation",
                        reason=f"Table {tbl_idx}: header row has >50% numeric cells — re-inferring header",
                        confidence=0.7,
                    ))
                    # Try to infer from all rows
                    header_idx, header, header_conf = infer_header_row(all_rows)
                    data_rows = all_rows[header_idx + 1:]
                    operator_notes.append(f"Table {tbl_idx}: header was re-inferred — please verify column names")
                else:
                    data_rows = text_body_rows
            else:
                # No header provided — synthesize from ontology
                if all_rows:
                    header = _synthesize_header(all_rows[0])
                    data_rows = all_rows
                    operator_notes.append(f"Table {tbl_idx}: header synthesized via ontology — verify column names")
                    traces.append(ExplainabilityTrace(
                        step="docai_header_synthesis",
                        reason=f"No header row provided by Document AI — synthesized using ontology column inference",
                        confidence=0.5,
                    ))
                else:
                    continue

            # Normalize col count
            num_cols = len(header) if header else (len(data_rows[0]) if data_rows else 0)
            normalized_rows = []
            for row in data_rows:
                if len(row) < num_cols:
                    row = row + [""] * (num_cols - len(row))
                elif len(row) > num_cols:
                    row = row[:num_cols]
                normalized_rows.append(row)

            candidates, cells, recon_traces = reconstruct_table(normalized_rows, header, expected_total)
            traces.extend(recon_traces)

            validation = validate_table(normalized_rows, header, expected_total)

            table = ExtractedTable(
                table_id=f"docai_t{tbl_idx}",
                detected_header_row=header,
                reconstructed=len(candidates) > 1,
                reconstruction_candidates=candidates,
                cells=cells,
                validation=validation,
            )
            tables.append(table)

    return tables


def _process_docai_form_fields(
    pages: List[Dict],
    full_response: Dict,
    ocr_blocks: List[OCRBlock],
    traces: List[ExplainabilityTrace],
) -> Dict[str, FieldExtraction]:
    """Extract fields from Document AI formFields."""
    fields: Dict[str, FieldExtraction] = {}

    for page_data in pages:
        form_fields = page_data.get("formFields", [])
        for ff in form_fields:
            field_name_text = _get_docai_text(ff.get("fieldName", {}), full_response).strip()
            field_value_text = _get_docai_text(ff.get("fieldValue", {}), full_response).strip()

            if not field_name_text or not field_value_text:
                continue

            conf = ff.get("fieldName", {}).get("confidence", 0.0)
            if isinstance(conf, dict):
                conf = conf.get("confidence", 0.5)

            canonical, map_conf = map_to_canonical(field_name_text)
            fname = canonical if canonical else field_name_text.lower().replace(" ", "_")[:30]

            fields[fname] = FieldExtraction(
                value=field_value_text,
                confidence=round(float(conf) if isinstance(conf, (int, float)) else 0.5, 3),
                explain=f"Document AI formField: '{field_name_text}' → '{canonical or fname}'",
                canonical_name=canonical,
            )

    # Supplement with regex extraction
    full_text = "\n".join(b.text for b in ocr_blocks if b.text)
    regex_fields, _, regex_traces = extract_fields_from_text(full_text)
    traces.extend(regex_traces)

    for fname, fval in regex_fields.items():
        if fname not in fields:
            fields[fname] = fval

    if fields:
        traces.append(ExplainabilityTrace(
            step="docai_field_extraction",
            reason=f"Extracted {len(fields)} fields from Document AI formFields + regex fallback",
            confidence=0.85,
        ))

    return fields


def _synthesize_header(sample_row: List[str]) -> List[str]:
    """Synthesize header names from column data types and ontology."""
    header = []
    type_counters: Dict[str, int] = {}

    for cell in sample_row:
        ct = infer_cell_type(cell)
        if ct == "string":
            header.append("Description")
        elif ct in ("integer",):
            count = type_counters.get("qty", 0) + 1
            type_counters["qty"] = count
            header.append("Quantity" if count == 1 else f"Value_{count}")
        elif ct in ("decimal", "currency"):
            count = type_counters.get("amount", 0) + 1
            type_counters["amount"] = count
            if count == 1:
                header.append("Rate")
            elif count == 2:
                header.append("Amount")
            else:
                header.append(f"Value_{count}")
        else:
            header.append(f"Column_{len(header) + 1}")

    return header
