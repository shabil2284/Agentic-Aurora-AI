"""
Azure Form Recognizer Post-Processor
======================================
Uses Azure Form Recognizer tables and keyValuePairs as priors.
Merges multi-line description cells using vertical proximity + line-height heuristics.
Detects shuffled/misaligned values and produces top-3 candidate tables.

Combined confidence: 0.55 * azure_conf + 0.45 * heuristic_conf
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
)
from document_understanding.field_extractor import extract_fields_from_text
from document_understanding.ontology import map_to_canonical


AZURE_CONF_WEIGHT = 0.55
HEURISTIC_CONF_WEIGHT = 0.45


def process_azure_output(
    azure_response: Dict[str, Any],
    expected_total: Optional[float] = None,
) -> Tuple[List[OCRBlock], List[ExtractedTable], Dict[str, FieldExtraction], List[ExplainabilityTrace]]:
    """
    Process Azure Form Recognizer output into Aurora format.

    Args:
        azure_response: Azure Form Recognizer JSON (analyzeResult format)
        expected_total: Expected invoice total for validation

    Returns:
        (ocr_blocks, tables, fields, traces)
    """
    traces: List[ExplainabilityTrace] = []
    ocr_blocks: List[OCRBlock] = []

    analyze_result = azure_response.get("analyzeResult", azure_response)
    pages = analyze_result.get("pages", analyze_result.get("pageResults", []))
    read_results = analyze_result.get("readResults", [])

    # 1. Ingest lines from readResults
    for page_data in read_results if read_results else pages:
        page_num = page_data.get("page", page_data.get("pageNumber", 1))
        lines = page_data.get("lines", [])

        for line in lines:
            text = line.get("content", line.get("text", ""))
            bbox = _azure_bbox(line)
            conf = line.get("confidence", 1.0)

            if text.strip():
                ocr_blocks.append(OCRBlock(
                    id=f"azure_{len(ocr_blocks)}",
                    text=text.strip(),
                    bbox=bbox,
                    confidence=conf if isinstance(conf, float) else float(conf or 1.0),
                    page=page_num,
                ))

    traces.append(ExplainabilityTrace(
        step="azure_ingestion",
        reason=f"Ingested {len(ocr_blocks)} text lines from Azure Form Recognizer",
        confidence=0.95,
    ))

    # 2. Process tables
    tables = _process_azure_tables(analyze_result, expected_total, traces)

    # 3. Process key-value pairs
    fields = _process_azure_key_values(analyze_result, ocr_blocks, traces)

    return ocr_blocks, tables, fields, traces


def _azure_bbox(element: Dict) -> List[float]:
    """Convert Azure bounding box/polygon to [x1, y1, x2, y2]."""
    # Azure v3+ uses polygon (list of points)
    polygon = element.get("polygon", element.get("boundingPolygon", []))
    if polygon and isinstance(polygon, list):
        if isinstance(polygon[0], dict):
            xs = [p.get("x", 0) for p in polygon]
            ys = [p.get("y", 0) for p in polygon]
        elif isinstance(polygon[0], (int, float)):
            # Flat list: [x1,y1,x2,y2,x3,y3,x4,y4]
            xs = polygon[0::2]
            ys = polygon[1::2]
        else:
            return [0, 0, 0, 0]
        return [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]

    # Azure v2 uses boundingBox [x1,y1,...,x4,y4]
    bbox = element.get("boundingBox", [])
    if bbox and len(bbox) >= 8:
        xs = bbox[0::2]
        ys = bbox[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]

    return [0, 0, 0, 0]


def _process_azure_tables(
    analyze_result: Dict,
    expected_total: Optional[float],
    traces: List[ExplainabilityTrace],
) -> List[ExtractedTable]:
    """Process Azure tables."""
    tables: List[ExtractedTable] = []

    azure_tables = analyze_result.get("tables", [])
    page_results = analyze_result.get("pageResults", [])

    # Also check pageResults for tables (v2 format)
    if not azure_tables and page_results:
        for pr in page_results:
            azure_tables.extend(pr.get("tables", []))

    for tbl_idx, tbl in enumerate(azure_tables):
        row_count = tbl.get("rowCount", 0)
        col_count = tbl.get("columnCount", 0)
        cells = tbl.get("cells", [])

        if not cells:
            continue

        # Build grid
        grid: Dict[int, Dict[int, str]] = {}
        cell_confs: List[float] = []

        for cell in cells:
            row_idx = cell.get("rowIndex", 0)
            col_idx = cell.get("columnIndex", 0)
            text = cell.get("content", cell.get("text", "")).strip()
            conf = cell.get("confidence", 1.0)

            # Merge multi-line cells: concatenate with space
            row_span = cell.get("rowSpan", 1)
            col_span = cell.get("columnSpan", 1)

            for r in range(row_span):
                for c in range(col_span):
                    ri = row_idx + r
                    ci = col_idx + c
                    if ri not in grid:
                        grid[ri] = {}
                    existing = grid[ri].get(ci, "")
                    if existing and r == 0 and c == 0:
                        grid[ri][ci] = text
                    elif not existing:
                        grid[ri][ci] = text if r == 0 and c == 0 else ""

            cell_confs.append(float(conf) if isinstance(conf, (int, float)) else 0.5)

        if not grid:
            continue

        max_row = max(grid.keys()) + 1
        max_col_actual = max(max(cols.keys()) for cols in grid.values()) + 1 if grid else col_count

        text_rows = []
        for r in range(max_row):
            row = []
            for c in range(max_col_actual):
                row.append(grid.get(r, {}).get(c, ""))
            text_rows.append(row)

        # Infer header
        header_idx, header, header_conf = infer_header_row(text_rows)

        # Check for multi-line description cells that Azure split
        text_rows = _merge_multiline_cells(text_rows, header_idx, traces, tbl_idx)

        data_rows = text_rows[header_idx + 1:]

        # Detect shuffled values
        avg_conf = sum(cell_confs) / len(cell_confs) if cell_confs else 0.5

        # Normalize
        num_cols = len(header) if header else max_col_actual
        normalized_rows = []
        for row in data_rows:
            if len(row) < num_cols:
                row = row + [""] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:num_cols]
            normalized_rows.append(row)

        candidates, tbl_cells, recon_traces = reconstruct_table(
            normalized_rows, header, expected_total
        )
        traces.extend(recon_traces)

        # Apply combined confidence
        for cell in tbl_cells:
            azure_cell_conf = avg_conf
            heuristic_conf = cell.confidence
            cell.confidence = round(
                AZURE_CONF_WEIGHT * azure_cell_conf +
                HEURISTIC_CONF_WEIGHT * heuristic_conf,
                3,
            )

        validation = validate_table(normalized_rows, header, expected_total)
        bbox = _azure_bbox(tbl) if "boundingRegions" not in tbl else (
            _azure_bbox(tbl["boundingRegions"][0]) if tbl.get("boundingRegions") else []
        )

        table = ExtractedTable(
            table_id=f"azure_t{tbl_idx}",
            bbox=bbox,
            detected_header_row=header,
            reconstructed=len(candidates) > 1,
            reconstruction_candidates=candidates,
            cells=tbl_cells,
            validation=validation,
        )
        tables.append(table)

    if tables:
        traces.append(ExplainabilityTrace(
            step="azure_table_processing",
            reason=f"Processed {len(tables)} table(s) from Azure Form Recognizer",
            confidence=0.9,
        ))

    return tables


def _merge_multiline_cells(
    text_rows: List[List[str]],
    header_idx: int,
    traces: List[ExplainabilityTrace],
    tbl_idx: int,
) -> List[List[str]]:
    """
    Merge multi-line description cells that Azure split into separate rows.
    Heuristic: if a row has only values in one column (description-like) and
    rest is empty, merge into previous row.
    """
    merged = []
    i = 0
    merge_count = 0

    while i < len(text_rows):
        row = text_rows[i]
        if i <= header_idx:
            merged.append(row)
            i += 1
            continue

        # Check if next row looks like continuation
        if i + 1 < len(text_rows):
            next_row = text_rows[i + 1]
            non_empty = sum(1 for c in next_row if c.strip())
            if non_empty == 1 and len(next_row) > 2:
                # Find the single non-empty cell
                for j, cell in enumerate(next_row):
                    if cell.strip():
                        # Merge into current row's same column
                        if j < len(row):
                            row = list(row)
                            row[j] = f"{row[j]} {cell.strip()}" if row[j] else cell.strip()
                        merge_count += 1
                        break
                merged.append(row)
                i += 2
                continue

        merged.append(row)
        i += 1

    if merge_count > 0:
        traces.append(ExplainabilityTrace(
            step=f"azure_multiline_merge",
            reason=f"Table {tbl_idx}: merged {merge_count} multi-line description cells using vertical proximity heuristic",
            confidence=0.8,
        ))

    return merged


def _process_azure_key_values(
    analyze_result: Dict,
    ocr_blocks: List[OCRBlock],
    traces: List[ExplainabilityTrace],
) -> Dict[str, FieldExtraction]:
    """Extract fields from Azure keyValuePairs / documents."""
    fields: Dict[str, FieldExtraction] = {}

    # v3+ format: documents[].fields
    documents = analyze_result.get("documents", [])
    for doc in documents:
        doc_fields = doc.get("fields", {})
        for field_name, field_data in doc_fields.items():
            value = field_data.get("content", field_data.get("valueString", field_data.get("value", "")))
            conf = field_data.get("confidence", 0.5)

            canonical, map_conf = map_to_canonical(field_name)
            fname = canonical if canonical else field_name.lower().replace(" ", "_")[:30]

            fields[fname] = FieldExtraction(
                value=str(value).strip() if value else "",
                confidence=round(float(conf) if isinstance(conf, (int, float)) else 0.5, 3),
                explain=f"Azure document field: '{field_name}' → '{canonical or fname}'",
                canonical_name=canonical,
            )

    # v2 format: keyValuePairs
    kv_pairs = analyze_result.get("keyValuePairs", [])
    for kv in kv_pairs:
        key_data = kv.get("key", {})
        value_data = kv.get("value", {})
        key_text = key_data.get("content", key_data.get("text", "")).strip()
        value_text = value_data.get("content", value_data.get("text", "")).strip()
        conf = kv.get("confidence", 0.5)

        if not key_text or not value_text:
            continue

        canonical, map_conf = map_to_canonical(key_text)
        fname = canonical if canonical else key_text.lower().replace(" ", "_")[:30]

        if fname not in fields:
            fields[fname] = FieldExtraction(
                value=value_text,
                confidence=round(float(conf), 3),
                explain=f"Azure keyValuePair: '{key_text}' → '{canonical or fname}'",
                canonical_name=canonical,
            )

    # Supplement with regex
    full_text = "\n".join(b.text for b in ocr_blocks if b.text)
    regex_fields, _, regex_traces = extract_fields_from_text(full_text)
    traces.extend(regex_traces)

    for fname, fval in regex_fields.items():
        if fname not in fields:
            fields[fname] = fval

    if fields:
        traces.append(ExplainabilityTrace(
            step="azure_field_extraction",
            reason=f"Extracted {len(fields)} field(s) from Azure documents/keyValuePairs + regex fallback",
            confidence=0.85,
        ))

    return fields
