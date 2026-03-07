"""
AWS Textract Post-Processor
============================
Ingests AWS Textract JSON blocks (BlockType: PAGE/LINE/TABLE/CELL/WORD with Geometry).
Uses Geometry.Polygon centroids for column clustering. Re-infers headers when
confidence < 0.85. Includes textract_block_ids for traceability.

Combined confidence: 0.6 * textract_conf + 0.4 * heuristic_conf
"""

from typing import Any, Dict, List, Optional, Tuple

from document_understanding.models import (
    ExplainabilityTrace,
    ExtractedTable,
    FieldExtraction,
    OCRBlock,
    PageResult,
)
from document_understanding.table_reconstructor import (
    extract_tables_from_text,
    infer_cell_type,
    infer_header_row,
    parse_numeric,
    reconstruct_table,
    validate_table,
)
from document_understanding.field_extractor import extract_fields_from_text


TEXTRACT_CONF_WEIGHT = 0.6
HEURISTIC_CONF_WEIGHT = 0.4


def process_textract_output(
    textract_blocks: List[Dict[str, Any]],
    expected_total: Optional[float] = None,
) -> Tuple[List[OCRBlock], List[ExtractedTable], Dict[str, FieldExtraction], List[ExplainabilityTrace]]:
    """
    Process AWS Textract JSON blocks into Aurora format.

    Args:
        textract_blocks: List of Textract Block objects
        expected_total:  Expected invoice total for validation

    Returns:
        (ocr_blocks, tables, fields, traces)
    """
    traces: List[ExplainabilityTrace] = []
    ocr_blocks: List[OCRBlock] = []
    block_map: Dict[str, Dict] = {}

    # 1. Ingest and index blocks
    for block in textract_blocks:
        block_id = block.get("Id", "")
        block_map[block_id] = block

        if block.get("BlockType") in ("LINE", "WORD"):
            bbox = _geometry_to_bbox(block.get("Geometry", {}))
            conf = block.get("Confidence", 0) / 100.0  # Textract gives 0-100
            ocr_block = OCRBlock(
                id=block_id,
                text=block.get("Text", ""),
                bbox=bbox,
                confidence=conf,
                page=block.get("Page", 1),
            )
            ocr_blocks.append(ocr_block)

    traces.append(ExplainabilityTrace(
        step="textract_ingestion",
        reason=f"Ingested {len(ocr_blocks)} LINE/WORD blocks from {len(textract_blocks)} total Textract blocks",
        confidence=0.95,
    ))

    # 2. Process TABLE blocks
    tables = _process_textract_tables(textract_blocks, block_map, expected_total, traces)

    # 3. Extract key-value pairs
    fields = _process_textract_key_values(textract_blocks, block_map, ocr_blocks, traces)

    return ocr_blocks, tables, fields, traces


def _geometry_to_bbox(geometry: Dict) -> List[float]:
    """Convert Textract Geometry to [x1, y1, x2, y2] bbox."""
    bbox_data = geometry.get("BoundingBox", {})
    if bbox_data:
        left = bbox_data.get("Left", 0) * 1000  # Scale to pixel-like coords
        top = bbox_data.get("Top", 0) * 1000
        width = bbox_data.get("Width", 0) * 1000
        height = bbox_data.get("Height", 0) * 1000
        return [left, top, left + width, top + height]

    # Fallback to polygon centroids
    polygon = geometry.get("Polygon", [])
    if polygon:
        xs = [p.get("X", 0) * 1000 for p in polygon]
        ys = [p.get("Y", 0) * 1000 for p in polygon]
        return [min(xs), min(ys), max(xs), max(ys)]

    return [0, 0, 0, 0]


def _polygon_centroid(geometry: Dict) -> Tuple[float, float]:
    """Get centroid from Textract Geometry polygon."""
    polygon = geometry.get("Polygon", [])
    if polygon:
        xs = [p.get("X", 0) for p in polygon]
        ys = [p.get("Y", 0) for p in polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    bbox_data = geometry.get("BoundingBox", {})
    cx = bbox_data.get("Left", 0) + bbox_data.get("Width", 0) / 2
    cy = bbox_data.get("Top", 0) + bbox_data.get("Height", 0) / 2
    return (cx, cy)


def _process_textract_tables(
    blocks: List[Dict],
    block_map: Dict[str, Dict],
    expected_total: Optional[float],
    traces: List[ExplainabilityTrace],
) -> List[ExtractedTable]:
    """Process Textract TABLE/CELL blocks into ExtractedTables."""
    tables: List[ExtractedTable] = []

    table_blocks = [b for b in blocks if b.get("BlockType") == "TABLE"]

    for tbl_block in table_blocks:
        table_id = tbl_block.get("Id", "")
        cell_ids = []
        for rel in tbl_block.get("Relationships", []):
            if rel.get("Type") == "CHILD":
                cell_ids.extend(rel.get("Ids", []))

        # Collect cells
        rows_dict: Dict[int, Dict[int, str]] = {}
        cell_confs: List[float] = []

        for cell_id in cell_ids:
            cell_block = block_map.get(cell_id, {})
            if cell_block.get("BlockType") != "CELL":
                continue

            row_idx = cell_block.get("RowIndex", 1)
            col_idx = cell_block.get("ColumnIndex", 1)
            cell_conf = cell_block.get("Confidence", 0) / 100.0
            cell_confs.append(cell_conf)

            # Get cell text from child WORD blocks
            cell_text = _get_block_text(cell_block, block_map)
            if row_idx not in rows_dict:
                rows_dict[row_idx] = {}
            rows_dict[row_idx][col_idx] = cell_text

        if not rows_dict:
            continue

        # Convert to ordered rows
        max_row = max(rows_dict.keys())
        max_col = max(max(cols.keys()) for cols in rows_dict.values())
        text_rows = []
        for r in range(1, max_row + 1):
            row = []
            for c in range(1, max_col + 1):
                row.append(rows_dict.get(r, {}).get(c, ""))
            text_rows.append(row)

        # Infer header
        header_idx, header, header_conf = infer_header_row(text_rows)

        # Re-infer header if Textract confidence < 0.85
        avg_conf = sum(cell_confs) / len(cell_confs) if cell_confs else 0
        if avg_conf < 0.85:
            traces.append(ExplainabilityTrace(
                step="textract_header_reinference",
                reason=f"Textract avg cell confidence ({avg_conf:.2f}) < 0.85 — re-inferring header with Aurora heuristics",
                confidence=header_conf,
            ))

        data_rows = text_rows[header_idx + 1:]

        # Reconstruct
        candidates, cells, recon_traces = reconstruct_table(data_rows, header, expected_total)
        traces.extend(recon_traces)

        # Apply combined confidence
        for cell in cells:
            textract_cell_conf = avg_conf
            heuristic_conf = cell.confidence
            cell.confidence = round(
                TEXTRACT_CONF_WEIGHT * textract_cell_conf +
                HEURISTIC_CONF_WEIGHT * heuristic_conf,
                3,
            )

        validation = validate_table(data_rows, header, expected_total)
        bbox = _geometry_to_bbox(tbl_block.get("Geometry", {}))

        table = ExtractedTable(
            table_id=f"t_{table_id[:8]}",
            bbox=bbox,
            detected_header_row=header,
            reconstructed=len(candidates) > 1,
            reconstruction_candidates=candidates,
            cells=cells,
            validation=validation,
        )
        tables.append(table)

    if tables:
        traces.append(ExplainabilityTrace(
            step="textract_table_processing",
            reason=f"Processed {len(tables)} table(s) from Textract TABLE blocks",
            confidence=0.9,
        ))

    return tables


def _process_textract_key_values(
    blocks: List[Dict],
    block_map: Dict[str, Dict],
    ocr_blocks: List[OCRBlock],
    traces: List[ExplainabilityTrace],
) -> Dict[str, FieldExtraction]:
    """
    Extract key-value pairs from Textract KEY_VALUE_SET blocks.
    Use as prior, but override if regex evidence contradicts.
    """
    fields: Dict[str, FieldExtraction] = {}

    kv_blocks = [b for b in blocks if b.get("BlockType") == "KEY_VALUE_SET"]
    key_blocks = [b for b in kv_blocks if b.get("EntityTypes") and "KEY" in b["EntityTypes"]]

    for key_block in key_blocks:
        key_text = _get_block_text(key_block, block_map)
        if not key_text:
            continue

        # Find associated VALUE
        value_text = ""
        value_conf = 0.0
        for rel in key_block.get("Relationships", []):
            if rel.get("Type") == "VALUE":
                for val_id in rel.get("Ids", []):
                    val_block = block_map.get(val_id, {})
                    value_text = _get_block_text(val_block, block_map)
                    value_conf = val_block.get("Confidence", 0) / 100.0
                    break

        if not value_text:
            continue

        field_name = key_text.strip().lower().replace(" ", "_").replace(".", "").replace("#", "")[:30]
        textract_conf = key_block.get("Confidence", 0) / 100.0

        fields[field_name] = FieldExtraction(
            value=value_text.strip(),
            confidence=round(textract_conf, 3),
            source_block_id=key_block.get("Id", ""),
            explain=f"Textract KEY_VALUE_SET: key='{key_text}', conf={textract_conf:.2f}",
        )

    # Override with regex-based extraction if contradiction found
    full_text = "\n".join(b.text for b in ocr_blocks if b.text)
    regex_fields, _, regex_traces = extract_fields_from_text(full_text)
    traces.extend(regex_traces)

    for fname, fval in regex_fields.items():
        if fname in fields:
            existing = fields[fname]
            if fval.confidence > existing.confidence:
                traces.append(ExplainabilityTrace(
                    step=f"textract_override:{fname}",
                    reason=f"Regex extraction (conf={fval.confidence:.2f}) overrides Textract (conf={existing.confidence:.2f}) for field '{fname}'",
                    confidence=fval.confidence,
                ))
                fields[fname] = fval
        else:
            fields[fname] = fval

    if fields:
        traces.append(ExplainabilityTrace(
            step="textract_kv_extraction",
            reason=f"Extracted {len(fields)} key-value pairs (Textract KV + regex fallback)",
            confidence=0.85,
        ))

    return fields


def _get_block_text(block: Dict, block_map: Dict[str, Dict]) -> str:
    """Recursively get text from a block via its CHILD relationships."""
    if "Text" in block:
        return block["Text"]

    text_parts = []
    for rel in block.get("Relationships", []):
        if rel.get("Type") == "CHILD":
            for child_id in rel.get("Ids", []):
                child = block_map.get(child_id, {})
                if child.get("BlockType") == "WORD":
                    text_parts.append(child.get("Text", ""))
                elif child.get("BlockType") == "SELECTION_ELEMENT":
                    text_parts.append("☑" if child.get("SelectionStatus") == "SELECTED" else "☐")

    return " ".join(text_parts)
