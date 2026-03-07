"""
Aurora Table Reconstructor
===========================
Core table detection and reconstruction engine implementing Legacy Antigravity Rules:

- Detection: 3+ column alignments or repeated numeric patterns across consecutive lines
- Header inference: row with >2 non-numeric tokens matching synonyms
- Shuffled value reconstruction via data-type heuristics, positional clustering,
  header-to-cell similarity, and global constraints (sum ≈ total)
- Produces top-3 ranked reconstructions
- Column mapping priorities: Description > Quantity > Unit Price/Rate > Tax > Line Total
"""

import csv
import io
import re
from typing import Any, Dict, List, Optional, Tuple

from document_understanding.models import (
    ExplainabilityTrace,
    ExtractedTable,
    LineSumCheck,
    OCRBlock,
    ReconstructionCandidate,
    TableCell,
    TableValidation,
)

# ── Header synonym groups ─────────────────────────────────────────────────────

HEADER_SYNONYMS: Dict[str, List[str]] = {
    "description": ["item", "description", "particulars", "product", "service", "detail", "details", "name"],
    "quantity": ["qty", "quantity", "units", "pcs", "nos", "count", "pieces"],
    "unit_price": ["rate", "price", "unit price", "unit rate", "unit cost", "cost", "mrp", "each"],
    "tax": ["tax", "gst", "vat", "sgst", "cgst", "igst", "tax%", "tax rate", "tax amount"],
    "line_total": ["amount", "total", "line total", "net", "value", "ext", "extended", "subtotal"],
    "discount": ["discount", "disc", "disc%", "rebate"],
    "hsn": ["hsn", "hsn code", "hsn/sac", "sac", "hs code"],
    "serial": ["s.no", "sno", "sl", "sl.no", "sr", "sr.no", "#", "no", "serial"],
}

# Flattened set of all synonym tokens for quick lookup
_ALL_HEADER_TOKENS = set()
for synonyms in HEADER_SYNONYMS.values():
    for s in synonyms:
        _ALL_HEADER_TOKENS.update(s.lower().split())

# Column mapping priority order (for reconstruction scoring)
_COLUMN_PRIORITY = ["description", "quantity", "unit_price", "tax", "line_total"]

# ── Data-type classifiers ─────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"^[\$₹€£¥]?\s*[\d,]+\.?\d*$|^\d[\d.,]+$")
_INTEGER_RE = re.compile(r"^\d+$")
_PERCENTAGE_RE = re.compile(r"^\d+\.?\d*\s*%$")
_DATE_RE = re.compile(
    r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$|"
    r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    re.IGNORECASE,
)


def infer_cell_type(text: str) -> str:
    """Infer the data type of a cell value."""
    t = text.strip()
    if not t:
        return "empty"
    if _INTEGER_RE.match(t):
        return "integer"
    if _PERCENTAGE_RE.match(t):
        return "percentage"
    if _DATE_RE.match(t):
        return "date"
    if _CURRENCY_RE.match(t):
        # Distinguish currency from plain decimal
        if any(c in t for c in "$₹€£¥") or ("," in t and "." in t):
            return "currency"
        return "decimal"
    return "string"


def parse_numeric(text: str) -> Optional[float]:
    """Parse a numeric string, handling European and US formats."""
    t = text.strip()
    t = re.sub(r"[\$₹€£¥\s]", "", t)
    if not t:
        return None
    try:
        # European format: 1.234,56 -> 1234.56
        if re.match(r"^\d{1,3}(\.\d{3})*(,\d+)?$", t):
            return float(t.replace(".", "").replace(",", "."))
        # US format: 1,234.56 -> 1234.56
        t = t.replace(",", "")
        return float(t)
    except ValueError:
        return None


# ── Table Detection ───────────────────────────────────────────────────────────


def detect_tables_from_blocks(
    blocks: List[OCRBlock],
) -> Tuple[List[List[OCRBlock]], List[ExplainabilityTrace]]:
    """
    Legacy Antigravity Rule: detect tables by finding 3+ vertical alignments
    or repeated numeric patterns across consecutive lines.

    Groups blocks into horizontal bands (rows), checks for column alignment.
    Returns list of block-groups that form tables + traces.
    """
    traces: List[ExplainabilityTrace] = []
    if not blocks:
        return [], traces

    # Sort blocks by y then x (top→bottom, left→right)
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1] if len(b.bbox) >= 4 else 0,
                                                    b.bbox[0] if len(b.bbox) >= 4 else 0))

    # Group into horizontal bands (rows) based on y-proximity
    bands: List[List[OCRBlock]] = []
    current_band: List[OCRBlock] = []
    prev_y = None
    y_tolerance = 8.0  # pixels

    for block in sorted_blocks:
        if not block.bbox or len(block.bbox) < 4:
            continue
        y = block.bbox[1]
        if prev_y is not None and abs(y - prev_y) > y_tolerance:
            if current_band:
                bands.append(current_band)
            current_band = [block]
        else:
            current_band.append(block)
        prev_y = y

    if current_band:
        bands.append(current_band)

    if len(bands) < 2:
        return [], traces

    # Check vertical alignment across bands
    table_bands = _find_aligned_bands(bands)

    if table_bands:
        traces.append(ExplainabilityTrace(
            step="table_detection",
            reason=f"Found {len(table_bands)} rows with {_count_column_alignments(table_bands)} vertical alignments and repeated numeric patterns",
            confidence=0.9 if len(table_bands) >= 4 else 0.75,
        ))
        # Flatten into single table group
        return [table_bands], traces

    return [], traces


def detect_tables_from_text(
    text: str,
) -> Tuple[List[List[List[str]]], List[ExplainabilityTrace]]:
    """
    Plain text table detection: find runs of lines with consistent separators
    and numeric patterns.

    Returns list of tables (each table = list of rows, each row = list of cells) + traces.
    """
    traces: List[ExplainabilityTrace] = []
    lines = text.split("\n")
    tables: List[List[List[str]]] = []
    current_table: List[List[str]] = []
    numeric_streak = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_table and len(current_table) >= 3:
                tables.append(current_table)
            current_table = []
            numeric_streak = 0
            continue

        # Try splitting by tab, pipe, or multi-space
        cells = _split_table_row(stripped)

        if len(cells) >= 2:
            numeric_count = sum(1 for c in cells if re.match(r"^[\d,.%$₹€£¥\s-]+$", c.strip()))
            if numeric_count >= 1 or len(cells) >= 3:
                current_table.append(cells)
                numeric_streak += 1
            else:
                if current_table and len(current_table) >= 3:
                    tables.append(current_table)
                current_table = []
                numeric_streak = 0
        else:
            if current_table and len(current_table) >= 3:
                tables.append(current_table)
            current_table = []
            numeric_streak = 0

    if current_table and len(current_table) >= 3:
        tables.append(current_table)

    if tables:
        traces.append(ExplainabilityTrace(
            step="table_detection",
            reason=f"Plain text: found {len(tables)} table(s) with consistent separators and numeric patterns",
            confidence=0.7,
        ))

    return tables, traces


# ── Header Inference ──────────────────────────────────────────────────────────


def infer_header_row(rows: List[List[str]]) -> Tuple[int, List[str], float]:
    """
    Legacy Antigravity Rule: prefer row with >2 non-numeric tokens matching header synonyms.

    Returns: (header_row_index, header_cells, confidence)
    """
    best_idx = 0
    best_score = 0.0
    best_cells: List[str] = []

    for i, row in enumerate(min(rows[:5], key=lambda x: 0) if not rows else rows[:5]):
        score = 0.0
        non_numeric_count = 0

        for cell in row:
            cell_lower = cell.strip().lower()
            cell_type = infer_cell_type(cell)

            if cell_type == "string":
                non_numeric_count += 1

            # Check against header synonyms
            for _, synonyms in HEADER_SYNONYMS.items():
                if cell_lower in synonyms or any(s in cell_lower for s in synonyms):
                    score += 2.0
                    break
            else:
                # Check individual tokens
                tokens = set(cell_lower.split())
                if tokens & _ALL_HEADER_TOKENS:
                    score += 1.0

        if non_numeric_count >= 2 and score > best_score:
            best_score = score
            best_idx = i
            best_cells = [c.strip() for c in row]

    confidence = min(best_score / (len(rows[0]) * 2 if rows and rows[0] else 1), 1.0) if best_score > 0 else 0.3

    # Fallback: if no header detected, use first row
    if not best_cells and rows:
        best_cells = [c.strip() for c in rows[0]]
        confidence = 0.3

    return best_idx, best_cells, confidence


# ── Shuffled Value Reconstruction ─────────────────────────────────────────────


def reconstruct_table(
    rows: List[List[str]],
    header: List[str],
    expected_total: Optional[float] = None,
) -> Tuple[List[ReconstructionCandidate], List[TableCell], List[ExplainabilityTrace]]:
    """
    When table values are shuffled: use data-type heuristics, header-to-cell similarity,
    positional clustering, and global constraints to reassign cells to columns.

    Produces top-3 ranked reconstructions.
    """
    traces: List[ExplainabilityTrace] = []
    cells: List[TableCell] = []

    if not rows or not header:
        return [], cells, traces

    num_cols = len(header)
    data_rows = rows  # Assume header already separated

    # ── Build column type profile from header ─────────────────────────────
    col_expected_types = _infer_column_types_from_header(header)

    # ── Candidate 1: Original order ───────────────────────────────────────
    original_csv = _rows_to_csv(header, data_rows)
    original_score = _score_reconstruction(data_rows, col_expected_types, expected_total)

    # ── Candidate 2: Type-sorted reconstruction ───────────────────────────
    type_sorted_rows = _reconstruct_by_type(data_rows, col_expected_types)
    type_sorted_csv = _rows_to_csv(header, type_sorted_rows)
    type_sorted_score = _score_reconstruction(type_sorted_rows, col_expected_types, expected_total)

    # ── Candidate 3: Priority-based reconstruction ────────────────────────
    priority_rows = _reconstruct_by_priority(data_rows, header)
    priority_csv = _rows_to_csv(header, priority_rows)
    priority_score = _score_reconstruction(priority_rows, col_expected_types, expected_total)

    candidates = [
        ReconstructionCandidate(score=round(original_score, 3), csv=original_csv),
        ReconstructionCandidate(score=round(type_sorted_score, 3), csv=type_sorted_csv),
        ReconstructionCandidate(score=round(priority_score, 3), csv=priority_csv),
    ]

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Build cells from best candidate
    best_rows = data_rows  # default
    if candidates[0].score == type_sorted_score:
        best_rows = type_sorted_rows
    elif candidates[0].score == priority_score:
        best_rows = priority_rows

    for r_idx, row in enumerate(best_rows):
        for c_idx, cell_text in enumerate(row):
            cells.append(TableCell(
                row=r_idx + 1,
                col=c_idx + 1,
                text=cell_text.strip() if isinstance(cell_text, str) else str(cell_text),
                confidence=round(candidates[0].score, 2),
                inferred_type=infer_cell_type(str(cell_text)),
            ))

    is_shuffled = candidates[0].score != original_score and candidates[0].score > original_score
    traces.append(ExplainabilityTrace(
        step="table_reconstruction",
        reason=f"{'Detected shuffled values — applied type-based reassignment' if is_shuffled else 'Original order is optimal'} "
               f"(scores: original={original_score:.2f}, type_sorted={type_sorted_score:.2f}, priority={priority_score:.2f})",
        confidence=candidates[0].score if candidates else 0.0,
    ))

    return candidates, cells, traces


def validate_table(
    rows: List[List[str]],
    header: List[str],
    expected_total: Optional[float] = None,
) -> TableValidation:
    """
    Validate table by checking if line totals sum to the expected total.
    Flags >1% discrepancy.
    """
    if expected_total is None:
        return TableValidation()

    # Find the total/amount column
    total_col_idx = None
    for i, h in enumerate(header):
        h_lower = h.lower()
        if any(s in h_lower for s in ["total", "amount", "value", "net"]):
            total_col_idx = i
            break

    if total_col_idx is None:
        return TableValidation()

    computed = 0.0
    for row in rows:
        if total_col_idx < len(row):
            val = parse_numeric(row[total_col_idx])
            if val is not None:
                computed += val

    passed = True
    if expected_total > 0:
        discrepancy = abs(computed - expected_total) / expected_total
        passed = discrepancy <= 0.01  # 1% tolerance

    return TableValidation(
        line_sum_check=LineSumCheck(
            expected_total=expected_total,
            computed_total=round(computed, 2),
            passed=passed,
        )
    )


# ── Full table extraction pipeline ───────────────────────────────────────────


def extract_tables_from_text(
    text: str,
    expected_total: Optional[float] = None,
) -> Tuple[List[ExtractedTable], List[ExplainabilityTrace]]:
    """
    Complete table extraction pipeline for plain text input.
    """
    all_traces: List[ExplainabilityTrace] = []

    raw_tables, detect_traces = detect_tables_from_text(text)
    all_traces.extend(detect_traces)

    extracted: List[ExtractedTable] = []

    for tbl_rows in raw_tables:
        header_idx, header, header_conf = infer_header_row(tbl_rows)
        data_rows = tbl_rows[header_idx + 1:] if header_idx < len(tbl_rows) else tbl_rows[1:]

        # Normalize row lengths
        num_cols = len(header)
        normalized_rows = []
        for row in data_rows:
            if len(row) < num_cols:
                row = row + [""] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:num_cols]
            normalized_rows.append(row)

        candidates, cells, recon_traces = reconstruct_table(
            normalized_rows, header, expected_total
        )
        all_traces.extend(recon_traces)

        validation = validate_table(normalized_rows, header, expected_total)
        is_large = num_cols > 40 or len(normalized_rows) > 5000

        table = ExtractedTable(
            detected_header_row=header,
            reconstructed=any(c.score > 0 for c in candidates) and len(candidates) > 1,
            reconstruction_candidates=candidates,
            cells=cells if not is_large else cells[:50 * num_cols],
            validation=validation,
            large_table=is_large,
        )
        extracted.append(table)

    return extracted, all_traces


def extract_tables_from_blocks(
    blocks: List[OCRBlock],
    expected_total: Optional[float] = None,
) -> Tuple[List[ExtractedTable], List[ExplainabilityTrace]]:
    """
    Complete table extraction pipeline for OCR block input.
    """
    all_traces: List[ExplainabilityTrace] = []

    table_groups, detect_traces = detect_tables_from_blocks(blocks)
    all_traces.extend(detect_traces)

    extracted: List[ExtractedTable] = []

    for group in table_groups:
        # Convert block groups (bands) into text rows
        if isinstance(group, list) and group and isinstance(group[0], list):
            # Already banded
            text_rows = [_band_to_row(band) for band in group]
        else:
            # Flat list of blocks — re-band them
            bands = _group_into_bands(group)
            text_rows = [_band_to_row(band) for band in bands]

        if not text_rows:
            continue

        header_idx, header, header_conf = infer_header_row(text_rows)
        data_rows = text_rows[header_idx + 1:]

        num_cols = len(header)
        normalized_rows = []
        for row in data_rows:
            if len(row) < num_cols:
                row = row + [""] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:num_cols]
            normalized_rows.append(row)

        candidates, cells, recon_traces = reconstruct_table(
            normalized_rows, header, expected_total
        )
        all_traces.extend(recon_traces)

        validation = validate_table(normalized_rows, header, expected_total)
        bbox = _compute_group_bbox(group if not isinstance(group[0], list) else [b for band in group for b in band])

        table = ExtractedTable(
            bbox=bbox,
            detected_header_row=header,
            reconstructed=any(c.score > 0 for c in candidates) and len(candidates) > 1,
            reconstruction_candidates=candidates,
            cells=cells,
            validation=validation,
        )
        extracted.append(table)

    return extracted, all_traces


# ── Internal helpers ──────────────────────────────────────────────────────────


def _split_table_row(line: str) -> List[str]:
    """Split a text line into table cells using common delimiters."""
    # Tab-separated
    if "\t" in line:
        return [c.strip() for c in line.split("\t") if c.strip()]
    # Pipe-separated
    if "|" in line:
        cells = [c.strip() for c in line.split("|")]
        return [c for c in cells if c]  # filter empty edge cells
    # Multi-space separated (2+ spaces)
    parts = re.split(r"\s{2,}", line.strip())
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]
    return [line.strip()]


def _find_aligned_bands(bands: List[List[OCRBlock]]) -> List[List[OCRBlock]]:
    """Find bands with vertically aligned x-positions (column structure)."""
    if len(bands) < 3:
        return []

    # Collect x-centroids for each band
    band_centroids: List[List[float]] = []
    for band in bands:
        centroids = []
        for b in band:
            if b.bbox and len(b.bbox) >= 4:
                centroids.append((b.bbox[0] + b.bbox[2]) / 2)
        band_centroids.append(sorted(centroids))

    # Count alignment matches between consecutive bands
    alignment_count = 0
    x_tolerance = 15.0

    for i in range(len(band_centroids) - 1):
        if not band_centroids[i] or not band_centroids[i + 1]:
            continue
        for cx in band_centroids[i]:
            for nx in band_centroids[i + 1]:
                if abs(cx - nx) < x_tolerance:
                    alignment_count += 1
                    break

    # Legacy Antigravity: 3+ alignments → table
    if alignment_count >= 3:
        return bands

    return []


def _count_column_alignments(bands: List[Any]) -> int:
    """Count vertically aligned positions across all bands."""
    if not bands:
        return 0
    count = 0
    for i in range(len(bands) - 1):
        if isinstance(bands[i], list) and isinstance(bands[i + 1], list):
            for b1 in bands[i]:
                if hasattr(b1, 'bbox') and b1.bbox and len(b1.bbox) >= 4:
                    cx1 = (b1.bbox[0] + b1.bbox[2]) / 2
                    for b2 in bands[i + 1]:
                        if hasattr(b2, 'bbox') and b2.bbox and len(b2.bbox) >= 4:
                            cx2 = (b2.bbox[0] + b2.bbox[2]) / 2
                            if abs(cx1 - cx2) < 15.0:
                                count += 1
                                break
    return count


def _group_into_bands(blocks: List[OCRBlock], y_tolerance: float = 8.0) -> List[List[OCRBlock]]:
    """Group blocks into horizontal bands by y-proximity."""
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1] if b.bbox and len(b.bbox) >= 4 else 0))
    bands: List[List[OCRBlock]] = []
    current: List[OCRBlock] = []
    prev_y = None

    for b in sorted_blocks:
        if not b.bbox or len(b.bbox) < 4:
            continue
        y = b.bbox[1]
        if prev_y is not None and abs(y - prev_y) > y_tolerance:
            if current:
                bands.append(sorted(current, key=lambda x: x.bbox[0] if x.bbox else 0))
            current = [b]
        else:
            current.append(b)
        prev_y = y

    if current:
        bands.append(sorted(current, key=lambda x: x.bbox[0] if x.bbox else 0))
    return bands


def _band_to_row(band: List[OCRBlock]) -> List[str]:
    """Convert a band of blocks to a row of cell texts."""
    sorted_band = sorted(band, key=lambda b: b.bbox[0] if b.bbox and len(b.bbox) >= 4 else 0)
    return [b.text.strip() for b in sorted_band if b.text.strip()]


def _compute_group_bbox(blocks: List[OCRBlock]) -> List[float]:
    """Compute bounding box encompassing all blocks."""
    if not blocks:
        return []
    xs, ys = [], []
    for b in blocks:
        if b.bbox and len(b.bbox) >= 4:
            xs.extend([b.bbox[0], b.bbox[2]])
            ys.extend([b.bbox[1], b.bbox[3]])
    if xs and ys:
        return [min(xs), min(ys), max(xs), max(ys)]
    return []


def _infer_column_types_from_header(header: List[str]) -> List[str]:
    """Map header names to expected column data types."""
    types = []
    for h in header:
        h_lower = h.lower().strip()
        if any(s in h_lower for s in HEADER_SYNONYMS.get("description", [])):
            types.append("string")
        elif any(s in h_lower for s in HEADER_SYNONYMS.get("quantity", [])):
            types.append("integer")
        elif any(s in h_lower for s in HEADER_SYNONYMS.get("unit_price", [])):
            types.append("decimal")
        elif any(s in h_lower for s in HEADER_SYNONYMS.get("tax", [])):
            types.append("decimal")
        elif any(s in h_lower for s in HEADER_SYNONYMS.get("line_total", [])):
            types.append("decimal")
        elif any(s in h_lower for s in HEADER_SYNONYMS.get("serial", [])):
            types.append("integer")
        else:
            types.append("any")
    return types


def _score_reconstruction(
    rows: List[List[str]],
    expected_types: List[str],
    expected_total: Optional[float] = None,
) -> float:
    """Score a table reconstruction based on type-match rate and total consistency."""
    if not rows or not expected_types:
        return 0.5

    type_matches = 0
    total_cells = 0

    for row in rows:
        for col_idx, cell in enumerate(row):
            if col_idx >= len(expected_types):
                break
            total_cells += 1
            cell_type = infer_cell_type(cell)
            expected = expected_types[col_idx]

            if expected == "any":
                type_matches += 1
            elif expected == "string" and cell_type == "string":
                type_matches += 1
            elif expected == "integer" and cell_type in ("integer", "decimal"):
                type_matches += 1
            elif expected == "decimal" and cell_type in ("decimal", "currency", "integer"):
                type_matches += 1
            elif cell_type == expected:
                type_matches += 1

    type_score = type_matches / total_cells if total_cells > 0 else 0.5

    # Global constraint: sum check
    total_score = 1.0
    if expected_total and expected_total > 0:
        # Find amount/total column
        total_col = None
        for i, et in enumerate(expected_types):
            if et == "decimal" and i == len(expected_types) - 1:
                total_col = i
                break
        if total_col is not None:
            computed = sum(
                parse_numeric(row[total_col]) or 0
                for row in rows
                if total_col < len(row)
            )
            if computed > 0:
                discrepancy = abs(computed - expected_total) / expected_total
                total_score = max(0, 1.0 - discrepancy)

    return 0.7 * type_score + 0.3 * total_score


def _reconstruct_by_type(
    rows: List[List[str]],
    expected_types: List[str],
) -> List[List[str]]:
    """Reconstruct by sorting cells within each row to match expected column types."""
    reconstructed = []
    for row in rows:
        if len(row) != len(expected_types):
            reconstructed.append(row)
            continue

        # Classify each cell
        cell_types = [(i, cell, infer_cell_type(cell)) for i, cell in enumerate(row)]

        # Greedy assignment: for each expected type, pick best matching cell
        used = set()
        new_row = [""] * len(expected_types)

        for col_idx, expected in enumerate(expected_types):
            best_cell = None
            best_idx = None

            for orig_idx, cell, ct in cell_types:
                if orig_idx in used:
                    continue

                if expected == "any":
                    if best_cell is None:
                        best_cell = cell
                        best_idx = orig_idx
                elif expected == "string" and ct == "string":
                    best_cell = cell
                    best_idx = orig_idx
                    break
                elif expected in ("decimal", "currency") and ct in ("decimal", "currency", "integer"):
                    if best_cell is None or ct == expected:
                        best_cell = cell
                        best_idx = orig_idx
                elif expected == "integer" and ct in ("integer", "decimal"):
                    if best_cell is None:
                        best_cell = cell
                        best_idx = orig_idx

            if best_idx is not None:
                new_row[col_idx] = best_cell
                used.add(best_idx)

        # Fill remaining with unassigned cells
        unassigned = [cell for i, cell, _ in cell_types if i not in used]
        for col_idx in range(len(new_row)):
            if not new_row[col_idx] and unassigned:
                new_row[col_idx] = unassigned.pop(0)

        reconstructed.append(new_row)

    return reconstructed


def _reconstruct_by_priority(
    rows: List[List[str]],
    header: List[str],
) -> List[List[str]]:
    """
    Reconstruct using column priority order:
    Description > Quantity > Unit Price/Rate > Tax > Line Total
    """
    # Map header to priority categories
    col_categories = []
    for h in header:
        h_lower = h.lower().strip()
        matched = False
        for cat in _COLUMN_PRIORITY:
            if any(s in h_lower for s in HEADER_SYNONYMS.get(cat, [])):
                col_categories.append(cat)
                matched = True
                break
        if not matched:
            col_categories.append("other")

    # For each row, attempt to place the longest string in description slot,
    # largest number in total slot, etc.
    reconstructed = []
    for row in rows:
        if len(row) != len(header):
            reconstructed.append(row)
            continue

        cells_with_type = [(cell, infer_cell_type(cell)) for cell in row]
        string_cells = [c for c, t in cells_with_type if t == "string"]
        numeric_cells = [(c, parse_numeric(c)) for c, t in cells_with_type if t in ("decimal", "currency", "integer")]
        numeric_cells.sort(key=lambda x: x[1] or 0)

        new_row = list(row)  # Start with original
        # Only rearrange if we have clear type mismatches
        for col_idx, cat in enumerate(col_categories):
            if cat == "description" and col_idx < len(row):
                ct = infer_cell_type(row[col_idx])
                if ct != "string" and string_cells:
                    new_row[col_idx] = string_cells[0]

        reconstructed.append(new_row)

    return reconstructed


def _rows_to_csv(header: List[str], rows: List[List[str]]) -> str:
    """Convert header + rows to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    return output.getvalue()
