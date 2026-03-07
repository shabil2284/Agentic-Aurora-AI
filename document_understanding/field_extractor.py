"""
Aurora Field Extractor
=======================
Key-value field extraction with regex pattern matching + spatial proximity heuristics.
Generates candidate_fields arrays when uncertain, with confidence scoring
based on regex match strength and spatial distance.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from document_understanding.models import (
    CandidateField,
    ExplainabilityTrace,
    FieldExtraction,
    OCRBlock,
)

# ── Regex patterns for common fields ──────────────────────────────────────────

FIELD_PATTERNS: Dict[str, List[re.Pattern]] = {
    "invoice_number": [
        re.compile(r"(?:invoice|inv)\s*(?:no|number|#|num)[.:# ]*\s*([\w\-/]+)", re.IGNORECASE),
        re.compile(r"(?:bill|receipt)\s*(?:no|number|#)[.:# ]*\s*([\w\-/]+)", re.IGNORECASE),
        re.compile(r"\b(INV[-/]?\d{2,})\b"),
    ],
    "invoice_date": [
        re.compile(r"(?:invoice\s+)?date[.: ]*\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})", re.IGNORECASE),
        re.compile(r"(?:date|dated)[.: ]*\s*(\d{1,2}\s+\w{3,9}\s+\d{2,4})", re.IGNORECASE),
        re.compile(r"(?:invoice\s+)?date[.: ]*\s*(\w{3,9}\s+\d{1,2},?\s+\d{4})", re.IGNORECASE),
    ],
    "due_date": [
        re.compile(r"due\s*date[.: ]*\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})", re.IGNORECASE),
        re.compile(r"due\s*date[.: ]*\s*(\d{1,2}\s+\w{3,9}\s+\d{2,4})", re.IGNORECASE),
    ],
    "total": [
        re.compile(r"(?:grand\s+)?total[.: ]*\s*[\$₹€£¥]?\s*([\d,]+\.?\d*)", re.IGNORECASE),
        re.compile(r"(?:amount\s+)?(?:due|payable)[.: ]*\s*[\$₹€£¥]?\s*([\d,]+\.?\d*)", re.IGNORECASE),
    ],
    "subtotal": [
        re.compile(r"sub\s*total[.: ]*\s*[\$₹€£¥]?\s*([\d,]+\.?\d*)", re.IGNORECASE),
    ],
    "tax": [
        re.compile(r"(?:tax|vat|gst)\s*(?:amount)?[.: ]*\s*[\$₹€£¥]?\s*([\d,]+\.?\d*)", re.IGNORECASE),
        re.compile(r"(?:sgst|cgst|igst)\s*(?:@\d+%)?[.: ]*\s*[\$₹€£¥]?\s*([\d,]+\.?\d*)", re.IGNORECASE),
    ],
    "tax_rate": [
        re.compile(r"(?:tax|vat|gst)\s*(?:rate)?[.: ]*\s*(\d+\.?\d*\s*%)", re.IGNORECASE),
    ],
    "gstin": [
        re.compile(r"(?:GSTIN|GST\s*(?:No|Number|#|IN))[.: ]*\s*(\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\d)", re.IGNORECASE),
        re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\d)\b"),  # standalone GSTIN
    ],
    "pan": [
        re.compile(r"(?:PAN|PAN\s*(?:No|Number|#))[.: ]*\s*([A-Z]{5}\d{4}[A-Z])", re.IGNORECASE),
        re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b"),  # standalone PAN
    ],
    "vendor": [
        re.compile(r"(?:vendor|seller|supplier|from|company)[.: ]*\s*(.{3,60}?)(?:\n|$)", re.IGNORECASE),
    ],
    "payer": [
        re.compile(r"(?:bill\s*to|buyer|customer|client|payer)[.: ]*\s*(.{3,60}?)(?:\n|$)", re.IGNORECASE),
    ],
    "address": [
        re.compile(r"(?:address)[.: ]*\s*(.{10,150}?)(?:\n\n|\n(?=[A-Z]))", re.IGNORECASE | re.DOTALL),
    ],
    "email": [
        re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),
    ],
    "phone": [
        re.compile(r"(?:phone|tel|mobile|contact)[.: ]*\s*(\+?[\d\s\-()]{7,20})", re.IGNORECASE),
        re.compile(r"\b(\+\d{1,3}[-\s]?\d[\d\s\-()]{6,18})\b"),
    ],
    "po_number": [
        re.compile(r"(?:P\.?O\.?\s*(?:No|Number|#)?|Purchase\s+Order)[.: ]*\s*([\w\-/]+)", re.IGNORECASE),
    ],
}


def extract_fields_from_text(
    text: str,
) -> Tuple[Dict[str, FieldExtraction], List[CandidateField], List[ExplainabilityTrace]]:
    """
    Extract key-value fields from plain text using regex patterns.

    Returns:
        (fields dict, candidate_fields list, traces list)
    """
    fields: Dict[str, FieldExtraction] = {}
    candidates: List[CandidateField] = []
    traces: List[ExplainabilityTrace] = []

    for field_name, patterns in FIELD_PATTERNS.items():
        all_matches: List[Tuple[str, float, str]] = []  # (value, confidence, explain)

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                # Confidence based on specificity of the pattern
                conf = 0.90 if len(patterns) == 1 else 0.85
                # Boost for longer / more specific patterns
                if len(pattern.pattern) > 50:
                    conf = min(conf + 0.05, 0.99)
                explain = f"matched regex {pattern.pattern[:60]}"
                all_matches.append((value, conf, explain))

        if len(all_matches) == 1:
            val, conf, explain = all_matches[0]
            fields[field_name] = FieldExtraction(
                value=val,
                confidence=round(conf, 3),
                explain=explain,
            )
            traces.append(ExplainabilityTrace(
                step=f"field_extraction:{field_name}",
                reason=explain,
                confidence=conf,
            ))

        elif len(all_matches) > 1:
            # Multiple matches → pick best, put others in candidates
            all_matches.sort(key=lambda x: x[1], reverse=True)
            best = all_matches[0]
            fields[field_name] = FieldExtraction(
                value=best[0],
                confidence=round(best[1], 3),
                explain=best[2],
            )
            for val, conf, explain in all_matches[1:]:
                candidates.append(CandidateField(
                    field_name=field_name,
                    value=val,
                    confidence=round(conf, 3),
                    explain=explain,
                ))
            traces.append(ExplainabilityTrace(
                step=f"field_extraction:{field_name}",
                reason=f"Multiple matches ({len(all_matches)}): selected '{best[0]}' (conf={best[1]:.2f})",
                confidence=best[1],
            ))

    return fields, candidates, traces


def extract_fields_from_blocks(
    blocks: List[OCRBlock],
) -> Tuple[Dict[str, FieldExtraction], List[CandidateField], List[ExplainabilityTrace]]:
    """
    Extract key-value fields from OCR blocks using regex + spatial proximity.

    Legacy Antigravity Rule: prefer nearby text to label; fallback to pattern matching.
    """
    traces: List[ExplainabilityTrace] = []

    # 1. Try spatial label→value pairing
    spatial_fields, spatial_candidates = _extract_spatial_pairs(blocks, traces)

    # 2. Fallback: full text regex extraction
    full_text = "\n".join(b.text for b in blocks if b.text)
    regex_fields, regex_candidates, regex_traces = extract_fields_from_text(full_text)
    traces.extend(regex_traces)

    # Merge: spatial takes priority, regex fills gaps
    merged_fields: Dict[str, FieldExtraction] = {**regex_fields}
    for k, v in spatial_fields.items():
        if k not in merged_fields or v.confidence > merged_fields[k].confidence:
            merged_fields[k] = v

    all_candidates = spatial_candidates + regex_candidates

    return merged_fields, all_candidates, traces


def _extract_spatial_pairs(
    blocks: List[OCRBlock],
    traces: List[ExplainabilityTrace],
) -> Tuple[Dict[str, FieldExtraction], List[CandidateField]]:
    """
    Scan for label-value spatial groupings.
    A label block followed by a value block to its right or below.
    """
    fields: Dict[str, FieldExtraction] = {}
    candidates: List[CandidateField] = []

    # Label patterns (simplified)
    label_keywords: Dict[str, List[str]] = {
        "invoice_number": ["invoice no", "invoice number", "inv no", "inv#", "invoice #"],
        "invoice_date": ["invoice date", "date", "dated"],
        "due_date": ["due date", "payment due"],
        "total": ["grand total", "total amount", "total due", "amount due", "total"],
        "subtotal": ["subtotal", "sub total", "sub-total"],
        "tax": ["tax amount", "vat amount", "gst amount", "tax"],
        "vendor": ["vendor", "seller", "from", "supplier"],
        "payer": ["bill to", "buyer", "customer", "client", "ship to"],
        "gstin": ["gstin", "gst no", "gst number"],
        "pan": ["pan", "pan no", "pan number"],
    }

    for i, block in enumerate(blocks):
        if not block.text or not block.bbox or len(block.bbox) < 4:
            continue

        text_lower = block.text.strip().lower().rstrip(":")

        for field_name, keywords in label_keywords.items():
            if field_name in fields:
                continue  # Already found

            if any(text_lower == kw or text_lower.startswith(kw) for kw in keywords):
                # Look for value in same block (after the label)
                parts = re.split(r"[:]\s*", block.text, maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    value = parts[1].strip()
                    fields[field_name] = FieldExtraction(
                        value=value,
                        confidence=0.9,
                        bbox=block.bbox,
                        source_block_id=block.id,
                        explain=f"spatial: label '{text_lower}' in same block, value='{value}'",
                    )
                    continue

                # Look for nearest block to the right or below
                nearest = _find_nearest_value_block(block, blocks[i + 1:])
                if nearest:
                    fields[field_name] = FieldExtraction(
                        value=nearest.text.strip(),
                        confidence=0.8,
                        bbox=nearest.bbox,
                        source_block_id=nearest.id,
                        explain=f"spatial: label '{text_lower}' → nearest value block '{nearest.text.strip()[:30]}'",
                    )

    if fields:
        traces.append(ExplainabilityTrace(
            step="spatial_field_extraction",
            reason=f"Found {len(fields)} fields via spatial label→value pairing",
            confidence=0.85,
        ))

    return fields, candidates


def _find_nearest_value_block(
    label_block: OCRBlock,
    candidates: List[OCRBlock],
    max_distance: float = 100.0,
) -> Optional[OCRBlock]:
    """Find the nearest block to the right of or below the label block."""
    if not label_block.bbox or len(label_block.bbox) < 4:
        return None

    lx2, ly1, ly2 = label_block.bbox[2], label_block.bbox[1], label_block.bbox[3]
    best: Optional[OCRBlock] = None
    best_dist = max_distance

    for block in candidates:
        if not block.text.strip() or not block.bbox or len(block.bbox) < 4:
            continue

        bx1, by1 = block.bbox[0], block.bbox[1]

        # To the right on same line
        if abs(by1 - ly1) < 15 and bx1 > lx2:
            dist = bx1 - lx2
            if dist < best_dist:
                best_dist = dist
                best = block

        # Below
        elif by1 > ly2 and by1 - ly2 < 30 and abs(block.bbox[0] - label_block.bbox[0]) < 50:
            dist = by1 - ly2
            if dist < best_dist:
                best_dist = dist
                best = block

    return best
