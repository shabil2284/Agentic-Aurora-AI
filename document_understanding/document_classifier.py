"""
Aurora Document Classifier
===========================
Detects document class from OCR blocks or plain text using keyword scoring,
structural pattern analysis, and heuristic confidence computation.

Supported classes:
  invoice, receipt, purchase-order, contract, form, report, resume, table-only, mixed
"""

import re
from typing import Dict, List, Optional, Tuple

from document_understanding.models import (
    DocumentTypeResult,
    ExplainabilityTrace,
    OCRBlock,
)

# ── Keyword dictionaries (case-insensitive) ───────────────────────────────────

_KEYWORD_MAP: Dict[str, List[str]] = {
    "invoice": [
        "invoice", "inv no", "inv#", "invoice number", "invoice no",
        "bill to", "ship to", "due date", "payment terms", "subtotal",
        "tax invoice", "proforma", "gst invoice", "vat invoice",
    ],
    "receipt": [
        "receipt", "receipt no", "received from", "payment received",
        "cash receipt", "transaction id", "thank you for your purchase",
        "amount paid", "change due",
    ],
    "purchase-order": [
        "purchase order", "po number", "po no", "po#", "p.o.",
        "delivery date", "ship via", "requisition", "buyer",
    ],
    "contract": [
        "agreement", "contract", "terms and conditions", "parties",
        "whereas", "hereinafter", "effective date", "termination",
        "governing law", "indemnification", "liability", "arbitration",
        "witness whereof", "signatory",
    ],
    "form": [
        "application form", "registration form", "enrollment",
        "please fill", "applicant", "signature", "date of birth",
        "nationality", "passport no", "id number", "checkbox",
    ],
    "report": [
        "report", "executive summary", "findings", "conclusion",
        "recommendations", "analysis", "quarterly", "annual report",
        "fiscal year", "appendix", "figure", "chart",
    ],
    "resume": [
        "resume", "curriculum vitae", "cv", "objective", "experience",
        "education", "skills", "references", "certifications",
        "professional summary", "work history", "qualifications",
    ],
}

# Weighted keywords — these are strong signals worth more
_STRONG_SIGNALS: Dict[str, List[str]] = {
    "invoice": ["tax invoice", "invoice number", "invoice no", "inv#", "gst invoice"],
    "receipt": ["receipt no", "payment received", "cash receipt"],
    "purchase-order": ["purchase order", "po number"],
    "contract": ["terms and conditions", "whereas", "hereinafter", "witness whereof"],
    "resume": ["curriculum vitae", "professional summary", "work history"],
}


def classify_document(
    blocks: Optional[List[OCRBlock]] = None,
    plain_text: Optional[str] = None,
) -> Tuple[DocumentTypeResult, List[ExplainabilityTrace]]:
    """
    Classify a document into one of the supported types.

    Args:
        blocks:     List of OCRBlock objects (from OCR provider).
        plain_text: Raw document text (fallback if no blocks provided).

    Returns:
        (DocumentTypeResult, list of ExplainabilityTrace entries)
    """
    traces: List[ExplainabilityTrace] = []

    # Build full text for keyword scanning
    if blocks:
        full_text = " ".join(b.text for b in blocks if b.text).lower()
    elif plain_text:
        full_text = plain_text.lower()
    else:
        result = DocumentTypeResult(label="mixed", confidence=0.1)
        traces.append(ExplainabilityTrace(
            step="document_classification",
            reason="No input text or blocks provided — defaulting to 'mixed'",
            confidence=0.1,
        ))
        return result, traces

    # ── Score each document type ──────────────────────────────────────────
    scores: Dict[str, float] = {}

    for doc_type, keywords in _KEYWORD_MAP.items():
        score = 0.0
        matched_keywords: List[str] = []

        for kw in keywords:
            if kw in full_text:
                score += 1.0
                matched_keywords.append(kw)

        # Strong signals get a 2x boost
        strong_kws = _STRONG_SIGNALS.get(doc_type, [])
        for skw in strong_kws:
            if skw in full_text:
                score += 2.0  # additional boost on top of base

        scores[doc_type] = score

        if matched_keywords:
            traces.append(ExplainabilityTrace(
                step="keyword_scoring",
                reason=f"'{doc_type}' matched keywords: {matched_keywords} (score={score:.1f})",
                confidence=min(score / 8.0, 1.0),
            ))

    # ── Structural heuristics ─────────────────────────────────────────────
    has_table_patterns = _detect_table_structure(full_text)
    line_count = full_text.count("\n") + 1

    if has_table_patterns and all(scores.get(t, 0) < 2.0 for t in scores):
        scores["table-only"] = 3.0
        traces.append(ExplainabilityTrace(
            step="structural_analysis",
            reason="Detected tabular patterns (aligned columns/numbers) with no strong doc-type keywords",
            confidence=0.7,
        ))

    # ── Pick winner ───────────────────────────────────────────────────────
    if not scores or max(scores.values()) == 0:
        result = DocumentTypeResult(label="mixed", confidence=0.3)
        traces.append(ExplainabilityTrace(
            step="document_classification",
            reason="No keywords matched any document type — classified as 'mixed'",
            confidence=0.3,
        ))
        return result, traces

    winner = max(scores, key=scores.get)  # type: ignore[arg-type]
    max_score = scores[winner]
    total_score = sum(scores.values()) or 1.0

    # Confidence: ratio of winner to total, scaled
    raw_confidence = max_score / total_score
    # Also factor in absolute score strength
    absolute_factor = min(max_score / 6.0, 1.0)
    confidence = round(0.5 * raw_confidence + 0.5 * absolute_factor, 3)
    confidence = max(0.1, min(confidence, 0.99))

    # Check for mixed: if runner-up is close
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        runner_up_score = sorted_scores[1][1]
        if runner_up_score > 0 and (max_score - runner_up_score) / max_score < 0.2:
            winner = "mixed"
            confidence = round(confidence * 0.7, 3)
            traces.append(ExplainabilityTrace(
                step="ambiguity_detection",
                reason=f"Top two types are close: {sorted_scores[0]} vs {sorted_scores[1]} — classified as 'mixed'",
                confidence=confidence,
            ))

    result = DocumentTypeResult(label=winner, confidence=confidence)
    traces.append(ExplainabilityTrace(
        step="document_classification",
        reason=f"Final classification: '{winner}' with scores={dict(sorted_scores[:3])}",
        confidence=confidence,
    ))

    return result, traces


def _detect_table_structure(text: str) -> bool:
    """
    Legacy Antigravity Rule: detect table presence by looking for
    3+ column alignments or repeated numeric patterns across consecutive lines.
    """
    lines = text.split("\n")
    numeric_line_count = 0
    separator_line_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Count lines with multiple numeric tokens
        numeric_tokens = re.findall(r"\b\d[\d,.]*\b", stripped)
        if len(numeric_tokens) >= 2:
            numeric_line_count += 1
        # Count lines with tabular separators
        if "\t" in stripped or "  " in stripped or "|" in stripped:
            separator_line_count += 1

    return numeric_line_count >= 3 or separator_line_count >= 3
