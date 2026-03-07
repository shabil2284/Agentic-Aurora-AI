"""
Aurora Hierarchy Extractor
===========================
Extracts hierarchical section structure (titles, section headings, paragraphs)
from OCR blocks or plain text using font-size, boldness, and spatial heuristics.
"""

import re
from typing import List, Optional, Tuple

from document_understanding.models import (
    ExplainabilityTrace,
    HierarchyNode,
    OCRBlock,
)

# ── Heading detection patterns ────────────────────────────────────────────────

_HEADING_PATTERNS = [
    re.compile(r"^(CHAPTER|SECTION|PART|APPENDIX)\s+[\dIVXA-Z]+", re.IGNORECASE),
    re.compile(r"^\d+\.\s+[A-Z]"),            # "1. Introduction"
    re.compile(r"^\d+\.\d+\s+"),               # "1.1 Subsection"
    re.compile(r"^\d+\.\d+\.\d+\s+"),          # "1.1.1 Sub-subsection"
    re.compile(r"^[IVX]+\.\s+", re.IGNORECASE),  # "IV. Roman numeral heading"
    re.compile(r"^[A-Z]\.\s+"),                # "A. Letter heading"
]

_DOCUMENT_TITLE_WORDS = {
    "invoice", "receipt", "contract", "agreement", "report",
    "statement", "purchase order", "resume", "cv", "form",
    "certificate", "memo", "memorandum", "notice", "policy",
}


def extract_hierarchy(
    blocks: Optional[List[OCRBlock]] = None,
    plain_text: Optional[str] = None,
) -> Tuple[List[HierarchyNode], List[ExplainabilityTrace]]:
    """
    Extract document hierarchy from OCR blocks or plain text.

    Returns:
        (list of HierarchyNode, list of ExplainabilityTrace)
    """
    traces: List[ExplainabilityTrace] = []

    if blocks and any(b.font_size for b in blocks):
        nodes = _extract_from_styled_blocks(blocks, traces)
    elif blocks:
        nodes = _extract_from_unstyled_blocks(blocks, traces)
    elif plain_text:
        nodes = _extract_from_plain_text(plain_text, traces)
    else:
        traces.append(ExplainabilityTrace(
            step="hierarchy_extraction",
            reason="No input data — returning empty hierarchy",
            confidence=0.0,
        ))
        return [], traces

    return nodes, traces


def _extract_from_styled_blocks(
    blocks: List[OCRBlock],
    traces: List[ExplainabilityTrace],
) -> List[HierarchyNode]:
    """Use font_size and font_style metadata for hierarchy detection."""
    nodes: List[HierarchyNode] = []
    if not blocks:
        return nodes

    # Compute font size statistics
    font_sizes = [b.font_size for b in blocks if b.font_size and b.font_size > 0]
    if not font_sizes:
        return _extract_from_unstyled_blocks(blocks, traces)

    max_fs = max(font_sizes)
    median_fs = sorted(font_sizes)[len(font_sizes) // 2]

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        fs = block.font_size or median_fs
        is_bold = block.font_style and "bold" in block.font_style.lower()

        # Determine hierarchy level based on font size relative to median
        if fs >= max_fs * 0.9 or _is_document_title(text):
            level = 1
        elif fs > median_fs * 1.2 or is_bold:
            level = 2
        elif fs > median_fs * 1.05:
            level = 3
        elif _matches_heading_pattern(text):
            level = _infer_heading_level(text)
        else:
            continue  # body text — skip

        node = HierarchyNode(
            level=level,
            title=text[:200],
            bbox=block.bbox,
            text=text,
        )
        nodes.append(node)

    traces.append(ExplainabilityTrace(
        step="hierarchy_extraction",
        reason=f"Font-size based extraction: max_fs={max_fs}, median_fs={median_fs}, found {len(nodes)} heading nodes",
        confidence=0.85,
    ))

    return nodes


def _extract_from_unstyled_blocks(
    blocks: List[OCRBlock],
    traces: List[ExplainabilityTrace],
) -> List[HierarchyNode]:
    """Fallback: use text patterns and uppercase heuristics."""
    nodes: List[HierarchyNode] = []

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        level = _classify_text_level(text)
        if level is not None:
            nodes.append(HierarchyNode(
                level=level,
                title=text[:200],
                bbox=block.bbox,
                text=text,
            ))

    traces.append(ExplainabilityTrace(
        step="hierarchy_extraction",
        reason=f"Pattern-based extraction (no font metadata): found {len(nodes)} heading nodes",
        confidence=0.65,
    ))

    return nodes


def _extract_from_plain_text(
    text: str,
    traces: List[ExplainabilityTrace],
) -> List[HierarchyNode]:
    """Extract hierarchy from plain text line-by-line."""
    nodes: List[HierarchyNode] = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        level = _classify_text_level(stripped)
        if level is not None:
            nodes.append(HierarchyNode(
                level=level,
                title=stripped[:200],
                bbox=[],
                text=stripped,
            ))

    traces.append(ExplainabilityTrace(
        step="hierarchy_extraction",
        reason=f"Plain text extraction: {len(lines)} lines scanned, {len(nodes)} headings found",
        confidence=0.5,
    ))

    return nodes


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_document_title(text: str) -> bool:
    """Check if text looks like a document title."""
    t_lower = text.lower().strip()
    # Short all-uppercase text
    if text.isupper() and len(text.split()) <= 6:
        return True
    # Known title words
    return any(kw in t_lower for kw in _DOCUMENT_TITLE_WORDS)


def _matches_heading_pattern(text: str) -> bool:
    """Check if text matches any known heading pattern."""
    return any(p.match(text.strip()) for p in _HEADING_PATTERNS)


def _infer_heading_level(text: str) -> int:
    """Infer heading level from numbering pattern."""
    stripped = text.strip()
    if re.match(r"^\d+\.\d+\.\d+\s+", stripped):
        return 4
    if re.match(r"^\d+\.\d+\s+", stripped):
        return 3
    if re.match(r"^\d+\.\s+", stripped):
        return 2
    return 2


def _classify_text_level(text: str) -> Optional[int]:
    """
    Classify a line of text as a heading level or None (body text).
    Uses pattern matching, uppercase detection, and length heuristics.
    """
    stripped = text.strip()
    if not stripped:
        return None

    # Document title
    if _is_document_title(stripped) and len(stripped.split()) <= 8:
        return 1

    # All uppercase short text (likely heading)
    if stripped.isupper() and 2 <= len(stripped.split()) <= 10 and len(stripped) <= 80:
        return 2

    # Numbered heading patterns
    if _matches_heading_pattern(stripped):
        return _infer_heading_level(stripped)

    # Single short line ending with colon — could be a label/section
    if stripped.endswith(":") and len(stripped.split()) <= 6:
        return 3

    return None
