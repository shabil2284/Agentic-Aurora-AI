"""
Aurora Document Understanding — Output Schema Models
=====================================================
Pydantic v2 models implementing the strict output JSON schema.
Every field, table, and patch carries confidence scores and
explainability traces per the Universal Document Understanding spec.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Atomic building blocks ────────────────────────────────────────────────────


class DocumentTypeResult(BaseModel):
    """Classification label with confidence."""
    label: str = Field(..., description="One of: invoice, receipt, purchase-order, contract, form, report, resume, table-only, mixed")
    confidence: float = Field(..., ge=0.0, le=1.0)


class OCRBlock(BaseModel):
    """Single OCR block with spatial + style metadata."""
    id: str = Field(default_factory=lambda: f"b{uuid.uuid4().hex[:6]}")
    text: str = ""
    bbox: List[float] = Field(default_factory=list, description="[x1, y1, x2, y2]")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    page: int = 1
    font_size: Optional[float] = None
    font_style: Optional[str] = None  # bold, italic, normal


class HierarchyNode(BaseModel):
    """Section / subsection in the document hierarchy."""
    level: int = Field(..., ge=1)
    title: str = ""
    bbox: List[float] = Field(default_factory=list)
    text: str = ""


class FieldExtraction(BaseModel):
    """A single extracted key-value field with provenance."""
    value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: List[float] = Field(default_factory=list)
    source_block_id: Optional[str] = None
    explain: str = ""
    canonical_name: Optional[str] = None
    inferred_type: Optional[str] = None
    unit: Optional[str] = None


class CandidateField(BaseModel):
    """Plausible extraction when uncertain."""
    field_name: str
    value: Any
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explain: str = ""


# ── Table models ──────────────────────────────────────────────────────────────


class TableCell(BaseModel):
    """Single table cell with type inference."""
    row: int
    col: int
    text: str = ""
    bbox: List[float] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    inferred_type: str = "string"  # string, integer, decimal, currency, date, percentage


class ReconstructionCandidate(BaseModel):
    """One possible table reconstruction with score."""
    score: float = Field(..., ge=0.0, le=1.0)
    csv: str = ""


class LineSumCheck(BaseModel):
    """Validation: do line totals sum to invoice total?"""
    expected_total: float = 0.0
    computed_total: float = 0.0
    passed: bool = True


class TableValidation(BaseModel):
    """Aggregated validation results for a table."""
    line_sum_check: LineSumCheck = Field(default_factory=LineSumCheck)


class ExtractedTable(BaseModel):
    """A fully reconstructed table with candidates and validation."""
    table_id: str = Field(default_factory=lambda: f"t{uuid.uuid4().hex[:4]}")
    bbox: List[float] = Field(default_factory=list)
    detected_header_row: List[str] = Field(default_factory=list)
    reconstructed: bool = False
    reconstruction_candidates: List[ReconstructionCandidate] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)
    validation: TableValidation = Field(default_factory=TableValidation)
    large_table: bool = False


# ── Explainability ────────────────────────────────────────────────────────────


class ExplainabilityTrace(BaseModel):
    """Audit trace for a non-trivial decision."""
    step: str
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Edit patches ──────────────────────────────────────────────────────────────


class EditPatch(BaseModel):
    """OT-style operational diff for chatbot apply/reject."""
    patch_id: str = Field(default_factory=lambda: f"p{uuid.uuid4().hex[:4]}")
    target: str = ""  # e.g. "table:t1", "field:invoice_number"
    op: str = ""  # reconstruct_columns, fix_field_value, revert_legacy_settings
    description: str = ""
    candidate_index: Optional[int] = None
    diff_preview: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    candidates: List[Dict[str, Any]] = Field(default_factory=list)


# ── Page-level result ─────────────────────────────────────────────────────────


class PageResult(BaseModel):
    """All extractions for a single page."""
    page_num: int = 1
    blocks: List[OCRBlock] = Field(default_factory=list)
    hierarchy: List[HierarchyNode] = Field(default_factory=list)
    fields: Dict[str, FieldExtraction] = Field(default_factory=dict)
    candidate_fields: List[CandidateField] = Field(default_factory=list)
    tables: List[ExtractedTable] = Field(default_factory=list)
    explainability: List[ExplainabilityTrace] = Field(default_factory=list)


# ── Top-level output ──────────────────────────────────────────────────────────


class OutputBundle(BaseModel):
    """Multi-format output container."""
    json_data: Optional[Dict[str, Any]] = None  # serialized top-level result
    csv_files: Dict[str, str] = Field(default_factory=dict)  # table_id -> csv string
    human_summary: str = ""
    repro_command: str = "REVERT_LEGACY=antigravity_legacy && RUN_AURORA_POSTPROCESS --mode legacy --emit-json --emit-csv --include-explainability"


class DocumentUnderstandingResult(BaseModel):
    """
    Top-level output schema for Aurora Universal Document Understanding.
    This is the single canonical response object.
    """
    document_type: DocumentTypeResult = Field(default_factory=lambda: DocumentTypeResult(label="mixed", confidence=0.0))
    pages: List[PageResult] = Field(default_factory=list)
    outputs: OutputBundle = Field(default_factory=OutputBundle)
    edit_patches: List[EditPatch] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reliability: str = "high"  # high, medium, low
