"""
Aurora Universal Document Understanding
========================================
Deterministic, single-pass extraction pipeline for structured document analysis.
Detects document type, captures hierarchical structure, reconstructs tables
(including shuffled/misaligned values), extracts key-value fields, and emits
edit-sync patches for chatbot integration.

Legacy Antigravity Mode enforced.
"""

from document_understanding.pipeline import AuroraDocumentPipeline

__all__ = ["AuroraDocumentPipeline"]
