"""
Aurora Output Formatter
========================
Multi-format output generation:
  - JSON: primary DocumentUnderstandingResult serialized
  - CSV: one CSV string per detected table
  - Human summary: one-line natural language description
  - Repro command for legacy mode re-execution
"""

import csv
import io
from typing import Dict, List

from document_understanding.models import (
    DocumentUnderstandingResult,
    ExtractedTable,
    OutputBundle,
)


def format_outputs(result: DocumentUnderstandingResult) -> OutputBundle:
    """
    Generate all output formats from the top-level result.
    """
    csv_files = generate_csv_files(result)
    human_summary = generate_human_summary(result)

    return OutputBundle(
        json_data=result.model_dump(mode="json"),
        csv_files=csv_files,
        human_summary=human_summary,
    )


def generate_csv_files(result: DocumentUnderstandingResult) -> Dict[str, str]:
    """Generate one CSV string per detected table."""
    csv_files: Dict[str, str] = {}

    for page in result.pages:
        for table in page.tables:
            csv_str = _table_to_csv(table)
            csv_files[f"{table.table_id}.csv"] = csv_str

    return csv_files


def generate_human_summary(result: DocumentUnderstandingResult) -> str:
    """Generate a one-line human-readable validation summary."""
    doc_type = result.document_type.label
    doc_conf = result.document_type.confidence

    # Count tables and fields
    total_tables = sum(len(p.tables) for p in result.pages)
    total_fields = sum(len(p.fields) for p in result.pages)
    total_rows = 0
    for page in result.pages:
        for table in page.tables:
            total_rows += len(set(c.row for c in table.cells))

    # Check for field extractions
    field_summary_parts = []
    for page in result.pages:
        for fname, fval in page.fields.items():
            if fname in ("invoice_number", "invoice_date", "total"):
                field_summary_parts.append(f"{fname}={fval.value}")

    field_str = ", ".join(field_summary_parts[:3]) if field_summary_parts else "no key fields found"

    # Check for patches
    patch_count = len(result.edit_patches)
    patch_str = f" {patch_count} edit patch(es) proposed." if patch_count > 0 else ""

    # Check for warnings
    warning_str = f" Warnings: {', '.join(result.warnings[:2])}." if result.warnings else ""

    # Validation issues
    validation_issues = []
    for page in result.pages:
        for table in page.tables:
            if not table.validation.line_sum_check.passed:
                validation_issues.append(f"table {table.table_id} total mismatch")

    validation_str = f" Validation issues: {', '.join(validation_issues)}." if validation_issues else ""

    summary = (
        f"Found {doc_type} (conf={doc_conf:.2f}) with {total_tables} table(s) "
        f"({total_rows} rows) and {total_fields} field(s). "
        f"{field_str}.{patch_str}{warning_str}{validation_str}"
    )

    return summary.strip()


def _table_to_csv(table: ExtractedTable) -> str:
    """Convert an ExtractedTable to CSV string."""
    # If we have reconstruction candidates, use the best one
    if table.reconstruction_candidates:
        best = max(table.reconstruction_candidates, key=lambda c: c.score)
        return best.csv

    # Otherwise build from cells
    if not table.cells:
        return ""

    max_row = max(c.row for c in table.cells)
    max_col = max(c.col for c in table.cells)

    # Build grid
    grid = [[""] * max_col for _ in range(max_row)]
    for cell in table.cells:
        if 1 <= cell.row <= max_row and 1 <= cell.col <= max_col:
            grid[cell.row - 1][cell.col - 1] = cell.text

    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    if table.detected_header_row:
        writer.writerow(table.detected_header_row)

    for row in grid:
        writer.writerow(row)

    return output.getvalue()
