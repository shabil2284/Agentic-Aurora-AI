"""
Aurora Edit Patch Generator
=============================
Generates OT-style edit patches compatible with chatbot_editor.py AtomicOperations.
Each patch includes human label, minimal diff preview, candidate fixes, and confidence.

Never deletes original OCR text — only proposes patches.
"""

import uuid
from typing import Any, Dict, List, Optional

from document_understanding.models import (
    EditPatch,
    ExplainabilityTrace,
    ExtractedTable,
    FieldExtraction,
    ReconstructionCandidate,
)


def generate_table_patches(
    tables: List[ExtractedTable],
) -> List[EditPatch]:
    """
    Generate edit patches for table reconstructions.
    If a table was reconstructed (values were shuffled), propose a patch.
    """
    patches: List[EditPatch] = []

    for table in tables:
        if not table.reconstructed or len(table.reconstruction_candidates) < 2:
            continue

        best = table.reconstruction_candidates[0]
        original = table.reconstruction_candidates[-1]  # lowest score = likely original

        if best.score <= original.score:
            continue  # No improvement

        # Build diff preview
        best_lines = best.csv.strip().split("\n")
        orig_lines = original.csv.strip().split("\n")

        diff_lines = []
        for i, (b_line, o_line) in enumerate(zip(best_lines, orig_lines)):
            if b_line != o_line:
                diff_lines.append(f"row_{i}: [{o_line}] -> [{b_line}]")
                if len(diff_lines) >= 3:
                    break

        diff_preview = "; ".join(diff_lines) if diff_lines else "no visible changes"

        patch = EditPatch(
            target=f"table:{table.table_id}",
            op="reconstruct_columns",
            description=f"Detected shuffled values — propose reassign from candidate #1 (score={best.score:.2f})",
            candidate_index=0,
            diff_preview=diff_preview,
            confidence=round(best.score, 3),
            rationale=f"Type-based column reassignment improved score from {original.score:.2f} to {best.score:.2f}",
            candidates=[
                {"index": i, "score": c.score, "csv_preview": c.csv[:200]}
                for i, c in enumerate(table.reconstruction_candidates[:3])
            ],
        )
        patches.append(patch)

    return patches


def generate_field_patches(
    fields: Dict[str, FieldExtraction],
    candidate_fields: List[Any] = None,
) -> List[EditPatch]:
    """
    Generate edit patches for uncertain field extractions.
    When confidence < 0.8 or when candidate alternatives exist.
    """
    patches: List[EditPatch] = []
    candidate_fields = candidate_fields or []

    # Group candidates by field name
    candidates_by_field: Dict[str, List[Any]] = {}
    for cf in candidate_fields:
        fname = cf.field_name if hasattr(cf, "field_name") else cf.get("field_name", "")
        if fname not in candidates_by_field:
            candidates_by_field[fname] = []
        candidates_by_field[fname].append(cf)

    for field_name, extraction in fields.items():
        should_patch = False
        reason = ""

        if extraction.confidence < 0.8:
            should_patch = True
            reason = f"Low confidence ({extraction.confidence:.2f}) for field '{field_name}'"

        if field_name in candidates_by_field:
            should_patch = True
            reason = f"Multiple candidates found for field '{field_name}'"

        if not should_patch:
            continue

        alt_candidates = candidates_by_field.get(field_name, [])
        candidates_list = [
            {"value": extraction.value, "confidence": extraction.confidence, "explain": extraction.explain}
        ]
        for alt in alt_candidates[:2]:  # Up to 3 total candidates
            if hasattr(alt, "value"):
                candidates_list.append({
                    "value": alt.value,
                    "confidence": alt.confidence if hasattr(alt, "confidence") else 0.5,
                    "explain": alt.explain if hasattr(alt, "explain") else "",
                })

        patch = EditPatch(
            target=f"field:{field_name}",
            op="fix_field_value",
            description=reason,
            diff_preview=f"current: '{extraction.value}' (conf={extraction.confidence:.2f})",
            confidence=extraction.confidence,
            rationale=extraction.explain,
            candidates=candidates_list,
        )
        patches.append(patch)

    return patches


def generate_legacy_revert_patch(
    mismatched_rules: Optional[List[str]] = None,
) -> Optional[EditPatch]:
    """
    If pipeline configuration mismatches legacy antigravity settings,
    generate a revert patch.
    """
    if not mismatched_rules:
        return None

    return EditPatch(
        target="pipeline:config",
        op="revert_legacy_settings",
        description=f"Re-enabled legacy antigravity rules: {', '.join(mismatched_rules)}",
        diff_preview=f"rules re-enabled: {mismatched_rules}",
        confidence=1.0,
        rationale="Pipeline deviates from legacy antigravity configuration — auto-reverting",
    )


def generate_validation_patches(
    tables: List[ExtractedTable],
) -> List[EditPatch]:
    """
    Generate patches when numeric totals disagree with invoice-level totals by >1%.
    """
    patches: List[EditPatch] = []

    for table in tables:
        check = table.validation.line_sum_check
        if check.expected_total > 0 and not check.passed:
            diff = abs(check.computed_total - check.expected_total)
            pct = (diff / check.expected_total) * 100

            patch = EditPatch(
                target=f"table:{table.table_id}",
                op="fix_total_discrepancy",
                description=f"Line total sum ({check.computed_total}) disagrees with invoice total ({check.expected_total}) by {pct:.1f}%",
                diff_preview=f"computed={check.computed_total}, expected={check.expected_total}, diff={diff:.2f}",
                confidence=0.7,
                rationale="Probable explanations: missing tax line, unrecognized discounts, or rounding differences",
                candidates=[
                    {"explanation": "Missing tax/duty line not captured in table", "likelihood": 0.4},
                    {"explanation": "Discount applied but not recognized as line item", "likelihood": 0.3},
                    {"explanation": "Rounding differences in European number format", "likelihood": 0.3},
                ],
            )
            patches.append(patch)

    return patches


def generate_all_patches(
    tables: List[ExtractedTable],
    fields: Dict[str, FieldExtraction],
    candidate_fields: List[Any] = None,
    legacy_mismatches: Optional[List[str]] = None,
) -> List[EditPatch]:
    """
    Generate all edit patches in one call.
    """
    patches: List[EditPatch] = []

    patches.extend(generate_table_patches(tables))
    patches.extend(generate_field_patches(fields, candidate_fields))
    patches.extend(generate_validation_patches(tables))

    legacy_patch = generate_legacy_revert_patch(legacy_mismatches)
    if legacy_patch:
        patches.append(legacy_patch)

    return patches
