"""
Aurora Tabularizer — utils/tabularizer.py
==========================================
Non-destructive tabular detection and flattening module.
Reads master_json, produces a tabular_preview in memory.
NEVER modifies master_json unless apply_edits() is explicitly called
and confirmed by the user.
"""

import re
import json
import uuid
import copy
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─── Currency / Date helpers ───────────────────────────────────────────────────

CURRENCY_RE = re.compile(r"[\$€£₹¥]?\s*[\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?")
AMOUNT_RE   = re.compile(r"[\$€£₹]?[\d,\.]+(?:\.\d{1,2})?")

def _parse_currency(s: Any) -> Optional[float]:
    """Convert formatted currency string to float. Handles both , and . separators."""
    if s is None:
        return None
    raw = str(s).strip()
    raw = re.sub(r"[€$£₹¥\s]", "", raw)
    # European format: 3.878,50 → 3878.50
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d{1,2})?$", raw):
        raw = raw.replace(".", "").replace(",", ".")
    # US format: 3,878.50 → 3878.50
    elif re.match(r"^\d{1,3}(,\d{3})+(\.\d{1,2})?$", raw):
        raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d",
]

def _parse_date(s: Any) -> str:
    """Try to parse s into ISO8601 date string. Returns raw string on failure."""
    if s is None:
        return ""
    raw = str(s).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw


# ─── Tabular Detection ────────────────────────────────────────────────────────

TABULAR_ARRAY_KEYS = {
    "line items", "lineitems", "items", "transactions",
    "entries", "rows", "table", "records", "line_items",
    "transactions", "data", "contacts", "employees",
}

TABULAR_DOC_KEYWORDS = {
    "invoice", "bill", "statement", "transaction", "payment",
    "line items", "subtotal", "balance", "receipt", "purchase order",
}

HR_NAME_KEYS  = {"name", "full name", "employee name", "contact name"}
HR_EMAIL_KEYS = {"email", "email address", "e-mail"}
HR_PHONE_KEYS = {"phone", "mobile", "contact", "phone number"}

NEGATIVE_KEYWORDS = {
    "summary", "objective", "experience", "education", "skills",
    "references", "dear", "sincerely", "whereas", "party", "clause",
}

# ─── Strict Disallow Keywords (hard gate — override all positive signals) ──────
_DISALLOW_KEYWORDS = {
    "resume", "curriculum vitae", "cv", "cover letter",
    "letter of recommendation", "experience letter",
    "contract", "agreement", "legal notice", "memorandum",
    "certificate of employment", "to whom it may concern",
    "dear sir", "dear madam", "yours sincerely", "yours faithfully",
}

# ─── Positive Transactional Keywords (boost detection) ─────────────────────────
_TRANSACTIONAL_KEYWORDS = {
    "invoice", "bill", "bank statement", "transaction statement",
    "payroll", "employee list", "vendor list", "inventory",
    "item list", "contact list", "customer list", "purchase order",
    "sales order", "delivery note", "receipt", "quotation",
    "line items", "debit", "credit", "balance",
}

# ─── SQL-safe table name mapping ──────────────────────────────────────────────
_DOC_TYPE_TABLE_NAME = {
    "invoice":        "invoice_table",
    "bank_statement": "bank_transactions",
    "hr_list":        "employees",
    "contact_list":   "contacts",
    "generic_table":  "document_table",
}


class TabularDetector:
    """
    Analyses master_json (dict or list of dicts) and returns a
    (is_tabular, confidence, reason) tuple.
    """

    @staticmethod
    def detect(master_json: Any) -> Tuple[bool, float, str]:
        if not master_json:
            return False, 0.0, "empty"

        # ── HARD GATE: Strict disallow for narrative document types ─────────
        text_blob_check = json.dumps(master_json).lower()
        disallow_hits = [kw for kw in _DISALLOW_KEYWORDS if kw in text_blob_check]
        # Strong signals: a single hit of these blocks table creation
        _strong_disallow = {"resume", "cv", "curriculum vitae", "cover letter", "legal notice"}
        strong_hits = [h for h in disallow_hits if h in _strong_disallow]
        if strong_hits or len(disallow_hits) >= 2:
            return False, 0.0, f"disallowed_doc_type: {disallow_hits[:3]}"

        score = 0.0
        reasons = []

        # Normalise to dict for analysis
        if isinstance(master_json, list):
            # TABLE_MODE already produces flat list → high confidence
            if len(master_json) >= 2 and all(isinstance(r, dict) for r in master_json):
                score += 0.50
                reasons.append("flat list of dicts")
            obj = master_json[0] if master_json else {}
        elif isinstance(master_json, dict):
            obj = master_json
        else:
            return False, 0.0, "not a dict or list"

        # Scan top-level keys for known array headers (name-based)
        for k in obj.keys():
            if k.lower().strip() in TABULAR_ARRAY_KEYS:
                val = obj[k]
                if isinstance(val, list) and len(val) >= 1:
                    score += 0.50
                    reasons.append(f"array key '{k}'")
                    if TabularDetector._has_amount_fields(val):
                        score += 0.30
                        reasons.append("currency/numeric fields in array")
                    break

        # ── UNIVERSAL STRUCTURE-BASED SCAN (name-independent) ─────────────
        # Recursively find ANY array of objects with ≥2 consistent keys,
        # regardless of the field name. This catches Items, LineItems,
        # Data, Entries, etc. that aren't in TABULAR_ARRAY_KEYS.
        struct_hit = TabularDetector._find_repeating_arrays(master_json)
        if struct_hit:
            _s_key, _s_count, _s_keys = struct_hit
            if _s_count >= 1 and len(_s_keys) >= 2:
                # Only add if the name-based check didn't already fire
                if not any("array key" in r for r in reasons):
                    score += 0.55
                    reasons.append(f"struct_array:'{_s_key}' ({_s_count} objs, {len(_s_keys)} keys)")
                    if TabularDetector._has_amount_fields(
                        TabularDetector._extract_array(master_json, _s_key)
                    ):
                        score += 0.20
                        reasons.append("currency/numeric in struct_array")

        # Document-level keywords
        text_blob = json.dumps(master_json).lower()
        matches = [kw for kw in TABULAR_DOC_KEYWORDS if kw in text_blob]
        if matches:
            score += min(len(matches) * 0.08, 0.30)
            reasons.append(f"doc keywords: {matches[:3]}")

        # Positive transactional keyword boost
        trans_matches = [kw for kw in _TRANSACTIONAL_KEYWORDS if kw in text_blob]
        if trans_matches:
            score += min(len(trans_matches) * 0.06, 0.20)
            reasons.append(f"transactional: {trans_matches[:3]}")

        # HR / contact list detection
        if isinstance(master_json, list) and len(master_json) >= 2:
            if TabularDetector._is_hr_list(master_json):
                score += 0.35              # was 0.20
                reasons.append("HR/contact list pattern")

        # Negative signals
        neg_matches = [kw for kw in NEGATIVE_KEYWORDS if kw in text_blob]
        if neg_matches:
            score -= min(len(neg_matches) * 0.10, 0.30)
            reasons.append(f"negative signals: {neg_matches[:3]}")

        # Clamp
        confidence = max(0.0, min(1.0, score))
        is_tabular = confidence >= 0.75
        return is_tabular, round(confidence, 3), ", ".join(reasons)

    @staticmethod
    def _has_amount_fields(rows: List[dict]) -> bool:
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            for v in row.values():
                if AMOUNT_RE.match(str(v).strip()):
                    return True
        return False

    @staticmethod
    def _is_hr_list(records: List[dict]) -> bool:
        name_hit = email_hit = phone_hit = 0
        for record in records[:10]:
            if not isinstance(record, dict):
                continue
            keys_lower = {k.lower() for k in record.keys()}
            if keys_lower & HR_NAME_KEYS:
                name_hit += 1
            if keys_lower & HR_EMAIL_KEYS:
                email_hit += 1
            if keys_lower & HR_PHONE_KEYS:
                phone_hit += 1
        n = min(len(records), 10)
        return name_hit / n >= 0.5 and (email_hit / n >= 0.4 or phone_hit / n >= 0.4)

    @staticmethod
    def _find_repeating_arrays(obj: Any, max_depth: int = 5) -> Optional[Tuple[str, int, set]]:
        """
        Recursively scan JSON for ANY array of objects with ≥2 consistent keys.
        Returns (field_name, num_objects, common_keys) or None.
        Structure-based — completely independent of field names.
        """
        if max_depth <= 0:
            return None

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list) and len(v) >= 1:
                    # Check: are the elements objects with consistent keys?
                    dict_elements = [e for e in v if isinstance(e, dict)]
                    if len(dict_elements) >= 1:
                        # Gather all keys across elements
                        all_keys = [set(d.keys()) for d in dict_elements]
                        common = all_keys[0]
                        for ks in all_keys[1:]:
                            common = common & ks
                        if len(common) >= 2:
                            return (k, len(dict_elements), common)
                # Recurse into nested dicts
                elif isinstance(v, dict):
                    result = TabularDetector._find_repeating_arrays(v, max_depth - 1)
                    if result:
                        return result

        elif isinstance(obj, list):
            for item in obj[:5]:  # Only check first few
                if isinstance(item, dict):
                    result = TabularDetector._find_repeating_arrays(item, max_depth - 1)
                    if result:
                        return result

        return None

    @staticmethod
    def _extract_array(obj: Any, key: str) -> List[dict]:
        """Extract the array for a given key from a possibly nested structure."""
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], list):
                return [e for e in obj[key] if isinstance(e, dict)]
            for v in obj.values():
                if isinstance(v, dict):
                    result = TabularDetector._extract_array(v, key)
                    if result:
                        return result
        return []


# ─── Tabularizer ─────────────────────────────────────────────────────────────

MAX_ROWS_AUTO = 5000  # rows above this need explicit user confirm


class Tabularizer:
    """
    Flattens master_json into a tabular_preview without modifying master_json.
    Supports: invoice (Line Items array), bank statement, HR list, generic table.
    """

    @staticmethod
    def build_preview(master_json: Any, confidence: float = 1.0) -> Dict:
        """
        Returns a tabular_preview dict:
        {
          "rows"      : [ {col: val, ..., "__source_row_id": "row-0", "__master_path": "..."}, ...],
          "columns"   : [col_name, ...],    # UI display order
          "confidence": float,
          "doc_type"  : str,
          "preview_only": bool,             # True if > MAX_ROWS_AUTO rows
        }
        """
        if isinstance(master_json, dict):
            return Tabularizer._flatten_dict(master_json, confidence)
        elif isinstance(master_json, list):
            return Tabularizer._flatten_list(master_json, confidence)
        return {"rows": [], "columns": [], "confidence": confidence, "doc_type": "unknown", "preview_only": False}

    # ── Invoice / hierarchical dict ────────────────────────────────────────────
    @staticmethod
    def _flatten_dict(obj: dict, confidence: float) -> Dict:
        # Find the array key (Line Items etc.) — name-based first, structural fallback
        array_key = None
        for k in obj.keys():
            if k.lower().strip() in TABULAR_ARRAY_KEYS and isinstance(obj[k], list) and obj[k]:
                array_key = k
                break
        # Structure-based fallback: find ANY array of objects with consistent keys
        if array_key is None:
            struct_hit = TabularDetector._find_repeating_arrays(obj, max_depth=2)
            if struct_hit:
                _sk, _sc, _skeys = struct_hit
                if _sc >= 1 and len(_skeys) >= 2 and _sk in obj:
                    array_key = _sk

        # Scalar top-level fields (primitives only)
        scalars = {k: v for k, v in obj.items() if not isinstance(v, (list, dict))}

        # Summary fields that come after the array (Subtotal, Tax, Grand Total etc.)
        summary_keys = [k for k in obj.keys() if k != array_key and not isinstance(obj[k], (list, dict))]

        rows = []
        columns = []

        if array_key and obj[array_key]:
            items = obj[array_key]
            # Infer column order: scalars + item keys + (any remaining summary)
            item_keys = []
            for item in items[:5]:
                if isinstance(item, dict):
                    for ik in item.keys():
                        if ik not in item_keys:
                            item_keys.append(ik)

            columns = list(scalars.keys()) + item_keys
            # Add summary cols that are not already scalar cols
            for sk in summary_keys:
                if sk not in columns:
                    columns.append(sk)

            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                row = {}
                # Repeat scalar top-level into every row
                for k, v in scalars.items():
                    row[k] = v
                # Item-specific fields
                for ik in item_keys:
                    raw_val = item.get(ik, "")
                    row[ik] = raw_val
                # Summary fields (same for all rows)
                for sk in summary_keys:
                    if sk not in row:
                        row[sk] = obj.get(sk, "")
                # Metadata
                row["__source_row_id"]  = f"row-{idx}"
                row["__master_path"]    = f"/{array_key}/{idx}"
                row["__created_at_confidence"] = confidence
                rows.append(row)
        else:
            # No array — single row from scalar fields
            row = dict(scalars)
            row["__source_row_id"]  = "row-0"
            row["__master_path"]    = "/"
            row["__created_at_confidence"] = confidence
            rows = [row]
            columns = list(scalars.keys())

        preview_only = len(rows) > MAX_ROWS_AUTO
        if preview_only:
            rows = rows[:MAX_ROWS_AUTO]

        doc_type = Tabularizer._infer_doc_type(obj)
        return {
            "rows": rows,
            "columns": columns,
            "confidence": confidence,
            "doc_type": doc_type,
            "preview_only": preview_only,
        }

    # ── Flat list of dicts (TABLE_MODE / HR lists) ─────────────────────────────
    @staticmethod
    def _flatten_list(records: List[dict], confidence: float) -> Dict:
        columns = []
        for r in records[:5]:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in columns:
                        columns.append(k)

        rows = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            row = dict(record)
            row["__source_row_id"]  = f"row-{idx}"
            row["__master_path"]    = f"/{idx}"
            row["__created_at_confidence"] = confidence
            rows.append(row)

        preview_only = len(rows) > MAX_ROWS_AUTO
        if preview_only:
            rows = rows[:MAX_ROWS_AUTO]

        doc_type = Tabularizer._infer_doc_type(records[0] if records else {})
        return {
            "rows": rows,
            "columns": columns,
            "confidence": confidence,
            "doc_type": doc_type,
            "preview_only": preview_only,
        }

    # ── Apply staged edits back to master_json (atomic, pointer-based) ─────────
    @staticmethod
    def apply_edits(master_json: Any, staged_rows: List[dict]) -> Any:
        """
        Updates master_json in-place using __master_path from each staged row.
        Creates a deep-copy backup in OS temp before mutating.
        Returns the updated master_json.
        """
        # Snapshot backup
        Tabularizer._save_snapshot(master_json)

        updated = copy.deepcopy(master_json)

        for row in staged_rows:
            path = row.get("__master_path", "")
            rid  = row.get("__source_row_id", "")
            if not path:
                continue

            meta_keys = {"__source_row_id", "__master_path", "__created_at_confidence"}
            data_fields = {k: v for k, v in row.items() if k not in meta_keys}

            # Resolve the JSON pointer
            target = Tabularizer._resolve_path(updated, path)
            if target is None:
                # New row — append to the array
                Tabularizer._append_at_path(updated, path, data_fields)
                continue

            if isinstance(target, dict):
                for k, v in data_fields.items():
                    if k in target:        # only update fields that exist at this path
                        target[k] = v
            # Lists/arrays: we update the item, not a list itself

        return updated

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_path(obj: Any, path: str) -> Optional[Any]:
        """Simple JSON Pointer resolver (RFC 6901 subset)."""
        parts = [p for p in path.lstrip("/").split("/") if p]
        cur = obj
        try:
            for p in parts:
                if isinstance(cur, list):
                    cur = cur[int(p)]
                elif isinstance(cur, dict):
                    cur = cur[p]
                else:
                    return None
            return cur
        except (IndexError, KeyError, ValueError):
            return None

    @staticmethod
    def _append_at_path(obj: Any, path: str, new_item: dict) -> None:
        """Append new_item to the list one level above the path's leaf."""
        parts = [p for p in path.lstrip("/").split("/") if p]
        if not parts:
            return
        parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        parent = Tabularizer._resolve_path(obj, parent_path) if parent_path != "/" else obj
        if isinstance(parent, list):
            new_item["__source_row_id"] = f"row-{len(parent)}"
            new_item["__row_version"]   = 0
            parent.append(new_item)

    @staticmethod
    def _save_snapshot(master_json: Any) -> str:
        """Save master_json snapshot to OS temp dir. Returns file path."""
        tmp_dir = tempfile.gettempdir()
        snap_path = os.path.join(tmp_dir, f"aurora_snapshot_{uuid.uuid4().hex}.json")
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(master_json, f, indent=2, ensure_ascii=False)
        print(f"[Tabularizer] Snapshot saved → {snap_path}")
        return snap_path

    @staticmethod
    def _infer_doc_type(obj: Any) -> str:
        if not obj:
            return "generic_table"
        if isinstance(obj, dict):
            text = json.dumps(obj).lower()
            if any(k in text for k in ("invoice", "vendor", "line item")):
                return "invoice"
            if any(k in text for k in ("transaction", "debit", "credit", "balance")):
                return "bank_statement"
            if any(k in text for k in ("employee", "contact", "phone", "mobile")):
                return "hr_list"
            if any(k in text for k in ("contact", "email", "phone")):
                return "contact_list"
        return "generic_table"

    @staticmethod
    def infer_table_name(obj: Any) -> str:
        """Returns a SQL-safe table name based on document type."""
        doc_type = Tabularizer._infer_doc_type(obj)
        return _DOC_TYPE_TABLE_NAME.get(doc_type, "document_table")
