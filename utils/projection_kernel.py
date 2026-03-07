"""
Aurora Projection & Semantic Flattening Kernel  v2.0
======================================================
utils/projection_kernel.py

PROJECTION-ONLY module: reads master_json → produces table_df for CSV Editor.
master_json is NEVER mutated here under any path.

Feature flag:  PROJECTION_LAYER = True   (set False to bypass entirely)

Audit log returned alongside every table_df:
  projection_version, timestamp, last_action, document_title, fallback_reason
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# ── Feature / toggle flags ─────────────────────────────────────────────────────
PROJECTION_LAYER = True      # Master on/off toggle (replaces old PERF_SAFE_PROJ)

# ── Constants ─────────────────────────────────────────────────────────────────
INTERNAL_PREFIX = "__"       # Columns starting with this are hidden from UI/exports
VERSION = "2.0.0"

# ── Regex helpers ─────────────────────────────────────────────────────────────
_EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
_URL_RE    = re.compile(r"(https?://|www\.)[^\s]+", re.I)
_MAP_RE    = re.compile(
    r"(maps\.app\.goo\.gl|goo\.gl/maps|maps\.google\.com|google\.com/maps)", re.I)
_PHONE_RE  = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
_DATE_RE   = re.compile(
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)
_CURRENCY_RE = re.compile(r"([\$\€\£\₹\¥]|USD|EUR|INR|AED|GBP)\s*[\d,]+(\.\d+)?", re.I)
_QTY_RE      = re.compile(r"\b\d+(\.\d+)?\s*(pcs?|kg|g|ltr?|units?|nos?)\b", re.I)

# ── Noise tokens to drop when deriving company name from email/URL ────────────
_EMAIL_PREFIX_DROP = {
    "info", "contact", "hr", "careers", "admin", "hello", "mail",
    "enquiry", "enquiries", "jobs", "recruitment", "support", "cv",
    "noreply", "no-reply", "sales", "team",
}
_TLD_SET = {"com", "in", "net", "org", "io", "ae", "uk", "www", "co",
            "edu", "gov", "mil", "biz", "me"}

# ── Company-name indicator words (any indicator in key → likely company col) ──
_CO_INDICATORS = {
    "company", "company name", "organisation", "organization",
    "org", "firm", "pvt", "ltd", "limited", "inc", "corp",
    "technologies", "solutions", "services", "group", "associates",
    "employer", "establishment", "entity",
}

# ── Semantic column ordering preference (used for final reorder) ──────────────
_COL_ORDER = [
    "S No", "Sno", "SNo", "Sr No",                          # serial
    "Company Name", "Company", "Name", "Full Name",         # primary id
    "Email", "Phone", "Mobile",                              # contact
    "Google Maps URL", "URL", "Website", "Location URL",    # web
    "Role", "Designation", "Title", "Position",             # title/role
    "Amount", "Qty", "Quantity", "Price", "Rate", "Cost",   # numeric
    "Date", "Start Date", "End Date", "Issue Date",         # dates
    "Notes", "Remarks", "Description",                      # notes
]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Value-level type inference
# ──────────────────────────────────────────────────────────────────────────────

def _infer_dtype(value: Any) -> str:
    """Strict value-based type inference. Never guesses."""
    if not isinstance(value, str):
        if isinstance(value, bool):   return "boolean"
        if isinstance(value, int):    return "integer"
        if isinstance(value, float):  return "float"
        return "string"
    v = value.strip()
    if not v:                         return "string"
    if _EMAIL_RE.search(v):           return "email"
    if _MAP_RE.search(v):             return "maps_url"
    if _URL_RE.search(v):             return "url"
    if _PHONE_RE.search(v):           return "phone"
    if _DATE_RE.search(v):            return "date"
    if _CURRENCY_RE.search(v):        return "currency"
    if _QTY_RE.search(v):             return "quantity"
    try:
        int(v.replace(",", "")); return "integer"
    except Exception: pass
    try:
        float(v.replace(",", "")); return "float"
    except Exception: pass
    return "string"


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Column-key normalization
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_col(raw_key: str, sample_value: Any = None) -> str:
    """
    Normalize a JSON key to a human-readable column name.
    Uses sample_value to upgrade URL → Google Maps URL when applicable.
    """
    k = str(raw_key).strip()

    # Maps URL — key-based
    if _MAP_RE.search(k) or any(kw in k.lower()
                                for kw in ["google map", "location url", "maps url"]):
        return "Google Maps URL"

    # Email — key-based
    if re.search(r"\be[-\s]?mail\b", k, re.I):
        return "Email"

    # URL — key-based (may be upgraded by value below)
    if any(kw in k.lower() for kw in ["url", "link", "website", "site"]):
        col = "URL"
        # Upgrade to Google Maps URL if the value says so
        if isinstance(sample_value, str) and _MAP_RE.search(sample_value):
            return "Google Maps URL"
        return col

    # Phone — key-based
    if re.search(r"\bphone|mobile|tel\b", k, re.I):
        return "Phone"

    # Generic TitleCase normalization
    k = re.sub(r"[_\-]+", " ", k)
    k = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", k)   # camelCase split
    return k.title().strip()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Nested object flattening (dot-notation)
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_value(key: str, value: Any, parent: str = "") -> Dict[str, Any]:
    """
    Recursively flatten nested objects into dot-notation columns.
    Single semantic nested values (e.g., maps URL dict) are collapsed to one col.
    """
    full_key = f"{parent}.{key}" if parent else key

    if isinstance(value, dict):
        # Check if it looks like a single-semantic value (e.g., {url: "maps..."})
        str_vals = [v for v in value.values() if isinstance(v, str)]
        if len(str_vals) == 1 and _MAP_RE.search(str_vals[0]):
            return {_normalize_col(full_key, str_vals[0]): str_vals[0]}

        # Otherwise recursively flatten
        result = {}
        for sub_k, sub_v in value.items():
            result.update(_flatten_value(sub_k, sub_v, full_key))
        return result

    if isinstance(value, list):
        # Keep as-is if non-object list (e.g., ["tag1", "tag2"])
        return {_normalize_col(full_key): ", ".join(str(v) for v in value)
                if all(not isinstance(v, (dict, list)) for v in value)
                else str(value)}

    return {_normalize_col(full_key, value): value}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Company name detection & semantic recovery
# ──────────────────────────────────────────────────────────────────────────────

def _detect_company_col(row: dict) -> Optional[str]:
    """Return the key in `row` most likely to be the company name column."""
    for key in row:
        if any(ind in str(key).lower() for ind in _CO_INDICATORS):
            return key
    return None


def _derive_company_from_email(email: str) -> Tuple[str, float]:
    match = _EMAIL_RE.search(email)
    if not match:
        return "", 0.0
    _, domain = match.group(0).split("@", 1)
    host = domain.rsplit(".", 1)[0]
    parts = re.split(r"[.\-_]", host)
    filtered = [p for p in parts if p.lower() not in _EMAIL_PREFIX_DROP and len(p) > 1]
    if not filtered:
        return "", 0.0
    name = " ".join(w.title() for w in filtered)
    return name, (0.75 if len(filtered) >= 2 else 0.55)


def _derive_company_from_url(url: str) -> Tuple[str, float]:
    if _MAP_RE.search(url):
        return "", 0.0
    m = re.search(r"(?:https?://|www\.)([^/\s]+)", url)
    if not m:
        return "", 0.0
    host = m.group(1)
    parts = re.split(r"[.\-_]", host)
    filtered = [p for p in parts if p.lower() not in {*_EMAIL_PREFIX_DROP, *_TLD_SET}
                and len(p) > 1]
    if not filtered:
        return "", 0.0
    return " ".join(w.title() for w in filtered), 0.60


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: Document-level title extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_document_title(master: dict) -> Optional[str]:
    for k, v in master.items():
        if isinstance(v, str) and any(kw in k.lower()
                                      for kw in ["title", "document title", "heading"]):
            return v
    return None


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Column semantic ordering
# ──────────────────────────────────────────────────────────────────────────────

def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns: known semantics first, then remaining alpha."""
    priority = []
    seen = set()
    for pref in _COL_ORDER:
        for col in df.columns:
            if col.lower() == pref.lower() and col not in seen:
                priority.append(col)
                seen.add(col)
    remaining = [c for c in df.columns if c not in seen]
    return df[priority + sorted(remaining)]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: Entity-list flattening
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_entity_list(array_data: List[Any], array_key: str) -> pd.DataFrame:
    """
    Convert array-of-dicts → flat DataFrame.
    - Preserves __source_order
    - Flattens nested objects (dot-notation → semantic column names)
    - Applies Maps URL detection per value
    - Performs in-place semantic company recovery
    """
    rows: List[dict] = []

    for idx, item in enumerate(array_data):
        if not isinstance(item, dict):
            continue

        row: dict = {"__source_order": idx}

        # Flatten item (handles nested objects)
        for raw_k, val in item.items():
            flat = _flatten_value(raw_k, val)
            # Value-level upgrade: rename URL → Google Maps URL if value matches
            upgraded = {}
            for col, cell in flat.items():
                if col == "URL" and isinstance(cell, str) and _MAP_RE.search(cell):
                    col = "Google Maps URL"
                upgraded[col] = cell
            row.update(upgraded)

        # ── Semantic Recovery: fill blank Company Name in-place ────────────
        company_col = _detect_company_col(row)
        company_val = row.get(company_col, "") if company_col else ""
        is_missing = (
            not company_val
            or str(company_val).strip().lower() in {"", "none", "null", "n/a", "nan"}
        )

        if is_missing:
            recovered, confidence = "", 0.0

            for k, v in row.items():
                if k.startswith("__"):
                    continue
                sv = str(v) if v is not None else ""
                if _EMAIL_RE.search(sv) and confidence < 0.75:
                    r, c = _derive_company_from_email(sv)
                    if c > confidence:
                        recovered, confidence = r, c
                if _URL_RE.search(sv) and not _MAP_RE.search(sv) and confidence < 0.75:
                    r, c = _derive_company_from_url(sv)
                    if c > confidence:
                        recovered, confidence = r, c

            if recovered:
                # Update in-place — use company_col key if exists, else "Company Name"
                target = company_col if company_col else "Company Name"
                row[target] = recovered
                if confidence < 0.70:
                    row["__needs_review"] = True
            else:
                row["__needs_review"] = True

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by original extraction order
    if "__source_order" in df.columns:
        df = df.sort_values("__source_order", kind="stable").reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: TABLE-LIKE text detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_table_like(master_json: Any) -> bool:
    """
    Lightweight heuristic: if the master json is a list of dicts
    where ≥ 60% of keys are the same across records, treat as table-like.
    """
    if not isinstance(master_json, list) or len(master_json) < 2:
        return False
    key_sets = [frozenset(r.keys()) for r in master_json if isinstance(r, dict)]
    if not key_sets:
        return False
    union_keys = frozenset().union(*key_sets)
    intersection_keys = key_sets[0].intersection(*key_sets[1:])
    if not union_keys:
        return False
    overlap = len(intersection_keys) / len(union_keys)
    return overlap >= 0.6


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: Main Projection Kernel Class
# ──────────────────────────────────────────────────────────────────────────────

class ProjectionKernel:
    """
    Reads master_json, returns (table_df, audit_log).
    master_json is NEVER mutated.

    Projection is lazy (only triggered by caller).
    Set PROJECTION_LAYER = False to bypass entirely.
    """

    VERSION = VERSION

    @staticmethod
    def project(
        master_json: Any,
        doc_mode: str = "TABLE_MODE",
        projection_layer: bool = PROJECTION_LAYER,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Returns (table_df, audit_log).

        table_df:   flat, user-visible DataFrame (no __ columns, semantically ordered).
        audit_log:  { projection_version, timestamp, last_action,
                      document_title, fallback_reason }
        """
        audit: dict = {
            "projection_version": ProjectionKernel.VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "last_action": "none",
            "document_title": None,
            "fallback_reason": None,
        }

        # ── Feature flag bypass ────────────────────────────────────────────────
        if not projection_layer:
            audit["last_action"] = "bypass"
            if isinstance(master_json, list):
                return pd.DataFrame(master_json), audit
            if isinstance(master_json, dict):
                return pd.DataFrame([master_json]), audit
            return pd.DataFrame(), audit

        try:
            return ProjectionKernel._execute(master_json, doc_mode, audit)
        except Exception as e:
            # Graceful failure — return empty df, caller shows JSON fallback
            audit["last_action"] = "error"
            audit["fallback_reason"] = str(e)
            print(f"[ProjectionKernel] Projection failed (non-fatal): {e}")
            return pd.DataFrame(), audit

    @staticmethod
    def _execute(master_json: Any, doc_mode: str, audit: dict) -> Tuple[pd.DataFrame, dict]:

        # ── HIERARCHICAL / KEY_VALUE / ENTITY_LIST: look for entity-list arrays ─
        if doc_mode in ("HIERARCHICAL_MODE", "KEY_VALUE_MODE", "ENTITY_LIST_MODE") and isinstance(master_json, dict):

            doc_title = _extract_document_title(master_json)
            audit["document_title"] = doc_title

            array_key, array_data = ProjectionKernel._find_best_array(master_json)

            if array_data:
                df_raw = _flatten_entity_list(array_data, array_key)
                audit["last_action"] = f"entity_list_flatten:{array_key}"

                # Inject scalar global meta (excluding the array and the title value)
                for meta_k, meta_v in master_json.items():
                    if meta_k == array_key:
                        continue
                    if isinstance(meta_v, (list, dict)):
                        continue
                    if doc_title and meta_v == doc_title:
                        continue  # Don't repeat the global title in every row
                    norm_k = _normalize_col(meta_k, meta_v)
                    if norm_k not in df_raw.columns:
                        df_raw[norm_k] = meta_v

                df_visible = ProjectionKernel._hide_internal(df_raw)
                df_visible = _reorder_columns(df_visible)

                # Validate: no duplicate primary key
                ProjectionKernel._validate(df_visible, audit)
                return df_visible, audit

            # No array: scalar-only projection
            scalar = {k: v for k, v in master_json.items()
                      if not isinstance(v, (list, dict))}
            if scalar:
                audit["last_action"] = "scalar_flat"
                df_visible = ProjectionKernel._hide_internal(pd.DataFrame([scalar]))
                df_visible = _reorder_columns(df_visible)
                return df_visible, audit

            audit["last_action"] = "empty"
            return pd.DataFrame(), audit

        # ── TABLE_MODE / TABLE-LIKE list of records ────────────────────────────
        if isinstance(master_json, list):
            rows = [r for r in master_json if isinstance(r, dict)]
            if not rows:
                audit["last_action"] = "empty"
                return pd.DataFrame(), audit

            # Flatten each row (handles nested objects per row)
            flat_rows = []
            for idx, r in enumerate(rows):
                flat_row: dict = {"__source_order": idx}
                for k, v in r.items():
                    flat = _flatten_value(k, v)
                    upgraded = {}
                    for col, cell in flat.items():
                        if col == "URL" and isinstance(cell, str) and _MAP_RE.search(cell):
                            col = "Google Maps URL"
                        upgraded[col] = cell
                    flat_row.update(upgraded)
                flat_rows.append(flat_row)

            df = pd.DataFrame(flat_rows)
            if "__source_order" in df.columns:
                df = df.sort_values("__source_order", kind="stable").reset_index(drop=True)

            is_tl = _detect_table_like(master_json)
            audit["last_action"] = "table_like_flat" if is_tl else "table_flat"

            df_visible = ProjectionKernel._hide_internal(df)
            df_visible = _reorder_columns(df_visible)
            ProjectionKernel._validate(df_visible, audit)
            return df_visible, audit

        # ── Unsupported structure ──────────────────────────────────────────────
        audit["last_action"] = "unsupported"
        audit["fallback_reason"] = "Neither dict nor list"
        return pd.DataFrame(), audit

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _find_best_array(master: dict) -> Tuple[Optional[str], List]:
        """Return (key, array_of_dicts) for the best entity-list candidate."""
        best_key, best_data = None, []
        for k, v in master.items():
            if isinstance(v, list) and len(v) > 0:
                # Count how many elements are dicts
                n_dicts = sum(1 for x in v if isinstance(x, dict))
                if n_dicts >= len(v) * 0.5 and n_dicts > len(best_data):
                    best_key, best_data = k, v
        return best_key, best_data

    @staticmethod
    def _hide_internal(df: pd.DataFrame) -> pd.DataFrame:
        """Drop all columns starting with INTERNAL_PREFIX."""
        visible = [c for c in df.columns if not str(c).startswith(INTERNAL_PREFIX)]
        return df[visible].copy()

    @staticmethod
    def _validate(df: pd.DataFrame, audit: dict) -> None:
        """
        Non-fatal validation pass.
        Adds 'validation_warnings' to audit if issues found.
        """
        warnings = []

        # Check for [object Object] style cells (un-flattened dicts)
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, (dict, list))).any():
                warnings.append(f"Column '{col}' contains un-flattened objects")

        # Check for internal columns leaking
        for col in df.columns:
            if str(col).startswith(INTERNAL_PREFIX):
                warnings.append(f"Internal column leaked to visible: '{col}'")

        if warnings:
            audit["validation_warnings"] = warnings
