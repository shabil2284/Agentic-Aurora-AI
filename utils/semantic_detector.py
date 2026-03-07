"""
Aurora Universal Semantic Field Detector
----------------------------------------
Detects field types based purely on VALUE SEMANTICS — never on column name assumptions.

Detection Priority (highest to lowest):
  1. Email        → value contains "@"
  2. Google Maps  → maps.app.goo.gl / maps.google.com / goo.gl/maps
  3. LinkedIn     → linkedin.com
  4. Career Page  → /careers, /jobs, /join-us, /work-with-us
  5. Website      → http(s)+any URL not matched above
  6. Phone        → digits + phone pattern (+91, +, -, parentheses, 7-15 digits)
  7. Address      → street, road, city, nagar, layout, India location keywords
  8. SNo / ID     → pure numeric or leading-number pattern
  9. Company Name → explicit suffix words: Ltd, LLC, Pvt, Inc, Solutions, etc.
  10. Document Title → first line with heading words like Companies, Directory, Statement
  11. Other        → unclassified (column name preserved as-is)
"""

import re
from typing import Any, List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"^\+?[\d\s\-\(\)]{7,20}$")
URL_RE   = re.compile(r"(https?://|www\.)[^\s]+", re.IGNORECASE)

MAPS_KEYWORDS = ("maps.app.goo.gl", "maps.google.com", "goo.gl/maps", "google.com/maps")
LINKEDIN_KEYWORDS = ("linkedin.com",)
CAREER_KEYWORDS = ("/careers", "/jobs", "/join-us", "/join_us", "/work-with-us", "/openings", "/vacancies")

DOC_TITLE_WORDS = (
    "companies", "directory", "statement", "invoice", "report", "list",
    "register", "catalogue", "catalog", "manifest", "roster", "census",
)

INDIA_LOCATION_KEYWORDS = (
    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "gurugram", "noida", "india", "hsr",
    "koramangala", "whitefield", "indiranagar", "jayanagar", "btm",
)

COMPANY_SUFFIXES = (
    " ltd", " llc", " llp", " pvt", " inc", " corp", " co.",
    " technologies", " technology", " solutions", " systems",
    " consulting", " consultancy", " services", " group",
    " enterprises", " industries", " international", " global",
    " digital", " software", " networks", " associates",
    " management", " outsourcing", " ventures", " holdings",
    " foundation", " agency", " studios", " labs",
)

ADDRESS_KEYWORDS = (
    "street", "st.", "road", "rd.", "avenue", "ave", "lane",
    "layout", "nagar", "colony", "district", "city", "state",
    "province", "zip", "pincode", "po box", "floor", "building",
    "block", "sector", "phase", "plot",
)

MARKDOWN_MAILTO_RE = re.compile(r"\[([^\]]*)\]\(mailto:([^)]+)\)", re.IGNORECASE)
MARKDOWN_URL_RE    = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)", re.IGNORECASE)


# ── Cleaner helpers ────────────────────────────────────────────────────────────

def clean_markdown_email(value: str) -> str:
    """
    Extracts the raw email from markdown mailto links.
    e.g.  [careers@tasc.com](mailto:careers@tasc.com)  →  careers@tasc.com
    """
    if not isinstance(value, str):
        return value
    m = MARKDOWN_MAILTO_RE.search(value)
    if m:
        # Prefer the link URL portion as it's always the raw email
        email_candidate = m.group(2).strip()
        return email_candidate
    return value


def clean_markdown_url(value: str) -> str:
    """
    Extracts raw URL from markdown links.
    e.g.  [Visit Us](https://tasc.com)  →  https://tasc.com
    """
    if not isinstance(value, str):
        return value
    m = MARKDOWN_URL_RE.search(value)
    if m:
        return m.group(2).strip()
    return value


def clean_value(value: str) -> str:
    """Apply all cleaning rules and return the canonical value for detection."""
    if not isinstance(value, str):
        return value
    v = clean_markdown_email(value)
    v = clean_markdown_url(v)
    return v.strip()


# ── Core detector ──────────────────────────────────────────────────────────────

def detect_field_type(value: Any) -> str:
    """
    Detect the semantic field type of a single value.

    Returns one of:
      "Email", "Phone", "Google Maps", "LinkedIn", "Career Page",
      "Website", "Address", "SNo", "ID", "Company Name",
      "Document Title", "Other"
    """
    if value is None:
        return "Other"

    raw = str(value).strip()
    if not raw:
        return "Other"

    v = clean_value(raw)
    v_lower = v.lower()

    # RULE 1: Email — must contain valid email pattern
    if EMAIL_RE.search(v):
        return "Email"

    # RULE 2: Google Maps — most specific URL label (check BEFORE generic Website)
    if any(kw in v_lower for kw in MAPS_KEYWORDS):
        return "Google Maps"

    # RULE 3: LinkedIn
    if any(kw in v_lower for kw in LINKEDIN_KEYWORDS):
        return "LinkedIn"

    # RULE 4: Career / Jobs page
    if any(kw in v_lower for kw in CAREER_KEYWORDS) and URL_RE.search(v):
        return "Career Page"

    # RULE 5: Generic Website URL (http/https/www that didn't match above)
    if URL_RE.search(v) or v_lower.startswith("www.") or (
        "." in v and "@" not in v and
        any(ext in v_lower for ext in [".com", ".net", ".org", ".io", ".ae", ".uk", ".in"])
    ):
        return "Website"

    # RULE 6: Phone — +91 / 10-digit Indian number / international pattern
    digits_only = re.sub(r"\D", "", v)
    if PHONE_RE.match(v) and len(digits_only) >= 7:
        return "Phone"
    if v_lower.startswith("+91") and len(digits_only) >= 10:
        return "Phone"
    if re.match(r"^[6-9]\d{9}$", digits_only) and not re.search(r"[a-zA-Z]", v):
        return "Phone"

    # RULE 7: Address — keyword match OR Indian location + sufficient length
    # Document Title is checked FIRST to avoid short headings being labelled as Address
    if any(kw in v_lower for kw in DOC_TITLE_WORDS) and len(v.split()) <= 8:
        return "Document Title"
    if (any(kw in v_lower for kw in ADDRESS_KEYWORDS) and len(v) > 8) or \
       (any(kw in v_lower for kw in INDIA_LOCATION_KEYWORDS) and len(v) > 6):
        return "Address"

    # RULE 8: SNo / ID — pure numeric or short alphanumeric ID
    if re.match(r"^\d{1,10}$", v.strip()):
        return "SNo"
    if re.match(r"^[A-Z]{0,3}\d{3,10}$", v.strip(), re.IGNORECASE):
        return "ID"

    # RULE 9: Company Name — MUST have explicit company suffix words
    if any(suffix in v_lower for suffix in COMPANY_SUFFIXES):
        return "Company Name"

    # RULE 10: Document Title — short heading line with known category words
    if any(kw in v_lower for kw in DOC_TITLE_WORDS) and len(v.split()) <= 8:
        return "Document Title"

    # RULE 11: Unclassified — preserve as-is
    return "Other"


def detect_column_type(values: List[Any], threshold: float = 0.4) -> str:
    """
    Detect the dominant semantic field type for a column by voting across all values.
    
    Args:
        values:    List of raw cell values from this column.
        threshold: Fraction of rows that must match for that type to win (default 40%).
    
    Returns the winning type name, or "Other" if nothing wins.
    """
    if not values:
        return "Other"

    votes: dict = {}
    total = 0

    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "n/a", "na", "-", "—", "", "undefined"}:
            continue
        t = detect_field_type(v)
        votes[t] = votes.get(t, 0) + 1
        total += 1

    if total == 0:
        return "Other"

    # Find the winner
    winner, count = max(votes.items(), key=lambda kv: kv[1])
    if count / total >= threshold:
        return winner
    return "Other"


def suggest_column_name(current_name: str, values: List[Any]) -> Optional[str]:
    """
    Given a generic/unknown column name and its values, suggest a semantic name.
    Only suggests a rename if the column name looks generic (col_0, field_1, value, text, etc.)
    For columns that already have meaningful names (SNo, Company Name, Email), returns None.

    Returns suggested name or None if no rename is appropriate.
    """
    GENERIC_PATTERNS = re.compile(
        r"^(col_?\d+|field_?\d+|column_?\d+|value_?\d*|text_?\d*|key_?\d*|data_?\d*|item_?\d*|attr_?\d*)$",
        re.IGNORECASE
    )
    # If the column already has a meaningful name, skip it
    if not GENERIC_PATTERNS.match(current_name.strip()):
        return None

    detected = detect_column_type(values)
    if detected == "Other":
        return None
    return detected


def derive_schema_from_records(records: List[dict]) -> List[dict]:
    """
    Dynamically generate a schema list from extracted records.
    
    Returns a list like:
        [
          {"field": "SNo",          "type": "number"},
          {"field": "Company Name", "type": "string"},
          {"field": "Email",        "type": "string"},
        ]
    """
    if not records:
        return []

    # Collect all keys in order of first appearance
    seen_keys = []
    for record in records:
        if isinstance(record, dict):
            for k in record.keys():
                if k not in seen_keys:
                    seen_keys.append(k)

    schema = []
    for field in seen_keys:
        # Gather column values
        col_values = [
            r.get(field) for r in records
            if isinstance(r, dict) and r.get(field) is not None
        ]
        field_type = detect_column_type(col_values)

        # Map semantic type → JSON schema type
        TYPE_MAP = {
            "Email":          "string",
            "Phone":          "string",
            "Website":        "string",
            "Google Maps":    "string",
            "LinkedIn":       "string",
            "Career Page":    "string",
            "Document Title": "string",
            "Company Name":   "string",
            "SNo":            "number",
            "ID":             "string",
            "Address":        "string",
            "Other":          "string",
        }

        # SNo / numeric strings → number
        if field.lower() in ("sno", "s.no", "s no", "serial", "serial no", "sl", "sl.no"):
            json_type = "number"
        else:
            json_type = TYPE_MAP.get(field_type, "string")

        schema.append({"field": field, "type": json_type})

    return schema
