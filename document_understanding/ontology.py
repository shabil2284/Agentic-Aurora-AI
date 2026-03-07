"""
Aurora Canonical Field Ontology
================================
Maps raw extracted field names to normalized canonical names with type inference
and unit specifications. Provides synonym resolution for consistent field naming
across document types.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from document_understanding.models import ExplainabilityTrace, FieldExtraction

# ── Canonical field definitions ───────────────────────────────────────────────
# Each entry: canonical_name -> { synonyms, type, unit (optional) }

ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "invoice_number": {
        "synonyms": [
            "invoice no", "invoice number", "inv no", "inv#", "inv num",
            "bill no", "bill number", "bill#", "receipt no", "receipt number",
            "document no", "doc no", "reference no", "ref no", "ref#",
        ],
        "type": "string",
        "unit": None,
    },
    "invoice_date": {
        "synonyms": [
            "invoice date", "date", "dated", "bill date", "document date",
            "issue date", "issued on", "created date",
        ],
        "type": "date",
        "unit": None,
    },
    "due_date": {
        "synonyms": [
            "due date", "payment due", "due by", "pay by", "expiry date",
            "valid until",
        ],
        "type": "date",
        "unit": None,
    },
    "vendor_name": {
        "synonyms": [
            "vendor", "seller", "supplier", "from", "company",
            "vendor name", "seller name", "supplier name", "issued by",
            "issuer", "issuer organization",
        ],
        "type": "string",
        "unit": None,
    },
    "payer_name": {
        "synonyms": [
            "bill to", "buyer", "customer", "client", "payer",
            "buyer name", "customer name", "ship to", "consignee",
            "recipient", "billed to",
        ],
        "type": "string",
        "unit": None,
    },
    "billing_address": {
        "synonyms": [
            "billing address", "bill to address", "address",
            "buyer address", "customer address",
        ],
        "type": "string",
        "unit": None,
    },
    "shipping_address": {
        "synonyms": [
            "shipping address", "ship to address", "delivery address",
            "consignee address",
        ],
        "type": "string",
        "unit": None,
    },
    "subtotal": {
        "synonyms": ["subtotal", "sub total", "sub-total", "net amount"],
        "type": "currency",
        "unit": "auto_detect",
    },
    "tax_amount": {
        "synonyms": [
            "tax", "tax amount", "vat", "gst", "vat amount", "gst amount",
            "total tax", "tax total",
        ],
        "type": "currency",
        "unit": "auto_detect",
    },
    "tax_rate": {
        "synonyms": ["tax rate", "vat rate", "gst rate", "tax%", "vat%"],
        "type": "percentage",
        "unit": "%",
    },
    "total": {
        "synonyms": [
            "total", "grand total", "total amount", "amount due",
            "total due", "balance due", "net payable", "amount payable",
            "invoice total",
        ],
        "type": "currency",
        "unit": "auto_detect",
    },
    "discount": {
        "synonyms": ["discount", "disc", "rebate", "discount amount"],
        "type": "currency",
        "unit": "auto_detect",
    },
    "gstin": {
        "synonyms": ["gstin", "gst no", "gst number", "gst in", "gst identification"],
        "type": "string",
        "unit": None,
    },
    "pan": {
        "synonyms": ["pan", "pan no", "pan number", "permanent account number"],
        "type": "string",
        "unit": None,
    },
    "email": {
        "synonyms": ["email", "e-mail", "email address", "mail"],
        "type": "email",
        "unit": None,
    },
    "phone": {
        "synonyms": [
            "phone", "telephone", "tel", "mobile", "contact",
            "phone no", "phone number", "contact number",
        ],
        "type": "phone",
        "unit": None,
    },
    "po_number": {
        "synonyms": [
            "po number", "po no", "po#", "purchase order", "purchase order number",
            "p.o.", "p.o. no",
        ],
        "type": "string",
        "unit": None,
    },
    "payment_terms": {
        "synonyms": [
            "payment terms", "terms", "payment method", "payment mode",
            "mode of payment",
        ],
        "type": "string",
        "unit": None,
    },
    "currency": {
        "synonyms": ["currency", "ccy", "cur"],
        "type": "string",
        "unit": None,
    },
}

# Build reverse lookup: synonym -> canonical_name
_SYNONYM_TO_CANONICAL: Dict[str, str] = {}
for canonical, info in ONTOLOGY.items():
    # Add the canonical name itself as a synonym
    _SYNONYM_TO_CANONICAL[canonical.lower()] = canonical
    # Add the canonical name with underscores replaced by spaces
    _SYNONYM_TO_CANONICAL[canonical.lower().replace("_", " ")] = canonical
    for syn in info["synonyms"]:
        _SYNONYM_TO_CANONICAL[syn.lower()] = canonical

# ── Currency detection ────────────────────────────────────────────────────────

_CURRENCY_SYMBOLS = {
    "$": "USD", "₹": "INR", "€": "EUR", "£": "GBP", "¥": "JPY",
    "AED": "AED", "SAR": "SAR", "Rs": "INR", "Rs.": "INR",
}


def detect_currency(text: str) -> str:
    """Detect currency from text context."""
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return "USD"  # default


# ── Public API ────────────────────────────────────────────────────────────────


def map_to_canonical(
    raw_field_name: str,
) -> Tuple[Optional[str], float]:
    """
    Map a raw extracted field name to its canonical ontology name.

    Returns: (canonical_name or None, confidence)
    """
    normalized = raw_field_name.strip().lower()

    # Exact match
    if normalized in _SYNONYM_TO_CANONICAL:
        return _SYNONYM_TO_CANONICAL[normalized], 0.99

    # Partial match — check if synonym starts at word boundary (not substring)
    for syn, canonical in _SYNONYM_TO_CANONICAL.items():
        # Only match if the synonym IS the normalized string
        # or they share full words, not just substrings
        if len(syn) > 3 and len(normalized) > 3:
            if normalized.startswith(syn + " ") or normalized.endswith(" " + syn):
                return canonical, 0.8

    # Token overlap
    raw_tokens = set(normalized.split())
    best_match = None
    best_overlap = 0

    for syn, canonical in _SYNONYM_TO_CANONICAL.items():
        syn_tokens = set(syn.split())
        overlap = len(raw_tokens & syn_tokens)
        if overlap > best_overlap and overlap >= 1:
            best_overlap = overlap
            best_match = canonical

    if best_match and best_overlap >= 1:
        confidence = min(best_overlap / max(len(raw_tokens), 1), 0.9)
        return best_match, round(confidence, 3)

    return None, 0.0


def apply_ontology(
    fields: Dict[str, FieldExtraction],
    full_text: str = "",
) -> Tuple[Dict[str, FieldExtraction], List[ExplainabilityTrace]]:
    """
    Apply ontology mapping to all extracted fields:
    - Map raw names to canonical names
    - Infer types and units
    - Add canonical_name, inferred_type, unit to each FieldExtraction

    Returns: (updated fields dict, traces)
    """
    traces: List[ExplainabilityTrace] = []
    mapped_fields: Dict[str, FieldExtraction] = {}
    currency = detect_currency(full_text)

    for raw_name, extraction in fields.items():
        canonical, conf = map_to_canonical(raw_name)

        if canonical:
            ontology_entry = ONTOLOGY.get(canonical, {})
            extraction.canonical_name = canonical
            extraction.inferred_type = ontology_entry.get("type", "string")
            unit = ontology_entry.get("unit")
            if unit == "auto_detect":
                extraction.unit = currency
            else:
                extraction.unit = unit
            extraction.explain += f" | ontology: mapped '{raw_name}' → '{canonical}' (conf={conf:.2f})"

            mapped_fields[canonical] = extraction
            traces.append(ExplainabilityTrace(
                step=f"ontology_mapping:{raw_name}",
                reason=f"Mapped '{raw_name}' → canonical '{canonical}' with type='{extraction.inferred_type}', unit='{extraction.unit}'",
                confidence=conf,
            ))
        else:
            # Keep original name
            mapped_fields[raw_name] = extraction
            traces.append(ExplainabilityTrace(
                step=f"ontology_mapping:{raw_name}",
                reason=f"No canonical mapping found for '{raw_name}' — keeping original",
                confidence=0.3,
            ))

    return mapped_fields, traces


def get_canonical_name(raw_name: str) -> str:
    """Quick lookup: return canonical name or original if not found."""
    canonical, _ = map_to_canonical(raw_name)
    return canonical if canonical else raw_name
