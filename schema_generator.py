import json
from typing import Dict, Any, List, Optional

from utils.semantic_detector import derive_schema_from_records

class SchemaGenerator:
    """
    Aurora v2 Dynamic Schema Generator.
    Derives schema from actual extracted data — never hardcodes field names.
    Supports both TABLE_MODE (list of records) and HIERARCHICAL_MODE (nested object).
    """

    @staticmethod
    def derive_from_data(records: List[Dict]) -> List[Dict]:
        """
        TABLE_MODE: derive schema from a list of flat records.
        Returns [{\"field\": \"SNo\", \"type\": \"number\"}, ...]
        """
        if not records:
            return []
        return derive_schema_from_records(records)

    @staticmethod
    def derive_from_hierarchical(obj: Dict) -> List[Dict]:
        """
        HIERARCHICAL_MODE / KEY_VALUE_MODE: derive schema from a nested JSON object.
        Top-level keys become schema fields. Arrays become type 'array', dicts become 'object'.

        Returns [{\"field\": \"Name\", \"type\": \"string\"}, {\"field\": \"Experience\", \"type\": \"array\"}, ...]
        """
        if not obj or not isinstance(obj, dict):
            return []

        schema = []
        for key, value in obj.items():
            if isinstance(value, list):
                # Inspect element type
                if value and isinstance(value[0], dict):
                    field_type = "array[object]"
                else:
                    field_type = "array"
            elif isinstance(value, dict):
                field_type = "object"
            elif isinstance(value, bool):
                field_type = "boolean"
            elif isinstance(value, int):
                field_type = "number"
            elif isinstance(value, float):
                field_type = "number"
            else:
                # String heuristics
                v_str = str(value).strip()
                if "@" in v_str:
                    field_type = "email"
                elif v_str.lstrip("+-").replace(" ", "").replace("-", "").isdigit() and len(v_str) >= 7:
                    field_type = "phone"
                elif v_str.startswith("http") or "www." in v_str:
                    field_type = "url"
                else:
                    field_type = "string"
            schema.append({"field": key, "type": field_type})

        return schema

    @staticmethod
    def derive_auto(extracted_data: Any, doc_mode: str = "TABLE_MODE") -> List[Dict]:
        """
        Auto-select derivation strategy based on doc_mode.
        Returns a list of {\"field\": ..., \"type\": ...} dicts.
        """
        if doc_mode in ("HIERARCHICAL_MODE", "KEY_VALUE_MODE"):
            if isinstance(extracted_data, dict):
                return SchemaGenerator.derive_from_hierarchical(extracted_data)
            elif isinstance(extracted_data, list) and extracted_data and isinstance(extracted_data[0], dict):
                # fallback: flatten first record
                return SchemaGenerator.derive_from_hierarchical(extracted_data[0])
            return []
        else:
            data = extracted_data if isinstance(extracted_data, list) else [extracted_data]
            return SchemaGenerator.derive_from_data(data)

    @staticmethod
    def schema_to_dict(schema: List[Dict]) -> Dict[str, str]:
        """Convert schema list -> dict for backwards compatibility."""
        return {item["field"]: item["type"] for item in schema}

    @staticmethod
    def apply_schema_rename(df, schema: List[Dict]):
        """
        Apply field renames from edited schema back to a DataFrame.
        """
        import pandas as pd
        if df is None or df.empty:
            return df
        current_cols = list(df.columns)
        schema_fields = [s["field"] for s in schema]
        rename_map = {}
        for i, col in enumerate(current_cols):
            if i < len(schema_fields) and schema_fields[i] != col:
                rename_map[col] = schema_fields[i]
        if rename_map:
            return df.rename(columns=rename_map)
        return df
