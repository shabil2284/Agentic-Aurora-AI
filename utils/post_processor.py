from typing import Dict, Any, List
import re

from utils.semantic_detector import (
    suggest_column_name,
    clean_markdown_email,
    clean_markdown_url,
    detect_column_type,
)

# Values that indicate a field was not present in the document
NULL_VALUES = {"null", "none", "n/a", "na", "-", "—", "", "undefined"}

class PostProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Removes OCR artifacts, broken symbols, and encoding noise."""
        if not text or not isinstance(text, str):
            return text
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
        text = re.sub(r'\\n|\\t|\\r', ' ', text)
        text = re.sub(r'={3,}', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def clean_value_links(value: Any) -> Any:
        """Remove markdown link wrappers from cell values."""
        if not isinstance(value, str):
            return value
        v = clean_markdown_email(value)
        v = clean_markdown_url(v)
        return v

    @staticmethod
    def is_null_value(v: Any) -> bool:
        """Returns True if the value represents a missing/null field."""
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in NULL_VALUES:
            return True
        return False

    @staticmethod
    def drop_null_columns(records: list) -> list:
        """
        Removes columns (keys) from all records where EVERY value is null/None/NULL.
        This prevents hallucinated columns from appearing in the table.
        """
        if not records or not isinstance(records, list):
            return records

        all_keys = []
        for r in records:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in all_keys:
                        all_keys.append(k)

        columns_with_data = set()
        for key in all_keys:
            for record in records:
                if isinstance(record, dict):
                    val = record.get(key)
                    if not PostProcessor.is_null_value(val):
                        columns_with_data.add(key)
                        break

        null_columns = set(all_keys) - columns_with_data
        if null_columns:
            print(f"[PostProcessor] Dropping all-NULL columns: {null_columns}")

        cleaned = []
        for record in records:
            if isinstance(record, dict):
                cleaned.append({k: v for k, v in record.items() if k in columns_with_data})
            else:
                cleaned.append(record)
        return cleaned

    @staticmethod
    def apply_semantic_schema(records: list) -> list:
        """
        Renames ONLY generic/unknown column names (col_0, field_1, value, text, etc.)
        based on value semantics. Preserves meaningful column names already set by the LLM
        (e.g. SNo, Company Name, Email, Phone) — these are NOT renamed.
        """
        if not records or not isinstance(records, list):
            return records

        # Collect all keys in order of first appearance
        all_keys = []
        for r in records:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in all_keys:
                        all_keys.append(k)

        rename_map = {}
        used_names = set(all_keys)  # Track names in use to avoid collisions

        for key in all_keys:
            # Collect non-null string values for this column
            values = []
            for r in records:
                if isinstance(r, dict):
                    val = r.get(key)
                    if not PostProcessor.is_null_value(val):
                        values.append(val)

            if not values:
                continue

            # Only suggest renames for generic/unknown column names
            suggested = suggest_column_name(key, values)
            if suggested and suggested not in used_names:
                rename_map[key] = suggested
                used_names.add(suggested)
                used_names.discard(key)

        if not rename_map:
            return records

        print(f"[PostProcessor] Semantic column renames (generic names only): {rename_map}")

        cleaned = []
        for record in records:
            if isinstance(record, dict):
                new_record = {}
                for k, v in record.items():
                    target_key = rename_map.get(k, k)
                    # Merge if target key collision (shouldn't happen but guard anyway)
                    if target_key in new_record and not PostProcessor.is_null_value(new_record[target_key]):
                        if not PostProcessor.is_null_value(v):
                            new_record[target_key] = v
                    else:
                        new_record[target_key] = v
                cleaned.append(new_record)
            else:
                cleaned.append(record)

        return cleaned

    @staticmethod
    def clean_all_link_values(records: list) -> list:
        """Strip markdown link wrappers from all cell values across all records."""
        if not records or not isinstance(records, list):
            return records
        cleaned = []
        for record in records:
            if isinstance(record, dict):
                cleaned.append({
                    k: PostProcessor.clean_value_links(v)
                    for k, v in record.items()
                })
            else:
                cleaned.append(record)
        return cleaned

    @staticmethod
    def normalize(data: Any) -> Any:
        """
        Normalizes numbers, dates, emails, and trims whitespace.
        Also drops all-NULL columns, cleans markdown links, and
        applies semantic column renaming for generic column names only.
        """
        if isinstance(data, list):
            out = []
            for item in data:
                normalized_item = PostProcessor.normalize(item)
                if normalized_item not in out:
                    out.append(normalized_item)
            # Drop columns that are entirely NULL
            out = PostProcessor.drop_null_columns(out)
            # Clean markdown links in values
            out = PostProcessor.clean_all_link_values(out)
            # Rename ONLY generic column names using value semantics
            out = PostProcessor.apply_semantic_schema(out)
            return out

        if not isinstance(data, dict):
            if isinstance(data, str):
                return PostProcessor.clean_text(data)
            return data

        cleaned = {}
        for k, v in data.items():
            normalized_v = PostProcessor.normalize(v)
            # Remove individual NULL-valued keys within a record
            if not PostProcessor.is_null_value(normalized_v):
                cleaned[k] = normalized_v
        return cleaned
