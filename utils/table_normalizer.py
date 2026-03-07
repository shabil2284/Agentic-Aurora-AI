import pandas as pd
from typing import Dict, Any, List, Tuple
import copy

class TableNormalizer:
    """
    Intelligently flattens hierarchical JSON documents into relational tables.
    Extracts repeatable structured records (e.g., Line Items, Transactions)
    and appends parent identifier columns (e.g., Invoice No, Date) to preserve
    relational integrity.
    """

    @staticmethod
    def is_repeatable_structure(value: Any) -> bool:
        """
        Check if a value is a list of at least 2 structurally similar dictionaries.
        """
        if not isinstance(value, list) or len(value) < 2:
            return False
            
        # Ensure all elements are dicts
        if not all(isinstance(v, dict) for v in value):
            return False
            
        # Optional: check schema similarity (at least one shared key)
        first_keys = set(value[0].keys())
        for v in value[1:]:
            if len(first_keys.intersection(set(v.keys()))) == 0:
                return False
                
        return True

    @staticmethod
    def extract_parent_identifiers(source_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts top-level scalar values (strings, numbers, booleans)
        to act as foreign key columns in child tables.
        """
        identifiers = {}
        for key, value in source_doc.items():
            if not isinstance(value, (list, dict)) and pd.notna(value):
                # Optionally filter for keys that sound like IDs (Invoice No, Date, etc.)
                # For now, append all top-level scalars to maintain maximum context
                identifiers[key] = value
        return identifiers

    @staticmethod
    def normalize_to_tables(master_json: Any, base_table_name: str = "document") -> Dict[str, pd.DataFrame]:
        """
        Scans master_json for any repeating lists of objects.
        Returns a dictionary mapping logical table names replacing space with underscores
        e.g. "invoice_line_items" -> DataFrame.
        """
        tables = {}
        
        # Guard: If data is already a list (Table Mode), it's already tabular
        if isinstance(master_json, list):
            # We don't normalize already flat tables
            return tables

        if not isinstance(master_json, dict):
            return tables

        parent_identifiers = TableNormalizer.extract_parent_identifiers(master_json)

        for key, value in master_json.items():
            if TableNormalizer.is_repeatable_structure(value):
                # We found a viable child table
                rows = []
                for child_record in value:
                    # Make a flat copy of the child
                    flat_row = copy.deepcopy(child_record)
                    # Prepend parent identifiers
                    for p_key, p_val in parent_identifiers.items():
                        # Don't overwrite if child somehow has same key
                        if p_key not in flat_row:
                            flat_row[p_key] = p_val
                    rows.append(flat_row)
                
                if rows:
                    df = pd.DataFrame(rows)
                    # Clean column name for SQL/CSV
                    clean_suffix = str(key).strip().lower().replace(" ", "_").replace("-", "_")
                    table_name = f"{base_table_name}_{clean_suffix}"
                    tables[table_name] = df

        return tables
