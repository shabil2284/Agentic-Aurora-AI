"""
Aurora Semantic Completer — utils/semantic_completer.py
=========================================================
Extracts universal implicit semantic entities from raw document text
and non-destructively merges them into the extracted master_json.
"""

import json
from typing import Any, Dict, List, Union
from utils.strict_llm_client import StrictLLMClient

# The universal schema we want the LLM to search for in the raw text.
# We explicitly ask it to only return fields it can confidently find.
UNIVERSAL_SCHEMA_PROMPT = """
You are a Universal Document Semantic Analyzer.
Your task is to thoroughly read the attached raw document text and extract specific universal entities.

Look for the following entities, even if they are buried in paragraphs, sentences, or implicitly stated:
- Document Title
- Document Type
- Issuer Organization
- Employee Name
- Designation
- Start Date
- End Date
- Relieved Date
- Certificate Purpose
- Authority Name
- Authority Role
- Signature Presence (e.g. "Yes", "No")
- Location
- Issue Date

RULES:
1. ONLY return entities you can confidently find in the text.
2. If an entity is not present, DO NOT include it in the JSON output. Omit the key entirely.
3. Keep values concise and exact.
4. Output MUST be ONLY valid JSON matching this structure:
{
  "Employee Name": "John Doe",
  "Start Date": "January 1, 2020",
  ... (only include found fields)
}
"""

class SemanticCompleter:
    """
    Analyzes raw text to find implicit universal entities and merges 
    them into existing extracted data without overwriting.
    """

    @staticmethod
    def complete(raw_text: str, current_data: Union[dict, list, None]) -> Union[dict, list, None]:
        """
        Coordinates the LLM extraction and the non-destructive merge.
        """
        if not raw_text or not raw_text.strip():
            return current_data
            
        if current_data is None:
            return current_data

        try:
            semantic_data = SemanticCompleter._extract_semantics(raw_text)
            if not semantic_data:
                return current_data
                
            return SemanticCompleter._merge_non_destructive(current_data, semantic_data)
        except Exception as e:
            print(f"[SemanticCompleter] Error during completion: {e}")
            return current_data

    @staticmethod
    def _extract_semantics(raw_text: str) -> Dict[str, Any]:
        """
        Calls the LLM to extract the universal schema from the text.
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", UNIVERSAL_SCHEMA_PROMPT),
            ("user", f"DOCUMENT RAW TEXT:\n\n{raw_text[:8000]}") # Safely truncate to 8000 chars
        ])
        
        llm = StrictLLMClient(require_json=True)
        response = llm.invoke(prompt, {})
        
        if response.get("status") in ["success", "repaired"]:
            data = response.get("data", {})
            if isinstance(data, dict):
                return data
        return {}

    @staticmethod
    def _merge_non_destructive(current: Union[dict, list], semantic: dict) -> Union[dict, list]:
        """
        Merges `semantic` into `current` WITHOUT overwriting any existing keys or values.
        If `current` is a list, it merges the semantic fields into EVERY dictionary 
        in the list (useful for TABLE_MODE where doc-level entities apply to all rows).
        """
        if not semantic:
            return current

        # Helper to clean keys for comparison (case-insensitive, strip whitespace)
        def _clean_key(k: str) -> str:
            return str(k).strip().lower()

        if isinstance(current, dict):
            # Create a shallow copy to avoid mutating the original reference directly
            # though in LangGraph state it usually replaces it.
            merged = dict(current)
            existing_keys_clean = {_clean_key(k) for k in merged.keys()}
            
            for sem_key, sem_val in semantic.items():
                if not sem_val: # Skip empty semantic values
                    continue
                # Only add if the key doesn't already exist (case-insensitive match)
                if _clean_key(sem_key) not in existing_keys_clean:
                    merged[sem_key] = sem_val
            return merged
            
        elif isinstance(current, list):
            # If it's a list (like extracted table rows), we inject these 
            # document-level semantic fields into every row.
            merged_list = []
            for item in current:
                if isinstance(item, dict):
                    merged_item = dict(item)
                    existing_keys_clean = {_clean_key(k) for k in merged_item.keys()}
                    for sem_key, sem_val in semantic.items():
                        if not sem_val:
                            continue
                        if _clean_key(sem_key) not in existing_keys_clean:
                            merged_item[sem_key] = sem_val
                    merged_list.append(merged_item)
                else:
                    merged_list.append(item)
            return merged_list
            
        return current
