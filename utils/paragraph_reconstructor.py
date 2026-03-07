"""
Aurora Semantic Paragraph Reconstruction & Completion Controller
================================================================
utils/paragraph_reconstructor.py

Post-processing layer to reconstruct fragmented paragraph fields
(Summary, Description, Purpose, etc.) by analyzing the raw OCR text.

STRICT NON-DESTRUCTIVE CONSOLIDATION:
- If confident > 75% and text is expanded > 30%, it replaces the original value
  (setting `_reconstructed = True` flags).
- Otherwise, it creates a `<field>_reconstructed` sibling field.
- Always generates an internal `__reconstruction_log` audit trail.
"""

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Union, Tuple, Optional
from utils.strict_llm_client import StrictLLMClient

# ── Feature flag ─────────────────────────────────────────────────────────────
RECONSTRUCTION_LAYER_ACTIVE = True

# ── Keys that suggest a paragraph/long-form field ──────────────────────────────
_PARAGRAPH_INDICATORS = {
    "summary", "description", "purpose", "remarks", "notes",
    "responsibilities", "objective", "profile", "clause", "terms"
}

# The LLM prompt to perform the contextual reconstruction without hallucination
_RECONSTRUCTION_PROMPT = """
You are a strict Document Text Reconstructor.
Your task is to take a truncated extracted field and find its FULL, original, contiguous paragraph in the provided RAW OCR TEXT.

RULES:
1. Find where the 'TRUNCATED_FIELD_VALUE' appears in the 'RAW_OCR_TEXT'.
2. Reconstruct the full paragraph from that point until the next major section heading or blank line break.
3. Fix soft line-breaks (join lines that were broken arbitrarily).
4. Remove hyphenation that splits words across lines (e.g. 'microcon-' 'troller' -> 'microcontroller').
5. Preserve deliberate styling like bullet points and numbered lists.
6. DO NOT FABRICATE OR PARAPHRASE. Use ONLY exact words found in the OCR text.
7. If you cannot find the text or it is ambiguous, return a confidence of 0.
8. Output MUST be ONLY valid JSON matching this structure:
{
  "reconstructed_text": "The full repaired paragraph string here...",
  "confidence": 85,
  "method": "llm_contextual_merge"
}
"""

class ParagraphReconstructor:
    """Analyzes extracted paragraph fields against raw text and reconstructs them."""

    @staticmethod
    def reconstruct(raw_text: str, current_data: Union[dict, list]) -> Union[dict, list]:
        """Entry point. Iterates over data to find truncated string fields and repairs them."""
        if not RECONSTRUCTION_LAYER_ACTIVE or not raw_text or not current_data:
            return current_data

        try:
            return ParagraphReconstructor._traverse_and_reconstruct(current_data, raw_text)
        except Exception as e:
            print(f"[ParagraphReconstructor] Global Error: {e}")
            return current_data

    @staticmethod
    def _traverse_and_reconstruct(data: Union[dict, list], raw_text: str) -> Union[dict, list]:
        if isinstance(data, dict):
            # Operate on a shallow copy
            reconstructed_dict = dict(data)
            keys_to_process = list(reconstructed_dict.keys())
            
            for key in keys_to_process:
                # Skip internal fields
                if str(key).startswith("__"):
                    continue
                    
                val = reconstructed_dict[key]
                
                # Check if it's a candidate for reconstruction
                if ParagraphReconstructor._is_candidate(key, val):
                    result = ParagraphReconstructor._process_field(key, val, raw_text)
                    if result:
                        ParagraphReconstructor._consolidate(reconstructed_dict, key, val, result)
                        
                # Recursively process nested dictionaries/lists
                elif isinstance(val, (dict, list)):
                    reconstructed_dict[key] = ParagraphReconstructor._traverse_and_reconstruct(val, raw_text)
                    
            return reconstructed_dict
            
        elif isinstance(data, list):
            return [ParagraphReconstructor._traverse_and_reconstruct(item, raw_text) for item in data]
            
        return data

    @staticmethod
    def _is_candidate(key: str, val: Any) -> bool:
        """Determines if a field should be sent to the reconstructor."""
        if not isinstance(val, str):
            return False
            
        k_lower = str(key).lower()
        
        # Is it a known long-form field?
        if any(ind in k_lower for ind in _PARAGRAPH_INDICATORS):
            return True
            
        # Or is it just suspiciously long but potentially cut off?
        # (Usually sentences end with punctuation. If it's long but has no terminal punctuation, 
        # or ends abruptly, it might be truncated.)
        if len(val) > 50 and not val.strip().endswith(('.', '?', '!', '"', "'")):
            return True
            
        return False

    @staticmethod
    def _process_field(field_name: str, extracted_val: str, raw_text: str) -> Optional[dict]:
        """Calls LLM to reconstruct the field from raw text."""
        from langchain_core.prompts import ChatPromptTemplate
        
        # Don't send massive payloads for small fields
        snippet = extracted_val[:200]
        
        # If the snippet isn't even in the raw text, OCR might be wildly different or hallucinated. Skip.
        # We do a loose check.
        if snippet[:50].lower() not in raw_text.lower() and len(snippet) > 50:
             # Try first 20 chars
             if snippet[:20].lower() not in raw_text.lower():
                 print(f"[ParagraphReconstructor] Value for '{field_name}' not found in raw text. Skipping.")
                 return None

        prompt = ChatPromptTemplate.from_messages([
            ("system", _RECONSTRUCTION_PROMPT),
            ("user", f"FIELD_NAME: {field_name}\nTRUNCATED_FIELD_VALUE:\n{extracted_val}\n\nRAW_OCR_TEXT:\n{raw_text[:8000]}")
        ])
        
        llm = StrictLLMClient(require_json=True)
        response = llm.invoke(prompt, {})
        
        if response.get("status") in ["success", "repaired"]:
            data = response.get("data", {})
            if isinstance(data, dict) and "reconstructed_text" in data and "confidence" in data:
                return data
        return None

    @staticmethod
    def _consolidate(current: dict, field_name: str, original_val: str, result: dict):
        """Merges the reconstruction result safely."""
        recon_text = result.get("reconstructed_text", "").strip()
        confidence = result.get("confidence", 0)
        method = result.get("method", "unknown")
        
        if not recon_text or recon_text == original_val.strip():
            return # Nothing to change
            
        orig_len = len(original_val.strip())
        recon_len = len(recon_text)
        
        # Calculate % increase
        increase_pct = ((recon_len - orig_len) / orig_len * 100) if orig_len > 0 else 100
        
        # Initialize audit log if missing
        if "__reconstruction_log" not in current:
            current["__reconstruction_log"] = []
            
        log_entry = {
            "field": field_name,
            "original_length": orig_len,
            "reconstructed_length": recon_len,
            "confidence": confidence,
            "method": method,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Rule: Expand original if increase > 10% and confident >= 75%
        if confidence >= 75 and increase_pct > 10:
            current[field_name] = recon_text
            current[f"{field_name}_reconstructed"] = True
            current[f"{field_name}_reconstruction_confidence"] = confidence
            log_entry["action"] = "replaced_original"
            print(f"[ParagraphReconstructor] Replaced '{field_name}' (Expanded by {increase_pct:.1f}%, Conf: {confidence})")
        else:
            # Low confidence or minor change -> safely add as sibling
            sibling_key = f"{field_name}_reconstructed_text"
            current[sibling_key] = recon_text
            current[f"{sibling_key}_confidence"] = confidence
            log_entry["action"] = "created_sibling"
            print(f"[ParagraphReconstructor] Created sibling '{sibling_key}' (Conf: {confidence})")
            
        current["__reconstruction_log"].append(log_entry)
