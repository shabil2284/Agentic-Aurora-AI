import asyncio
import json
from config import settings
from extractor_graph import workflow
from utils.gemini_client import get_gemini_client

def test_anti_gravity_extraction():
    test_state = {
        "session_id": "test_1",
        "file_path": "dummy.txt",
        "raw_text": "Invoice 12345. Amount: $500. Date: 2025-01-01. Customer: Acme Corp.",
        "errors": [],
        "iteration": 0,
        "status": ""
    }
    
    print("Testing Anti-Gravity Extraction Pipeline (Schema-bypassed)...")
    res = workflow.invoke(test_state)
    
    print("\nFINAL PIPELINE STATE:")
    print("Errors:", res.get("errors"))
    print("Extracted Data:", json.dumps(res.get("extracted_data"), indent=2))

if __name__ == "__main__":
    test_anti_gravity_extraction()
