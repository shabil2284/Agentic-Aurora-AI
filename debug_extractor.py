import json
import time
from extractor_graph import workflow

def run_test():
    print("Testing extraction on raw text (Sync)...")
    
    start_time = time.time()
    initial_state = {
        "session_id": "debug_session",
        "file_path": "/Users/shabil/Downloads/Company Wise HR Contacts - HR Contacts.pdf",
        "document": None,
        "raw_text": "",
        "schema": None,
        "extracted_data": None,
        "errors": [],
        "iteration": 0,
        "status": "Initializing"
    }

    result = workflow.invoke(initial_state)
    end_time = time.time()
    
    print(f"\nPipeline completed in {end_time - start_time:.2f} seconds.")
    
    print("\n=== SCHEMA ===")
    print(json.dumps(result.get("schema"), indent=2))

    print("\n=== EXTRACTED DATA ===")
    print(json.dumps(result.get("extracted_data"), indent=2))

    print("\n=== ERRORS ===")
    errors = result.get("errors", [])
    if errors:
        for err in errors:
            print(f"- {err}")
    else:
        print("None")

if __name__ == "__main__":
    run_test()
