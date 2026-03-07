from extractor import Extractor
import json

extractor = Extractor()
print("Starting extraction test...")

test_text = """
Sno | Name | Email | Title | Company
1 | Akanksha Puri | akanksha.puri@sourcefuse.com | Associate Director HR | SourceFuse Technologies
2 | Akanksha Sogani | akanksha.sogani@perennialsys.com | Head HR | Perennial Systems
"""

# Call extract_data
res = extractor.extract_data(test_text)
print("\nFINAL RESULTS:")
print(json.dumps(res, indent=2))
