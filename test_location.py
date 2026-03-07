from utils.post_processor import PostProcessor
import json

data = [
    {
        "Field 1": "Techasoft Pvt. Ltd.",
        "Email": "HSR Layout",
        "Website": "https://maps.app.goo.gl/w4BMXZHYeWf1dGL39",
        "Attr 4": "http://example.com"
    }
]

print("BEFORE:")
print(json.dumps(data, indent=2))
out = PostProcessor.normalize(data)
print("\nAFTER:")
print(json.dumps(out, indent=2))

from utils.semantic_detector import derive_schema_from_records
schema = derive_schema_from_records(out)
print("\nSCHEMA:")
print(json.dumps(schema, indent=2))
