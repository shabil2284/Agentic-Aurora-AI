from utils.post_processor import PostProcessor
import json

data = [
    {
        "Field 1": "TASC Pvt Ltd",
        "Email": "https://maps.app.goo.gl/w4BMXZHYeWf1dGL39",
        "Phone": "info@tasc.com",
        "Column 4": "+971501234567"
    }
]

print("BEFORE:")
print(json.dumps(data, indent=2))
out = PostProcessor.normalize(data)
print("\nAFTER:")
print(json.dumps(out, indent=2))
