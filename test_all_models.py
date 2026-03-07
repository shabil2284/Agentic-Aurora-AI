import google.generativeai as genai
from config import settings
import time

genai.configure(api_key=settings.GEMINI_API_KEY)
# We test a smaller payload just to see if the model throws 429
prompt = "Return exactly this JSON: {\"test\": 1}"

print("Testing Google Core Models for Quota...")
models_to_test = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-flash-latest', 
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemma-3-27b-it'
]

for m in models_to_test:
    try:
        print(f"[{m}] Sending request...", end=" ")
        model = genai.GenerativeModel(m)
        res = model.generate_content(prompt)
        print(f"SUCCESS! {res.text.strip()[:30]}")
    except Exception as e:
        err = str(e).split('\n')[0][:100]
        print(f"FAIL: {err}")
    time.sleep(1) # Pace the requests
