import os
import sys
import time
from config import settings
import google.generativeai as genai

print("Loading PDF text...", flush=True)
try:
    import pdfplumber
    with pdfplumber.open("/Users/shabil/Downloads/Company Wise HR Contacts - HR Contacts.pdf") as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages[:20]) # Only 20 pages
except Exception as e:
    print("Cannot read pdf:", e)
    sys.exit(1)

print(f"Loaded {len(text)} characters.", flush=True)

print("Configuring Gemini with Native SDK...", flush=True)
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

prompt = "Extract all proper names and emails from this document into a structured JSON array.\n\nDOCUMENT:\n" + text

print("Sending 20-Page Payload to Gemini... (Timeout 180s)", flush=True)
start = time.time()
try:
    response = model.generate_content(prompt)
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} seconds!", flush=True)
    
    try:
        print("Response length:", len(response.text), flush=True)
        print("Preview:", response.text[:200], flush=True)
    except ValueError as e:
        print("Blocked or failed response:", response.candidates, flush=True)
        
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f}s:", repr(e), flush=True)
