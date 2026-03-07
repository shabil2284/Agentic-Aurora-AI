import time
import os
import pdfplumber
import json
from config import settings
from utils.gemini_client import get_gemini_client
from langchain_core.messages import SystemMessage, HumanMessage
import google.generativeai as genai

# Setup Native Gemini Client
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

pdf_path = "/Users/shabil/Downloads/Company Wise HR Contacts - HR Contacts.pdf"

print("1. Loading PDF...")
t0 = time.time()
try:
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
except Exception as e:
    print(f"Failed to load PDF: {e}")
    text = ""
t1 = time.time()
print(f"   Done. Loaded {len(text)} characters in {t1-t0:.2f}s.")

system_prompt = """ANTI-GRAVITY PROJECT — DIRECT KEY VALUE JSON EXTRACTION PROMPT

You are an Enterprise Document Key-Value Extraction Engine.

Your task is to extract all fields from the document and return them as DIRECT KEY-VALUE JSON.

---

## CRITICAL OUTPUT FORMAT RULE

Return output STRICTLY in this format:

{
"Key": "Value",
"Key": "Value"
}

NOT this format:

[
{"key": "Name", "value": "John"}
]

NOT this format:

{
"data": [
{"key": "", "value": ""}
]
}

---

## EXTRACTION RULES

1. Each field name must be JSON key

Example:

Name: John

Output:

"Name": "John"

2. Each field value must be JSON value

3. DO NOT wrap inside array

4. DO NOT use "key" and "value" labels

5. DO NOT nest inside "data"

6. Return single flat JSON object

---

## DOCUMENT TYPES

Works for:

Resume
Invoice
Form
OCR
PDF
Any document

---

## EXAMPLE CORRECT OUTPUT

{
"Name": "MUSTAFA ALTHAF AZEEZ",
"Designation": "Electronics and Communication Engineer",
"Phone": "+91 7338502290",
"Email": "althafazeez16@gmail.com"
}

---

## FINAL INSTRUCTION

Extract and return DIRECT KEY-VALUE JSON ONLY.

No explanation.

No extra structure.
"""

CHUNK_SIZE = 12000 # ~3-4 dense pages per chunk
MAX_CHUNKS = 15
text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, min(len(text), CHUNK_SIZE * MAX_CHUNKS), CHUNK_SIZE)]
print(f"   Total Chunks created: {len(text_chunks)}")

import concurrent.futures

from langchain_groq import ChatGroq

# Setup Native Groq Client for Benchmarking
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=settings.GROQ_API_KEY
)

def process_chunk(idx, chunk):
    prompt_str = f"DOCUMENT CONTENT:\n{chunk}\n\nRESPOND WITH DIRECT KEY-VALUE JSON ONLY:"
    start_time = time.time()
    
    for attempt in range(4):
        try:
            response = groq_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_str)
            ])
            res_text = response.content.strip('` \n')
            if res_text.startswith('json\n'):
                res_text = res_text[5:]
            data = json.loads(res_text)
            elapsed = time.time() - start_time
            return {"idx": idx, "status": "success", "data": data, "time": elapsed}
        except Exception as e:
            error_str = str(e)
            print(f"   [Chunk {idx+1} Attempt {attempt+1}] Caught ERROR: {error_str.splitlines()[0][:150]}")
            if "429" in error_str or "Quota" in error_str or "exhausted" in error_str.lower():
                time.sleep(15 * (attempt + 1))
            else:
                elapsed = time.time() - start_time
                return {"idx": idx, "status": "error", "error": error_str, "time": elapsed}
    
    elapsed = time.time() - start_time
    return {"idx": idx, "status": "error", "error": "Max retries exhausted for 429", "time": elapsed}

print("3. Executing Sequential Extraction (Paced to avoid 429 Rate Limits)...")
t0 = time.time()
results = []
for i, chunk in enumerate(text_chunks):
    print(f"   [Chunk {i+1}/{len(text_chunks)}] Sending 40,000 characters to Gemini...", flush=True)
    res = process_chunk(i, chunk)
    results.append(res)
    
    if i < len(text_chunks) - 1:
        print("   [Pacing] Sleeping 15 seconds to respect Gemini Free Tier TPM limits...", flush=True)
        time.sleep(15)

t1 = time.time()
print(f"   Done. Total sequential extraction time: {t1-t0:.2f}s.")

print("\n4. Analyzing Results...")
total_records = 0
total_dict_keys = 0

for res in sorted(results, key=lambda x: x['idx']):
    idx = res['idx']
    status = res['status']
    elapsed = res['time']
    if status == 'success':
        data = res['data']
        if isinstance(data, list):
            count = len(data)
            total_records += count
            print(f"   Chunk {idx+1}: Success in {elapsed:.2f}s. Extracted {count} TABLE records. Sample: {json.dumps(data[0]) if count > 0 else 'None'}")
        elif isinstance(data, dict):
            key_count = len(data.keys())
            total_dict_keys += key_count
            print(f"   Chunk {idx+1}: Success in {elapsed:.2f}s. Extracted {key_count} KEY-VALUE pairs. Sample: {json.dumps(list(data.items())[:2])}")
        else:
             print(f"   Chunk {idx+1}: Success in {elapsed:.2f}s. Returned unknown format: type {type(data)}")
    else:
        print(f"   Chunk {idx+1}: FAILED in {elapsed:.2f}s. Error: {res['error']}")

print(f"\nFINAL TALLY: {total_records} tabular records, {total_dict_keys} key-value pairs extracted.")
