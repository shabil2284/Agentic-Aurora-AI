import time
import os
import pdfplumber
import json
from config import settings
from utils.groq_client import get_groq_client
from langchain_core.messages import SystemMessage, HumanMessage

pdf_path = "/Users/shabil/Downloads/Company Wise HR Contacts - HR Contacts.pdf"

print("1. Loading PDF...")
try:
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
except Exception as e:
    text = ""
    print(e)

system_prompt = """You are a Universal Document Intelligence Engine.
Your task is to extract Key-Value pairs into a structured JSON array of objects.
CRITICAL RULES:
- Output ONLY valid JSON.
- DO NOT wrap in markdown blocks like ```json
- Extract ALL entries present in the text provided.
"""

CHUNK_SIZE = 15000 
text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, min(len(text), CHUNK_SIZE*4), CHUNK_SIZE)]
print(f"2. Testing {len(text_chunks)} chunks on Groq Llama 3.1 8B...")

client = get_groq_client('llama-3.1-8b-instant')

def process_chunk(idx, chunk):
    prompt_str = f"DOCUMENT CONTENT:\n{chunk}\n\nRESPOND WITH EXACTLY ONE JSON ARRAY AND NOTHING ELSE:"
    start_time = time.time()
    for attempt in range(4):
        try:
            res = client.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt_str)])
            elapsed = time.time() - start_time
            return {"idx": idx, "status": "success", "data": res.content, "time": elapsed}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                time.sleep(20) # Groq rate limits cool off exactly in 15-60s
            else:
                return {"idx": idx, "status": "error", "error": error_str, "time": time.time() - start_time}
    return {"idx": idx, "status": "error", "error": "Max retries", "time": time.time() - start_time}

results = []
for i, chunk in enumerate(text_chunks):
    res = process_chunk(i, chunk)
    results.append(res)
    if res['status'] == 'success':
        print(f" Chunk {res['idx']+1}: Success in {res['time']:.2f}s.")
    else:
        print(f" Chunk {res['idx']+1}: FAILED in {res['time']:.2f}s. Error: {res['error'][:100]}")
    time.sleep(5) # Add padding between chunks to avoid TPM limits

total_records = 0
for res in results:
    if res['status'] == 'success':
        try:
            d = json.loads(res['data'].strip('` \n'))
            if isinstance(d, list): total_records += len(d)
        except Exception as e:
            print(f" Failed to parse Chunk {res['idx']+1}: {e}")
            pass
        
print(f"FINAL TALLY: {total_records} records.")
