import os
import time
from config import settings
from utils.groq_client import get_groq_client
from langchain_core.messages import SystemMessage, HumanMessage

print("Testing Groq LLM Quotas...")
models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
prompt = "Return exactly this JSON: {\"test\": 1}"
messages = [SystemMessage(content="You are a JSON engine. Only return JSON."), HumanMessage(content=prompt)]

for m in models:
    try:
         print(f"[{m}] Sending request...", end=" ", flush=True)
         client = get_groq_client(m)
         res = client.invoke(messages)
         print(f"SUCCESS! {res.content.strip()}")
    except Exception as e:
         err = str(e).split('\n')[0][:100]
         print(f"FAIL: {err}")
    time.sleep(2)
