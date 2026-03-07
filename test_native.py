import os
from config import settings
import google.generativeai as genai

print("Configuring Gemini with Native SDK...")
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

print("Sending 'Hello' to Gemini...")
try:
    response = model.generate_content("Hello")
    print("Response received:", response.text)
except Exception as e:
    print("Error:", repr(e))
