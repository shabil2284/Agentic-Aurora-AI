from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
import os

# Disable langchain-google-genai's internal automatic nest_asyncio patching which conflicts with Streamlit uvloop
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # (optional, just good practice)
os.environ["LANGCHAIN_GOOGLE_GENAI_PATCH_ASYNCIO"] = "false"

def get_gemini_client(model_name: str = "gemini-1.5-flash-latest"):
    return ChatGoogleGenerativeAI(
        google_api_key=settings.GEMINI_API_KEY,
        model=model_name,
        temperature=0
    )
