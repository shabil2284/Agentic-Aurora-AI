from langchain_groq import ChatGroq
from config import settings

def get_groq_client(model_name: str = None):
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=model_name or settings.PRIMARY_MODEL,
        temperature=0,
        max_tokens=8192
    )
