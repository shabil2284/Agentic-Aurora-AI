import sys
import pydantic
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Objective 1: Load .env at app startup
load_dotenv()

class Settings(BaseSettings):
    """
    Central application configuration.
    Strict, explicit, enterprise-safe.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid"
    )

    # Core LLM Keys
    GROQ_API_KEY: str = Field(..., min_length=10)
    GEMINI_API_KEY: str = Field(..., min_length=10)
    LANGSMITH_API_KEY: str | None = None

    # LangChain / LangSmith Tracing
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_PROJECT: str | None = None

    # Storage & DB Config
    CHROMA_DB_PATH: str = Field(default="./vectorstore/chroma")
    SQL_DATABASE_URL: str = Field(default="sqlite:///:memory:", description="Full SQLAlchemy database connection string")

    # Model Configuration 
    PRIMARY_MODEL: str = Field(default="llama-3.3-70b-versatile")
    FALLBACK_MODEL: str = Field(default="mixtral-8x7b-32768")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    # System Concurrency Limits
    TOKENIZERS_PARALLELISM: str = Field(default="false")


def _validate_runtime():
    """
    Runtime safety checks for enterprise stability.
    """

    # Ensure Pydantic v2

_validate_runtime()

_validate_runtime()

# Instantiate settings and fail fast
try:
    settings = Settings()
    if settings.GROQ_API_KEY.startswith("PLACEHOLDER") or settings.GEMINI_API_KEY.startswith("PLACEHOLDER"):
        raise ValueError("API Keys are still set to PLACEHOLDER values.")
except Exception as e:
    import sys
    print(f"\n[CRITICAL STARTUP ERROR] Environment Validation Failed: {e}\n"
          f"Please ensure .env contains valid GROQ_API_KEY and GEMINI_API_KEY.\n")
    sys.exit(1)
