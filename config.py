"""Central configuration: one place for paths, models and RAG parameters.

Every value can be overridden with an environment variable (or a local .env file).
This module has no heavy imports, so it is safe to import anywhere, including tests.
"""
import os
from pathlib import Path

try:  # optional: load variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Config:
    """Configuration for the RAG project."""

    # Paths (PROJECT_ROOT is the folder that contains this file)
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PDF_DIR = RAW_DATA_DIR / "pdfs"
    VECTORSTORE_DIR = Path(
        os.getenv("VECTORSTORE_PATH", DATA_DIR / "vectorstore" / "chroma_db")
    )

    # Secrets
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # Retrieval: dense | hybrid | rerank
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

    # RAG parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    SEARCH_K = int(os.getenv("SEARCH_K", "3"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        """Fail early with a clear message if required settings are missing."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        cls.PDF_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    Config.validate()
    print("Configuration validated successfully")
