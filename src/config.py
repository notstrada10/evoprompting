"""
Configuration settings for the RAG system.
Centralizes all configurable parameters to avoid hard-coded values throughout the codebase.
"""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the RAG system."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Embeddings
    EMBEDDING_MODEL: str = "google/embeddinggemma-300m"
    EMBEDDING_DIM: int = 768
    USE_LOCAL_EMBEDDINGS: bool = True
    EMBEDDING_BATCH_SIZE: int = 1  # Set to 1 to disable batching

    # Chunking
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 50

    # LLM
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    LLM_PROVIDER: str = "deepseek"  # "groq" or "deepseek"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1024

    # Benchmarking
    RAGBENCH_DATASET: str = "rungalileo/ragbench"
    RAGBENCH_SUBSET: str = "hotpotqa"
    DEFAULT_RETRIEVAL_LIMIT: int = 5

    # Paths
    RESULTS_DIR: str = "results"

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration values are set."""
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL must be set in environment variables")
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set in environment variables")
