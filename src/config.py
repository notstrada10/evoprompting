"""
Configuration settings for the RAG system.
Centralizes all configurable parameters to avoid hard-coded values throughout the codebase.
"""
import os
from dataclasses import dataclass
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
    EMBEDDING_BATCH_SIZE: int = 1  # Set to 1 to disable batching

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 80

    # LLM
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1024

    # Benchmarking
    RAGBENCH_DATASET: str = "rungalileo/ragbench"
    RAGBENCH_SUBSET: str = "hotpotqa"
    DEFAULT_RETRIEVAL_LIMIT: int = 5
    BATCH_SIZE: int = 100 # Concurrent LLM requests

    # Paths
    RESULTS_DIR: str = "thesis_results"

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration values are set."""
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL must be set in environment variables")


@dataclass
class EvolutionConfig:
    """GA hyperparameters for evolutionary RAG variants."""
    population_size: int = 20
    k_initial: int = 5
    max_generations: int = 100
    mutation_rate: float = 0.1
    model_name: str = "deepseek-ai/DeepSeek-V3"
