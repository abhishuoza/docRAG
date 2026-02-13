"""Application configuration via pydantic-settings."""

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCRAG_", env_file=".env", extra="ignore")

    # API keys (no DOCRAG_ prefix — read directly from env / .env)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    # LLM
    llm_backend: LLMBackend = LLMBackend.OPENAI
    openai_model: str = "gpt-5-mini-2025-08-07"
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    local_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    local_max_tokens: int = 2048

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Text splitting
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    relevance_threshold: float = 1.0  # ChromaDB distance; chunks scoring above this are flagged

    # ChromaDB
    chroma_dir: Path = Path("./data/chroma")
    chroma_collection: str = "docrag_docs"

    # Cache
    cache_dir: Path = Path("./data/cache")
    cache_ttl_hours: int = 24

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def get_settings() -> Settings:
    """Create and return a Settings instance."""
    return Settings()
