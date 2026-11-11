"""Application settings, read from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "gemini" uses the Google Gemini API; "fake" runs fully offline (useful for tests and demos).
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Chunking: documents are split into overlapping windows of characters.
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval: how many chunks are sent to the model as context.
    top_k: int = 4
    # Chunks scoring below this cosine similarity are ignored.
    min_score: float = 0.2

    max_upload_mb: int = 20
    data_dir: Path = Path("data")
    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
