from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    qdrant_path: str = "./data/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    collection_name: str = "squad_rag"
    bm25_path: str = "./data/bm25.pkl"

    top_k_bm25: int = 20
    top_k_vector: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60

    chunk_size: int = 900
    chunk_overlap: int = 150

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def ensure_dirs(self) -> None:
        Path(self.bm25_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.qdrant_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
