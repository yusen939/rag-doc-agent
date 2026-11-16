from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    data_dir: Path
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    request_timeout: float = 120.0
    ollama_num_ctx: int = 8192
    chunk_size: int = 700
    chunk_overlap: int = 120
    retrieve_top_k: int = 8
    rerank_top_n: int = 4
    max_file_mb: int = 20
    vector_backend: str = "local"
    chroma_collection: str = "documents"
    redis_url: str | None = None
    session_ttl_seconds: int = 60 * 60 * 6
    history_messages: int = 10
    cors_origins: tuple[str, ...] = ("http://localhost:8000", "http://127.0.0.1:8000")
    allow_origins: bool = False

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.json"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "documents.json"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("DATA_DIR", "data")).expanduser()
        origins = tuple(
            item.strip()
            for item in os.getenv(
                "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
            ).split(",")
            if item.strip()
        )
        return cls(
            data_dir=data_dir,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chat_model=os.getenv("CHAT_MODEL", "qwen2.5:7b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "120")),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", "8")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "4")),
            max_file_mb=int(os.getenv("MAX_FILE_MB", "20")),
            vector_backend=os.getenv("VECTOR_BACKEND", "local").lower(),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "documents"),
            redis_url=os.getenv("REDIS_URL") or None,
            session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 6))),
            history_messages=int(os.getenv("HISTORY_MESSAGES", "10")),
            cors_origins=origins,
            allow_origins=_env_bool("ALLOW_ALL_ORIGINS", False),
        )
