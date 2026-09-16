"""Все настройки в одном месте, читаются из .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    database_url: str
    anthropic_model: str
    top_k: int

    # Модель эмбеддингов и её размерность.
    # ВАЖНО: dim обязан совпадать с VECTOR(n) в таблице chunks. Сменили модель —
    # запустите `python -m rag reset`, иначе Postgres честно скажет
    # "expected 384 dimensions, not 1024".
    embedding_model: str
    embedding_dim: int

    # Куда fastembed складывает скачанную ONNX-модель.
    embedding_cache_dir: Path

    # Параметры нарезки. target_chars — к какому размеру стремимся,
    # overlap_chars — сколько символов дублируем между соседними чанками,
    # чтобы мысль не обрывалась ровно на границе.
    chunk_target_chars: int = 1000
    chunk_overlap_chars: int = 150


def load_settings() -> Settings:
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5433/rag"
        ),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
        top_k=int(os.getenv("TOP_K", "5")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        embedding_cache_dir=PROJECT_ROOT / "local_cache",
        chunk_target_chars=int(os.getenv("CHUNK_TARGET_CHARS", "1000")),
        chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "150")),
    )


settings = load_settings()
