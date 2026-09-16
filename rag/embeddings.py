"""Шаг 3 конвейера: текст -> вектор.

У Anthropic нет embeddings API — это принципиально другая модель, и её берут
отдельно. Здесь считаем локально через fastembed (onnxruntime, без torch):
бесплатно, офлайн и абсолютно детерминированно.

Отдельная тонкость — асимметричный поиск. Вопрос и абзац из документа
выглядят по-разному ("Какой срок гарантии?" vs "Гарантийный срок составляет
24 месяца"), поэтому модели семейства E5 просят помечать их префиксами
"query: " и "passage: ". Для моделей, которые этого не ждут, префиксы,
наоборот, портят результат — поэтому проверяем имя модели.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Sequence

from fastembed import TextEmbedding

from .config import settings

# fastembed предупреждает, что сменил pooling с CLS на mean. Для нас это
# ничего не меняет (важно лишь, чтобы индексация и поиск шли одной версией),
# но сообщение вылезало бы на каждый запуск CLI и выглядело как поломка.
warnings.filterwarnings(
    "ignore", message=".*mean pooling instead of CLS.*", category=UserWarning
)


def _wants_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # Первый вызов скачивает модель в local_cache/ (дальше — из кеша).
    settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=settings.embedding_model,
        cache_dir=str(settings.embedding_cache_dir),
    )


def _embed(texts: Sequence[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _model().embed(list(texts))]


def embed_passages(texts: Sequence[str]) -> list[list[float]]:
    """Векторы для кусков документов (то, что кладём в базу)."""
    if _wants_e5_prefixes(settings.embedding_model):
        texts = [f"passage: {text}" for text in texts]
    return _embed(texts)


def embed_query(text: str) -> list[float]:
    """Вектор для вопроса пользователя (то, чем ищем)."""
    if _wants_e5_prefixes(settings.embedding_model):
        text = f"query: {text}"
    return _embed([text])[0]


def warm_up() -> None:
    """Прогреть модель заранее, чтобы скачивание не мешалось в середине вывода."""
    _model()
