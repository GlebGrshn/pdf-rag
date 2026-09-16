"""Индексация: PDF -> чанки -> векторы -> Postgres.

Дедупликация идёт по sha256 файла, а не по имени: переименованный дубликат
не попадёт в базу дважды, а изменённый файл будет переиндексирован.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psycopg

from . import db, embeddings, pdf
from .chunking import chunk_document
from .config import settings

BATCH_SIZE = 64

Progress = Callable[[str], None]


@dataclass
class IngestResult:
    filename: str
    status: str  # "indexed" | "skipped" | "empty"
    n_pages: int = 0
    n_chunks: int = 0


def ingest_file(
    conn: psycopg.Connection,
    path: Path,
    *,
    force: bool = False,
    progress: Progress = lambda _: None,
) -> IngestResult:
    progress(f"  читаю {path.name}")
    document = pdf.read_pdf(path)

    if not document.pages:
        return IngestResult(path.name, "empty")

    existing = db.find_document(conn, document.sha256)
    if existing and not force:
        return IngestResult(path.name, "skipped")
    if existing:
        db.delete_document(conn, existing["id"])

    chunks = chunk_document(
        document, settings.chunk_target_chars, settings.chunk_overlap_chars
    )
    if not chunks:
        return IngestResult(path.name, "empty", n_pages=len(document.pages))

    progress(f"  считаю эмбеддинги: {len(chunks)} чанков")
    # Всё одним куском: иначе падение на середине оставит документ в базе
    # частично проиндексированным, а дедупликация по sha256 решит, что он готов,
    # и больше никогда его не переиндексирует.
    with conn.transaction():
        document_id = db.insert_document(
            conn, document.filename, document.sha256, len(document.pages)
        )
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            vectors = embeddings.embed_passages([c.text for c in batch])
            db.insert_chunks(conn, document_id, batch, vectors)

    return IngestResult(
        path.name, "indexed", n_pages=len(document.pages), n_chunks=len(chunks)
    )


def ingest_path(
    conn: psycopg.Connection,
    target: Path,
    *,
    force: bool = False,
    progress: Progress = lambda _: None,
) -> list[IngestResult]:
    paths = pdf.find_pdfs(target)
    if not paths:
        return []

    # Прогреваем модель один раз до цикла — иначе загрузка ONNX-файла
    # случится посреди первого документа и будет выглядеть как зависание.
    progress(f"модель эмбеддингов: {settings.embedding_model}")
    embeddings.warm_up()

    results = []
    for index, path in enumerate(paths, start=1):
        progress(f"[{index}/{len(paths)}] {path.name}")
        results.append(ingest_file(conn, path, force=force, progress=progress))
    return results
