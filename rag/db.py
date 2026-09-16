"""Работа с Postgres + pgvector.

pgvector добавляет тип `vector(n)` и операторы расстояния. Нам нужен `<=>` —
косинусное расстояние: 0 = тексты про одно и то же, 1 = про разное, 2 =
противоположны. Поэтому сортировка всегда ORDER BY ... ASC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .chunking import Chunk
from .config import settings


class DimensionMismatch(RuntimeError):
    pass


def connect() -> psycopg.Connection:
    # connect_timeout, иначе на Windows неподнятая база отваливается только
    # через ~30 секунд: libpq сначала долго ждёт по IPv6, потом по IPv4.
    conn = psycopg.connect(
        settings.database_url,
        autocommit=True,
        row_factory=dict_row,
        connect_timeout=10,
    )
    with conn.cursor() as cur:
        # Расширение должно существовать до register_vector: адаптер ищет OID типа.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    dim = settings.embedding_dim
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          BIGSERIAL PRIMARY KEY,
                filename    TEXT        NOT NULL,
                sha256      TEXT        NOT NULL UNIQUE,
                n_pages     INT         NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INT    NOT NULL,
                page_start  INT    NOT NULL,
                page_end    INT    NOT NULL,
                content     TEXT   NOT NULL,
                embedding   VECTOR({dim}) NOT NULL,
                UNIQUE (document_id, chunk_index)
            )
            """
        )
        # HNSW — приблизительный поиск ближайших соседей. На сотне документов
        # разницы не увидеть, на миллионе чанков он и делает поиск возможным.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
    _assert_dimension(conn)


def _assert_dimension(conn: psycopg.Connection) -> None:
    """Ловим самую частую ошибку: сменили модель, забыли пересоздать таблицу."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod AS dim
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
    if row and row["dim"] != settings.embedding_dim:
        raise DimensionMismatch(
            f"В таблице chunks колонка embedding имеет размерность {row['dim']}, "
            f"а модель {settings.embedding_model} выдаёт {settings.embedding_dim}. "
            "Выполните `python -m rag reset` и переиндексируйте документы."
        )


def reset_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.execute("DROP TABLE IF EXISTS documents")


def find_document(conn: psycopg.Connection, sha256: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename FROM documents WHERE sha256 = %s", (sha256,))
        return cur.fetchone()


def insert_document(
    conn: psycopg.Connection, filename: str, sha256: str, n_pages: int
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (filename, sha256, n_pages)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (filename, sha256, n_pages),
        )
        return cur.fetchone()["id"]


def delete_document(conn: psycopg.Connection, document_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    # Vector(...) обязателен: без него psycopg отправит обычный float8[],
    # и Postgres не найдёт оператор для vector <-> double precision[].
    rows = [
        (document_id, c.index, c.page_start, c.page_end, c.text, Vector(list(v)))
        for c, v in zip(chunks, vectors, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, page_start, page_end, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            rows,
        )


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    filename: str
    page_start: int
    page_end: int
    content: str
    distance: float

    @property
    def similarity(self) -> float:
        """Косинусное сходство: 1.0 — идеальное совпадение."""
        return 1.0 - self.distance

    @property
    def source_label(self) -> str:
        pages = (
            f"стр. {self.page_start}"
            if self.page_start == self.page_end
            else f"стр. {self.page_start}–{self.page_end}"
        )
        return f"{self.filename}, {pages}"


def search(
    conn: psycopg.Connection, query_vector: Sequence[float], top_k: int
) -> list[SearchHit]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, d.filename, c.page_start, c.page_end, c.content,
                   c.embedding <=> %s AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY distance
            LIMIT %s
            """,
            (Vector(list(query_vector)), top_k),
        )
        return [
            SearchHit(
                chunk_id=row["id"],
                filename=row["filename"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                content=row["content"],
                distance=float(row["distance"]),
            )
            for row in cur.fetchall()
        ]


def stats(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.filename, d.n_pages, COUNT(c.id) AS n_chunks, d.created_at
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at
            """
        )
        return cur.fetchall()
