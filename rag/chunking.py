"""Шаг 2 конвейера: текст -> чанки.

Почему вообще нарезаем:
  1. Эмбеддинг целой страницы — это "среднее по больнице": один вектор на
     пять разных мыслей, поиск по нему работает плохо.
  2. В контекст модели надо класть только релевантное, а не весь документ.

Ключевая деталь реализации: режем по границам абзацев и предложений, а не по
голым символам, и для каждого чанка помним, на каких страницах он лежит.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Iterator

from .pdf import Document

MIN_CHUNK_CHARS = 60

_PARAGRAPH_RE = re.compile(r"[^\n]+")
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*\s*")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"стр. {self.page_start}"
        return f"стр. {self.page_start}–{self.page_end}"


def _iter_units(text: str, max_len: int) -> Iterator[tuple[int, str]]:
    """Разбиваем текст на минимальные неделимые куски со смещениями.

    Абзац -> если длинный, предложения -> если и они длинные, режем жёстко.
    Смещение нужно, чтобы потом восстановить номер страницы.
    """
    for para in _PARAGRAPH_RE.finditer(text):
        para_text, para_start = para.group(), para.start()
        if len(para_text) <= max_len:
            yield para_start, para_text
            continue

        for sentence in _SENTENCE_RE.finditer(para_text):
            sent_text = sentence.group().strip()
            if not sent_text:
                continue
            sent_start = para_start + sentence.start()
            if len(sent_text) <= max_len:
                yield sent_start, sent_text
                continue
            # Предложение длиннее целевого размера (таблица, список без точек) —
            # режем по символам, тут уже не до красоты.
            for offset in range(0, len(sent_text), max_len):
                yield sent_start + offset, sent_text[offset : offset + max_len]


def _build_page_map(document: Document) -> tuple[str, list[int], list[int]]:
    """Склеиваем страницы в один текст и запоминаем, где какая начинается."""
    offsets: list[int] = []
    numbers: list[int] = []
    cursor = 0
    for page in document.pages:
        offsets.append(cursor)
        numbers.append(page.number)
        cursor += len(page.text) + 2  # +2 — разделитель "\n\n"
    return "\n\n".join(page.text for page in document.pages), offsets, numbers


def chunk_document(
    document: Document, target_chars: int, overlap_chars: int
) -> list[Chunk]:
    text, page_offsets, page_numbers = _build_page_map(document)
    if not text:
        return []

    def page_at(offset: int) -> int:
        # bisect_right - 1 = последняя страница, начавшаяся не позже offset
        return page_numbers[max(0, bisect.bisect_right(page_offsets, offset) - 1)]

    chunks: list[Chunk] = []

    def flush(units: list[tuple[int, str]]) -> None:
        if not units:
            return
        start = units[0][0]
        end = units[-1][0] + len(units[-1][1])
        # Берём срез исходного текста, а не склейку кусков: так цитата в ответе
        # совпадёт с документом символ в символ.
        body = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            return
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                page_start=page_at(start),
                page_end=page_at(max(start, end - 1)),
            )
        )

    current: list[tuple[int, str]] = []
    current_len = 0

    for offset, unit in _iter_units(text, target_chars):
        if current and current_len + len(unit) > target_chars:
            flush(current)
            # Хвост предыдущего чанка переносим в новый — это и есть overlap.
            # Без него фраза, разрезанная по границе, не найдётся ни в одном чанке.
            tail: list[tuple[int, str]] = []
            tail_len = 0
            for prev_offset, prev_unit in reversed(current):
                if tail_len + len(prev_unit) > overlap_chars:
                    break
                tail.insert(0, (prev_offset, prev_unit))
                tail_len += len(prev_unit) + 1
            current, current_len = tail, tail_len

        current.append((offset, unit))
        current_len += len(unit) + 1

    flush(current)
    return chunks
