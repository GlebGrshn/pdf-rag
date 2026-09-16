"""Шаг 1 конвейера: PDF -> текст постранично.

Номер страницы тащим дальше через весь конвейер — без него ссылка на источник
в ответе будет бесполезной ("где-то в этом файле на 200 страниц").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, как видит человек в читалке
    text: str


@dataclass(frozen=True)
class Document:
    path: Path
    sha256: str
    pages: list[Page]

    @property
    def filename(self) -> str:
        return self.path.name


def _clean(text: str) -> str:
    """Приводим извлечённый текст в порядок.

    pypdf часто рвёт слова переносами и оставляет двойные пробелы из-за
    вёрстки в две колонки. Чистим, иначе мусор попадёт в эмбеддинги.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Склеиваем слова, разорванные переносом на границе строки: "инфор-\nмация"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Одиночный перенос внутри абзаца -> пробел; двойной оставляем как границу абзаца
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append(Page(number=index, text=text))
    return Document(path=path, sha256=file_sha256(path), pages=pages)


def find_pdfs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.pdf") if p.is_file())
