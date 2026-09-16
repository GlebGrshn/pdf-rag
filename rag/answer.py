"""Шаг 5 конвейера: найденные чанки + вопрос -> ответ Claude со ссылками.

Тут используется встроенный механизм цитирования Anthropic: каждый чанк
уходит как content-блок типа `document` с `citations: {"enabled": True}`.
В ответ модель возвращает текст, разбитый на блоки, и у процитированных
блоков есть поле `citations` с точной выдержкой (`cited_text`) и индексом
документа.

Почему это лучше, чем просить модель писать "[1]" руками: цитату формирует
API по реальному тексту документа, её нельзя выдумать. Классический вариант
с ручными маркерами работает, но модель иногда ссылается на источник, которого
в контексте не было.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import anthropic

from .config import settings
from .db import SearchHit

SYSTEM_PROMPT = """\
Ты отвечаешь на вопросы строго по приложенным фрагментам документов.

Правила:
- Используй только информацию из фрагментов. Не добавляй ничего из общих знаний.
- Если во фрагментах нет ответа, так и скажи: «В предоставленных документах \
ответа на этот вопрос нет». Не выдумывай и не строй догадок.
- Отвечай на языке вопроса, по делу, без вступлений вроде «Согласно документам».
- Если фрагменты противоречат друг другу, покажи оба варианта и укажи это явно.
"""


@dataclass(frozen=True)
class Citation:
    source_label: str
    cited_text: str
    hit_index: int


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def used_hits(self) -> list[int]:
        """Индексы чанков, на которые модель реально сослалась."""
        return sorted({c.hit_index for c in self.citations})


def _build_content(question: str, hits: Sequence[SearchHit]) -> list[dict]:
    """Документы идут ПЕРЕД вопросом — так модель сначала читает контекст."""
    blocks: list[dict] = [
        {
            "type": "document",
            "source": {
                "type": "content",
                "content": [{"type": "text", "text": hit.content}],
            },
            "title": hit.source_label,
            "citations": {"enabled": True},
        }
        for hit in hits
    ]
    blocks.append({"type": "text", "text": question})
    return blocks


def _collect(message, hits: Sequence[SearchHit]) -> Answer:
    text_parts: list[str] = []
    citations: list[Citation] = []
    seen: set[tuple[int, str]] = set()

    for block in message.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            index = getattr(citation, "document_index", None)
            quote = (getattr(citation, "cited_text", "") or "").strip()
            if index is None or index >= len(hits):
                continue
            key = (index, quote)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    source_label=hits[index].source_label,
                    cited_text=quote,
                    hit_index=index,
                )
            )

    return Answer(
        text="".join(text_parts).strip(),
        citations=citations,
        refused=message.stop_reason == "refusal",
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


def answer_question(
    question: str,
    hits: Sequence[SearchHit],
    *,
    on_text: Callable[[str], None] | None = None,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    if not hits:
        return Answer(text="В базе нет ни одного подходящего фрагмента. "
                           "Проиндексируйте документы: `python -m rag ingest ./docs`.")

    client = client or anthropic.Anthropic()

    with client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        # Adaptive thinking: модель сама решает, сколько думать. Здесь это
        # помогает честно определить, что ответа в контексте нет.
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": _build_content(question, hits)}],
    ) as stream:
        if on_text is not None:
            for delta in stream.text_stream:
                on_text(delta)
        message = stream.get_final_message()

    result = _collect(message, hits)
    if result.refused and not result.text:
        result.text = "Модель отказалась отвечать на этот запрос."
    return result
