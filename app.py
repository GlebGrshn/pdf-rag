"""Веб-интерфейс: streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from rag import answer as answer_mod
from rag import db, embeddings
from rag.config import settings

st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Подключаюсь к Postgres…")
def get_connection():
    # Схему создаём не здесь: ошибку размерности надо показать пользователю,
    # а не похоронить внутри закешированного ресурса.
    return db.connect()


@st.cache_resource(show_spinner="Загружаю модель эмбеддингов…")
def warm_embeddings() -> str:
    embeddings.warm_up()
    return settings.embedding_model


st.title("📄 Вопросы к своим PDF")
st.caption("Поиск по pgvector + ответ Claude со ссылкой на страницу источника.")

try:
    conn = get_connection()
except Exception as exc:  # noqa: BLE001
    st.error(
        f"Нет связи с Postgres по адресу `{settings.database_url}`.\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "Поднимите базу: `docker compose up -d`"
    )
    st.stop()

try:
    db.init_schema(conn)
except db.DimensionMismatch as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("База")
    rows = db.stats(conn)
    if rows:
        st.metric("Документов", len(rows))
        st.metric("Чанков", sum(r["n_chunks"] for r in rows))
        for row in rows:
            st.caption(f"{row['filename']} — {row['n_pages']} стр.")
    else:
        st.info("Пусто. Запустите:\n\n`python -m rag ingest ./docs`")

    st.subheader("Параметры")
    top_k = st.slider("Фрагментов в контекст", 1, 15, settings.top_k)
    show_chunks = st.checkbox("Показывать найденные фрагменты", value=False)

    st.subheader("Модели")
    st.caption(f"Эмбеддинги: `{settings.embedding_model}`")
    st.caption(f"Генерация: `{settings.anthropic_model}`")

if not rows:
    st.stop()

warm_embeddings()

question = st.text_input(
    "Вопрос", placeholder="Например: какие условия расторжения договора?"
)

if question:
    with st.spinner("Ищу в документах…"):
        hits = db.search(conn, embeddings.embed_query(question), top_k)

    if show_chunks:
        with st.expander(f"Найдено фрагментов: {len(hits)}", expanded=False):
            for position, hit in enumerate(hits, start=1):
                st.markdown(
                    f"**[{position}] {hit.source_label}** · сходство `{hit.similarity:.3f}`"
                )
                st.text(hit.content[:800])
                st.divider()

    st.subheader("Ответ")
    placeholder = st.empty()
    buffer: list[str] = []

    def on_text(delta: str) -> None:
        buffer.append(delta)
        placeholder.markdown("".join(buffer))

    try:
        with st.spinner("Claude читает фрагменты…"):
            result = answer_mod.answer_question(question, hits, on_text=on_text)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка при обращении к Anthropic API: {exc}")
        st.stop()

    placeholder.markdown(result.text)

    if result.citations:
        st.subheader("Источники")
        for citation in result.citations:
            with st.expander(citation.source_label):
                st.markdown(f"> {citation.cited_text}")
    else:
        st.info("Модель не сослалась ни на один фрагмент.")

    st.caption(
        f"Токенов: {result.input_tokens} на вход, {result.output_tokens} на выход."
    )
