"""Веб-интерфейс ко всему конвейеру: streamlit run app.py

Четыре вкладки: документы (загрузка и индексация), поиск (без модели),
вопрос (с моделью) и диагностика. Всё, кроме вкладки «Вопрос», работает
без ключа Anthropic — эмбеддинги считаются локально.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag import answer as answer_mod
from rag import db, embeddings
from rag.config import PROJECT_ROOT, settings
from rag.ingest import ingest_path

DOCS_DIR = PROJECT_ROOT / "docs"

st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Тяжёлые ресурсы держим в кеше Streamlit: иначе они пересоздавались бы
# на каждое нажатие кнопки — Streamlit перезапускает скрипт целиком.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Подключаюсь к Postgres…")
def get_connection():
    # Схему тут не создаём: ошибку размерности надо показать пользователю,
    # а не похоронить внутри закешированного ресурса.
    return db.connect()


@st.cache_resource(show_spinner="Загружаю модель эмбеддингов (первый раз — долго)…")
def warm_embeddings() -> str:
    embeddings.warm_up()
    return settings.embedding_model


def similarity_bar(value: float) -> None:
    """Сходство приходит в диапазоне примерно −1…1, прогрессбар хочет 0…1."""
    st.progress(min(max(value, 0.0), 1.0))


# --------------------------------------------------------------------------
# Подключение
# --------------------------------------------------------------------------
st.title("📄 PDF RAG")

try:
    conn = get_connection()
except Exception as exc:  # noqa: BLE001
    st.error(
        f"Нет связи с Postgres по адресу `{settings.database_url}`\n\n"
        f"`{type(exc).__name__}: {exc}`"
    )
    st.info("Поднимите базу командой `docker compose start` и обновите страницу.")
    st.stop()

try:
    db.init_schema(conn)
except db.DimensionMismatch as exc:
    st.error(str(exc))
    st.info("Выполните `python -m rag reset`, затем переиндексируйте документы.")
    st.stop()

documents = db.stats(conn)
total_chunks = sum(d["n_chunks"] for d in documents)
has_key = answer_mod.credentials_available()


# --------------------------------------------------------------------------
# Боковая панель
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("База")
    left, right = st.columns(2)
    left.metric("Документов", len(documents))
    right.metric("Чанков", total_chunks)

    st.subheader("Поиск")
    top_k = st.slider(
        "Фрагментов в контекст", 1, 15, settings.top_k,
        help="Сколько ближайших чанков доставать из базы.",
    )

    st.subheader("Состояние")
    st.caption(f"Эмбеддинги: `{settings.embedding_model.split('/')[-1]}` ({settings.embedding_dim})")
    st.caption(f"Чанк: {settings.chunk_target_chars} симв., перекрытие {settings.chunk_overlap_chars}")
    if has_key:
        st.caption(f"Модель ответа: `{settings.anthropic_model}` ✅")
    else:
        st.caption("Ключ Anthropic не найден — вкладка «Вопрос» отключена")


tab_docs, tab_search, tab_ask, tab_diag = st.tabs(
    ["📚 Документы", "🔍 Поиск", "💬 Вопрос", "🩺 Диагностика"]
)


# --------------------------------------------------------------------------
# Документы
# --------------------------------------------------------------------------
with tab_docs:
    st.subheader("Загрузить PDF")

    uploaded = st.file_uploader(
        "Перетащите файлы сюда",
        type=["pdf"],
        accept_multiple_files=True,
        help="Файлы сохраняются в папку docs/ проекта.",
    )

    col_a, col_b = st.columns([1, 3])
    force = col_b.checkbox(
        "Переиндексировать заново",
        help="По умолчанию уже загруженные файлы пропускаются по хешу содержимого.",
    )

    if col_a.button("Проиндексировать", type="primary", use_container_width=True):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for item in uploaded or []:
            (DOCS_DIR / Path(item.name).name).write_bytes(item.getbuffer())
        if uploaded:
            st.caption(f"Сохранено файлов: {len(uploaded)}")

        warm_embeddings()
        with st.status("Индексирую…", expanded=True) as status:
            try:
                results = ingest_path(conn, DOCS_DIR, force=force, progress=st.write)
            except Exception as exc:  # noqa: BLE001
                status.update(label="Ошибка при индексации", state="error")
                st.exception(exc)
                results = []
            else:
                status.update(label="Готово", state="complete")

        for item in results:
            if item.status == "indexed":
                st.success(f"{item.filename}: {item.n_pages} стр. → {item.n_chunks} чанков")
            elif item.status == "skipped":
                st.info(f"{item.filename}: уже в базе")
            else:
                st.warning(f"{item.filename}: текст не извлёкся — похоже, скан без OCR")
        if results:
            st.rerun()

    st.divider()
    st.subheader("В базе")

    if not documents:
        st.info(
            "Пока пусто. Загрузите PDF выше или выполните в терминале:\n\n"
            "`python scripts/make_sample_pdf.py` — создать тестовый договор"
        )
    else:
        for doc in documents:
            c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
            c1.markdown(f"**{doc['filename']}**")
            c1.caption(doc["created_at"].strftime("%d.%m.%Y %H:%M"))
            c2.metric("стр.", doc["n_pages"])
            c3.metric("чанков", doc["n_chunks"])
            if c4.button("Удалить", key=f"del-{doc['id']}", use_container_width=True):
                db.delete_document(conn, doc["id"])
                st.rerun()


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------
with tab_search:
    st.subheader("Что находит поиск")
    st.caption(
        "Только векторный поиск, без обращения к модели. Работает офлайн и бесплатно. "
        "Если ответ на вкладке «Вопрос» мимо — смотреть надо сюда: почти всегда "
        "проблема в выдаче, а не в промпте."
    )

    query = st.text_input(
        "Запрос", key="search_query",
        placeholder="Например: сколько дней на уведомление о расторжении?",
    )

    if query:
        if not documents:
            st.warning("База пуста — индексировать нечего.")
        else:
            warm_embeddings()
            hits = db.search(conn, embeddings.embed_query(query), top_k)
            for position, hit in enumerate(hits, start=1):
                with st.container(border=True):
                    head, score = st.columns([3, 1])
                    head.markdown(f"**[{position}] {hit.source_label}**")
                    score.markdown(f"сходство `{hit.similarity:.3f}`")
                    similarity_bar(hit.similarity)
                    st.text(hit.content)


# --------------------------------------------------------------------------
# Вопрос
# --------------------------------------------------------------------------
with tab_ask:
    st.subheader("Вопрос с ответом по документам")

    if not has_key:
        st.warning("Ключ Anthropic не найден — эта вкладка отключена.")
        st.markdown(
            "Впишите ключ в файл `.env` в корне проекта:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```\n\n"
            "и перезапустите `streamlit run app.py`. Файл в `.gitignore`, "
            "в репозиторий не попадёт.\n\n"
            "Индексация и вкладка «Поиск» работают и без ключа."
        )
    elif not documents:
        st.warning("База пуста — сначала загрузите документы на вкладке «Документы».")
    else:
        question = st.text_input(
            "Вопрос", key="ask_question",
            placeholder="Например: какой срок уведомления о расторжении договора?",
        )
        show_chunks = st.checkbox("Показать найденные фрагменты", value=False)

        if question:
            warm_embeddings()
            with st.spinner("Ищу в документах…"):
                hits = db.search(conn, embeddings.embed_query(question), top_k)

            if show_chunks:
                with st.expander(f"Найдено фрагментов: {len(hits)}"):
                    for position, hit in enumerate(hits, start=1):
                        st.markdown(
                            f"**[{position}] {hit.source_label}** · `{hit.similarity:.3f}`"
                        )
                        st.text(hit.content)
                        st.divider()

            st.markdown("### Ответ")
            placeholder = st.empty()
            buffer: list[str] = []

            def on_text(delta: str) -> None:
                buffer.append(delta)
                placeholder.markdown("".join(buffer))

            try:
                with st.spinner("Claude читает фрагменты…"):
                    result = answer_mod.answer_question(question, hits, on_text=on_text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка обращения к Anthropic API: {exc}")
                st.stop()

            placeholder.markdown(result.text)

            if result.citations:
                st.markdown("### Источники")
                st.caption(
                    "Цитаты формирует API по реальному тексту фрагмента — "
                    "выдумать источник модель не может."
                )
                for citation in result.citations:
                    with st.expander(citation.source_label):
                        st.markdown(f"> {citation.cited_text}")
            else:
                st.info("Модель не сослалась ни на один фрагмент.")

            st.caption(
                f"Токенов: {result.input_tokens} на вход, "
                f"{result.output_tokens} на выход."
            )


# --------------------------------------------------------------------------
# Диагностика
# --------------------------------------------------------------------------
with tab_diag:
    st.subheader("Состояние системы")

    info = db.server_info(conn)
    col_dim = db.column_dim(conn)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**База данных**")
        st.text(f"Postgres : {info['postgres']}")
        st.text(f"pgvector : {info['pgvector'] or 'не установлено'}")
        st.text(f"Адрес    : {settings.database_url}")

        if db.has_hnsw_index(conn):
            st.success("HNSW-индекс на месте")
        else:
            st.warning("HNSW-индекса нет — поиск будет перебирать все строки")

    with c2:
        st.markdown("**Эмбеддинги**")
        st.text(f"Модель   : {settings.embedding_model}")
        st.text(f"Модель даёт : {settings.embedding_dim}")
        st.text(f"В колонке   : {col_dim if col_dim is not None else '—'}")

        if col_dim is not None and col_dim != settings.embedding_dim:
            st.error("Размерности разошлись — нужен `python -m rag reset`")
        elif col_dim is not None:
            st.success("Размерности совпадают")

    st.divider()
    st.markdown("**Нарезка**")
    st.caption(
        "Размер чанка — главная ручка качества поиска. Слишком крупный кусок "
        "смешивает несколько тем в один вектор, и поиск промахивается."
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Целевой размер", f"{settings.chunk_target_chars} симв.")
    d2.metric("Перекрытие", f"{settings.chunk_overlap_chars} симв.")
    d3.metric("Чанков в базе", total_chunks)

    if documents:
        avg = total_chunks / sum(d["n_pages"] for d in documents)
        st.caption(f"В среднем {avg:.1f} чанков на страницу.")
        if avg < 1.2:
            st.warning(
                "Меньше ~1.2 чанка на страницу — куски, скорее всего, слишком крупные. "
                "Попробуйте уменьшить CHUNK_TARGET_CHARS в .env и переиндексировать."
            )
