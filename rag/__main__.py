"""CLI: python -m rag <команда>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import answer as answer_mod
from . import db, embeddings
from .config import settings
from .ingest import ingest_path


def _fix_console() -> None:
    # Без этого русский текст в старой консоли Windows превращается в вопросики.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def _connect_or_exit():
    try:
        conn = db.connect()
    except Exception as exc:  # noqa: BLE001 — хотим человеческое сообщение
        print(f"Не могу подключиться к Postgres ({settings.database_url}).", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  Проверьте, что база поднята: docker compose up -d", file=sys.stderr)
        raise SystemExit(1)
    return conn


def cmd_init(_: argparse.Namespace) -> None:
    conn = _connect_or_exit()
    db.init_schema(conn)
    print(f"Схема готова. Размерность вектора: {settings.embedding_dim}.")


def cmd_reset(_: argparse.Namespace) -> None:
    conn = _connect_or_exit()
    db.reset_schema(conn)
    db.init_schema(conn)
    print("Таблицы пересозданы, база пуста.")


def cmd_ingest(args: argparse.Namespace) -> None:
    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Путь не найден: {target}")

    conn = _connect_or_exit()
    db.init_schema(conn)

    results = ingest_path(conn, target, force=args.force, progress=print)
    if not results:
        raise SystemExit(f"PDF-файлов в {target} не нашлось.")

    print("\nИтог:")
    for item in results:
        if item.status == "indexed":
            print(f"  + {item.filename}: {item.n_pages} стр. -> {item.n_chunks} чанков")
        elif item.status == "skipped":
            print(f"  = {item.filename}: уже в базе (--force чтобы переиндексировать)")
        else:
            print(f"  ! {item.filename}: не удалось извлечь текст (скан без OCR?)")


def cmd_stats(_: argparse.Namespace) -> None:
    conn = _connect_or_exit()
    db.init_schema(conn)
    rows = db.stats(conn)
    if not rows:
        print("База пуста. Запустите: python -m rag ingest ./docs")
        return
    print(f"{'Документ':<45} {'Стр.':>6} {'Чанков':>8}")
    print("-" * 61)
    for row in rows:
        print(f"{row['filename'][:44]:<45} {row['n_pages']:>6} {row['n_chunks']:>8}")
    print("-" * 61)
    print(f"Всего чанков: {sum(r['n_chunks'] for r in rows)}")


def cmd_search(args: argparse.Namespace) -> None:
    """Голый поиск без модели — видно, что именно RAG находит."""
    conn = _connect_or_exit()
    db.init_schema(conn)
    hits = db.search(conn, embeddings.embed_query(args.query), args.top_k)
    if not hits:
        print("Ничего не нашлось. База пуста?")
        return
    for position, hit in enumerate(hits, start=1):
        preview = hit.content[:300].replace("\n", " ")
        print(f"\n[{position}] {hit.source_label}  (сходство {hit.similarity:.3f})")
        print(f"    {preview}{'…' if len(hit.content) > 300 else ''}")


def cmd_ask(args: argparse.Namespace) -> None:
    if not answer_mod.credentials_available():
        print("Ключ Anthropic не найден — команда ask без него не работает.", file=sys.stderr)
        print("  Впишите ANTHROPIC_API_KEY в файл .env (он в .gitignore),", file=sys.stderr)
        print("  либо выполните `ant auth login`.", file=sys.stderr)
        print("  Поиск работает и без ключа: python -m rag search \"ваш запрос\"", file=sys.stderr)
        raise SystemExit(1)

    conn = _connect_or_exit()
    db.init_schema(conn)

    hits = db.search(conn, embeddings.embed_query(args.question), args.top_k)

    if args.show_chunks:
        print("Найденные фрагменты:")
        for position, hit in enumerate(hits, start=1):
            print(f"  [{position}] {hit.source_label} (сходство {hit.similarity:.3f})")
        print()

    print("Ответ:\n")
    result = answer_mod.answer_question(
        args.question, hits, on_text=lambda text: print(text, end="", flush=True)
    )
    print("\n")

    if result.citations:
        print("Источники:")
        for citation in result.citations:
            quote = citation.cited_text
            if len(quote) > 200:
                quote = quote[:200] + "…"
            print(f"  • {citation.source_label}")
            print(f"    «{quote}»")
    elif hits:
        print("Источники: модель не сослалась ни на один фрагмент.")

    print(
        f"\nТокенов: {result.input_tokens} на вход, {result.output_tokens} на выход."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="Мини-RAG: PDF -> pgvector -> ответ Claude со ссылками на источник.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="создать таблицы и индекс").set_defaults(func=cmd_init)
    sub.add_parser("reset", help="удалить и создать таблицы заново").set_defaults(
        func=cmd_reset
    )
    sub.add_parser("stats", help="что лежит в базе").set_defaults(func=cmd_stats)

    ingest = sub.add_parser("ingest", help="проиндексировать PDF (файл или папку)")
    ingest.add_argument("path", nargs="?", default="./docs")
    ingest.add_argument(
        "--force", action="store_true", help="переиндексировать, даже если файл уже в базе"
    )
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="только поиск, без обращения к модели")
    search.add_argument("query")
    search.add_argument("-k", "--top-k", type=int, default=settings.top_k)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="поиск + ответ модели со ссылками")
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", type=int, default=settings.top_k)
    ask.add_argument("--show-chunks", action="store_true", help="показать, что нашлось")
    ask.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    _fix_console()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except db.DimensionMismatch as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
