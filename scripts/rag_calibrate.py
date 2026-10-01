"""Одноразовый скрипт калибровки RAG_MIN_SCORE (не часть бота).

Для каждого запроса из файла строит эмбеддинг, ищет топ-k чанков выбранной стратегии в
индексе RAG_INDEX_DIR и печатает оценку top-1, заголовок и номер чанка; в конце — сводку
по группам (A — по теме корпуса, B — пограничные, C — не по теме) и подсказку о границе.
Релевантность top-1 (отвечает ли чанк на вопрос) оценивается человеком — флаг --snippets
печатает начало текста чанка. Порог зависит от модели эмбеддингов и стратегии: при смене
EMBEDDINGS_MODEL, стратегии или RAG_FIXED_CHUNK_* калибровку нужно повторить.

Запуск из корня проекта (нужны .env бота и запущенный Ollama):
    python scripts/rag_calibrate.py [--strategy structural] [--top-k 3] [--snippets]
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# Файл запросов НЕ входит в репозиторий: запросы привязаны к корпусу документов оператора,
# поэтому он лежит в data/ (в .gitignore). Формат — строки `<группа><TAB><запрос>`, группа A (по
# теме корпуса), B (пограничные) или C (не по теме); пустые строки и строки с # пропускаются.
DEFAULT_QUERIES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "rag_eval", "calibration_queries.txt"
)
GROUPS = ("A", "B", "C")
SNIPPET_CHARS = 200


def load_queries(path: str) -> list[tuple[str, str]]:
    """Читает `<группа>\\t<запрос>`; пустые строки и строки с `#` пропускает."""
    queries: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as file:
        for line_number, raw in enumerate(file, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            group, separator, query = line.partition("\t")
            if not separator or group not in GROUPS or not query.strip():
                raise ValueError(
                    f"{path}:{line_number}: ожидается «<A|B|C><TAB><запрос>», получено {line!r}"
                )
            queries.append((group, query.strip()))
    return queries


def summarize(scores_by_group: dict[str, list[float]]) -> list[str]:
    lines = ["", "Сводка по группам (score top-1):"]
    for group in GROUPS:
        scores = scores_by_group.get(group, [])
        if not scores:
            lines.append(f"  {group}: нет запросов")
            continue
        lines.append(
            f"  {group}: n={len(scores)}  min={min(scores):.3f}  "
            f"median={statistics.median(scores):.3f}  max={max(scores):.3f}"
        )
    a_scores = scores_by_group.get("A", [])
    c_scores = scores_by_group.get("C", [])
    if a_scores and c_scores:
        a_min, c_max = min(a_scores), max(c_scores)
        if a_min > c_max:
            lines.append(
                f"\nГруппы A и C не пересекаются: зазор {c_max:.3f} … {a_min:.3f} "
                f"(середина {(a_min + c_max) / 2:.3f}). Группу B и релевантность top-1 "
                "оцените вручную."
            )
        else:
            lines.append(
                f"\nГруппы A и C ПЕРЕСЕКАЮТСЯ (min A = {a_min:.3f} <= max C = {c_max:.3f}). "
                "Порог выбирается по цене ошибки — см. design.md: пропуск дешевле ложного "
                "включения, поэтому смещайте порог к более строгому, пока большая часть A "
                "остаётся выше него."
            )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Калибровка RAG_MIN_SCORE")
    parser.add_argument("--strategy", default="structural", help="стратегия индекса")
    parser.add_argument("--top-k", type=int, default=3, help="сколько чанков искать")
    parser.add_argument("--queries", default=DEFAULT_QUERIES_FILE, help="файл с запросами")
    parser.add_argument("--snippets", action="store_true", help="печатать начало текста top-1")
    args = parser.parse_args()

    if not os.path.isfile(args.queries):
        print(
            f"Файл запросов не найден: {os.path.normpath(args.queries)}.\n"
            "Создайте его (он не входит в репозиторий) или укажите путь через --queries. Формат: "
            "строки `<A|B|C><TAB><запрос>`, где A — запросы по теме вашего корпуса, B — "
            "пограничные, C — не по теме; пустые строки и строки с # пропускаются.",
            file=sys.stderr,
        )
        return 2
    queries = load_queries(args.queries)

    # config.py читает .env и настраивает логирование — импорт после разбора аргументов,
    # чтобы ошибка в файле запросов не требовала окружения бота.
    import config
    from providers.embeddings_client import embed_texts
    from rag.index_store import index_exists, search

    if not index_exists(args.strategy, config.RAG_INDEX_DIR):
        print(
            f"Индекс стратегии «{args.strategy}» не найден в {config.RAG_INDEX_DIR}. "
            "Сначала запустите `make index`.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Стратегия: {args.strategy}; индекс: {config.RAG_INDEX_DIR}; "
        f"модель эмбеддингов: {config.EMBEDDINGS_MODEL}; запросов: {len(queries)}"
    )
    print(f"{'гр':<3} {'score':>6}  {'чанк':>4}  {'документ':<50}  запрос")

    scores_by_group: dict[str, list[float]] = {}
    for group, query in queries:
        vector = embed_texts([query])[0]
        results = search(args.strategy, config.RAG_INDEX_DIR, vector, args.top_k)
        if not results:
            print(f"{group:<3} {'—':>6}  {'—':>4}  {'(пустой индекс)':<50}  {query}")
            continue
        top = results[0]
        scores_by_group.setdefault(group, []).append(top.score)
        title = top.title if len(top.title) <= 50 else top.title[:47] + "..."
        print(f"{group:<3} {top.score:>6.3f}  {top.chunk_index:>4}  {title:<50}  {query}")
        if args.snippets:
            snippet = " ".join(top.text.split())[:SNIPPET_CHARS]
            print(f"      └ {snippet}")
        if args.top_k > 1 and len(results) > 1:
            rest = ", ".join(f"{item.score:.3f}" for item in results[1:])
            print(f"      (остальные в топе: {rest})")

    print("\n".join(summarize(scores_by_group)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
