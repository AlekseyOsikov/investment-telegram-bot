"""Одноразовая оценка качества разбора новостей (не часть бота; спека news-digest, требование
«Оценка качества разбора»).

Прогоняет новости из набора через ту же модель разбора, те же сообщения, проверки и повтор, что
в боте (правила — research/news_eval.py и news_digest/rules.py), и пишет отчёт оператору: доля
принятых кодом разборов, нарушения (направление, прогноз, совет, перечни, длины, ссылка) и, если
набор размечен, доля совпадения типа события. Скрипт ничего не применяет сам: модель, промпт и
настройки остаются решением оператора. Запускайте ДО включения NEWS_ENABLED.

Запуск из корня проекта (нужны .env бота и модель разбора NEWS_PROVIDER/NEWS_MODEL):
    python scripts/news_eval.py                          # все data/news_eval/*.json
    python scripts/news_eval.py data/news_eval/a.json    # конкретные наборы
    python scripts/news_eval.py --out-dir /путь/к/отчётам          (или: make news-eval)

Формат набора (НЕ коммитится: каталог data/ в .gitignore):
    {"name": "Октябрь",
     "items": [{"title": "...", "excerpt": "...", "source": "...", "link": "https://...",
                "tickers": ["SBER"], "expected_type": "dividends"}]}
expected_type — необязательная разметка из перечня типов событий. Если набора нет или каталог
пуст, скрипт сообщает, что оценивать нечего, и завершается с кодом 0. Тексты новостей уходят
модели разбора (при облачном NEWS_PROVIDER — провайдеру). Код выхода 1 — ошибка формата набора.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_SETS_GLOB = os.path.join(ROOT, "data", "news_eval", "*.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "news_eval", "reports")


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка качества разбора новостей.")
    parser.add_argument(
        "paths", nargs="*", help="файлы наборов (по умолчанию data/news_eval/*.json)"
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="каталог отчётов")
    args = parser.parse_args()

    # Формат проверяется до импорта config/клиентов: ошибка набора не должна требовать окружения.
    from research import news_eval as ev

    paths = args.paths or sorted(glob.glob(DEFAULT_SETS_GLOB))
    if not paths:
        print(ev.NO_DATA_TEXT)
        return 0
    datasets = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                datasets.append(ev.parse_dataset(json.load(handle), os.path.basename(path)))
        except (OSError, ValueError) as error:  # DatasetError и json — ValueError
            print(f"Набор {path}: {error}", file=sys.stderr)
            return 1

    from config import NEWS_TIMEOUT_SECONDS, SYSTEM_PROMPT
    from news_digest.analyzer import make_completer
    from providers.news_client import news_backend

    if news_backend is None:
        print(
            "Модель разбора не настроена (NEWS_PROVIDER/NEWS_MODEL и ключ провайдера): "
            "оценивать нечем.",
            file=sys.stderr,
        )
        return 1
    complete = make_completer(news_backend, NEWS_TIMEOUT_SECONDS)
    label = f"{news_backend.provider}:{news_backend.model}"

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for dataset in datasets:
        results = []
        for index, item in enumerate(dataset.items, start=1):
            result = ev.evaluate_item(item, index, complete, SYSTEM_PROMPT)
            results.append(result)
            print(
                f"[{dataset.name}] {index}/{len(dataset.items)}: "
                f"{'принято' if result.ok else 'отвергнуто' if not result.error else 'ошибка'}"
            )
        summary = ev.summarize(results)
        report = ev.render_report(dataset, results, summary, label)
        name = "".join(c if c.isalnum() else "_" for c in dataset.name)[:40] or "set"
        out_path = os.path.join(args.out_dir, f"{stamp}-{name}.md")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(report.split("\n\n## По новостям")[0])
        print(f"Отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
