"""Генерация НЕРАЗМЕЧЕННОГО корпуса для дистилляции.

В отличие от ``watch_parser/scripts/build_dataset*.py`` здесь мы НЕ пишем метки в
обучающие данные: метки потом поставит учитель (большая модель). Генератор
переиспользуется только чтобы получить разнообразные русские просьбы, а золотой
ответ (``meta.golden``) сохраняется ИСКЛЮЧИТЕЛЬНО для QA:

- проверки качества учителя («учитель ошибся в K из N»);
- строгого варианта фильтрации (``train_golden``).

В обучение ``golden`` не попадает. Тестовые примеры (``eval.jsonl`` +
``holdout.jsonl``) и прошлые train/valid (v1/v2/v3) исключаются: корпус — свежие
входы, которые ученик раньше не видел.

Запуск:  python data/make_inputs.py
Вывод:   data/corpus.jsonl  ({"user": ..., "meta": {"category", "golden"}})
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

# Единый источник правды задачи (spec) и генераторов (build_dataset) — в watch_parser/scripts.
_WATCH_SCRIPTS = Path(__file__).resolve().parents[2] / "watch_parser" / "scripts"
if str(_WATCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_WATCH_SCRIPTS))

import build_dataset as bd  # noqa: E402  isort: skip
import build_dataset_v3 as v3  # noqa: E402  isort: skip
from spec import normalize_text  # noqa: E402  isort: skip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WATCH_DATA = Path(__file__).resolve().parents[2] / "watch_parser" / "data"

SEED = 4242

# Категории и объёмы корпуса (по образцу v1 + точечные v3, свежий seed).
SIMPLE = 120
MULTI = 80
COLLOQUIAL = 80
CORRECTION = 50
INVALID_COUNTS = {
    "missing_tickers": 12,
    "missing_intervals": 12,
    "missing_poll_interval": 12,
    "missing_report_interval": 13,
    "bad_interval": 12,
    "bad_ticker": 11,
    "not_watch": 5,
    "ambiguous": 6,
    "empty": 1,
}
DUP_SUCCESS = 40
PERIOD_SUCCESS = 12


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ngrams(text: str, n: int = 8) -> set[str]:
    text = normalize_text(text)
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_prior_and_test() -> tuple[set[str], list[tuple[str, set[str]]]]:
    """Нормализованные тексты всех прошлых наборов + n-граммы теста."""
    prior: set[str] = set()
    for path in (
        WATCH_DATA / "eval.jsonl",
        WATCH_DATA / "holdout.jsonl",
        WATCH_DATA / "v1" / "train.jsonl",
        WATCH_DATA / "v1" / "valid.jsonl",
        WATCH_DATA / "v2" / "train.jsonl",
        WATCH_DATA / "v2" / "valid.jsonl",
        WATCH_DATA / "v3" / "train.jsonl",
        WATCH_DATA / "v3" / "valid.jsonl",
    ):
        if not path.exists():
            continue
        for row in load_rows(path):
            key = normalize_text(row["meta"]["user"])
            if key:
                prior.add(key)

    test_grams = [
        (normalize_text(row["meta"]["user"]), ngrams(row["meta"]["user"]))
        for row in load_rows(WATCH_DATA / "eval.jsonl") + load_rows(WATCH_DATA / "holdout.jsonl")
    ]
    return prior, test_grams


def main() -> None:
    rng = random.Random(SEED)
    bd.rng = rng
    bd.TICKER_USAGE = Counter()

    prior, test_grams = load_prior_and_test()

    generated: list[dict] = []
    generated += bd.generate_valid_category(SIMPLE, "simple", (1, 1), "formal")
    generated += bd.generate_valid_category(MULTI, "multi", (2, 4), "formal")
    generated += bd.generate_valid_category(COLLOQUIAL, "colloquial", (1, 3), "colloquial")
    generated += bd.generate_invalid(INVALID_COUNTS)
    generated += bd.generate_corrections(CORRECTION)
    generated += v3.duplicate_success(rng, DUP_SUCCESS)
    generated += v3.period_success(rng, PERIOD_SUCCESS)
    generated += v3.hard_invalid(rng)

    corpus: list[dict] = []
    seen: set[str] = set()
    skipped_exact, skipped_near_test = 0, 0
    for row in generated:
        user = row["meta"]["user"]
        key = normalize_text(user)
        if key in seen or key in prior:
            skipped_exact += 1
            continue
        grams = ngrams(user)
        if any(jaccard(grams, tg) >= 0.5 for _, tg in test_grams):
            skipped_near_test += 1
            continue
        seen.add(key)
        corpus.append(
            {
                "user": user,
                "meta": {
                    "category": row["meta"]["category"],
                    "golden": row["meta"]["expected"],
                },
            }
        )

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in corpus:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "seed": SEED,
        "generated": len(generated),
        "corpus": len(corpus),
        "skipped_exact_or_prior": skipped_exact,
        "skipped_near_test": skipped_near_test,
        "by_category": dict(sorted(Counter(row["meta"]["category"] for row in corpus).items())),
        "by_error": dict(
            sorted(
                Counter(
                    row["meta"]["golden"]["error"]
                    for row in corpus
                    if not row["meta"]["golden"].get("ok")
                ).items()
            )
        ),
    }
    (DATA / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n--- 5 случайных входов корпуса ---")
    for row in random.Random(7).sample(corpus, 5):
        print(row["meta"]["category"], "|", row["user"])


if __name__ == "__main__":
    main()
