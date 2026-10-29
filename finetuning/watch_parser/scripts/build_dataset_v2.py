"""Версия 2 датасета: баланс 50/50 и жёсткие негативные примеры.

Проблема v1: 216 успешных примеров против 54 ошибочных. Модель выучила
«всегда возвращай успешный JSON» и на плохих запросах придумывает правило.
Здесь из каждого успешного примера строятся близкие негативные варианты:
убрать тикер, убрать период, оставить только сводку, подменить тикер на
неизвестный, поставить «30 секунд», добавить третий период.

Итог: 240 успешных + 240 ошибочных = 480 примеров, train/valid 90/10.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_dataset as bd
from spec import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "v2"
SEED = 2027

UNKNOWN_TICKERS = ["Ромашка", "Апельсин", "Тесла", "TSLA", "Банан", "Криптовалюта", "Аэрофлот2"]
BAD_INTERVALS = ["30 секунд", "0 минут", "25 часов", "8 дней", "2 недели", "полторы минуты"]
NOT_WATCH = [
    "Какая погода в Москве?",
    "Привет! Как дела?",
    "Сколько стоит доллар?",
    "Расскажи, что такое облигация.",
    "Посоветуй фильм на вечер.",
    "Как пройти в библиотеку?",
    "Который сейчас час?",
    "Что ты умеешь?",
    "Помоги с домашним заданием по математике.",
    "Напиши стихотворение про осень.",
    "Кто выиграл вчерашний матч?",
    "Переведи слово «инвестиция» на английский.",
    "Сколько будет 17 умножить на 23?",
    "Какие новости в мире?",
    "Дай рецепт борща.",
    "Как настроение?",
    "Что нового?",
    "Помоги выбрать подарок другу.",
    "Объясни, как работает двигатель внутреннего сгорания.",
    "Сколько калорий в яблоке?",
    "Посоветуй книгу.",
    "Как называется столица Австралии?",
    "Напиши список покупок на неделю.",
    "Что такое фотосинтез?",
    "Как ухаживать за кактусом?",
    "Расскажи про историю Рима.",
    "Сделай мне план тренировки.",
    "Кто такой Юрий Гагарин?",
    "Что лучше: чай или кофе?",
    "Как приготовить пасту?",
    "Переведи «доброе утро» на французский.",
    "Сколько дней в феврале?",
    "Придумай название для кота.",
    "Какой сегодня праздник?",
    "Дай совет по сну.",
]

# Сколько негативных примеров каждого кода добавить к исходным 60.
COUNTS = {
    "missing_tickers": 24,
    "missing_intervals": 24,
    "missing_poll_interval": 24,
    "missing_report_interval": 24,
    "bad_interval": 24,
    "bad_ticker": 24,
    "ambiguous": 20,
    "not_watch": 15,
    "empty": 1,
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def variant(code: str, row: dict, rng: random.Random) -> tuple[str, str] | None:
    """Возвращает (user, code) для негативного варианта успешного примера."""
    expected = row["meta"]["expected"]
    tickers = expected["secids"]
    poll = expected["poll_interval"]
    report = expected["report_interval"]
    poll_text = bd.PERIOD_TEXT[poll][0]
    other_poll = rng.choice([p for p in bd.POLL_VALUES if p != poll])
    other_text = bd.PERIOD_TEXT[other_poll][0]

    if code == "missing_tickers":
        texts = [
            f"{bd.every(poll, 'formal')} смотри цены, {bd.report(report, 'formal')} присылай сводку.",
            f"Опрос {bd.every(poll, 'formal')}, сводка {bd.report(report, 'formal')}.",
            f"Проверяй цены {bd.every(poll, 'formal')} и присылай отчёт {bd.report(report, 'formal')}.",
        ]
        return rng.choice(texts), code

    if code == "missing_intervals":
        texts = [
            f"Следи за {bd.join_tickers(tickers, 'inst')}.",
            f"Поставь на контроль {bd.join_tickers(tickers, 'acc')}.",
            f"Интересуют {bd.join_tickers(tickers, 'nom')}.",
        ]
        return rng.choice(texts), code

    if code == "missing_poll_interval":
        texts = [
            f"По {bd.join_tickers(tickers, 'dat')}: сводку присылай {bd.report(report, 'formal')}.",
            f"Сводка по {bd.join_tickers(tickers, 'dat')} {bd.report(report, 'formal')}.",
        ]
        return rng.choice(texts), code

    if code == "missing_report_interval":
        texts = [
            f"Следи за {bd.join_tickers(tickers, 'inst')} {bd.every(poll, 'formal')}.",
            f"Проверяй {bd.join_tickers(tickers, 'acc')} {bd.every(poll, 'formal')}.",
        ]
        return rng.choice(texts), code

    if code == "bad_interval":
        bad = rng.choice(BAD_INTERVALS)
        return (
            f"Следи за {bd.join_tickers(tickers, 'inst')} каждые {bad}, "
            f"сводку {bd.report(report, 'formal')}.",
            code,
        )

    if code == "bad_ticker":
        unknown = rng.choice(UNKNOWN_TICKERS)
        forms = [rng.choice(bd.TICKER_FORMS[ticker]["nom"]) for ticker in tickers] + [unknown]
        joined = forms[0] if len(forms) == 1 else ", ".join(forms[:-1]) + " и " + forms[-1]
        return (
            f"Добавь в отслеживание: {joined}. "
            f"Проверка {bd.every(poll, 'formal')}, сводка {bd.report(report, 'formal')}.",
            code,
        )

    if code == "ambiguous":
        if rng.random() < 0.5:
            return (
                f"Следи за {bd.join_tickers(tickers, 'inst')}: опрос {poll_text} и {other_text}, "
                f"сводка {bd.report(report, 'formal')}.",
                code,
            )
        return (
            f"По {bd.join_tickers(tickers, 'dat')}: опрос {bd.every(poll, 'formal')}, "
            f"сводка {bd.report(report, 'formal')} и {bd.report(other_poll, 'formal')}.",
            code,
        )

    if code == "not_watch":
        return rng.choice(NOT_WATCH), code

    if code == "empty":
        return "", code

    raise ValueError(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-v1", default=str(DATA / "v1" / "train.jsonl"))
    parser.add_argument("--valid-v1", default=str(DATA / "v1" / "valid.jsonl"))
    parser.add_argument("--eval", default=str(DATA / "eval.jsonl"))
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bd.rng = rng

    v1 = load(Path(args.train_v1)) + load(Path(args.valid_v1))
    eval_rows = load(Path(args.eval))
    success_rows = [row for row in v1 if row["meta"]["expected"].get("ok")]
    original_invalid = [row for row in v1 if not row["meta"]["expected"].get("ok")]
    if len(success_rows) != 240:
        raise RuntimeError(f"ожидали 240 успешных примеров, нашли {len(success_rows)}")

    seen = {normalize_text(row["meta"]["user"]) for row in v1 + eval_rows if normalize_text(row["meta"]["user"])}
    generated: list[dict] = []
    code_counts: Counter[str] = Counter()

    for code, needed in COUNTS.items():
        attempts = 0
        while code_counts[code] < needed and attempts < needed * 100:
            attempts += 1
            row = rng.choice(success_rows)
            result = variant(code, row, rng)
            if result is None:
                continue
            user, error_code = result
            key = normalize_text(user)
            if key and key in seen:
                continue
            seen.add(key)
            generated.append(bd.make_example(user, {"ok": False, "error": error_code}, "invalid"))
            code_counts[code] += 1
        if code_counts[code] < needed:
            raise RuntimeError(f"не удалось сгенерировать {code}: {code_counts[code]}/{needed}")

    rows = success_rows + original_invalid + generated
    rng.shuffle(rows)

    # Стратифицированное 90/10: успех — по категории, ошибка — по коду.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        expected = row["meta"]["expected"]
        key = "ok:" + row["meta"]["category"] if expected.get("ok") else "err:" + expected["error"]
        buckets[key].append(row)

    train: list[dict] = []
    valid: list[dict] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_valid = max(1, round(len(bucket) * 0.10)) if len(bucket) > 1 else 0
        valid.extend(bucket[:n_valid])
        train.extend(bucket[n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "source_v1": len(v1),
        "success": len(success_rows),
        "invalid_original": len(original_invalid),
        "invalid_generated": len(generated),
        "total": len(rows),
        "train": len(train),
        "valid": len(valid),
        "generated_by_code": dict(sorted(code_counts.items())),
        "by_category": dict(sorted(Counter(row["meta"]["category"] for row in rows).items())),
        "by_error": dict(
            sorted(
                Counter(
                    row["meta"]["expected"]["error"]
                    for row in rows
                    if not row["meta"]["expected"].get("ok")
                ).items()
            )
        ),
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n--- 5 случайных train ---")
    for row in rng.sample(train, 5):
        print(row["meta"]["category"], "|", row["meta"]["user"])
        print("  →", json.dumps(row["meta"]["expected"], ensure_ascii=False))


if __name__ == "__main__":
    main()
