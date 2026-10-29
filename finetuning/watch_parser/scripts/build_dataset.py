"""Генерация train/valid для задачи «просьба → JSON /watch».

300 примеров, 5 категорий:
  simple       — один тикер, формальные периоды;
  multi        — 2–4 тикера;
  colloquial   — разговорные периоды («полчаса», «раз в сутки»);
  invalid      — ошибки всех кодов;
  correction   — исправления, дубли, нижний регистр.

Метку строит генератор, а не языковая модель: для каждого примера известны
тикеры и периоды по построению. Это важно — иначе ошибки модели попали бы
в эталон. Скрипт только пишет данные; проверка — в check_dataset.py.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from spec import (
    TICKER_FORMS,
    canonical_json,
    error,
    example_messages,
    normalize_text,
    success,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
rng = random.Random(SEED)

# value → (формально, разговорно)
PERIOD_TEXT: dict[str, tuple[str, str]] = {
    "5m": ("5 минут", "5 минут"),
    "10m": ("10 минут", "10 минут"),
    "15m": ("15 минут", "четверть часа"),
    "20m": ("20 минут", "20 минут"),
    "30m": ("30 минут", "полчаса"),
    "45m": ("45 минут", "45 минут"),
    "60m": ("60 минут", "60 минут"),
    "90m": ("90 минут", "90 минут"),
    "1h": ("час", "час"),
    "2h": ("2 часа", "два часа"),
    "3h": ("3 часа", "три часа"),
    "6h": ("6 часов", "шесть часов"),
    "12h": ("12 часов", "полсуток"),
    "1d": ("сутки", "сутки"),
    "2d": ("2 дня", "два дня"),
    "3d": ("3 дня", "трое суток"),
    "7d": ("7 дней", "неделю"),
}

POLL_VALUES = ["5m", "10m", "15m", "20m", "30m", "45m", "60m", "90m", "1h", "2h", "3h", "6h"]
REPORT_VALUES = [
    "15m", "30m", "45m", "60m", "90m", "1h", "2h", "3h", "6h", "12h", "1d", "2d", "3d", "7d"
]

VALID_TICKERS = sorted(TICKER_FORMS)
TICKER_USAGE: Counter[str] = Counter()


def pick_tickers(count: int) -> list[str]:
    """Берём наименее использованные тикеры, чтобы не было перекоса в один тикер."""
    if count <= 0:
        return []
    pool = sorted(VALID_TICKERS, key=lambda ticker: (TICKER_USAGE[ticker], rng.random()))
    chosen = pool[:count]
    rng.shuffle(chosen)
    TICKER_USAGE.update(chosen)
    return chosen


def minutes(value: str) -> int:
    number = int(value[:-1])
    unit = value[-1]
    return number * {"m": 1, "h": 60, "d": 1440}[unit]


def every(value: str, style: str) -> str:
    text = PERIOD_TEXT[value][1 if style == "colloquial" else 0]
    if style == "colloquial":
        if text == "четверть часа":
            return "раз в четверть часа"
        if text in {"час", "сутки"}:
            return f"каждый {'час' if text == 'час' else 'день'}"
        if text == "трое суток":
            return "раз в трое суток"
        if text == "неделю":
            return "раз в неделю"
        if text == "полсуток":
            return "каждые полсуток"
        return f"каждые {text}"
    if value in {"1h", "1d"}:
        return "каждый час" if value == "1h" else "каждый день"
    return f"каждые {text}"


def report(value: str, style: str) -> str:
    text = PERIOD_TEXT[value][1 if style == "colloquial" else 0]
    if style == "colloquial":
        if text == "неделю":
            return "раз в неделю"
        if text == "трое суток":
            return "раз в трое суток"
        if text == "полсуток":
            return "раз в полсуток"
        return f"раз в {text}"
    if value == "1h":
        return "раз в час"
    if value == "1d":
        return "раз в сутки"
    return f"раз в {text}"


def report_plain(value: str) -> str:
    return PERIOD_TEXT[value][0]


def join_tickers(tickers: list[str], case: str) -> str:
    forms = [rng.choice(TICKER_FORMS[t][case]) for t in tickers]
    if len(forms) == 1:
        return forms[0]
    if len(forms) == 2:
        return f"{forms[0]} и {forms[1]}"
    return ", ".join(forms[:-1]) + " и " + forms[-1]


def pick_periods() -> tuple[str, str]:
    poll = rng.choice(POLL_VALUES)
    report_candidates = [r for r in REPORT_VALUES if minutes(r) >= minutes(poll)]
    return poll, rng.choice(report_candidates)


# ---------------------------------------------------------------------------
# Шаблоны
# ---------------------------------------------------------------------------
FORMAL_TEMPLATES = [
    lambda t, p, r: f"Следи за {join_tickers(t, 'inst')} {every(p, 'formal')}, сводку присылай {report(r, 'formal')}.",
    lambda t, p, r: f"Настрой наблюдение за {join_tickers(t, 'inst')}: опрос {every(p, 'formal')}, сводка {report(r, 'formal')}.",
    lambda t, p, r: f"Хочу получать сводки по {join_tickers(t, 'dat')} {report(r, 'formal')}, а цены проверять {every(p, 'formal')}.",
    lambda t, p, r: f"{join_tickers(t, 'nom')} — следи {every(p, 'formal')}, итоги присылай {report(r, 'formal')}.",
    lambda t, p, r: f"Добавь в отслеживание {join_tickers(t, 'acc')}. Проверка цен — {every(p, 'formal')}, отчёт — {report(r, 'formal')}.",
    lambda t, p, r: f"Мониторь {join_tickers(t, 'acc')} с интервалом {PERIOD_TEXT[p][0]}, сводку — {report(r, 'formal')}.",
    lambda t, p, r: f"По {join_tickers(t, 'dat')}: {every(p, 'formal')} смотри цены, {report(r, 'formal')} присылай отчёт.",
    lambda t, p, r: f"За {join_tickers(t, 'inst')} наблюдай {every(p, 'formal')}; сводка — {report(r, 'formal')}.",
    lambda t, p, r: f"Следи за {join_tickers(t, 'inst')}. Опрос — {PERIOD_TEXT[p][0]}, сводка — {report_plain(r)}.",
    lambda t, p, r: f"Хочу watch на {join_tickers(t, 'acc')}: проверять {every(p, 'formal')}, отчитываться {report(r, 'formal')}.",
]

COLLOQUIAL_TEMPLATES = [
    lambda t, p, r: f"Следи за {join_tickers(t, 'inst')} {every(p, 'colloquial')}, сводку — {report(r, 'colloquial')}.",
    lambda t, p, r: f"По {join_tickers(t, 'dat')}: проверяй {every(p, 'colloquial')}, отчёт {report(r, 'colloquial')}.",
    lambda t, p, r: f"{join_tickers(t, 'nom')} мониторь {every(p, 'colloquial')}, итоги присылай {report(r, 'colloquial')}.",
    lambda t, p, r: f"Хочу следить за {join_tickers(t, 'inst')}: опрос {every(p, 'colloquial')}, сводка {report(r, 'colloquial')}.",
    lambda t, p, r: f"Наблюдай за {join_tickers(t, 'inst')} {every(p, 'colloquial')}, а сводку присылай {report(r, 'colloquial')}.",
    lambda t, p, r: f"Настрой watch на {join_tickers(t, 'acc')}: проверка {every(p, 'colloquial')}, сводка {report(r, 'colloquial')}.",
    lambda t, p, r: f"По {join_tickers(t, 'dat')} {every(p, 'colloquial')} смотри цены, {report(r, 'colloquial')} — отчёт.",
    lambda t, p, r: f"{join_tickers(t, 'nom')} — {every(p, 'colloquial')} проверяй, {report(r, 'colloquial')} присылай сводку.",
]


def make_example(user: str, assistant: dict, category: str) -> dict:
    user = user.strip()
    if user:
        user = user[0].upper() + user[1:]
    return {
        "messages": example_messages(user, assistant),
        "meta": {
            "category": category,
            "user": user,
            "expected": assistant,
        },
    }


def generate_valid_category(count: int, category: str, ticker_range: tuple[int, int], style: str) -> list[dict]:
    templates = COLLOQUIAL_TEMPLATES if style == "colloquial" else FORMAL_TEMPLATES
    examples: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count and attempts < count * 200:
        attempts += 1
        n = rng.randint(*ticker_range)
        tickers = pick_tickers(n)
        poll, report_value = pick_periods()
        template = rng.choice(templates)
        user = template(tickers, poll, report_value)
        key = normalize_text(user)
        if key in seen:
            continue
        seen.add(key)
        examples.append(make_example(user, success(tickers, poll, report_value), category))
    if len(examples) < count:
        raise RuntimeError(f"не удалось сгенерировать {category}: {len(examples)}/{count}")
    return examples


def generate_invalid(counts: dict[str, int]) -> list[dict]:
    examples: list[dict] = []
    seen: set[str] = set()

    def add(user: str, code: str) -> None:
        key = normalize_text(user)
        if key in seen:
            return
        seen.add(key)
        examples.append(make_example(user, error(code), "invalid"))

    unknown = ["Ромашка", "Апельсин", "Тесла", "TSLA", "Банан", "Криптовалюта", "Аэрофлот2"]
    bad_intervals = ["30 секунд", "0 минут", "25 часов", "8 дней", "2 недели", "полторы минуты"]

    # missing_tickers (повторяем, пока не наберём нужное число уникальных)
    missing_tickers_attempts = 0
    while (
        sum(1 for row in examples if row["meta"]["expected"].get("error") == "missing_tickers")
        < counts["missing_tickers"]
        and missing_tickers_attempts < 1000
    ):
        missing_tickers_attempts += 1
        poll, report_value = pick_periods()
        add(
            f"{every(poll, 'formal')} смотри цены, {report(report_value, 'formal')} присылай сводку.",
            "missing_tickers",
        )

    # missing_intervals
    for _ in range(counts["missing_intervals"]):
        tickers = pick_tickers(rng.randint(1, 3))
        add(f"Следи за {join_tickers(tickers, 'inst')}.", "missing_intervals")

    # missing_poll_interval
    for _ in range(counts["missing_poll_interval"]):
        tickers = pick_tickers(rng.randint(1, 2))
        report_value = rng.choice(REPORT_VALUES)
        add(
            f"По {join_tickers(tickers, 'dat')}: сводку присылай {report(report_value, 'formal')}.",
            "missing_poll_interval",
        )

    # missing_report_interval
    for _ in range(counts["missing_report_interval"]):
        tickers = pick_tickers(rng.randint(1, 2))
        poll = rng.choice(POLL_VALUES)
        add(
            f"Следи за {join_tickers(tickers, 'inst')} {every(poll, 'formal')}.",
            "missing_report_interval",
        )

    # bad_interval
    for _ in range(counts["bad_interval"]):
        tickers = pick_tickers(rng.randint(1, 2))
        bad = rng.choice(bad_intervals)
        report_value = rng.choice(REPORT_VALUES)
        add(
            f"Следи за {join_tickers(tickers, 'inst')} каждые {bad}, сводку {report(report_value, 'formal')}.",
            "bad_interval",
        )

    # bad_ticker
    for _ in range(counts["bad_ticker"]):
        good = pick_tickers(rng.randint(0, 1))
        bad = rng.choice(unknown)
        names = [rng.choice(TICKER_FORMS[t]["nom"]) for t in good] + [bad]
        rng.shuffle(names)
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " и " + names[-1]
        poll, report_value = pick_periods()
        add(
            f"Добавь в отслеживание: {joined}. Проверка {every(poll, 'formal')}, "
            f"сводка {report(report_value, 'formal')}.",
            "bad_ticker",
        )

    # not_watch
    for text in [
        "Что по погоде в Питере?",
        "Как пройти в библиотеку?",
        "Посоветуй фильм на вечер.",
        "Который сейчас час?",
        "Расскажи анекдот.",
    ][: counts["not_watch"]]:
        add(text, "not_watch")

    # ambiguous
    for text in [
        "Следи за GAZP: опрос 10 минут и 20 минут, сводка раз в час.",
        "Следи за LKOH каждые 5 минут, 15 минут и раз в час.",
        "Следи за ROSN: опрос каждые 30 минут, сводка раз в час и раз в 2 часа.",
        "Следи за SBER: опрос каждые 10 минут, потом 20 минут, сводка раз в час.",
        "Следи за GAZP: сводка раз в час и раз в 3 часа, опрос каждые 15 минут.",
        "Следи за LKOH: 5 минут, 10 минут, сводка раз в час.",
    ][: counts["ambiguous"]]:
        add(text, "ambiguous")

    # empty (ровно один пример: пустая строка в messages не несёт смысла)
    if counts.get("empty"):
        add(" ", "empty")

    return examples


def generate_corrections(count: int) -> list[dict]:
    examples: list[dict] = []
    seen: set[str] = set()

    def add(user: str, assistant: dict) -> None:
        key = normalize_text(user)
        if key in seen:
            return
        seen.add(key)
        examples.append(make_example(user, assistant, "correction"))

    # 10 исправлений периода опроса
    for _ in range(count // 3):
        ticker = pick_tickers(1)[0]
        old_poll, new_poll = rng.sample(POLL_VALUES, 2)
        report_value = rng.choice([r for r in REPORT_VALUES if minutes(r) >= minutes(new_poll)])
        old_text = PERIOD_TEXT[old_poll][0]
        new_text = PERIOD_TEXT[new_poll][0]
        add(
            f"Следи за {join_tickers([ticker], 'inst')} {every(old_poll, 'formal')}, "
            f"сводку {report(report_value, 'formal')}. Нет, не {old_text}, а {new_text}.",
            success([ticker], new_poll, report_value),
        )

    # 10 исправлений периода сводки
    for _ in range(count // 3):
        ticker = pick_tickers(1)[0]
        poll = rng.choice(POLL_VALUES)
        candidates = [r for r in REPORT_VALUES if minutes(r) >= minutes(poll)]
        old_report, new_report = rng.sample(candidates, 2)
        add(
            f"По {join_tickers([ticker], 'dat')}: опрос {every(poll, 'formal')}, "
            f"сводка {report(old_report, 'formal')}. Хотя нет, сводку лучше {report(new_report, 'formal')}.",
            success([ticker], poll, new_report),
        )

    # дубли, нижний регистр, смешанные тикеры
    for _ in range(count - 2 * (count // 3)):
        n = rng.randint(1, 3)
        tickers = pick_tickers(n)
        duplicated = tickers + [rng.choice(tickers)]
        rng.shuffle(duplicated)
        names = []
        for t in duplicated:
            form = rng.choice(TICKER_FORMS[t]["inst"])
            names.append(form.lower() if rng.random() < 0.5 else form)
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " и " + names[-1]
        poll, report_value = pick_periods()
        add(
            f"Следи за {joined} {every(poll, 'formal')}, сводку {report(report_value, 'formal')}.",
            success(tickers, poll, report_value),
        )

    return examples


def split_category(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * 0.10)) if len(shuffled) > 1 else 0
    return shuffled[n_valid:], shuffled[:n_valid]


def main() -> None:
    (DATA / "v1").mkdir(parents=True, exist_ok=True)

    simple = generate_valid_category(90, "simple", (1, 1), "formal")
    multi = generate_valid_category(60, "multi", (2, 4), "formal")
    colloquial = generate_valid_category(60, "colloquial", (1, 3), "colloquial")
    invalid = generate_invalid(
        {
            "missing_tickers": 8,
            "missing_intervals": 8,
            "missing_poll_interval": 8,
            "missing_report_interval": 9,
            "bad_interval": 8,
            "bad_ticker": 7,
            "not_watch": 5,
            "ambiguous": 6,
            "empty": 1,
        }
    )
    corrections = generate_corrections(30)
    corrections = corrections[:30]

    all_examples = simple + multi + colloquial + invalid + corrections
    if len(all_examples) != 300:
        raise RuntimeError(f"ожидали 300 примеров, получили {len(all_examples)}")

    train: list[dict] = []
    valid: list[dict] = []
    for category_examples in (simple, multi, colloquial, invalid, corrections):
        tr, va = split_category(category_examples)
        train.extend(tr)
        valid.extend(va)

    rng.shuffle(train)
    rng.shuffle(valid)

    for path, rows in ((DATA / "v1" / "train.jsonl", train), (DATA / "v1" / "valid.jsonl", valid)):
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "seed": SEED,
        "total": len(all_examples),
        "train": len(train),
        "valid": len(valid),
        "by_category": {
            category: sum(1 for row in all_examples if row["meta"]["category"] == category)
            for category in ("simple", "multi", "colloquial", "invalid", "correction")
        },
        "by_error": {
            code: sum(
                1
                for row in all_examples
                if row["meta"]["expected"].get("error") == code
            )
            for code in sorted({row["meta"]["expected"].get("error") for row in all_examples if not row["meta"]["expected"]["ok"]})
        },
    }
    (DATA / "v1" / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n--- 5 случайных train ---")
    for row in rng.sample(train, 5):
        print(row["meta"]["category"], "|", row["meta"]["user"])
        print("  →", canonical_json(row["meta"]["expected"]))


if __name__ == "__main__":
    main()
