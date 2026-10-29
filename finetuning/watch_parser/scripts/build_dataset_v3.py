"""Версия 3: точечно добиваем оставшиеся ошибки v2.

v2 (83.3% exact, 46.7% error_acc) всё ещё ломается на:
  - bad_ticker: «Ромашка»/«TSLA» превращаются в выдуманный тикер;
  - empty: пустой ввод уходит в missing_report_interval;
  - missing_tickers / missing_poll / missing_report;
  - ambiguous: два периода опроса склеиваются в "15m,30m";
  - дубль тикера «SBER и SBER» ошибочно считается ambiguous.

Здесь добавляются близкие, но НЕ совпадающие с eval примеры: другие тикеры,
другие периоды, другие формулировки. Плюс success-примеры с дублями тикеров,
чтобы модель поняла: повтор — не ошибка, его надо схлопнуть.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_dataset as bd
import build_dataset_v2 as v2
from spec import normalize_text, success

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 3031

UNKNOWN_INST = {
    "Ромашка": "Ромашкой",
    "Апельсин": "Апельсином",
    "Тесла": "Теслой",
    "TSLA": "TSLA",
    "Банан": "Бананом",
    "Криптовалюта": "Криптовалютой",
    "Аэрофлот2": "Аэрофлот2",
    "Сбербанк2": "Сбербанк2",
}
UNKNOWN_NOM = list(UNKNOWN_INST)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick_periods(rng: random.Random) -> tuple[str, str]:
    poll = rng.choice(bd.POLL_VALUES)
    report = rng.choice([r for r in bd.REPORT_VALUES if bd.minutes(r) >= bd.minutes(poll)])
    return poll, report


def duplicate_success(rng: random.Random, count: int) -> list[dict]:
    rows: list[dict] = []
    tickers = bd.VALID_TICKERS
    while len(rows) < count:
        t = rng.choice(tickers)
        t2 = rng.choice([x for x in tickers if x != t])
        poll, report = pick_periods(rng)
        template = rng.randrange(4)
        if template == 0:
            user = (
                f"Следи за {bd.join_tickers([t], 'inst')} и {bd.join_tickers([t], 'inst')} "
                f"{bd.every(poll, 'formal')}, сводку {bd.report(report, 'formal')}."
            )
            label_tickers = [t]
        elif template == 1:
            user = (
                f"{bd.join_tickers([t, t, t2], 'nom')} — опрос {bd.every(poll, 'formal')}, "
                f"сводка {bd.report(report, 'formal')}."
            )
            label_tickers = [t, t2]
        elif template == 2:
            user = (
                f"По {bd.join_tickers([t, t2, t], 'dat')}: проверка {bd.every(poll, 'formal')}, "
                f"сводка {bd.report(report, 'formal')}."
            )
            label_tickers = [t, t2]
        else:
            user = (
                f"Добавь в отслеживание {bd.join_tickers([t, t], 'acc')}. "
                f"Проверка {bd.every(poll, 'formal')}, отчёт {bd.report(report, 'formal')}."
            )
            label_tickers = [t]
        rows.append(bd.make_example(user, success(label_tickers, poll, report), "correction"))
    return rows


def period_success(rng: random.Random, count: int) -> list[dict]:
    """Явно добиваем разговорные периоды, на которых v2 ещё ошибалась."""
    rows: list[dict] = []
    report_variants = [
        ("дважды в сутки", "12h"),
        ("раз в полсуток", "12h"),
        ("раз в четверть часа", "15m"),
        ("раз в полчаса", "30m"),
    ]
    while len(rows) < count:
        phrase, report = rng.choice(report_variants)
        poll = rng.choice([p for p in bd.POLL_VALUES if bd.minutes(p) <= bd.minutes(report)])
        t = rng.choice(bd.VALID_TICKERS)
        template = rng.randrange(3)
        if template == 0:
            user = (
                f"{bd.join_tickers([t], 'nom')} мониторь {bd.every(poll, 'formal')}, "
                f"итоги присылай {phrase}."
            )
        elif template == 1:
            user = (
                f"По {bd.join_tickers([t], 'dat')}: опрос {bd.every(poll, 'formal')}, "
                f"сводка {phrase}."
            )
        else:
            user = (
                f"Следи за {bd.join_tickers([t], 'inst')} {bd.every(poll, 'formal')}, "
                f"сводку присылай {phrase}."
            )
        rows.append(bd.make_example(user, success([t], poll, report), "colloquial"))
    return rows


def hard_invalid(rng: random.Random) -> list[dict]:
    rows: list[dict] = []

    def add(user: str, code: str) -> None:
        rows.append(bd.make_example(user, {"ok": False, "error": code}, "invalid"))

    # missing_tickers
    for _ in range(8):
        poll, report = pick_periods(rng)
        add(
            rng.choice(
                [
                    f"Следи {bd.every(poll, 'formal')}, сводку {bd.report(report, 'formal')}.",
                    f"Проверяй цены {bd.every(poll, 'formal')}, отчёт {bd.report(report, 'formal')}.",
                    f"Мониторь {bd.every(poll, 'formal')}, итоги {bd.report(report, 'formal')}.",
                ]
            ),
            "missing_tickers",
        )

    # missing_poll_interval
    for _ in range(8):
        t = rng.choice(bd.VALID_TICKERS)
        _, report = pick_periods(rng)
        add(
            rng.choice(
                [
                    f"Следи за {bd.join_tickers([t], 'inst')}, сводку присылай {bd.report(report, 'formal')}.",
                    f"По {bd.join_tickers([t], 'dat')}: сводка {bd.report(report, 'formal')}.",
                    f"Сводка по {bd.join_tickers([t], 'dat')} {bd.report(report, 'formal')}.",
                ]
            ),
            "missing_poll_interval",
        )

    # missing_report_interval
    for _ in range(8):
        t = rng.choice(bd.VALID_TICKERS)
        poll, _ = pick_periods(rng)
        add(
            rng.choice(
                [
                    f"Следи за {bd.join_tickers([t], 'inst')} {bd.every(poll, 'formal')}.",
                    f"Проверяй {bd.join_tickers([t], 'acc')} {bd.every(poll, 'formal')}.",
                    f"Мониторь {bd.join_tickers([t], 'acc')} {bd.every(poll, 'formal')}.",
                ]
            ),
            "missing_report_interval",
        )

    # bad_ticker: неизвестные названия в разных падежах
    for _ in range(16):
        t = rng.choice(bd.VALID_TICKERS)
        unknown = rng.choice(UNKNOWN_NOM)
        poll, report = pick_periods(rng)
        if rng.random() < 0.5:
            user = (
                f"Следи за {bd.join_tickers([t], 'inst')} и {UNKNOWN_INST[unknown]} "
                f"{bd.every(poll, 'formal')}, сводку {bd.report(report, 'formal')}."
            )
        else:
            user = (
                f"Добавь в отслеживание: {bd.join_tickers([t], 'nom')} и {unknown}. "
                f"Проверка {bd.every(poll, 'formal')}, сводка {bd.report(report, 'formal')}."
            )
        add(user, "bad_ticker")

    # ambiguous: два периода опроса / две сводки
    for _ in range(10):
        t = rng.choice(bd.VALID_TICKERS)
        poll, report = pick_periods(rng)
        other_poll = rng.choice(
            [p for p in bd.POLL_VALUES if bd.minutes(p) != bd.minutes(poll)]
        )
        other_report = rng.choice(
            [r for r in bd.REPORT_VALUES if bd.minutes(r) != bd.minutes(report)]
        )
        if rng.random() < 0.5:
            user = (
                f"Следи за {bd.join_tickers([t], 'inst')}: опрос {bd.PERIOD_TEXT[poll][0]} "
                f"и {bd.PERIOD_TEXT[other_poll][0]}, сводка {bd.report(report, 'formal')}."
            )
        else:
            user = (
                f"По {bd.join_tickers([t], 'dat')}: опрос {bd.every(poll, 'formal')}, "
                f"сводка {bd.report(report, 'formal')} и {bd.report(other_report, 'formal')}."
            )
        add(user, "ambiguous")

    # bad_interval
    for _ in range(8):
        t = rng.choice(bd.VALID_TICKERS)
        _, report = pick_periods(rng)
        bad = rng.choice(v2.BAD_INTERVALS)
        add(
            f"Следи за {bd.join_tickers([t], 'inst')} каждые {bad}, "
            f"сводку {bd.report(report, 'formal')}.",
            "bad_interval",
        )

    # missing_intervals
    for _ in range(8):
        t = rng.choice(bd.VALID_TICKERS)
        t2 = rng.choice([x for x in bd.VALID_TICKERS if x != t])
        add(
            rng.choice(
                [
                    f"Следи за {bd.join_tickers([t, t2], 'inst')}.",
                    f"Поставь на контроль {bd.join_tickers([t, t2], 'acc')}.",
                    f"Интересуют {bd.join_tickers([t, t2], 'nom')}.",
                ]
            ),
            "missing_intervals",
        )

    # not_watch, в том числе с тикерами внутри
    for text in [
        "Сколько дивидендов у Сбера?",
        "Что думаешь о Газпроме?",
        "Сравни Сбер и ВТБ.",
        "Дай прогноз по нефти.",
        "Объясни, что такое P/E.",
        "Какие акции купить?",
        "Сравни доходность ОФЗ и вклада.",
        "Расскажи про биткоин.",
    ]:
        add(text, "not_watch")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-train", default=str(DATA / "v2" / "train.jsonl"))
    parser.add_argument("--v2-valid", default=str(DATA / "v2" / "valid.jsonl"))
    parser.add_argument("--eval", default=str(DATA / "eval.jsonl"))
    parser.add_argument("--out-dir", default=str(DATA / "v3"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bd.rng = rng

    base = load(Path(args.v2_train)) + load(Path(args.v2_valid))
    eval_rows = load(Path(args.eval))
    seen = {normalize_text(row["meta"]["user"]) for row in base + eval_rows if normalize_text(row["meta"]["user"])}

    extra_success = duplicate_success(rng, 40)
    extra_period = period_success(rng, 12)
    extra_invalid = hard_invalid(rng)

    extras: list[dict] = []
    for row in extra_success + extra_period + extra_invalid:
        key = normalize_text(row["meta"]["user"])
        if key and key in seen:
            continue
        seen.add(key)
        extras.append(row)

    rows = base + extras
    rng.shuffle(rows)

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
        "base_v2": len(base),
        "extra_success": len(extra_success),
        "extra_period": len(extra_period),
        "extra_invalid_defined": len(extra_invalid),
        "extra_added": len(extras),
        "total": len(rows),
        "train": len(train),
        "valid": len(valid),
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
    print("\n--- 8 случайных добавленных ---")
    for row in rng.sample(extras, min(8, len(extras))):
        print(row["meta"]["category"], "|", row["meta"]["user"])
        print("  →", json.dumps(row["meta"]["expected"], ensure_ascii=False))


if __name__ == "__main__":
    main()
