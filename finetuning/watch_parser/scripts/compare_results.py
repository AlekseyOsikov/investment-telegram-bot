"""Сводная таблица: база без промпта / база с промптом / дообученная модель.

Запуск:
    python scripts/compare_results.py \
        --base-none results/baseline/base_none.json \
        --base-prompt results/baseline/base_prompt.json \
        --tuned results/v1/tuned_prompt.json \
        --out results/v1/compare.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-none", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--out", default="results/v1/compare.md")
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def main() -> None:
    args = parse_args()
    runs = {
        "база без промпта": load(args.base_none),
        "база с промптом": load(args.base_prompt),
        "дообученная": load(args.tuned),
    }

    metrics = ["exact", "schema_ok", "json_strict", "json_extractable", "ticker_f1", "poll_acc", "report_acc", "error_acc"]
    lines = ["# Сравнение на eval-наборе (60 запросов)", "", "| метрика | " + " | ".join(runs) + " |"]
    lines.append("|---|" + "---|" * len(runs))
    for metric in metrics:
        lines.append(
            f"| {metric} | "
            + " | ".join(fmt(run["overall"].get(metric)) for run in runs.values())
            + " |"
        )

    categories = sorted({row["category"] for run in runs.values() for row in run["results"]})
    lines += ["", "## Exact match по категориям", "", "| категория | " + " | ".join(runs) + " |"]
    lines.append("|---|" + "---|" * len(runs))
    for category in categories:
        cells = []
        for run in runs.values():
            block = run["by_category"].get(category, {})
            cells.append(fmt(block.get("exact")))
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    none_rows = {row["n"]: row for row in runs["база без промпта"]["results"]}
    base_rows = {row["n"]: row for row in runs["база с промптом"]["results"]}
    tuned_rows = {row["n"]: row for row in runs["дообученная"]["results"]}
    worse = [
        (base_rows[n], tuned_rows[n])
        for n in base_rows
        if base_rows[n]["exact"] and not tuned_rows[n]["exact"]
    ]
    better = [
        (base_rows[n], tuned_rows[n])
        for n in base_rows
        if not base_rows[n]["exact"] and tuned_rows[n]["exact"]
    ]

    lines += [
        "",
        "## Где стало лучше",
        "",
        f"Всего: {len(better)}",
        "",
    ]
    for base, tuned in better[:10]:
        lines += [
            f"- **#{base['n']} [{base['category']}]** {base['user']}",
            f"  - эталон: `{json.dumps(base['expected'], ensure_ascii=False)}`",
            f"  - база: `{base['output'][:200]}`",
            f"  - тюн: `{tuned['output'][:200]}`",
        ]

    lines += [
        "",
        "## Где стало хуже",
        "",
        f"Всего: {len(worse)}",
        "",
    ]
    for base, tuned in worse[:10]:
        lines += [
            f"- **#{base['n']} [{base['category']}]** {base['user']}",
            f"  - эталон: `{json.dumps(base['expected'], ensure_ascii=False)}`",
            f"  - база: `{base['output'][:200]}`",
            f"  - тюн: `{tuned['output'][:200]}`",
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n", "category", "user", "expected", "base_none", "base_prompt", "tuned", "base_exact", "tuned_exact"])
        for n in sorted(base_rows):
            writer.writerow(
                [
                    n,
                    base_rows[n]["category"],
                    base_rows[n]["user"],
                    json.dumps(base_rows[n]["expected"], ensure_ascii=False),
                    none_rows[n]["output"],
                    base_rows[n]["output"],
                    tuned_rows[n]["output"],
                    base_rows[n]["exact"],
                    tuned_rows[n]["exact"],
                ]
            )

    print("\n".join(lines[:30]))
    print("отчёт:", out)
    print("таблица:", csv_path)


if __name__ == "__main__":
    main()
