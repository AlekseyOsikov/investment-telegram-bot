"""Сравнение v1 и v2: общие метрики, категории и точность по кодам ошибок.

Запуск:
    python scripts/compare_v2.py \
        --base-prompt results/baseline/base_prompt.json \
        --tuned-v1 results/v1/tuned_prompt.json \
        --tuned-v2 results/v2/tuned_prompt.json \
        --out results/v2/compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--tuned-v1", required=True)
    parser.add_argument("--tuned-v2", required=True)
    parser.add_argument("--out", default="results/v2/compare.md")
    args = parser.parse_args()

    runs = {
        "база с промптом": load(args.base_prompt),
        "v1": load(args.tuned_v1),
        "v2": load(args.tuned_v2),
    }
    metrics = ["exact", "schema_ok", "ticker_f1", "poll_acc", "report_acc", "error_acc"]

    lines = ["# Сравнение v1 и v2 на одном eval (60 запросов)", "", "| метрика | " + " | ".join(runs) + " |"]
    lines.append("|---|" + "---|" * len(runs))
    for metric in metrics:
        lines.append("| " + metric + " | " + " | ".join(pct(run["overall"].get(metric)) for run in runs.values()) + " |")

    categories = sorted({row["category"] for run in runs.values() for row in run["results"]})
    lines += ["", "## Exact match по категориям", "", "| категория | " + " | ".join(runs) + " |"]
    lines.append("|---|" + "---|" * len(runs))
    for category in categories:
        cells = [pct(run["by_category"].get(category, {}).get("exact")) for run in runs.values()]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    # Точность по каждому ожидаемому коду ошибки.
    codes = sorted(
        {
            row["expected"]["error"]
            for run in runs.values()
            for row in run["results"]
            if not row["expected"].get("ok")
        }
    )
    lines += ["", "## Точность по кодам ошибок (error_acc)", "", "| код | всего | " + " | ".join(runs) + " |"]
    lines.append("|---|---:|" + "---|" * len(runs))
    for code in codes:
        cells = []
        total = 0
        for run in runs.values():
            rows = [row for row in run["results"] if row["expected"].get("error") == code]
            total = len(rows)
            correct = sum(1 for row in rows if row.get("error_acc"))
            cells.append(f"{correct}/{len(rows)}")
        lines.append(f"| {code} | {total} | " + " | ".join(cells) + " |")

    v1 = {row["n"]: row for row in runs["v1"]["results"]}
    v2 = {row["n"]: row for row in runs["v2"]["results"]}
    better = [n for n in v1 if not v1[n]["exact"] and v2[n]["exact"]]
    worse = [n for n in v1 if v1[n]["exact"] and not v2[n]["exact"]]
    lines += [
        "",
        "## v1 → v2",
        "",
        f"- стало лучше по exact: **{len(better)}**",
        f"- стало хуже по exact: **{len(worse)}**",
        f"- верны у обеих: {sum(1 for n in v1 if v1[n]['exact'] and v2[n]['exact'])}",
        f"- неверны у обеих: {sum(1 for n in v1 if not v1[n]['exact'] and not v2[n]['exact'])}",
    ]

    for title, numbers in (("Где v2 лучше", better), ("Где v2 хуже", worse)):
        lines += ["", f"### {title}", ""]
        if not numbers:
            lines.append("нет")
            continue
        for n in numbers[:12]:
            row = v2[n]
            lines += [
                f"- **#{n} [{row['category']}]** {row['user']}",
                f"  - эталон: `{json.dumps(row['expected'], ensure_ascii=False)}`",
                f"  - v1: `{v1[n]['output'][:220]}`",
                f"  - v2: `{row['output'][:220]}`",
            ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:35]))
    print("отчёт:", out)


if __name__ == "__main__":
    main()
