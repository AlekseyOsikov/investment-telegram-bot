"""Универсальное сравнение нескольких прогонов на одном eval.

Пример:
    python scripts/compare_runs.py \
        --base results/baseline/base_prompt.json \
        --run v1=results/v1/tuned_prompt.json \
        --run v2=results/v2/tuned_prompt.json \
        --run "v2+guard=results/v2/tuned_prompt_guard.json" \
        --run v3=results/v3/tuned_prompt.json \
        --run "v3+guard=results/v3/tuned_prompt_guard.json" \
        --out results/compare/compare_all.md
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
    parser.add_argument("--base", required=True)
    parser.add_argument("--run", action="append", required=True, help="label=path")
    parser.add_argument("--out", default="results/compare/compare_all.md")
    args = parser.parse_args()

    runs = {"база": load(args.base)}
    for item in args.run:
        label, path = item.split("=", 1)
        runs[label] = load(path)

    metrics = ["exact", "schema_ok", "ticker_f1", "poll_acc", "report_acc", "error_acc"]
    lines = ["# Сравнение прогонов на одном eval (60 запросов)", "", "| метрика | " + " | ".join(runs) + " |"]
    lines.append("|---|" + "---|" * len(runs))
    for metric in metrics:
        lines.append("| " + metric + " | " + " | ".join(pct(run["overall"].get(metric)) for run in runs.values()) + " |")

    categories = sorted({row["category"] for run in runs.values() for row in run["results"]})
    lines += ["", "## Exact match по категориям", "", "| категория | " + " | ".join(runs) + " |"]
    lines.append("|---|" + "---|" * len(runs))
    for category in categories:
        lines.append(
            "| " + category + " | "
            + " | ".join(pct(run["by_category"].get(category, {}).get("exact")) for run in runs.values())
            + " |"
        )

    codes = sorted(
        {
            row["expected"]["error"]
            for run in runs.values()
            for row in run["results"]
            if not row["expected"].get("ok")
        }
    )
    lines += ["", "## Точность по кодам ошибок", "", "| код | всего | " + " | ".join(runs) + " |"]
    lines.append("|---|---:|" + "---|" * len(runs))
    for code in codes:
        cells = []
        total = 0
        for run in runs.values():
            rows = [row for row in run["results"] if row["expected"].get("error") == code]
            total = len(rows)
            cells.append(f"{sum(1 for row in rows if row.get('error_acc'))}/{len(rows)}")
        lines.append(f"| {code} | {total} | " + " | ".join(cells) + " |")

    labels = list(runs)
    if len(labels) >= 2:
        left, right = runs[labels[-2]], runs[labels[-1]]
        left_rows = {row["n"]: row for row in left["results"]}
        right_rows = {row["n"]: row for row in right["results"]}
        better = [n for n in left_rows if not left_rows[n]["exact"] and right_rows[n]["exact"]]
        worse = [n for n in left_rows if left_rows[n]["exact"] and not right_rows[n]["exact"]]
        lines += [
            "",
            f"## {labels[-2]} → {labels[-1]}",
            "",
            f"- стало лучше по exact: **{len(better)}**",
            f"- стало хуже по exact: **{len(worse)}**",
            f"- верны у обеих: {sum(1 for n in left_rows if left_rows[n]['exact'] and right_rows[n]['exact'])}",
            f"- неверны у обеих: {sum(1 for n in left_rows if not left_rows[n]['exact'] and not right_rows[n]['exact'])}",
        ]
        for title, numbers in (("Где лучше", better), ("Где хуже", worse)):
            lines += ["", f"### {title}", ""]
            if not numbers:
                lines.append("нет")
                continue
            for n in numbers[:12]:
                row = right_rows[n]
                lines += [
                    f"- **#{n} [{row['category']}]** {row['user']}",
                    f"  - эталон: `{json.dumps(row['expected'], ensure_ascii=False)}`",
                    f"  - {labels[-2]}: `{left_rows[n]['output'][:220]}`",
                    f"  - {labels[-1]}: `{row['output'][:220]}`",
                ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:35]))
    print("отчёт:", out)


if __name__ == "__main__":
    main()
