"""Применяет guard к уже сохранённым ответам модели и пересчитывает метрики.

Пример:
    python scripts/apply_guard_results.py \
        --input results/v2/tuned_prompt.json \
        --output results/v2/tuned_prompt_guard.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from guard import apply_guard
from metrics import extract_json, score_row
from spec import canonical_json


def aggregate(results: list[dict]) -> tuple[dict, dict]:
    overall = {"count": len(results)}
    for key in ("exact", "schema_ok", "json_strict", "json_extractable"):
        overall[key] = sum(row[key] for row in results) / len(results)
    for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
        values = [row[key] for row in results if key in row]
        overall[key] = sum(values) / len(values) if values else None

    by_category = {}
    for category in sorted({row["category"] for row in results}):
        subset = [row for row in results if row["category"] == category]
        block = {"count": len(subset)}
        for key in ("exact", "schema_ok", "json_strict"):
            block[key] = sum(row[key] for row in subset) / len(subset)
        for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
            values = [row[key] for row in subset if key in row]
            block[key] = sum(values) / len(values) if values else None
        by_category[category] = block
    return overall, by_category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = []
    for row in payload["results"]:
        parsed = extract_json(row["output"])
        guarded = apply_guard(row["user"], parsed) if parsed is not None else None
        raw = canonical_json(guarded) if guarded is not None and guarded != parsed else row["output"]
        metrics = score_row(row["expected"], raw)
        results.append({**row, **metrics, "output": raw})

    overall, by_category = aggregate(results)
    payload.update({"overall": overall, "by_category": by_category, "results": results})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["n", "category", "user", "expected", "output", "exact", "schema_ok", "json_strict"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **{key: row[key] for key in ("n", "category", "user", "output", "exact", "schema_ok", "json_strict")},
                    "expected": json.dumps(row["expected"], ensure_ascii=False),
                }
            )

    print(json.dumps({"overall": overall, "by_category": by_category}, ensure_ascii=False, indent=2))
    print("результаты:", out)


if __name__ == "__main__":
    main()
