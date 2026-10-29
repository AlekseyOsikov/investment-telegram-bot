"""Печать сравнения двух прогонов на одном eval в консоль.

Примеры:
    python scripts/compare_console.py
    python scripts/compare_console.py --tuned results/v3/tuned_prompt_guard.json
    python scripts/compare_console.py --category invalid --only-diff
    python scripts/compare_console.py --limit 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import semantic_equal

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def short(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(RESULTS / "baseline" / "base_prompt.json"))
    parser.add_argument("--tuned", default=str(RESULTS / "v3" / "tuned_prompt.json"))
    parser.add_argument("--limit", type=int, default=0, help="первые N запросов; 0 — все")
    parser.add_argument("--only", help="номера запросов через запятую, например 1,13,17")
    parser.add_argument("--category", default=None, help="simple/multi/invalid/...")
    parser.add_argument("--only-diff", action="store_true", help="только различающиеся ответы")
    parser.add_argument("--max-chars", type=int, default=0, help="обрезать ответ; 0 — полностью")
    args = parser.parse_args()

    base = load(args.base)
    tuned = load(args.tuned)
    base_rows = {row["n"]: row for row in base["results"]}
    tuned_rows = {row["n"]: row for row in tuned["results"]}

    numbers = sorted(set(base_rows) & set(tuned_rows))
    if args.only:
        wanted = {int(n) for n in args.only.split(",") if n.strip()}
        numbers = [n for n in numbers if n in wanted]
    if args.category:
        numbers = [n for n in numbers if base_rows[n]["category"] == args.category]
    if args.limit:
        numbers = numbers[: args.limit]
    if args.only_diff:
        numbers = [n for n in numbers if base_rows[n]["output"] != tuned_rows[n]["output"]]

    print(f"base : {args.base}")
    print(f"tuned: {args.tuned}")
    print(f"запросов: {len(numbers)}\n")

    counts = {
        "better_semantic": 0,
        "better_format": 0,
        "worse_semantic": 0,
        "worse_format": 0,
        "same_good": 0,
        "same_semantic_bad_format": 0,
        "same_bad": 0,
    }

    for n in numbers:
        base_row = base_rows[n]
        tuned_row = tuned_rows[n]
        expected = base_row["expected"]
        base_semantic = semantic_equal(expected, base_row["output"])
        tuned_semantic = semantic_equal(expected, tuned_row["output"])

        print(f"#{n:02d} [{base_row['category']}] {base_row['user']}")
        print(f"  эталон : {json.dumps(expected, ensure_ascii=False)}")
        print(f"  база   : {short(base_row['output'], args.max_chars)}")
        print(f"  тюн    : {short(tuned_row['output'], args.max_chars)}")

        if base_row["exact"] and tuned_row["exact"]:
            verdict = "так же"
            counts["same_good"] += 1
        elif tuned_row["exact"] and not base_row["exact"]:
            if base_semantic:
                verdict = "лучше (только формат)"
                counts["better_format"] += 1
            else:
                verdict = "лучше (смысл)"
                counts["better_semantic"] += 1
        elif base_row["exact"] and not tuned_row["exact"]:
            if tuned_semantic:
                verdict = "хуже (только формат)"
                counts["worse_format"] += 1
            else:
                verdict = "хуже (смысл)"
                counts["worse_semantic"] += 1
        elif base_semantic and tuned_semantic:
            verdict = "оба верны по смыслу, формат разный"
            counts["same_semantic_bad_format"] += 1
        elif tuned_semantic:
            verdict = "лучше (смысл)"
            counts["better_semantic"] += 1
        elif base_semantic:
            verdict = "хуже (смысл)"
            counts["worse_semantic"] += 1
        else:
            verdict = "оба неверны"
            counts["same_bad"] += 1
        print(f"  вердикт: {verdict}\n")

    total = len(numbers)
    print("--- итог ---")
    print(f"всего: {total}")
    print(f"база exact: {sum(1 for n in numbers if base_rows[n]['exact'])}/{total}")
    print(f"тюн  exact: {sum(1 for n in numbers if tuned_rows[n]['exact'])}/{total}")
    print(
        "лучше по смыслу: {better_semantic} | лучше только формат: {better_format}".format(**counts)
    )
    print(
        "хуже по смыслу: {worse_semantic} | хуже только формат: {worse_format}".format(**counts)
    )
    print(
        "так же верно: {same_good} | оба верны, формат разный: {same_semantic_bad_format} | "
        "оба неверны: {same_bad}".format(**counts)
    )


if __name__ == "__main__":
    main()
