"""Прогон eval/holdout через локальную Ollama-модель.

Пример:
    python scripts/ollama_eval.py --eval data/eval.jsonl \
        --model qwen25-watch-v3 --out results/v3/ollama.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from metrics import score_row
from spec import FEWSHOT, SYSTEM_PROMPT, canonical_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DATA / "eval.jsonl"))
    parser.add_argument("--model", default="qwen25-watch-v3")
    parser.add_argument("--prompt-mode", choices=["none", "system", "fewshot"], default="fewshot")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_messages(user: str, prompt_mode: str) -> list[dict]:
    if prompt_mode == "none":
        return [{"role": "user", "content": user}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prompt_mode == "fewshot":
        for example_user, example_answer in FEWSHOT:
            messages.append({"role": "user", "content": example_user})
            messages.append({"role": "assistant", "content": canonical_json(example_answer)})
    messages.append({"role": "user", "content": user})
    return messages


def chat(model: str, messages: list[dict], max_new_tokens: int, num_ctx: int) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
            "num_ctx": num_ctx,
            "seed": 42,
            # Ollama по умолчанию включает repeat_penalty=1.1; для greedy-JSON
            # это меняет ответ. Повторяем условия HF: без штрафа, top_k=1.
            "repeat_penalty": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    return payload["message"]["content"].strip()


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
    args = parse_args()
    rows = load_rows(args.eval)
    if args.limit:
        rows = rows[: args.limit]

    results = []
    started = time.time()
    for index, row in enumerate(rows, 1):
        user = row["meta"]["user"]
        output = chat(args.model, build_messages(user, args.prompt_mode), args.max_new_tokens, args.num_ctx)
        metrics = score_row(row["meta"]["expected"], output)
        results.append(
            {
                "n": index,
                "category": row["meta"]["category"],
                "user": user,
                "expected": row["meta"]["expected"],
                "output": output,
                **metrics,
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} exact={metrics['exact']} schema={metrics['schema_ok']} "
                  f"({round(time.time() - started)} с)", flush=True)

    overall, by_category = aggregate(results)
    payload = {
        "config": {
            "model": args.model,
            "eval": args.eval,
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": args.max_new_tokens,
            "num_ctx": args.num_ctx,
            "seconds": round(time.time() - started),
        },
        "overall": overall,
        "by_category": by_category,
        "results": results,
    }
    out = Path(args.out)
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
