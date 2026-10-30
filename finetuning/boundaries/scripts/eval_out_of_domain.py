"""Прогон задач ВНЕ домена через Ollama-модель — замер «что модель потеряла».

Оценка детерминированная: задача решена, если в ответе присутствуют все
``keywords`` (без учёта регистра). Прогоняйте одну и ту же модель базы и
дообученную и сравнивайте точность по категориям.

Пример:
    python scripts/eval_out_of_domain.py --model qwen2.5:1.5b-instruct-fp16 \
        --out results/ood/base.json
    python scripts/eval_out_of_domain.py --model qwen25-facts-v1 \
        --out results/ood/tuned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ollama_chat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DATA / "out_of_domain.json"))
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct-fp16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _has_all_keywords(answer: str, keywords: list[str]) -> bool:
    lowered = answer.lower()
    return all(key.lower() in lowered for key in keywords)


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    system_prompt = payload["system_prompt"]
    items = payload["items"]
    if args.limit:
        items = items[: args.limit]

    results = []
    started = time.time()
    for index, item in enumerate(items, 1):
        answer = ollama_chat.chat(
            args.model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ],
        )
        results.append(
            {
                "n": index,
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": answer,
                "ok": _has_all_keywords(answer, item["keywords"]),
                "keywords": item["keywords"],
            }
        )
        if index % 10 == 0 or index == len(items):
            print(f"{index}/{len(items)} ({round(time.time() - started)} с)", flush=True)

    def acc(rows: list[dict]) -> float:
        return sum(r["ok"] for r in rows) / len(rows) if rows else 0.0

    overall = acc(results)
    by_category = {
        category: {
            "n": len(subset),
            "accuracy": acc(subset),
        }
        for category in sorted({r["category"] for r in results})
        if (subset := [r for r in results if r["category"] == category])
    }

    out = {
        "config": {"model": args.model, "seconds": round(time.time() - started)},
        "overall": overall,
        "by_category": by_category,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "by_category": by_category}, ensure_ascii=False, indent=2))
    print("результаты:", out_path)


if __name__ == "__main__":
    main()
