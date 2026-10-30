"""Прогон фактов через Ollama-модель: три «плеча» эксперимента.

Плечи (``--arm``):

- ``base_none``  — база без фактов (показывает, что модель фактов не знает);
- ``base_facts`` — база, факт вставлен в системный промпт (baseline «факты
  в промпте» — то, что на проде делает RAG);
- ``tuned``      — дообученная facts-модель без фактов в промпте.

Метрики на каждый вопрос:

- ``keys_ok`` — детерминированная проверка: все ``keys`` присутствуют в ответе;
- ``coverage`` — judge-покрытие ``judge_facts`` (слепая оценка).

Пример:
    python scripts/eval_facts.py --arm base_none --model qwen2.5:1.5b-instruct-fp16 \
        --out results/facts/base_none.json
    python scripts/eval_facts.py --arm base_facts --model qwen2.5:1.5b-instruct-fp16 \
        --out results/facts/base_facts.json
    python scripts/eval_facts.py --arm tuned --model qwen25-facts-v1 \
        --out results/facts/tuned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import judge
import ollama_chat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ARMS = ("base_none", "base_facts", "tuned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DATA / "eval_facts.json"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct-fp16")
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--no-judge", action="store_true", help="только keys_ok, без LLM-судьи")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _has_all_keys(answer: str, keys: list[str]) -> bool:
    lowered = answer.lower()
    return all(key.lower() in lowered for key in keys)


def build_messages(item: dict, arm: str, system_prompt: str) -> list[dict]:
    if arm == "base_facts":
        system = system_prompt + "\n\nСправочный факт:\n" + item["fact"]
    else:
        system = system_prompt
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": item["question"]},
    ]


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
        answer = ollama_chat.chat(args.model, build_messages(item, args.arm, system_prompt))
        row = {
            "n": index,
            "id": item["id"],
            "kind": item["kind"],
            "mode": item["mode"],
            "question": item["question"],
            "expected": item["answer"],
            "answer": answer,
            "keys_ok": _has_all_keys(answer, item["keys"]),
            "coverage": None,
            "judge_note": None,
        }
        if not args.no_judge:
            verdict = judge.judge_coverage(
                item["question"],
                item["judge_facts"],
                answer,
                provider=args.judge_provider,
                model=args.judge_model,
            )
            if verdict is not None:
                row["coverage"] = verdict["coverage"]
                row["judge_note"] = verdict["note"]
        results.append(row)
        print(
            f"{index}/{len(items)} keys={int(row['keys_ok'])} cov={row['coverage']} ({round(time.time() - started)} с)",
            flush=True,
        )

    def mean(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    def group(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "keys_ok": mean([r["keys_ok"] for r in rows]),
            "coverage": mean([r["coverage"] for r in rows]),
        }

    overall = group(results)
    by_kind = {k: group([r for r in results if r["kind"] == k]) for k in ("synthetic", "real")}
    by_mode = {m: group([r for r in results if r["mode"] == m]) for m in ("train_like", "paraphrase")}

    out = {
        "config": {
            "arm": args.arm,
            "model": args.model,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "seconds": round(time.time() - started),
        },
        "overall": overall,
        "by_kind": by_kind,
        "by_mode": by_mode,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"overall": overall, "by_kind": by_kind, "by_mode": by_mode},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("результаты:", out_path)


if __name__ == "__main__":
    main()
