"""Прогон «стиль и тон» через Ollama: три плеча, как в эксперименте «формат».

Плечи (``--arm``):

- ``base_none``  — база, нейтральный промпт (стиля нет);
- ``base_prompt`` — база + ``style_spec`` (хороший промпт задаёт стиль);
- ``tuned``      — дообученная roleplay-модель + ``style_spec``.

Метрики на каждый промпт:

- ``coverage`` — judge-покрытие ``criteria`` (слепая оценка);
- ``short`` — автоматическая проверка краткости (≤ 2 предложения и ≤ 45 слов).

Пример:
    python scripts/eval_style.py --arm base_none --model qwen2.5:1.5b-instruct-fp16 \
        --out results/style/base_none.json
    python scripts/eval_style.py --arm base_prompt --model qwen2.5:1.5b-instruct-fp16 \
        --out results/style/base_prompt.json
    python scripts/eval_style.py --arm tuned --model qwen25-roleplay \
        --out results/style/tuned.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import judge
import ollama_chat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ARMS = ("base_none", "base_prompt", "tuned")

_SENT_END = re.compile(r"[.!?…]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DATA / "style_eval.json"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct-fp16")
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _is_short(answer: str) -> bool:
    words = len(answer.split())
    sentences = len(_SENT_END.findall(answer))
    return sentences <= 2 and words <= 45


def build_messages(item: dict, arm: str, style_spec: str, neutral: str) -> list[dict]:
    if arm == "base_none":
        system = neutral
    else:
        system = style_spec
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": item["prompt"]},
    ]


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    style_spec = payload["style_spec"]
    neutral = payload["neutral_prompt"]
    items = payload["items"]
    if args.limit:
        items = items[: args.limit]

    results = []
    started = time.time()
    for index, item in enumerate(items, 1):
        answer = ollama_chat.chat(args.model, build_messages(item, args.arm, style_spec, neutral))
        row = {
            "n": index,
            "id": item["id"],
            "kind": item["kind"],
            "prompt": item["prompt"],
            "answer": answer,
            "short": _is_short(answer),
            "coverage": None,
            "judge_note": None,
        }
        if not args.no_judge:
            verdict = judge.judge_coverage(
                item["prompt"],
                item["criteria"],
                answer,
                provider=args.judge_provider,
                model=args.judge_model,
            )
            if verdict is not None:
                row["coverage"] = verdict["coverage"]
                row["judge_note"] = verdict["note"]
        results.append(row)
        print(
            f"{index}/{len(items)} short={int(row['short'])} cov={row['coverage']} ({round(time.time() - started)} с)",
            flush=True,
        )

    def mean(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    def group(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "short": mean([r["short"] for r in rows]),
            "coverage": mean([r["coverage"] for r in rows]),
        }

    overall = group(results)
    by_kind = {k: group([r for r in results if r["kind"] == k]) for k in ("in_role", "boundary")}

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
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "by_kind": by_kind}, ensure_ascii=False, indent=2))
    print("результаты:", out_path)


if __name__ == "__main__":
    main()
