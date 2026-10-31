"""Прогон трёх вариантов (база / учитель / дистиллированный) на тестовом наборе.

Один и тот же промпт (``SYSTEM_PROMPT`` + 2 few-shot) и одни и те же параметры
жадной генерации для всех моделей — чтобы сравнивать только модель, а не условия.
Для каждого ответа фиксируются качество (``metrics.score_row``), латентность и
токены (вход/выход) — они нужны для оценки скорости и стоимости.

Пишет ``results/<split>/<variant>.json``. Дистиллированная модель появится после
Colab-обучения; до этого запускайте только базу и учителя.

Запуск:
    python scripts/eval_three.py --variants base,teacher
    python scripts/eval_three.py --variants base,teacher,distilled
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_WATCH_SCRIPTS = Path(__file__).resolve().parents[2] / "watch_parser" / "scripts"
if str(_WATCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_WATCH_SCRIPTS))

from metrics import score_row  # noqa: E402  isort: skip
from ollama_client import chat  # noqa: E402  isort: skip
from spec import FEWSHOT, SYSTEM_PROMPT, canonical_json  # noqa: E402  isort: skip

ROOT = Path(__file__).resolve().parents[1]
WATCH_DATA = Path(__file__).resolve().parents[2] / "watch_parser" / "data"
RESULTS = ROOT / "results"

DEFAULT_MODELS = {
    "base": "qwen2.5:1.5b-instruct-fp16",
    "teacher": "qwen2.5:7b",
    "distilled": "qwen25-distill-v1",
    "golden": "qwen25-golden-v1",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_messages(user: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_user, example_answer in FEWSHOT:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": canonical_json(example_answer)})
    messages.append({"role": "user", "content": user})
    return messages


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    quality = {"count": len(rows)}
    for key in ("exact", "schema_ok", "json_strict", "json_extractable"):
        quality[key] = sum(row[key] for row in rows) / len(rows)
    for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
        values = [row[key] for row in rows if key in row]
        quality[key] = sum(values) / len(values) if values else None

    by_category: dict[str, dict] = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        block = {"count": len(subset), "exact": sum(row["exact"] for row in subset) / len(subset)}
        for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
            values = [row[key] for row in subset if key in row]
            block[key] = sum(values) / len(values) if values else None
        by_category[category] = block

    latencies = [row["latency_s"] for row in rows]
    tps = [row["tokens_per_sec"] for row in rows if row["tokens_per_sec"] is not None]
    speed = {
        "requests": len(rows),
        "total_seconds": round(sum(latencies), 2),
        "mean_latency_s": round(mean(latencies), 3) if latencies else None,
        "median_latency_s": round(statistics.median(latencies), 3) if latencies else None,
        "mean_tokens_per_sec": round(mean(tps), 2) if tps else None,
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "total_output_tokens": sum(row["output_tokens"] for row in rows),
        "mean_prompt_tokens": round(mean([row["prompt_tokens"] for row in rows]), 1),
        "mean_output_tokens": round(mean([row["output_tokens"] for row in rows]), 1),
    }
    return {"quality": quality, "by_category": by_category, "speed": speed}


def run_split(variant: str, model: str, split: str, rows: list[dict], num_predict: int, num_ctx: int) -> dict:
    results = []
    started = time.time()
    for index, row in enumerate(rows, 1):
        user = row["meta"]["user"]
        expected = row["meta"]["expected"]
        reply = chat(model, build_messages(user), num_predict=num_predict, num_ctx=num_ctx)
        raw = reply["content"]
        m = score_row(expected, raw)
        eval_duration_s = (reply.get("eval_duration") or 0) / 1e9
        results.append(
            {
                "n": index,
                "category": row["meta"]["category"],
                "user": user,
                "expected": expected,
                "output": raw,
                "exact": m["exact"],
                "schema_ok": m["schema_ok"],
                "json_strict": m["json_strict"],
                "json_extractable": m["json_extractable"],
                **{key: m[key] for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc") if key in m},
                "latency_s": round((reply.get("total_duration") or 0) / 1e9, 3),
                "tokens_per_sec": round(reply.get("eval_count") / eval_duration_s, 2) if eval_duration_s > 0 else None,
                "prompt_tokens": reply.get("prompt_eval_count") or 0,
                "output_tokens": reply.get("eval_count") or 0,
            }
        )
        if index % 20 == 0 or index == len(rows):
            print(f"  {variant}/{split} {index}/{len(rows)} ({round(time.time() - started)} с)", flush=True)

    agg = aggregate(results)
    return {
        "variant": variant,
        "split": split,
        "model": model,
        "prompt_mode": "fewshot",
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "quality": agg["quality"],
        "by_category": agg["by_category"],
        "speed": agg["speed"],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="base,teacher")
    parser.add_argument("--splits", default="eval,holdout")
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for variant in variants:
        model = DEFAULT_MODELS.get(variant)
        if not model:
            raise SystemExit(f"неизвестный вариант {variant!r}; знаю {list(DEFAULT_MODELS)}")
        for split in splits:
            rows = load_jsonl(WATCH_DATA / f"{split}.jsonl")
            if args.limit is not None:
                rows = rows[: args.limit]
            print(f"== {variant} ({model}) на {split} ({len(rows)}) ==")
            payload = run_split(variant, model, split, rows, args.num_predict, args.num_ctx)
            out = RESULTS / split / f"{variant}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload["quality"], ensure_ascii=False))
            print(json.dumps(payload["speed"], ensure_ascii=False))
            print("результаты:", out)


if __name__ == "__main__":
    main()
