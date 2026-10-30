"""Прогон eval через HF-адаптер в Colab — без слияния и GGUF.

Нужен torch/transformers/peft (они есть в Colab). Пишет тот же JSON, что и
Ollama-скрипты, поэтому `compare_three.py` и отчёт его понимают.

Примеры (в Colab, из каталога проекта):
    # дообученная facts-модель, без фактов в промпте
    python scripts/eval_hf.py --task facts --adapter /content/adapter_facts \
        --out results/facts/tuned.json

    # дообученная facts-модель на задачах вне домена
    python scripts/eval_hf.py --task ood --adapter /content/adapter_facts \
        --out results/ood/tuned.json

    # (для сверки) база + факт в промпте
    python scripts/eval_hf.py --task facts --prompt-facts \
        --out results/facts/base_facts_hf.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import judge
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["facts", "ood"], required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt-facts", action="store_true", help="вставить факт в системный промпт")
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def build_messages_facts(item: dict, system_prompt: str, prompt_facts: bool) -> list[dict]:
    system = system_prompt
    if prompt_facts:
        system = system_prompt + "\n\nСправочный факт:\n" + item["fact"]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": item["question"]},
    ]


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new_tokens: int) -> str:
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(
        model.device
    )
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def _has_all(answer: str, keys: list[str]) -> bool:
    lowered = answer.lower()
    return all(key.lower() in lowered for key in keys)


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def run_facts(model, tok, args) -> dict:
    payload = json.loads((DATA / "eval_facts.json").read_text(encoding="utf-8"))
    system_prompt = payload["system_prompt"]
    items = payload["items"]
    if args.limit:
        items = items[: args.limit]

    results = []
    started = time.time()
    for index, item in enumerate(items, 1):
        answer = generate(model, tok, build_messages_facts(item, system_prompt, args.prompt_facts), args.max_new_tokens)
        row = {
            "n": index,
            "id": item["id"],
            "kind": item["kind"],
            "mode": item["mode"],
            "question": item["question"],
            "expected": item["answer"],
            "answer": answer,
            "keys_ok": _has_all(answer, item["keys"]),
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

    def group(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "keys_ok": mean([r["keys_ok"] for r in rows]),
            "coverage": mean([r["coverage"] for r in rows]),
        }

    return {
        "config": {
            "task": "facts",
            "adapter": args.adapter,
            "prompt_facts": args.prompt_facts,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "seconds": round(time.time() - started),
        },
        "overall": group(results),
        "by_kind": {k: group([r for r in results if r["kind"] == k]) for k in ("synthetic", "real")},
        "by_mode": {m: group([r for r in results if r["mode"] == m]) for m in ("train_like", "paraphrase")},
        "results": results,
    }


def run_ood(model, tok, args) -> dict:
    payload = json.loads((DATA / "out_of_domain.json").read_text(encoding="utf-8"))
    system_prompt = payload["system_prompt"]
    items = payload["items"]
    if args.limit:
        items = items[: args.limit]

    results = []
    started = time.time()
    for index, item in enumerate(items, 1):
        answer = generate(
            model,
            tok,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": item["prompt"]}],
            args.max_new_tokens,
        )
        results.append(
            {
                "n": index,
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": answer,
                "ok": _has_all(answer, item["keywords"]),
                "keywords": item["keywords"],
            }
        )
        if index % 10 == 0 or index == len(items):
            print(f"{index}/{len(items)} ({round(time.time() - started)} с)", flush=True)

    def acc(rows: list[dict]) -> float:
        return sum(r["ok"] for r in rows) / len(rows) if rows else 0.0

    return {
        "config": {"task": "ood", "adapter": args.adapter, "seconds": round(time.time() - started)},
        "overall": acc(results),
        "by_category": {
            category: {"n": len(subset), "accuracy": acc(subset)}
            for category in sorted({r["category"] for r in results})
            if (subset := [r for r in results if r["category"] == category])
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("GPU не найден — считаю на CPU (медленнее, но работает).")

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base_model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, device_map=device_map)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True

    out = run_facts(model, tok, args) if args.task == "facts" else run_ood(model, tok, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = out["overall"]
    if args.task == "facts":
        summary = {"overall": out["overall"], "by_kind": out["by_kind"], "by_mode": out["by_mode"]}
    else:
        summary = {"overall": out["overall"], "by_category": out["by_category"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("результаты:", out_path)


if __name__ == "__main__":
    main()
