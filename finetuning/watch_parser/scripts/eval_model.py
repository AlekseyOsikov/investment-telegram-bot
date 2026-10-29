"""Прогон eval-набора через базовую или дообученную модель.

Примеры:
    python scripts/eval_model.py --prompt-mode none  --out results/baseline/base_none.json
    python scripts/eval_model.py --prompt-mode system --out results/baseline/base_prompt.json
    python scripts/eval_model.py --adapter /content/adapter_watch_json --prompt-mode system \
        --out results/v1/tuned_prompt.json

Метрики:
  json_strict      — весь ответ парсится как JSON без вырезания;
  json_extractable — JSON удалось вытащить из текста/ограждений;
  schema_ok        — ответ соответствует схеме spec.validate_output;
  exact            — объект совпал с эталоном;
  ticker_f1, poll_acc, report_acc — по успешным примерам;
  error_acc        — по примерам с ожидаемой ошибкой.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from guard import apply_guard
from metrics import extract_json, score_row
from peft import PeftModel
from spec import FEWSHOT, SYSTEM_PROMPT, canonical_json
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DATA / "eval.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt-mode", choices=["none", "system", "fewshot"], default="system")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--guard", action="store_true", help="включить детерминированную страховку")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_eval(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_messages(row: dict, prompt_mode: str) -> list[dict]:
    user = row["messages"][1]["content"]
    if prompt_mode == "none":
        return [{"role": "user", "content": user}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prompt_mode == "fewshot":
        for example_user, example_answer in FEWSHOT:
            messages.append({"role": "user", "content": example_user})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example_answer, ensure_ascii=False, separators=(",", ":")),
                }
            )
    messages.append({"role": "user", "content": user})
    return messages


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new_tokens: int) -> str:
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    rows = load_eval(args.eval)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer_path = args.tokenizer or args.adapter or args.base_model
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
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

    results = []
    for index, row in enumerate(rows, 1):
        messages = build_messages(row, args.prompt_mode)
        raw = generate(model, tok, messages, args.max_new_tokens)
        if args.guard:
            parsed = extract_json(raw)
            if parsed is not None:
                guarded = apply_guard(row["messages"][1]["content"], parsed)
                if guarded != parsed:
                    raw = canonical_json(guarded)
        expected = row["meta"]["expected"]
        metrics = score_row(expected, raw)
        results.append(
            {
                "n": index,
                "category": row["meta"]["category"],
                "user": row["meta"]["user"],
                "expected": expected,
                "output": raw,
                **metrics,
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} exact={metrics['exact']} schema={metrics['schema_ok']}", flush=True)

    overall = {
        "count": len(results),
        "exact": sum(r["exact"] for r in results) / len(results),
        "schema_ok": sum(r["schema_ok"] for r in results) / len(results),
        "json_strict": sum(r["json_strict"] for r in results) / len(results),
        "json_extractable": sum(r["json_extractable"] for r in results) / len(results),
    }
    for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
        values = [r[key] for r in results if key in r]
        overall[key] = sum(values) / len(values) if values else None

    by_category: dict[str, dict] = {}
    for category in sorted({r["category"] for r in results}):
        subset = [r for r in results if r["category"] == category]
        block = {
            "count": len(subset),
            "exact": sum(r["exact"] for r in subset) / len(subset),
            "schema_ok": sum(r["schema_ok"] for r in subset) / len(subset),
            "json_strict": sum(r["json_strict"] for r in subset) / len(subset),
        }
        for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
            values = [r[key] for r in subset if key in r]
            block[key] = sum(values) / len(values) if values else None
        by_category[category] = block

    payload = {
        "config": vars(args),
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
    print("таблица:", csv_path)


if __name__ == "__main__":
    main()
