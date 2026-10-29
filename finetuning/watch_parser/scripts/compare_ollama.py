"""Живое сравнение базовой и дообученной моделей в Ollama.

Берёт запросы из eval/holdout JSONL, задаёт каждый обеим моделям с одинаковым
system-промптом и параметрами и печатает ответы парами. Результаты попутно
пишутся в JSON.

Примеры:
    python scripts/compare_ollama.py --limit 5
    python scripts/compare_ollama.py --category invalid --only-diff
    python scripts/compare_ollama.py --guard --tuned qwen25-watch-v3
    python scripts/compare_ollama.py --input data/holdout.jsonl --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from guard import apply_guard
from metrics import extract_json, score_row, semantic_equal
from ollama_eval import build_messages
from spec import canonical_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_INPUT = ROOT / "data" / "eval.jsonl"
DEFAULT_BASE = "qwen2.5:1.5b-instruct-fp16"
DEFAULT_TUNED = "qwen25-watch-v3"
OLLAMA_URL = "http://localhost:11434"
OPTIONS = {
    "temperature": 0,
    "repeat_penalty": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "num_predict": 128,
    "num_ctx": 4096,
    "seed": 42,
}


def list_models(base_url: str) -> set[str]:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=30) as response:
        names = {item["name"] for item in json.load(response)["models"]}
    return {name.removesuffix(":latest") for name in names}


def chat(base_url: str, model: str, messages: list[dict], timeout: int) -> str:
    body = {"model": model, "messages": messages, "stream": False, "options": OPTIONS}
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]["content"].strip()


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def short(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--tuned", default=DEFAULT_TUNED)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(RESULTS / "v3" / "compare_ollama_live.json"))
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--prompt-mode", choices=["system", "fewshot"], default="fewshot")
    parser.add_argument("--guard", action="store_true", help="применить guard к ответу тюна")
    parser.add_argument("--only", help="номера запросов через запятую, например 1,3,17")
    parser.add_argument("--limit", type=int, default=0, help="первые N запросов")
    parser.add_argument("--category", default=None, help="simple/multi/invalid/...")
    parser.add_argument("--only-diff", action="store_true", help="печатать только различающиеся ответы")
    parser.add_argument("--show-system", action="store_true", help="печатать system целиком")
    parser.add_argument("--max-chars", type=int, default=0, help="обрезать ответ; 0 — полностью")
    parser.add_argument("--timeout", type=int, default=900, help="таймаут одного запроса, с")
    args = parser.parse_args()

    rows = load_rows(args.input)
    for number, row in enumerate(rows, 1):
        row["_n"] = number
    if args.only:
        wanted = {int(n) for n in args.only.split(",") if n.strip()}
        rows = [row for row in rows if row["_n"] in wanted]
    if args.category:
        rows = [row for row in rows if row["meta"]["category"] == args.category]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Нет запросов под указанный фильтр.", file=sys.stderr)
        return 1

    try:
        models = list_models(args.url)
    except urllib.error.URLError as exc:
        print(f"Ollama недоступна по {args.url}: {exc}", file=sys.stderr)
        return 1

    missing = [model for model in (args.base, args.tuned) if model.removesuffix(":latest") not in models]
    if missing:
        print("В Ollama нет моделей: " + ", ".join(missing), file=sys.stderr)
        for model in missing:
            print(f"  docker exec ollama ollama pull {model}", file=sys.stderr)
        return 1

    out: list[dict] = []
    counts = {
        "better_semantic": 0,
        "better_format": 0,
        "worse_semantic": 0,
        "worse_format": 0,
        "same_good": 0,
        "same_semantic_bad_format": 0,
        "same_bad": 0,
    }
    started = time.time()

    for index, row in enumerate(rows, 1):
        meta = row["meta"]
        n = row["_n"]
        user = meta["user"]
        expected = meta["expected"]
        messages = build_messages(user, args.prompt_mode)

        print(f"\n{'=' * 78}")
        print(f"[{index}/{len(rows)}] запрос №{n} · {meta['category']}")
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        shown = system if args.show_system else system.replace("\n", " ")[:160]
        suffix = "" if args.show_system or len(system) <= 160 else " …"
        if system:
            print(f"SYSTEM: {shown}{suffix}")
        print(f"USER:   {user}")
        print(f"ЭТАЛОН: {json.dumps(expected, ensure_ascii=False)}\n")

        answers: dict[str, str] = {}
        metrics: dict[str, dict] = {}
        for label, model in (("БАЗА", args.base), ("ТЮН", args.tuned)):
            began = time.time()
            try:
                answer = chat(args.url, model, messages, args.timeout)
            except urllib.error.URLError as exc:
                print(f"--- {label} ({model}) --- ошибка: {exc}", file=sys.stderr)
                answer = ""
            if label == "ТЮН" and args.guard:
                parsed = extract_json(answer)
                if parsed is not None:
                    guarded = apply_guard(user, parsed)
                    if guarded != parsed:
                        answer = canonical_json(guarded)
            answers[label] = answer
            metrics[label] = score_row(expected, answer)
            print(f"--- {label}: {model} ({time.time() - began:.1f} с) ---")
            print(short(answer, args.max_chars))
            print()

        base_exact = metrics["БАЗА"]["exact"]
        tuned_exact = metrics["ТЮН"]["exact"]
        base_semantic = semantic_equal(expected, answers["БАЗА"])
        tuned_semantic = semantic_equal(expected, answers["ТЮН"])
        if base_exact and tuned_exact:
            verdict = "так же"
            counts["same_good"] += 1
        elif tuned_exact and not base_exact:
            if base_semantic:
                verdict = "лучше (только формат)"
                counts["better_format"] += 1
            else:
                verdict = "лучше (смысл)"
                counts["better_semantic"] += 1
        elif base_exact and not tuned_exact:
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

        out.append(
            {
                "n": n,
                "category": meta["category"],
                "user": user,
                "expected": expected,
                "base_model": args.base,
                "tuned_model": args.tuned,
                "base": answers["БАЗА"],
                "tuned": answers["ТЮН"],
                "base_exact": base_exact,
                "tuned_exact": tuned_exact,
                "base_semantic": base_semantic,
                "tuned_semantic": tuned_semantic,
                "tuned_guard": args.guard,
                "verdict": verdict,
            }
        )

        if not args.only_diff or answers["БАЗА"] != answers["ТЮН"]:
            print(f"вердикт: {verdict}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(out)
    print(f"\n{'=' * 78}")
    print(f"Готово: {total} запросов за {time.time() - started:.0f} с. JSON: {args.out}")
    print(f"база exact: {sum(1 for row in out if row['base_exact'])}/{total}")
    print(f"тюн  exact: {sum(1 for row in out if row['tuned_exact'])}/{total}")
    print("лучше по смыслу: {better_semantic} | лучше только формат: {better_format}".format(**counts))
    print("хуже по смыслу: {worse_semantic} | хуже только формат: {worse_format}".format(**counts))
    print(
        "так же верно: {same_good} | оба верны, формат разный: {same_semantic_bad_format} | "
        "оба неверны: {same_bad}".format(**counts)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
