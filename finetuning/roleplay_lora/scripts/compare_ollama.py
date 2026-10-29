"""Сравнение базовой Qwen2.5-1.5B-Instruct и дообученной qwen25-roleplay в Ollama.

Берёт запросы из results/compare_20.json, задаёт каждый обеим моделям с
одинаковыми system-промптом и параметрами (как в scripts/ollama_run.py)
и печатает ответы парами. Результаты попутно пишутся в JSON.

Примеры:
    python scripts/compare_ollama.py                 # все 20 запросов
    python scripts/compare_ollama.py --only 1,13,17  # выбранные номера
    python scripts/compare_ollama.py --limit 3       # первые три
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_INPUT = RESULTS / "compare_20.json"
DEFAULT_OUT = RESULTS / "compare_ollama.json"
DEFAULT_BASE = "qwen2.5:1.5b-instruct-fp16"
DEFAULT_TUNED = "qwen25-roleplay"
OLLAMA_URL = "http://localhost:11434"
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
OPTIONS = {"temperature": 0, "repeat_penalty": 1.0, "num_predict": 200, "seed": 42}


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


def show(text: str) -> str:
    return text if text else "(пустой ответ)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="базовая модель")
    parser.add_argument("--tuned", default=DEFAULT_TUNED, help="дообученная модель")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="JSON с запросами")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="куда сохранить ответы")
    parser.add_argument("--only", help="номера запросов через запятую, например 1,13,17")
    parser.add_argument("--limit", type=int, help="взять только первые N запросов")
    parser.add_argument("--show-system", action="store_true", help="печатать system целиком")
    parser.add_argument("--timeout", type=int, default=900, help="таймаут одного запроса, с")
    parser.add_argument("--url", default=OLLAMA_URL, help="адрес Ollama API")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.only:
        wanted = {int(n) for n in args.only.split(",") if n.strip()}
        rows = [row for row in rows if int(row["n"]) in wanted]
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

    missing = [
        model for model in (args.base, args.tuned) if model.removesuffix(":latest") not in models
    ]
    if missing:
        print("В Ollama нет моделей: " + ", ".join(missing), file=sys.stderr)
        print("Загрузите их, например:", file=sys.stderr)
        for model in missing:
            print(f"  docker exec ollama ollama pull {model}", file=sys.stderr)
        return 1

    out: list[dict] = []
    started = time.time()
    for index, row in enumerate(rows, 1):
        system = row["system"] or QWEN_DEFAULT_SYSTEM
        messages = [{"role": "system", "content": system}, {"role": "user", "content": row["user"]}]

        print(f"\n{'=' * 78}")
        print(f"[{index}/{len(rows)}] запрос №{row['n']} · {row['group']}")
        print(f"{'=' * 78}")
        printed_system = system if args.show_system else system.replace("\n", " ")[:160]
        suffix = "" if args.show_system or len(system) <= 160 else " …"
        print(f"SYSTEM: {printed_system}{suffix}")
        print(f"USER:   {row['user']}\n")

        answers: dict[str, str] = {}
        for label, model in (("БАЗА", args.base), ("ДООБУЧЕННАЯ", args.tuned)):
            began = time.time()
            try:
                answer = chat(args.url, model, messages, args.timeout)
            except urllib.error.URLError as exc:
                print(f"--- {label} ({model}) --- ошибка: {exc}", file=sys.stderr)
                answer = ""
            answers[label] = answer
            print(f"--- {label}: {model} ({time.time() - began:.1f} с) ---")
            print(show(answer))
            print()

        out.append(
            {
                "n": row["n"],
                "group": row["group"],
                "system": row["system"],
                "user": row["user"],
                "base": answers["БАЗА"],
                "tuned": answers["ДООБУЧЕННАЯ"],
            }
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'=' * 78}")
    print(f"Готово: {len(out)} запросов за {time.time() - started:.0f} с. JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
