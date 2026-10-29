"""Прогон 20 запросов roleplay_lora через Ollama (модель qwen25-roleplay).

Режим A повторяет условия Colab: жадная генерация, repeat_penalty 1.0,
для запросов без роли подставляется системное сообщение Qwen по умолчанию.
Режим B — настройки Ollama по умолчанию, только для запросов, где в Colab
нашлись регрессы.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_MODEL = "qwen25-roleplay"
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
REGRESS = (15, 17, 19, 20)


def chat(model: str, messages: list[dict], options: dict) -> str:
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(request, timeout=900))["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default=str(RESULTS / "compare_20.json"))
    parser.add_argument("--out", default=str(RESULTS / "ollama_20.json"))
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = []
    started = time.time()
    for row in rows:
        user = [{"role": "user", "content": row["user"]}]
        system = row["system"] or QWEN_DEFAULT_SYSTEM
        mode_a = chat(
            args.model,
            [{"role": "system", "content": system}] + user,
            {"temperature": 0, "repeat_penalty": 1.0, "num_predict": 200, "seed": 42},
        )
        item = {
            "n": row["n"],
            "group": row["group"],
            "user": row["user"],
            "colab": row["tuned"],
            "ollama_a": mode_a,
        }
        if row["n"] in REGRESS:
            messages_b = (
                [{"role": "system", "content": row["system"]}] if row["system"] else []
            ) + user
            item["ollama_b"] = chat(
                args.model, messages_b, {"temperature": 0, "num_predict": 200, "seed": 42}
            )
        out.append(item)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(row["n"], round(time.time() - started), "с", flush=True)


if __name__ == "__main__":
    main()
