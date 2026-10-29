"""Один вопрос локальной Ollama-модели с тем же системным промптом, что в eval.

Пример:
    python scripts/ollama_ask.py "Следи за Сбером каждые 15 минут, сводку раз в час."
    python scripts/ollama_ask.py --guard "Следи за SBER и SBER каждые 15 минут, сводку раз в час."
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from guard import apply_guard
from metrics import extract_json
from spec import SYSTEM_PROMPT, canonical_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--model", default="qwen25-watch-v3")
    parser.add_argument("--guard", action="store_true")
    args = parser.parse_args()

    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.question},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
            "num_ctx": 4096,
            "repeat_penalty": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
        },
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        raw = json.load(response)["message"]["content"].strip()

    if args.guard:
        parsed = extract_json(raw)
        if parsed is not None:
            guarded = apply_guard(args.question, parsed)
            if guarded != parsed:
                raw = canonical_json(guarded)

    print(raw)


if __name__ == "__main__":
    main()
