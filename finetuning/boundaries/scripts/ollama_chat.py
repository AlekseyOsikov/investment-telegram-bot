"""Общий вызов локальной Ollama-модели (нативный ``/api/chat``).

Повторяет проверенные в ``watch_parser`` параметры жадной генерации: без
штрафа за повторы, ``top_k=1``, ``top_p=1.0``, ``seed`` — чтобы ответы были
детерминированы и совпадали с условиями HF (см. ``watch_parser/scripts/ollama_eval.py``).
"""

from __future__ import annotations

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"


def chat(
    model: str,
    messages: list[dict],
    *,
    num_predict: int = 256,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    seed: int = 42,
    timeout: int = 600,
) -> str:
    """Возвращает текст ответа модели (без ``.strip()`` вызывающего)."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "seed": seed,
            "repeat_penalty": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["message"]["content"].strip()
