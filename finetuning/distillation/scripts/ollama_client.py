"""Вызов локальной Ollama-модели (нативный ``/api/chat``) с замером токенов и времени.

В отличие от ``watch_parser/scripts/ollama_eval.py`` возвращает не только текст,
но и счётчики/тайминги из ответа Ollama — они нужны для оценки скорости и
стоимости запроса (задание «Дистилляция»):

- ``prompt_eval_count`` — токены входа (промпт);
- ``eval_count`` — токены выхода (ответ);
- ``prompt_eval_duration`` / ``eval_duration`` / ``total_duration`` — наносекунды.

Параметры генерации жадные и повторяют условия HF/Ollama из ``watch_parser``:
``repeat_penalty=1.0``, ``top_k=1``, ``top_p=1.0``, ``seed`` — чтобы ответы были
детерминированы и сравнимы между моделями.
"""

from __future__ import annotations

import json
import urllib.request

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def chat(
    model: str,
    messages: list[dict],
    *,
    num_predict: int = 256,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    seed: int = 42,
    timeout: int = 900,
) -> dict:
    """Возвращает словарь ``{content, prompt_eval_count, eval_count, ...durations_ns}``."""
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
        OLLAMA_CHAT_URL,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {
        "content": (payload.get("message", {}).get("content") or "").strip(),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "prompt_eval_duration": payload.get("prompt_eval_duration"),
        "eval_duration": payload.get("eval_duration"),
        "total_duration": payload.get("total_duration"),
        "load_duration": payload.get("load_duration"),
    }
