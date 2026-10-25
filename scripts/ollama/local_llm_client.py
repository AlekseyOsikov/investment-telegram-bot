"""Тонкий клиент нативного API Ollama для оснастки `scripts/ollama/*.py` (не часть бота).

Отдельный модуль, чтобы чистые правила (`local_llm_lib.py`) не знали о сети. Адрес сервера
передаётся аргументом `--url` скриптов: `config.py` бота здесь не нужен (он требует токен бота
при импорте), а окружение скрипты не читают.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://localhost:11434"
TIMEOUT_SECONDS = 3600


class OllamaError(RuntimeError):
    """Ответ сервера с кодом ошибки; тело ответа сохраняется (у gpt-oss это часто разбор вызова)."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


def post(url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST JSON на сервер Ollama (`stream=False`), ответ — словарь."""
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise OllamaError(exc.code, exc.read().decode("utf-8", "replace")[:500]) from exc


def chat(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """`/api/chat`: сообщения, инструменты, параметры."""
    return post(url, "/api/chat", {**body, "stream": False})


def generate(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """`/api/generate`: один промпт без истории."""
    return post(url, "/api/generate", {**body, "stream": False})
