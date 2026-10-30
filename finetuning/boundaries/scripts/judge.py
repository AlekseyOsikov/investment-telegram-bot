"""Слепой judge по чек-листу: покрытие ожидаемых фактов/критериев в ответе.

Самодостаточный аналог ``research/rag_compare``: промпт и разбор ответа
повторяют ``RAG_COMPARE_JUDGE_SYSTEM_PROMPT`` и ``parse_judge_response`` из
``src/``, но без импорта бота (граница ``finetuning/`` — см. README проекта).

Judge обращается к LLM по OpenAI-совместимому эндпоинту:

- ``deepseek`` — ``DEEPSEEK_API_KEY``, база ``https://api.deepseek.com``;
- ``ollama`` — локально ``http://localhost:11434/v1`` (модель задаётся явно).

Оценка вслепую по построению: судье передаются только вопрос, пронумерованные
ожидаемые факты и ответ — без указания, как получен ответ (база/промпт/тюн).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

# Корень репозитория (finetuning/boundaries/scripts/judge.py → repo root) — оттуда
# при необходимости читаем .env, чтобы судья работал без экспорта переменных вручную.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_env_file() -> None:
    """Дочитывает ``.env`` из корня репозитория в ``os.environ`` (не перезаписывая уже
    заданные переменные). Минимальный парсер ``KEY=value`` без зависимости от dotenv."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


_load_env_file()

JUDGE_SYSTEM_PROMPT = (
    "Ты — СЛУЖЕБНЫЙ оценщик ответов в исследовательском эксперименте. Твой ответ "
    "пользователю не показывается — его разбирает код.\n\n"
    "Тебе дают вопрос, пронумерованный список ОЖИДАЕМЫХ ФАКТОВ и ответ ассистента. "
    "Определи, какие из ожидаемых фактов присутствуют в ответе, и верни ОДИН "
    "JSON-объект с полями:\n"
    '- "facts" — массив объектов {"index": номер факта из списка, "present": '
    "true или false}, по одному на КАЖДЫЙ факт списка;\n"
    '- "contradicts" — true, если в ответе есть утверждение, ПРЯМО противоречащее '
    "какому-либо ожидаемому факту, иначе false;\n"
    '- "note" — одна короткая фраза на русском о главном расхождении (или пустая '
    "строка).\n\n"
    "Правила:\n"
    "1. Факт присутствует, если ответ сообщает ту же информацию, пусть другими "
    "словами. Числа, даты и названия должны совпадать: другое число или другая дата "
    "— факт отсутствует.\n"
    "2. Факт не присутствует, если ответ о нём молчит, говорит слишком общо или "
    "частично: оценивай только то, что реально написано в ответе.\n"
    "3. Не оценивай стиль, полноту сверх списка и общую правильность — только "
    "наличие перечисленных фактов и прямые противоречия им.\n"
    '4. В "index" используй только номера из присланного списка.\n'
    "5. Верни ТОЛЬКО JSON-объект, без пояснений и без markdown-разметки."
)

_BASES = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
_DEFAULT_MODELS = {"deepseek": "deepseek-flash", "ollama": ""}


def _chat_completion(provider: str, model: str, messages: list[dict], max_tokens: int) -> str:
    """Один вызов OpenAI-совместимого эндпоинта. Возвращает ``content`` ответа."""
    base_url, key_env = _BASES[provider]
    api_key = os.environ.get(key_env) if key_env else "ollama"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if provider == "deepseek":
        body["extra_body"] = {"thinking": {"type": "disabled"}}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"] or ""


def _build_user_content(question: str, facts: list[str], answer: str) -> str:
    numbered = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, start=1))
    return f"Вопрос:\n{question}\n\nОжидаемые факты:\n{numbered}\n\nОтвет ассистента:\n<<<\n{answer}\n>>>"


def _parse(data: Any, fact_count: int) -> dict[str, Any] | None:
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        return None
    flags: dict[int, bool] = {}
    for entry in data["facts"]:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if 1 <= index <= fact_count and index not in flags:
            flags[index] = entry.get("present") is True
    present = [flags.get(i, False) for i in range(1, fact_count + 1)]
    return {
        "coverage": sum(present) / len(present) if present else 0.0,
        "present": present,
        "contradicts": data.get("contradicts") is True,
        "note": (data.get("note") or "").strip(),
    }


def judge_coverage(
    question: str,
    facts: list[str],
    answer: str,
    *,
    provider: str = "deepseek",
    model: str | None = None,
    max_tokens: int = 600,
) -> dict[str, Any] | None:
    """Покрытие фактов в ответе. ``None`` — судья не дал пригодного ответа."""
    model = model or _DEFAULT_MODELS.get(provider, "")
    if not model:
        raise ValueError(f"для provider={provider!r} нужно явно задать model")
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(question, facts, answer)},
    ]
    try:
        raw = _chat_completion(provider, model, messages, max_tokens)
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        return None
    return _parse(data, len(facts))
