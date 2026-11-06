#!/usr/bin/env python3
"""Сравнение ответа локальной модели БЕЗ гейтвея и ЧЕРЕЗ LLM Gateway.

Задаёте один промпт — скрипт делает два запроса:
1. напрямую в Ollama (как будто гейтвея нет) — секрет уходит в модель как есть;
2. через LLM Gateway — Input Guard блокирует (block) или маскирует (mask) секрет,
   Output Guard проверяет ответ.

Примеры:
    .venv/bin/python scripts/llm_gateway_compare.py "Мой ключ sk-proj-abc1234567890"
    .venv/bin/python scripts/llm_gateway_compare.py --guard-mode mask "..."
    .venv/bin/python scripts/llm_gateway_compare.py --examples
    .venv/bin/python scripts/llm_gateway_compare.py --gateway-url http://127.0.0.1:8000 "..."
    .venv/bin/python scripts/llm_gateway_compare.py --no-direct "..."   # без отправки секрета

Если промпт не передан аргументом, скрипт спросит его интерактивно.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from llm_gateway import Settings, create_app
from llm_gateway.detectors import scan_messages

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = ROOT / "reports" / "llm-gateway" / "audit-demo.jsonl"

EXAMPLES = [
    ("AWS-ключ", "Мой AWS ключ AKIAIOSFODNN7EXAMPLE, проверь его"),
    ("Карта", "Оплати картой 4111 1111 1111 1111"),
    ("Base64", "Данные: " + base64.b64encode(b"sk-proj-abc1234567890").decode()),
    ("Разбитый ключ", 'мой ключ: "sk-" + "proj-abc1234567890"'),
    ("Чистый промпт", "Сколько будет 2+2? Ответь одним словом."),
]


# ---------------------------------------------------------------------------
# Вызовы
# ---------------------------------------------------------------------------


def call_direct(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> httpx.Response:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with httpx.Client(timeout=180, trust_env=False) as client:
        return client.post(
            base_url.rstrip("/") + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )


def call_gateway_remote(
    gateway_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> httpx.Response:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with httpx.Client(timeout=180, trust_env=False) as client:
        return client.post(
            gateway_url.rstrip("/") + "/v1/chat/completions", json=payload
        )


def build_local_gateway(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def call_gateway_local(
    client: TestClient,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )


# ---------------------------------------------------------------------------
# Печать
# ---------------------------------------------------------------------------


def _content(body: dict) -> str:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    if not content and reasoning:
        return f"[reasoning] {reasoning}"
    return content


def print_direct(response: httpx.Response, secret_types: list[str]) -> None:
    print("\n=== 1. БЕЗ гейтвея (напрямую в Ollama) ===")
    print(f"HTTP {response.status_code}")
    if secret_types:
        print(
            "⚠️  В промпте был секрет: "
            + ", ".join(secret_types)
            + " — он ушёл в модель как есть."
        )
    try:
        body = response.json()
    except ValueError:
        print(response.text[:1000])
        return
    if response.status_code >= 400:
        print(f"Ошибка backend'а: {body}")
        return
    print("Ответ модели:")
    print(_content(body))
    usage = body.get("usage") or {}
    print(
        f"Токены: prompt={usage.get('prompt_tokens', '?')}, "
        f"completion={usage.get('completion_tokens', '?')}"
    )


def print_gateway(response: httpx.Response, mode: str) -> None:
    print(f"\n=== 2. С гейтвеем (guard_mode={mode}) ===")
    print(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        print(response.text[:1000])
        return

    blocked = response.headers.get("x-gateway-blocked")
    if blocked:
        print(f"Сработал guard: {blocked}")

    error = body.get("error")
    if error:
        print(f"Ошибка: {error.get('type')}: {error.get('message')}")
        for finding in error.get("findings", []):
            print(
                f"  - {finding['type']} -> {finding['redaction']} "
                f"(preview={finding['preview']})"
            )
        return

    print("Ответ модели:")
    print(_content(body))

    input_types = response.headers.get("x-gateway-input-finding-types", "")
    output_types = response.headers.get("x-gateway-output-finding-types", "")
    if input_types:
        print(f"Input findings: {input_types}")
    if output_types:
        print(f"Output findings: {output_types}")
    print(
        "Токены: "
        f"prompt={response.headers.get('x-gateway-prompt-tokens', '?')}, "
        f"completion={response.headers.get('x-gateway-completion-tokens', '?')}"
    )
    print(f"Стоимость: ${response.headers.get('x-gateway-cost-usd', '?')}")


def print_last_audit(path: Path) -> None:
    if not path.exists():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError:
        return
    print("\nАудит (последняя запись):")
    print(
        json.dumps(
            {
                "status": record.get("status"),
                "input_findings": record.get("input_findings"),
                "output_findings": record.get("output_findings"),
                "tokens": record.get("tokens"),
                "cost_usd": record.get("cost_usd"),
                "request_preview": record.get("request_preview"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Один прогон
# ---------------------------------------------------------------------------


def run_one(
    *,
    title: str | None,
    prompt: str,
    model: str,
    guard_mode: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    gateway_url: str | None,
    no_direct: bool,
    local_client: TestClient | None,
    audit_path: Path,
    backend_url: str,
    backend_api_key: str,
) -> None:
    if title:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print("Промпт:")
    print(prompt)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    secret_types = sorted({finding.type for finding in scan_messages(messages)})

    if no_direct:
        print("\n=== 1. БЕЗ гейтвея — пропущено (--no-direct) ===")
    else:
        direct = call_direct(
            backend_url, backend_api_key, model, messages, max_tokens, temperature
        )
        print_direct(direct, secret_types)

    if gateway_url:
        gateway = call_gateway_remote(
            gateway_url, model, messages, max_tokens, temperature
        )
    else:
        assert local_client is not None
        gateway = call_gateway_local(
            local_client, model, messages, max_tokens, temperature
        )
    print_gateway(gateway, guard_mode)

    if local_client is not None:
        print_last_audit(audit_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнить ответ локальной модели без гейтвея и через LLM Gateway"
    )
    parser.add_argument("prompt", nargs="?", help="промпт; если не задан — спросим")
    parser.add_argument(
        "--model", default="qwen2.5:1.5b-instruct-fp16", help="модель Ollama"
    )
    parser.add_argument(
        "--guard-mode",
        choices=["block", "mask"],
        default="block",
        help="режим Input/Output Guard",
    )
    parser.add_argument("--system", default=None, help="system prompt (необязательно)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--gateway-url",
        default=None,
        help="если задан, зовём запущенный сервер, иначе поднимаем гейтвей в процессе",
    )
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="не отправлять промпт напрямую в модель (если в нём реальный секрет)",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="прогнать встроенные примеры (AWS, карта, Base64, разбитый ключ, чистый)",
    )
    args = parser.parse_args()

    backend = Settings.from_env()
    backend_url = backend.backend_url
    backend_api_key = backend.backend_api_key

    local_client: TestClient | None = None
    audit_path = DEFAULT_AUDIT
    if not args.gateway_url:
        gateway_settings = Settings.from_env()
        gateway_settings.default_model = args.model
        gateway_settings.guard_mode = args.guard_mode
        gateway_settings.audit_log_path = DEFAULT_AUDIT
        gateway_settings.canary = "GW-CANARY-CONSOLE-" + secrets.token_hex(4)
        gateway_settings.request_timeout = 180.0
        local_client = build_local_gateway(gateway_settings)
        audit_path = gateway_settings.audit_log_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            audit_path.unlink()

    common = dict(
        model=args.model,
        guard_mode=args.guard_mode,
        system=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gateway_url=args.gateway_url,
        no_direct=args.no_direct,
        local_client=local_client,
        audit_path=audit_path,
        backend_url=backend_url,
        backend_api_key=backend_api_key,
    )

    if args.examples:
        for title, prompt in EXAMPLES:
            run_one(title=title, prompt=prompt, **common)
        return 0

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input("Промпт: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nОтменено.")
            return 1
    if not prompt:
        print("Пустой промпт.")
        return 1

    run_one(title=None, prompt=prompt, **common)
    return 0


if __name__ == "__main__":
    sys.exit(main())
