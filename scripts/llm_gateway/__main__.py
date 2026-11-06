"""Запуск LLM Gateway через uvicorn.

Из каталога ``scripts/``:
    python -m llm_gateway --host 127.0.0.1 --port 8000
    python -m llm_gateway --guard-mode mask --rate-limit 30 --model qwen2.5:1.5b-instruct-fp16

Из корня проекта:
    PYTHONPATH=scripts python -m llm_gateway
"""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Gateway (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--guard-mode", choices=["block", "mask"], default=None)
    parser.add_argument("--rate-limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend-url", default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.guard_mode:
        settings.guard_mode = args.guard_mode
    if args.rate_limit is not None:
        settings.rate_limit_per_minute = args.rate_limit
    if args.model:
        settings.default_model = args.model
    if args.backend_url:
        settings.backend_url = args.backend_url

    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
