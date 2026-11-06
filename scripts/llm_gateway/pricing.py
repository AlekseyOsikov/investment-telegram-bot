"""Cost tracking: токены и стоимость запроса.

Стоимость считается по ``usage`` из ответа backend'а (prompt_tokens /
completion_tokens) и таблице цен за 1M токенов. Для локальной Ollama цена 0,
но токены всё равно логируются — это и есть «стоимость» с точки зрения аренды
железа/электричества.
"""

from __future__ import annotations

import os

# USD за 1M токенов: (вход, выход). Значения ориентировочные, для отчёта.
PRICES: dict[str, tuple[float, float]] = {
    # Локальные модели — платим только за железо.
    "qwen3:30b-a3b": (0.0, 0.0),
    "qwen2.5:1.5b-instruct-fp16": (0.0, 0.0),
    "qwen2.5:7b": (0.0, 0.0),
    "gpt-oss:20b": (0.0, 0.0),
    # DeepSeek (USD / 1M).
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # OpenAI.
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    # Anthropic.
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}

# Переопределение для неизвестной модели через окружение (USD / 1M).
_ENV_INPUT = "LLM_GATEWAY_PRICE_INPUT_PER_1M"
_ENV_OUTPUT = "LLM_GATEWAY_PRICE_OUTPUT_PER_1M"


def price_for(model: str) -> tuple[float, float]:
    """Цена (вход, выход) для модели; поддержан prefix-match версий."""
    env_input = os.getenv(_ENV_INPUT)
    env_output = os.getenv(_ENV_OUTPUT)
    if env_input is not None and env_output is not None:
        try:
            return float(env_input), float(env_output)
        except ValueError:
            pass
    if model in PRICES:
        return PRICES[model]
    for name, price in PRICES.items():
        if model.startswith(name):
            return price
    return 0.0, 0.0


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Стоимость в USD по usage ответа."""
    input_price, output_price = price_for(model)
    return (
        max(prompt_tokens, 0) / 1_000_000 * input_price
        + max(completion_tokens, 0) / 1_000_000 * output_price
    )
