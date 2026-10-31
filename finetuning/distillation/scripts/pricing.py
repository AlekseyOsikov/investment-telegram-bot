"""Пересчёт фактических токенов в «облачный эквивалент» стоимости.

Учитель и ученик работают локально (Docker-Ollama, CPU), поэтому реального
долларового счёта нет. Для отчёта «сколько удалось сэкономить» переводим
измеренные токены входа/выхода в доллары по справочным тарифам близких по
размеру моделей Qwen2.5.

Цифры ниже — ОРИЕНТИРОВОЧНЫЕ справочные значения уровня Alibaba Cloud Model
Studio (international) / OpenRouter на 1 млн токенов. Перед финальным отчётом
их нужно сверить с актуальной тарифной картой вендора и проставить ссылку —
см. TODO в ``report.md``.
"""

from __future__ import annotations

# $ за 1 млн токенов: {"input": ..., "output": ...}
USD_PER_1M: dict[str, dict[str, float]] = {
    "qwen2.5-1.5b": {"input": 0.10, "output": 0.10},
    "qwen2.5-7b": {"input": 0.14, "output": 0.28},
}

# Какой тарифный класс соответствует каждой Ollama-модели в eval.
MODEL_PRICE_CLASS = {
    "qwen2.5:1.5b-instruct-fp16": "qwen2.5-1.5b",
    "qwen25-distill-v1": "qwen2.5-1.5b",
    "qwen25-golden-v1": "qwen2.5-1.5b",
    "qwen2.5:7b": "qwen2.5-7b",
}


def cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Стоимость одного запроса в долларах по справочным тарифам."""
    price_class = MODEL_PRICE_CLASS.get(model, "qwen2.5-1.5b")
    rates = USD_PER_1M[price_class]
    return (prompt_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
