"""Eval-набор для замера «стиль и тон» — соответствие роли и манере.

Целевой стиль задаёт ``STYLE_SPEC`` (смотритель маяка): короткие ответы, в
образе, отказ на темы вне роли. Три прогона повторяют схему «формат»:

- ``base_none`` — база без роли (нейтральный системный промпт);
- ``base_prompt`` — база + ``STYLE_SPEC`` (хороший промпт задаёт стиль);
- ``tuned`` — дообученная roleplay-модель + ``STYLE_SPEC``.

Оценка — покрытие ``criteria`` (judge) + автоматическая ``brevity`` (доля
коротких ответов, ≤ 2 предложений по грубой эвристике).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STYLE_SPEC = (
    "Ты — смотритель маяка по имени Фёдор. Отвечай коротко (не больше двух "
    "предложений), простыми словами, в образе смотрителя маяка. Если вопрос не "
    "про море, погоду, корабли или маяк — скажи, что это не твоё дело, и не "
    "отвечай по существу."
)

NEUTRAL_PROMPT = "Отвечай кратко и по существу."

# Критерии: общие для всех + добавочный критерий отказа для boundary-вопросов.
_IN_ROLE = "ответ выдержан в образе смотрителя маяка (говорит от лица Фёдора, не выходит из роли)"
_SHORT = "ответ короткий — не длиннее двух предложений"
_REDIRECT = "модель не отвечает по существу, а указывает, что тема вне её дела"

ITEMS: list[dict[str, Any]] = [
    {
        "id": "style_01",
        "kind": "in_role",
        "prompt": "Какая сегодня будет погода на море?",
        "criteria": [_IN_ROLE, _SHORT],
    },
    {"id": "style_02", "kind": "in_role", "prompt": "Расскажи, как устроен маяк.", "criteria": [_IN_ROLE, _SHORT]},
    {
        "id": "style_03",
        "kind": "in_role",
        "prompt": "Что делать, если корабль сел на мель?",
        "criteria": [_IN_ROLE, _SHORT],
    },
    {"id": "style_04", "kind": "in_role", "prompt": "Ты видел сегодня дельфинов?", "criteria": [_IN_ROLE, _SHORT]},
    {
        "id": "style_05",
        "kind": "in_role",
        "prompt": "Какой у тебя распорядок дня на маяке?",
        "criteria": [_IN_ROLE, _SHORT],
    },
    {"id": "style_06", "kind": "boundary", "prompt": "Как настроить телеграм-бота?", "criteria": [_IN_ROLE, _REDIRECT]},
    {
        "id": "style_07",
        "kind": "boundary",
        "prompt": "Что такое диверсификация портфеля?",
        "criteria": [_IN_ROLE, _REDIRECT],
    },
    {
        "id": "style_08",
        "kind": "boundary",
        "prompt": "Ты человек или программа? Назови своё настоящее имя.",
        "criteria": [_IN_ROLE, _REDIRECT],
    },
]


def main() -> None:
    out = {
        "style_spec": STYLE_SPEC,
        "neutral_prompt": NEUTRAL_PROMPT,
        "items": ITEMS,
    }
    path = Path(__file__).resolve().parent / "style_eval.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("записан style_eval.json:", len(ITEMS), "промптов")


if __name__ == "__main__":
    main()
