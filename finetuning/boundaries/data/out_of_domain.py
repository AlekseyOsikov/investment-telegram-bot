"""Eval-набор задач ВНЕ домена — для замера «что модель потеряла после тюна».

Задачи не из инвестиционного домена, с проверяемыми ответами. Оценка —
детерминированная: все ``keywords`` должны присутствовать в ответе (без учёта
регистра). Для арифметики/логики это точное число, для перевода/знаний —
ключевое слово, для кода — обязательный токен синтаксиса.

Категории: ``math``, ``translation``, ``general``, ``logic``, ``code``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ITEMS: list[dict[str, Any]] = [
    # ------------------------- арифметика ------------------------- #
    {"id": "math_01", "category": "math", "prompt": "Сколько будет 17 умножить на 23?", "keywords": ["391"]},
    {"id": "math_02", "category": "math", "prompt": "Сколько будет 144 разделить на 12?", "keywords": ["12"]},
    {"id": "math_03", "category": "math", "prompt": "Сколько будет 15% от 200?", "keywords": ["30"]},
    {"id": "math_04", "category": "math", "prompt": "Чему равен квадрат числа 13?", "keywords": ["169"]},
    {"id": "math_05", "category": "math", "prompt": "Сколько будет 25 умножить на 4?", "keywords": ["100"]},
    {"id": "math_06", "category": "math", "prompt": "Сколько будет 1000 минус 357?", "keywords": ["643"]},
    # ------------------------- перевод RU → EN ------------------------- #
    {
        "id": "trans_01",
        "category": "translation",
        "prompt": "Переведи на английский: «Доброе утро».",
        "keywords": ["good morning"],
    },
    {
        "id": "trans_02",
        "category": "translation",
        "prompt": "Переведи на английский слово «спасибо».",
        "keywords": ["thank"],
    },
    {
        "id": "trans_03",
        "category": "translation",
        "prompt": "Переведи на английский слово «кошка».",
        "keywords": ["cat"],
    },
    {
        "id": "trans_04",
        "category": "translation",
        "prompt": "Переведи на английский слово «яблоко».",
        "keywords": ["apple"],
    },
    {
        "id": "trans_05",
        "category": "translation",
        "prompt": "Переведи на английский слово «река».",
        "keywords": ["river"],
    },
    # ------------------------- общие знания (до 2026) ------------------------- #
    {"id": "gen_01", "category": "general", "prompt": "Назови столицу Японии.", "keywords": ["токио"]},
    {"id": "gen_02", "category": "general", "prompt": "Назови столицу Великобритании.", "keywords": ["лондон"]},
    {"id": "gen_03", "category": "general", "prompt": "Кто написал роман «Война и мир»?", "keywords": ["толстой"]},
    {
        "id": "gen_04",
        "category": "general",
        "prompt": "Какая самая большая планета Солнечной системы?",
        "keywords": ["юпитер"],
    },
    {"id": "gen_05", "category": "general", "prompt": "Кто написал «Евгения Онегина»?", "keywords": ["пушкин"]},
    {"id": "gen_06", "category": "general", "prompt": "Назови столицу Австралии.", "keywords": ["канберра"]},
    # ------------------------- логика / здравый смысл ------------------------- #
    {
        "id": "logic_01",
        "category": "logic",
        "prompt": "У Маши 3 яблока, она отдала 1. Сколько осталось?",
        "keywords": ["2"],
    },
    {
        "id": "logic_02",
        "category": "logic",
        "prompt": "Сегодня вторник. Какой день недели был вчера?",
        "keywords": ["понедельник"],
    },
    {"id": "logic_03", "category": "logic", "prompt": "Сколько лап у трёх собак?", "keywords": ["12"]},
    {
        "id": "logic_04",
        "category": "logic",
        "prompt": "Если сейчас 14:00, сколько будет через 3 часа?",
        "keywords": ["17"],
    },
    # ------------------------- код ------------------------- #
    {
        "id": "code_01",
        "category": "code",
        "prompt": "Напиши на Python функцию, которая возвращает квадрат числа.",
        "keywords": ["def", "return"],
    },
    {
        "id": "code_02",
        "category": "code",
        "prompt": "Напиши на Python команду, которая выводит строку на экран.",
        "keywords": ["print"],
    },
    {
        "id": "code_03",
        "category": "code",
        "prompt": "Напиши на Python цикл по числам от 0 до 9.",
        "keywords": ["range"],
    },
    {
        "id": "code_04",
        "category": "code",
        "prompt": "Напиши на Python условие: если x больше нуля.",
        "keywords": ["if"],
    },
    {"id": "code_05", "category": "code", "prompt": "Как в Python открыть файл для чтения?", "keywords": ["open"]},
]

SYSTEM_PROMPT = "Отвечай кратко и по существу."


def main() -> None:
    out = {"system_prompt": SYSTEM_PROMPT, "items": ITEMS}
    path = Path(__file__).resolve().parent / "out_of_domain.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("записан out_of_domain.json:", len(ITEMS), "задач")


if __name__ == "__main__":
    main()
