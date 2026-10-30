"""Сводная таблица «три типа задач»: прирост от тюна против прироста от промпта.

Читает результаты прогонов фактов и стиля (JSON с полем ``overall``) и берёт
числа «формат» из ``watch_parser`` (аргументы). Печатает Markdown-таблицу:

    тип задачи | база без промпта | база + промпт | дообученная | прирост промпта | прирост тюна

«Прирост промпта» = (промпт − без промпта), «прирост тюна» = (тюн − промпт).

Пример:
    python scripts/compare_three.py \
        --facts results/facts/base_none.json results/facts/base_facts.json results/facts/tuned.json \
        --style results/style/base_none.json results/style/base_prompt.json results/style/tuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--facts",
        nargs=3,
        metavar=("NONE", "PROMPT", "TUNED"),
        help="три JSON-результата фактов: base_none, base_facts, tuned",
    )
    parser.add_argument(
        "--style",
        nargs=3,
        metavar=("NONE", "PROMPT", "TUNED"),
        help="три JSON-результата стиля: base_none, base_prompt, tuned",
    )
    # Числа «формат» из watch_parser (exact на dev-eval): base_none / base_fewshot / v3.
    parser.add_argument("--format-none", type=float, default=0.0)
    parser.add_argument("--format-prompt", type=float, default=46.7)
    parser.add_argument("--format-tuned", type=float, default=85.0)
    return parser.parse_args()


def _overall(path: str, key: str = "coverage") -> float:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    value = data.get("overall", {}).get(key)
    return float(value) if value is not None else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta(prompt: float, none: float) -> str:
    return f"{(prompt - none) * 100:+.1f} п.п."


def main() -> None:
    args = parse_args()

    rows: list[tuple[str, str, float, float, float]] = []

    # Формат — числа из watch_parser (exact, %; приводим к долям).
    rows.append(
        (
            "Формат (JSON для /watch)",
            "exact",
            args.format_none / 100,
            args.format_prompt / 100,
            args.format_tuned / 100,
        )
    )

    if args.facts:
        rows.append(
            (
                "Фактические знания",
                "покрытие фактов",
                _overall(args.facts[0], "coverage"),
                _overall(args.facts[1], "coverage"),
                _overall(args.facts[2], "coverage"),
            )
        )

    if args.style:
        rows.append(
            (
                "Стиль и тон",
                "покрытие критериев",
                _overall(args.style[0], "coverage"),
                _overall(args.style[1], "coverage"),
                _overall(args.style[2], "coverage"),
            )
        )

    header = (
        "| Тип задачи | Метрика | База без промпта | База + промпт | Дообученная | Прирост промпта | Прирост тюна |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for label, metric, none, prompt, tuned in rows:
        lines.append(
            f"| {label} | {metric} | {_pct(none)} | {_pct(prompt)} | {_pct(tuned)} | "
            f"{_delta(prompt, none)} | {_delta(tuned, prompt)} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
