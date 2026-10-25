"""Замер скорости локальных моделей Ollama: загрузка, префилл и генерация (не часть бота).

Промпт набирается из исходников проекта до нужного числа токенов, в конце — задача на код;
начало промпта уникально, чтобы кеш префикса Ollama не искажал замер. Результат — JSON-строки в
stdout и в файл `--out`. Отчёт с таблицами — `docs/local-llm.md`.

Запуск из корня проекта (нужен запущенный Ollama с моделями):
    python scripts/ollama/speed_bench.py short --models qwen3-coder:30b gpt-oss:20b
    python scripts/ollama/speed_bench.py long --models qwen3-coder:30b --out /tmp/long.jsonl

Режимы: `short` — промпты 0,5k / 2k / 4k токенов при `num_ctx` 8192 (типично для чата IDE);
`long` — окна 8k / 16k / 32k, промпт ≈ 75% окна (на CPU 32k может занять десятки минут).
Токены оцениваются по символам (`--chars-per-token`), точное число возвращает Ollama
(`prompt_tokens`); если исходников не хватает, промпт получается короче цели.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import local_llm_client as client
import local_llm_lib as lib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ["qwen2.5-coder:3b", "gpt-oss:20b", "qwen3-coder:30b"]
# (num_ctx, целевое число токенов промпта)
SHORT = [(8192, 500), (8192, 2000), (8192, 4000)]
LONG = [(8192, 6000), (16384, 12000), (32768, 24000)]
TASK = (
    "\n\n---\nЗадача: добавь в price_watch/parse.py функцию, которая принимает строку периода "
    "вроде '15m' или '2h' и возвращает минуты (int) или None. Только код, без пояснений."
)


def source_text() -> str:
    """Исходники `src/` одним текстом: достаточно длинный контекст для любого режима."""
    parts = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT)
        parts.append(f"# файл {rel}\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def measure(url: str, model: str, source: str, num_ctx: int, tokens: int, cpt: float) -> dict:
    """Один запрос `/api/generate` и разбор таймингов."""
    prompt = f"# run {uuid.uuid4()}\n{source[: int(tokens * cpt)]}{TASK}"
    body = {
        "model": model,
        "prompt": prompt,
        "options": {"num_ctx": num_ctx, "temperature": 0.2, "top_p": 0.9, "num_predict": 200},
    }
    think = lib.default_think(model)
    if think is not None:
        body["think"] = think
    started = time.time()
    data = client.generate(url, body)
    return {
        "model": model,
        "num_ctx": num_ctx,
        "target_tokens": tokens,
        "prompt_tokens": data["prompt_eval_count"],
        "load_s": round(data["load_duration"] / 1e9, 1),
        "prefill_s": round(data["prompt_eval_duration"] / 1e9, 1),
        "prefill_tps": lib.rate(data["prompt_eval_count"], data["prompt_eval_duration"]),
        "gen_tokens": data["eval_count"],
        "gen_tps": lib.rate(data["eval_count"], data["eval_duration"]),
        "wall_s": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["short", "long"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--url", default=client.DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=Path("speed_bench.jsonl"))
    parser.add_argument("--chars-per-token", type=float, default=2.3)
    args = parser.parse_args()

    source = source_text()
    grid = SHORT if args.mode == "short" else LONG
    with args.out.open("a", encoding="utf-8") as out:
        for model in args.models:
            for num_ctx, tokens in grid:
                try:
                    row = measure(args.url, model, source, num_ctx, tokens, args.chars_per_token)
                except Exception as exc:  # noqa: BLE001 — таймаут/сбой одной точки не рушит сетку
                    row = {"model": model, "num_ctx": num_ctx, "error": repr(exc)}
                line = json.dumps(row, ensure_ascii=False)
                print(line, flush=True)
                out.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
