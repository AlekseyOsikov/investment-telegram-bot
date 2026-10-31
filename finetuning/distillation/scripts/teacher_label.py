"""Учитель (большая модель) размечает неразмеченный корпус.

Для каждого входа из ``data/corpus.jsonl`` вызывает учителя через Ollama
(``/api/chat``) с тем же промптом, что и на тесте: ``SYSTEM_PROMPT`` + 2 few-shot.
Ответы и тайминги пишутся в ``data/teacher_raw.jsonl`` построчно — скрипт
идемпотентен: уже размеченные входы (по нормализованному тексту) пропускаются,
поэтому прерывание/таймаут не теряет работу.

Запуск:
    python scripts/teacher_label.py
    python scripts/teacher_label.py --model qwen2.5:7b --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_WATCH_SCRIPTS = Path(__file__).resolve().parents[2] / "watch_parser" / "scripts"
if str(_WATCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_WATCH_SCRIPTS))

from ollama_client import chat  # noqa: E402  isort: skip
from spec import FEWSHOT, SYSTEM_PROMPT, canonical_json, normalize_text  # noqa: E402  isort: skip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_MODEL = "qwen2.5:7b"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_messages(user: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_user, example_answer in FEWSHOT:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": canonical_json(example_answer)})
    messages.append({"role": "user", "content": user})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default=str(DATA / "corpus.jsonl"))
    parser.add_argument("--out", default=str(DATA / "teacher_raw.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus = load_jsonl(Path(args.corpus))
    out_path = Path(args.out)
    done = {normalize_text(row["user"]) for row in load_jsonl(out_path)}

    todo = [row for row in corpus if normalize_text(row["user"]) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"корпус={len(corpus)} уже размечено={len(done)} осталось={len(todo)} модель={args.model}")

    started = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(todo, 1):
            user = row["user"]
            result = chat(args.model, build_messages(user))
            fh.write(
                json.dumps(
                    {
                        "user": user,
                        "meta": row["meta"],
                        "teacher": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            if index % 10 == 0 or index == len(todo):
                print(
                    f"{index}/{len(todo)} | {user[:40]!r} -> {result['content'][:40]!r} "
                    f"({round(time.time() - started)} с)",
                    flush=True,
                )

    print(f"готово за {round(time.time() - started)} с. выход: {out_path}")


if __name__ == "__main__":
    main()
