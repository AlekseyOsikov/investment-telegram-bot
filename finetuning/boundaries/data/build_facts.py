"""Генерация датасетов фактов из ``facts.py``.

Пишет:
- ``train.jsonl`` — обучающие формулировки (2 на факт);
- ``valid.jsonl`` — перефразировки для контроля лосса (1 на факт);
- ``eval_facts.json`` — eval-набор: обучающая формулировка + перефразировка
  на каждый факт, с контекстом для baseline «факты в промпте», чек-листом
  judge и ключами детерминированной проверки.

Проверка контаминации: перефразировки eval не должны дословно совпадать
ни с одной формулировкой train/valid.
"""

from __future__ import annotations

import json
from pathlib import Path

from facts import SYSTEM_PROMPT, all_train, all_valid, eval_items, to_messages

ROOT = Path(__file__).resolve().parent


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def check_contamination() -> int:
    """Возвращает число перефразировок eval, дословно совпадающих с train/valid."""
    train_valid_q = {_normalize(qa["q"]) for qa in all_train() + all_valid()}
    dupes = 0
    for item in eval_items():
        if item["mode"] == "paraphrase" and _normalize(item["question"]) in train_valid_q:
            print(f"ПЕРЕСЕЧЕНИЕ: {item['id']} «{item['question']}»")
            dupes += 1
    return dupes


def main() -> None:
    train_rows = [{"messages": to_messages(qa)} for qa in all_train()]
    valid_rows = [{"messages": to_messages(qa)} for qa in all_valid()]

    write_jsonl(ROOT / "train.jsonl", train_rows)
    write_jsonl(ROOT / "valid.jsonl", valid_rows)

    (ROOT / "eval_facts.json").write_text(
        json.dumps(
            {"system_prompt": SYSTEM_PROMPT, "items": eval_items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dupes = check_contamination()
    counts = {"train": len(train_rows), "valid": len(valid_rows), "eval": len(eval_items())}
    print("сгенерировано:", counts, "| контаминация перефразировок:", dupes)
    if dupes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
