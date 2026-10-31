"""Проверка ответов учителя и сборка train/valid в двух вариантах.

Задание «Дистилляция»: «собери ответы большой модели для обучения, проверь их
и удали плохие». Проверка здесь двуслойная:

- **форма** — ``spec.validate_output`` (валидный JSON, диапазоны, регистр тикеров);
- **смысл** — ответ совпал с золотым оракулом (``meta.golden``). Оракул доступен
  только потому, что входы порождены слот-генератором; в обучение он не идёт.

Из этого делаются два варианта обучающих данных:

- ``train_distill`` / ``valid_distill`` — оставляем всё, что прошло ФОРМУ. Ученик
  копирует учителя «как есть», включая его редкие смысловые ошибки. Это и есть
  честная дистилляция на задаче с дешёвой проверкой формы.
- ``train_golden`` / ``valid_golden`` — оставляем только ответы, совпавшие со
  смыслом. Это верхняя граница «что даёт строгая чистка».

Запуск:
    python scripts/filter_labels.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_WATCH_SCRIPTS = Path(__file__).resolve().parents[2] / "watch_parser" / "scripts"
if str(_WATCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_WATCH_SCRIPTS))

from metrics import extract_json  # noqa: E402  isort: skip
from spec import SYSTEM_PROMPT, canonical_json, normalize_text, validate_output  # noqa: E402  isort: skip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def messages(user: str, label: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": canonical_json(label)},
    ]


def split_stratified(rows: list[dict], label_key: str) -> tuple[list[dict], list[dict]]:
    """Стратификация 90/10 по (ok?, категория) метки, как в watch_parser v3."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row["meta"][label_key]
        key = "ok:" + row["meta"]["category"] if label.get("ok") else "err:" + label.get("error", "?")
        buckets[key].append(row)

    train: list[dict] = []
    valid: list[dict] = []
    rng = random.Random(42)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_valid = max(1, round(len(bucket) * 0.10)) if len(bucket) > 1 else 0
        valid.extend(bucket[:n_valid])
        train.extend(bucket[n_valid:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(DATA / "corpus.jsonl"))
    parser.add_argument("--teacher", default=str(DATA / "teacher_raw.jsonl"))
    args = parser.parse_args()

    corpus = {normalize_text(row["user"]): row for row in load_jsonl(Path(args.corpus))}
    teacher_rows = load_jsonl(Path(args.teacher))

    distill_rows: list[dict] = []
    golden_rows: list[dict] = []
    format_ok = 0
    semantic_ok = 0
    teacher_mistakes: list[dict] = []

    for row in teacher_rows:
        user = row["user"]
        meta = row["meta"]
        golden = meta["golden"]
        parsed = extract_json(row["teacher"]["content"])
        fmt = parsed is not None and not validate_output(parsed)
        sem = parsed is not None and parsed == golden
        format_ok += int(fmt)
        semantic_ok += int(sem)

        if fmt:
            distill_rows.append(
                {
                    "messages": messages(user, parsed),
                    "meta": {
                        "category": meta["category"],
                        "user": user,
                        "golden": golden,
                        "label": parsed,
                        "label_source": "teacher",
                    },
                }
            )
            if not sem:
                teacher_mistakes.append({"user": user, "golden": golden, "teacher": parsed})
        if sem:
            golden_rows.append(
                {
                    "messages": messages(user, golden),
                    "meta": {
                        "category": meta["category"],
                        "user": user,
                        "golden": golden,
                        "label": golden,
                        "label_source": "golden",
                    },
                }
            )

    distill_train, distill_valid = split_stratified(distill_rows, "label")
    golden_train, golden_valid = split_stratified(golden_rows, "label")

    write_jsonl(DATA / "train_distill.jsonl", distill_train)
    write_jsonl(DATA / "valid_distill.jsonl", distill_valid)
    write_jsonl(DATA / "train_golden.jsonl", golden_train)
    write_jsonl(DATA / "valid_golden.jsonl", golden_valid)

    report = {
        "corpus": len(corpus),
        "labeled": len(teacher_rows),
        "format_ok": format_ok,
        "semantic_ok": semantic_ok,
        "teacher_error_rate": 1 - semantic_ok / len(teacher_rows) if teacher_rows else None,
        "distill": {"train": len(distill_train), "valid": len(distill_valid)},
        "golden": {"train": len(golden_train), "valid": len(golden_valid)},
        "teacher_mistakes": len(teacher_mistakes),
        "by_category": dict(sorted(Counter(row["meta"]["category"] for row in distill_rows).items())),
    }
    (DATA / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n--- примеры ошибок учителя (прошли форму, но не смысл) ---")
    for row in teacher_mistakes[:10]:
        print("  ", row["user"])
        print("    golden :", json.dumps(row["golden"], ensure_ascii=False))
        print("    teacher:", json.dumps(row["teacher"], ensure_ascii=False))


if __name__ == "__main__":
    main()
