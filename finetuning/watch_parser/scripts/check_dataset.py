"""Проверка датасета: схема, дубли, почти-дубли, контаминация eval → train, баланс.

Запуск: python scripts/check_dataset.py
Код возврата 1, если есть ошибки схемы, точные дубли или контаминация eval.
Почти-дубли и перекосы печатаются как предупреждения: шаблонные примеры
могут быть структурно похожи, это нормально, если меняются слоты.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from spec import SYSTEM_PROMPT, TICKERS, canonical_json, normalize_text, validate_output

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"нет файла {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ngrams(text: str, n: int = 5) -> set[str]:
    text = normalize_text(text)
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_rows(name: str, rows: list[dict], errors: list[str]) -> None:
    for i, row in enumerate(rows, 1):
        where = f"{name}:{i}"
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            errors.append(f"{where}: ожидали system/user/assistant")
            continue
        if messages[0]["content"] != SYSTEM_PROMPT:
            errors.append(f"{where}: системный промпт отличается от spec.SYSTEM_PROMPT")
        user = messages[1]["content"]
        if not isinstance(user, str):
            errors.append(f"{where}: user не строка")
        try:
            assistant = json.loads(messages[2]["content"])
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: assistant не JSON ({exc})")
            continue
        problems = validate_output(assistant)
        if problems:
            errors.append(f"{where}: {problems}")
        expected = row.get("meta", {}).get("expected")
        if expected != assistant:
            errors.append(f"{where}: meta.expected не совпадает с assistant")
        if messages[2]["content"] != canonical_json(assistant):
            errors.append(f"{where}: assistant не в каноническом формате")


def exact_duplicates(rows: list[dict], name: str) -> list[tuple[str, str]]:
    seen: dict[str, int] = {}
    dupes: list[tuple[str, str]] = []
    for i, row in enumerate(rows, 1):
        key = normalize_text(row["meta"]["user"])
        if not key:  # пустой ввод — отдельный класс, не дубль
            continue
        if key in seen:
            dupes.append((f"{name}:{seen[key]}", f"{name}:{i}"))
        else:
            seen[key] = i
    return dupes


def near_duplicates(rows: list[dict], name: str, threshold: float = 0.90) -> list[tuple[float, int, int]]:
    grams = [ngrams(row["meta"]["user"]) for row in rows]
    found: list[tuple[float, int, int]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = jaccard(grams[i], grams[j])
            if score >= threshold:
                found.append((score, i + 1, j + 1))
    return sorted(found, reverse=True)


def contamination(train: list[dict], valid: list[dict], eval_rows: list[dict]) -> list[tuple[float, str, str]]:
    eval_grams = [(row["meta"]["user"], ngrams(row["meta"]["user"], 8)) for row in eval_rows]
    found: list[tuple[float, str, str]] = []
    for name, rows in (("train", train), ("valid", valid)):
        for i, row in enumerate(rows, 1):
            grams = ngrams(row["meta"]["user"], 8)
            for user, egrams in eval_grams:
                score = jaccard(grams, egrams)
                if score >= 0.5:
                    found.append((score, f"{name}:{i}", user))
    return sorted(found, reverse=True)


def exact_contamination(train: list[dict], valid: list[dict], eval_rows: list[dict]) -> list[tuple[str, str]]:
    """Точное совпадение нормализованного текста eval и train/valid — это утечка."""
    eval_keys = {normalize_text(row["meta"]["user"]): row["meta"]["user"] for row in eval_rows}
    found: list[tuple[str, str]] = []
    for name, rows in (("train", train), ("valid", valid)):
        for i, row in enumerate(rows, 1):
            key = normalize_text(row["meta"]["user"])
            if key and key in eval_keys:
                found.append((f"{name}:{i}", eval_keys[key]))
    return found


def counts(rows: list[dict]) -> dict:
    by_category = Counter(row["meta"]["category"] for row in rows)
    by_error = Counter(
        row["meta"]["expected"].get("error")
        for row in rows
        if not row["meta"]["expected"].get("ok")
    )
    ticker_counts = Counter()
    poll_units = Counter()
    report_units = Counter()
    for row in rows:
        expected = row["meta"]["expected"]
        if expected.get("ok"):
            ticker_counts.update(expected["secids"])
            poll_units[expected["poll_interval"][-1]] += 1
            report_units[expected["report_interval"][-1]] += 1
    return {
        "by_category": dict(sorted(by_category.items())),
        "by_error": dict(sorted(by_error.items())),
        "tickers": dict(sorted(ticker_counts.items())),
        "poll_units": dict(sorted(poll_units.items())),
        "report_units": dict(sorted(report_units.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DATA / "v1" / "train.jsonl"))
    parser.add_argument("--valid", default=str(DATA / "v1" / "valid.jsonl"))
    parser.add_argument("--eval", default=str(DATA / "eval.jsonl"))
    parser.add_argument("--report", default=str(DATA / "v1" / "check_report.json"))
    args = parser.parse_args()

    train = load(args.train)
    valid = load(args.valid)
    eval_rows = load(args.eval)
    errors: list[str] = []

    check_rows("train", train, errors)
    check_rows("valid", valid, errors)
    check_rows("eval", eval_rows, errors)

    for name, rows in (("train", train), ("valid", valid), ("eval", eval_rows)):
        for a, b in exact_duplicates(rows, name):
            errors.append(f"точный дубль внутри {name}: {a} = {b}")

    print(f"train={len(train)} valid={len(valid)} eval={len(eval_rows)}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for error in errors:
            print(" -", error)
    else:
        print("схема и точные дубли: OK")

    print("\nпочти-дубли внутри train (Jaccard 5-грамм >= 0.90, топ-5):")
    near = near_duplicates(train, "train")
    if not near:
        print("  нет")
    for score, i, j in near[:5]:
        print(f"  {score:.2f}  train:{i} ~ train:{j}")
        print(f"      {train[i - 1]['meta']['user']}")
        print(f"      {train[j - 1]['meta']['user']}")

    print("\nконтаминация eval → train/valid (Jaccard 8-грамм >= 0.50):")
    contam = contamination(train, valid, eval_rows)
    if not contam:
        print("  нет")
    for score, where, user in contam[:10]:
        print(f"  {score:.2f}  {where} ~ eval «{user}»")
    exact_contam = exact_contamination(train, valid, eval_rows)
    print("  точных совпадений:", len(exact_contam))
    for where, user in exact_contam:
        print(f"    {where} = eval «{user}»")
    high_contam = [item for item in contam if item[0] >= 0.85]
    print("  пар с Jaccard >= 0.85:", len(high_contam))

    print("\nбаланс train:")
    print(json.dumps(counts(train), ensure_ascii=False, indent=2))

    print("\nпокрытие тикеров в train (минимум 3):")
    ticker_counts = Counter()
    for row in train:
        if row["meta"]["expected"].get("ok"):
            ticker_counts.update(row["meta"]["expected"]["secids"])
    rare = [t for t in TICKERS if ticker_counts[t] < 3]
    print("  редко встречаются:", rare or "нет")

    print("\n--- 5 случайных train ---")
    for row in random.Random(7).sample(train, 5):
        print(row["meta"]["category"], "|", row["meta"]["user"])
        print("  →", canonical_json(row["meta"]["expected"]))

    print("\n--- 5 случайных eval ---")
    for row in random.Random(7).sample(eval_rows, 5):
        print(row["meta"]["category"], "|", row["meta"]["user"])
        print("  →", canonical_json(row["meta"]["expected"]))

    stats = {
        "train": len(train),
        "valid": len(valid),
        "eval": len(eval_rows),
        "train_balance": counts(train),
        "eval_balance": counts(eval_rows),
        "near_duplicates_train_090": len(near),
        "eval_contamination_050": len(contam),
        "eval_contamination_085": len(high_contam),
        "eval_exact_contamination": len(exact_contam),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors or exact_contam or high_contam:
        sys.exit(1)


if __name__ == "__main__":
    main()
