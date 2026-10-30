"""Консольное сравнение моделей по трём типам задач: формат, стиль, факты.

Печатает проверочные вопросы и ответы двух моделей рядом — как
``watch_parser/scripts/compare_ollama.py`` — и выносит вердикт по каждому:

- **ФОРМАТ** — база + few-shot vs ``qwen25-watch-v3`` (свободный текст → JSON
  ``/watch``), метрика — точное совпадение JSON;
- **СТИЛЬ** — база + ролевой промпт vs ``qwen25-roleplay``, метрика — краткость
  (или покрытие критериев с ``--judge``);
- **ФАКТЫ** — база + факт в промпте vs ``qwen25-facts-v1``, метрика — наличие
  ключевых значений (или покрытие фактов с ``--judge``).

По умолчанию — компактная выборка: до ``--per-group`` (2) вопросов на каждую
категорию (формат) / вид (стиль) / вид+режим (факты). ``--all`` — весь набор,
``--limit N`` — жёстко первые N.

Примеры:
    python scripts/compare_console.py
    python scripts/compare_console.py --task facts --all
    python scripts/compare_console.py --task style --judge
    python scripts/compare_console.py --only-diff --max-chars 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import judge
import ollama_chat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WATCH = ROOT.parent / "watch_parser"
OLLAMA_URL = "http://localhost:11434"

DEFAULT_BASE = "qwen2.5:1.5b-instruct-fp16"
DEFAULT_FORMAT = "qwen25-watch-v3"
DEFAULT_STYLE = "qwen25-roleplay"
DEFAULT_FACTS = "qwen25-facts-v1"

_SENT_END = re.compile(r"[.!?…]+")


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #
def list_models(base_url: str) -> set[str]:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=30) as response:
        names = {item["name"] for item in json.load(response)["models"]}
    return {name.removesuffix(":latest") for name in names}


def short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1] + "…"


def is_short(answer: str) -> bool:
    return len(_SENT_END.findall(answer)) <= 2 and len(answer.split()) <= 45


def has_all_keys(answer: str, keys: list[str]) -> bool:
    lowered = answer.lower()
    return all(key.lower() in lowered for key in keys)


def verdict(base_score: float, tuned_score: float, eps: float = 1e-9) -> str:
    if tuned_score > base_score + eps:
        return "лучше (тюн)"
    if base_score > tuned_score + eps:
        return "хуже (тюн)"
    return "так же"


def pick(items: list, group, args: argparse.Namespace) -> list:
    """Компактный отбор: ``--all`` — всё, ``--limit N`` — первые N, иначе до
    ``--per-group`` вопросов на каждую группу (категорию/вид)."""
    if args.all:
        return items
    if args.limit:
        return items[: args.limit]
    picked: list = []
    seen: dict = {}
    for item in items:
        key = group(item)
        if seen.get(key, 0) < args.per_group:
            picked.append(item)
            seen[key] = seen.get(key, 0) + 1
    return picked


def _print_item(
    index: int,
    total: int,
    label: str,
    system: str,
    user: str,
    reference: str,
    base_label: str,
    tuned_label: str,
    base_answer: str,
    tuned_answer: str,
    base_seconds: float,
    tuned_seconds: float,
    item_verdict: str,
    max_chars: int,
    show_system: bool,
) -> None:
    print(f"\n{'=' * 78}")
    print(f"[{index}/{total}] {label}")
    if system:
        shown = system if show_system else short(system, 160)
        print(f"SYSTEM: {shown}")
    print(f"USER:   {user}")
    print(f"ЭТАЛОН: {reference}")
    print()
    print(f"--- {base_label} ({base_seconds:.1f} с) ---")
    print(short(base_answer, max_chars))
    print()
    print(f"--- {tuned_label} ({tuned_seconds:.1f} с) ---")
    print(short(tuned_answer, max_chars))
    print(f"\nвердикт: {item_verdict}")


def _judge_score(
    enabled: bool, question: str, facts: list[str], answer: str, provider: str, model: str | None
) -> float | None:
    if not enabled:
        return None
    result = judge.judge_coverage(question, facts, answer, provider=provider, model=model)
    return result["coverage"] if result is not None else None


# --------------------------------------------------------------------------- #
# Три типа задач
# --------------------------------------------------------------------------- #
def run_format(args: argparse.Namespace, models: set[str]) -> dict:
    """Формат: JSON /watch. Данные и промпт — из watch_parser."""
    if str(WATCH) not in sys.path:
        sys.path.insert(0, str(WATCH / "scripts"))
    try:
        import metrics  # noqa: PLC0415 — watch_parser/scripts/metrics.py
        import spec  # noqa: PLC0415 — watch_parser/scripts/spec.py
    except ImportError as exc:  # pragma: no cover
        print(f"ФОРМАТ: не удалось импортировать watch_parser ({exc}) — пропуск.", file=sys.stderr)
        return {}

    rows = [
        json.loads(line)
        for line in (WATCH / "data" / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = pick(rows, lambda row: row["meta"]["category"], args)

    def messages(user: str) -> list[dict]:
        msgs = [{"role": "system", "content": spec.SYSTEM_PROMPT}]
        for example_user, example_answer in spec.FEWSHOT:
            msgs.append({"role": "user", "content": example_user})
            msgs.append({"role": "assistant", "content": spec.canonical_json(example_answer)})
        msgs.append({"role": "user", "content": user})
        return msgs

    records = []
    started = time.time()
    for index, row in enumerate(rows, 1):
        meta = row["meta"]
        msgs = messages(meta["user"])
        base_answer, base_seconds = _timed(args, args.base, msgs, 128)
        tuned_answer, tuned_seconds = _timed(args, args.format_model, msgs, 128)
        base_exact = metrics.score_row(meta["expected"], base_answer)["exact"]
        tuned_exact = metrics.score_row(meta["expected"], tuned_answer)["exact"]
        item_verdict = verdict(float(base_exact), float(tuned_exact))
        _print_item(
            index,
            len(rows),
            f"ФОРМАТ · {meta['category']}",
            msgs[0]["content"],
            meta["user"],
            json.dumps(meta["expected"], ensure_ascii=False),
            f"БАЗА+PROMPT: {args.base}",
            f"ТЮН: {args.format_model}",
            base_answer,
            tuned_answer,
            base_seconds,
            tuned_seconds,
            item_verdict,
            args.max_chars,
            args.show_system,
        )
        if args.only_diff and base_answer == tuned_answer:
            print("(ответы совпали)")
        records.append(
            {
                "task": "format",
                "n": index,
                "category": meta["category"],
                "base_exact": base_exact,
                "tuned_exact": tuned_exact,
                "verdict": item_verdict,
            }
        )
    return _summary("ФОРМАТ", records, "base_exact", "tuned_exact", started)


def run_style(args: argparse.Namespace, models: set[str]) -> dict:
    """Стиль: роль смотрителя маяка. База + ролевой промпт vs roleplay-тюн."""
    payload = json.loads((DATA / "style_eval.json").read_text(encoding="utf-8"))
    style_spec = payload["style_spec"]
    items = pick(payload["items"], lambda item: item["kind"], args)

    records = []
    started = time.time()
    for index, item in enumerate(items, 1):
        msgs = [{"role": "system", "content": style_spec}, {"role": "user", "content": item["prompt"]}]
        base_answer, base_seconds = _timed(args, args.base, msgs, 256)
        tuned_answer, tuned_seconds = _timed(args, args.style_model, msgs, 256)
        base_judge = _judge_score(
            args.judge, item["prompt"], item["criteria"], base_answer, args.judge_provider, args.judge_model
        )
        tuned_judge = _judge_score(
            args.judge, item["prompt"], item["criteria"], tuned_answer, args.judge_provider, args.judge_model
        )
        base_score = base_judge if base_judge is not None else float(is_short(base_answer))
        tuned_score = tuned_judge if tuned_judge is not None else float(is_short(tuned_answer))
        item_verdict = verdict(base_score, tuned_score)
        reference = "; ".join(item["criteria"])
        _print_item(
            index,
            len(items),
            f"СТИЛЬ · {item['kind']}",
            style_spec,
            item["prompt"],
            reference,
            f"БАЗА+РОЛЬ: {args.base}",
            f"ТЮН: {args.style_model}",
            base_answer,
            tuned_answer,
            base_seconds,
            tuned_seconds,
            item_verdict,
            args.max_chars,
            args.show_system,
        )
        print(
            f"краткость: база={is_short(base_answer)}, тюн={is_short(tuned_answer)}"
            + (f" · покрытие: база={base_judge:.2f}, тюн={tuned_judge:.2f}" if args.judge else "")
        )
        records.append(
            {
                "task": "style",
                "n": index,
                "kind": item["kind"],
                "base_score": base_score,
                "tuned_score": tuned_score,
                "verdict": item_verdict,
            }
        )
    return _summary("СТИЛЬ", records, "base_score", "tuned_score", started)


def run_facts(args: argparse.Namespace, models: set[str]) -> dict:
    """Факты: база + факт в промпте vs facts-тюн без факта в промпте."""
    payload = json.loads((DATA / "eval_facts.json").read_text(encoding="utf-8"))
    system_prompt = payload["system_prompt"]
    items = pick(payload["items"], lambda item: (item["kind"], item["mode"]), args)

    records = []
    started = time.time()
    for index, item in enumerate(items, 1):
        base_msgs = [
            {"role": "system", "content": system_prompt + "\n\nСправочный факт:\n" + item["fact"]},
            {"role": "user", "content": item["question"]},
        ]
        tuned_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]},
        ]
        base_answer, base_seconds = _timed(args, args.base, base_msgs, 256)
        tuned_answer, tuned_seconds = _timed(args, args.facts_model, tuned_msgs, 256)
        base_keys = float(has_all_keys(base_answer, item["keys"]))
        tuned_keys = float(has_all_keys(tuned_answer, item["keys"]))
        base_judge = _judge_score(
            args.judge, item["question"], item["judge_facts"], base_answer, args.judge_provider, args.judge_model
        )
        tuned_judge = _judge_score(
            args.judge, item["question"], item["judge_facts"], tuned_answer, args.judge_provider, args.judge_model
        )
        base_score = base_judge if base_judge is not None else base_keys
        tuned_score = tuned_judge if tuned_judge is not None else tuned_keys
        item_verdict = verdict(base_score, tuned_score)
        reference = f"{item['answer']}  [keys: {', '.join(item['keys'])}]"
        _print_item(
            index,
            len(items),
            f"ФАКТЫ · {item['kind']}/{item['mode']}",
            base_msgs[0]["content"],
            item["question"],
            reference,
            f"БАЗА+ФАКТ: {args.base}",
            f"ТЮН: {args.facts_model}",
            base_answer,
            tuned_answer,
            base_seconds,
            tuned_seconds,
            item_verdict,
            args.max_chars,
            args.show_system,
        )
        print(
            f"ключи: база={bool(base_keys)}, тюн={bool(tuned_keys)}"
            + (f" · покрытие: база={base_judge:.2f}, тюн={tuned_judge:.2f}" if args.judge else "")
        )
        records.append(
            {
                "task": "facts",
                "n": index,
                "kind": item["kind"],
                "mode": item["mode"],
                "base_score": base_score,
                "tuned_score": tuned_score,
                "verdict": item_verdict,
            }
        )
    return _summary("ФАКТЫ", records, "base_score", "tuned_score", started)


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #
def _timed(args: argparse.Namespace, model: str, messages: list[dict], num_predict: int) -> tuple[str, float]:
    began = time.time()
    try:
        answer = ollama_chat.chat(model, messages, num_predict=num_predict, timeout=args.timeout)
    except urllib.error.URLError as exc:
        print(f"--- ошибка {model}: {exc}", file=sys.stderr)
        answer = ""
    return answer, time.time() - began


def _summary(name: str, records: list[dict], base_key: str, tuned_key: str, started: float) -> dict:
    if not records:
        return {}
    n = len(records)
    base = sum(float(r[base_key]) for r in records) / n
    tuned = sum(float(r[tuned_key]) for r in records) / n
    better = sum(1 for r in records if r["verdict"] == "лучше (тюн)")
    worse = sum(1 for r in records if r["verdict"] == "хуже (тюн)")
    same = sum(1 for r in records if r["verdict"] == "так же")
    print(f"\n{'-' * 78}")
    print(
        f"{name}: база {base:.2f}, тюн {tuned:.2f} | лучше {better}, хуже {worse}, так же {same} ({time.time() - started:.0f} с)"
    )
    return {"task": name, "n": n, "base": base, "tuned": tuned, "better": better, "worse": worse, "same": same}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["format", "style", "facts", "all"], default="all")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--format-model", default=DEFAULT_FORMAT)
    parser.add_argument("--style-model", default=DEFAULT_STYLE)
    parser.add_argument("--facts-model", default=DEFAULT_FACTS)
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--judge", action="store_true", help="считать покрытие критериев/фактов через DeepSeek")
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--all", action="store_true", help="прогнать все вопросы (не компактный дефолт)")
    parser.add_argument(
        "--per-group",
        type=int,
        default=2,
        help="компактный дефолт: сколько вопросов на категорию/вид (формат — категория, стиль — вид, факты — вид+режим)",
    )
    parser.add_argument("--limit", type=int, default=0, help="жёсткий предел: первые N вопросов каждого типа")
    parser.add_argument("--only-diff", action="store_true", help="пометка, если ответы совпали")
    parser.add_argument("--show-system", action="store_true")
    parser.add_argument("--max-chars", type=int, default=0, help="обрезать ответ; 0 — полностью")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        models = list_models(args.url)
    except urllib.error.URLError as exc:
        print(f"Ollama недоступна по {args.url}: {exc}", file=sys.stderr)
        return 1

    needed = {args.base}
    if args.task in ("format", "all"):
        needed.add(args.format_model)
    if args.task in ("style", "all"):
        needed.add(args.style_model)
    if args.task in ("facts", "all"):
        needed.add(args.facts_model)
    missing = sorted(model for model in needed if model.removesuffix(":latest") not in models)
    if missing:
        print("В Ollama нет моделей: " + ", ".join(missing), file=sys.stderr)
        return 1

    summaries = []
    if args.task in ("format", "all"):
        print(f"\n{'#' * 78}\n# ФОРМАТ: JSON /watch\n{'#' * 78}")
        summaries.append(run_format(args, models))
    if args.task in ("style", "all"):
        print(f"\n{'#' * 78}\n# СТИЛЬ: роль и тон\n{'#' * 78}")
        summaries.append(run_style(args, models))
    if args.task in ("facts", "all"):
        print(f"\n{'#' * 78}\n# ФАКТЫ: знания\n{'#' * 78}")
        summaries.append(run_facts(args, models))

    print(f"\n{'=' * 78}\nИТОГИ")
    for summary in summaries:
        if summary:
            print(
                f"{summary['task']}: база {summary['base']:.2f}, тюн {summary['tuned']:.2f} "
                f"(лучше {summary['better']}, хуже {summary['worse']}, так же {summary['same']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
