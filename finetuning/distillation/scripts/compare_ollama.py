"""Живое сравнение вариантов дистилляции в Ollama прямо в консоли.

Берёт проверочные вопросы (``eval``/``holdout``), задаёт каждый моделям с
одинаковым системным промптом и жадной генерацией и печатает:

- ответ каждой модели на вопрос;
- качество по каждому вопросу (``exact`` / ``schema_ok``);
- скорость (латентность, токенов/с) и стоимость запроса в $;
- итоговую таблицу «качество / скорость / стоимость» по вариантам;
- «зоны замены»: где ученик уже заменяет учителя, где ещё ошибается.

Задание просит три варианта — маленькую до, после обучения и большую. Это
``base,distilled,teacher``; ``golden`` (строгая чистка) — бонусный четвёртый.
Оставьте ровно три через ``--variants base,distilled,teacher``.

Скорость меряется «проходами по моделям»: каждая модель прогоняет весь набор
подряд (как ``eval_three.py``), поэтому 7b не перезагружается между вопросами и
цифры сопоставимы с ``results/compare.md``. После сбора ответы печатаются по
вопросам.

Примеры:
    python scripts/compare_ollama.py --limit 5
    python scripts/compare_ollama.py --variants base,distilled,teacher --limit 10
    python scripts/compare_ollama.py --category invalid --only-diff
    python scripts/compare_ollama.py --splits holdout --max-chars 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_WATCH_SCRIPTS = Path(__file__).resolve().parents[2] / "watch_parser" / "scripts"
if str(_WATCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_WATCH_SCRIPTS))

from metrics import score_row  # noqa: E402  isort: skip
from ollama_client import chat  # noqa: E402  isort: skip
from pricing import MODEL_PRICE_CLASS, USD_PER_1M, cost_usd  # noqa: E402  isort: skip
from spec import FEWSHOT, SYSTEM_PROMPT, canonical_json  # noqa: E402  isort: skip

ROOT = Path(__file__).resolve().parents[1]
WATCH_DATA = Path(__file__).resolve().parents[2] / "watch_parser" / "data"
RESULTS = ROOT / "results"

MODELS = {
    "base": "qwen2.5:1.5b-instruct-fp16",
    "distilled": "qwen25-distill-v1",
    "golden": "qwen25-golden-v1",
    "teacher": "qwen2.5:7b",
}
LABELS = {"base": "база", "distilled": "distilled", "golden": "golden", "teacher": "учитель"}
DEFAULT_VARIANTS = "base,distilled,golden,teacher"
DEFAULT_OUT = RESULTS / "compare_live.json"
DEFAULT_URL = "http://localhost:11434"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_messages(user: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_user, example_answer in FEWSHOT:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": canonical_json(example_answer)})
    messages.append({"role": "user", "content": user})
    return messages


def short(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1] + "…"


def fmt_money(value: float) -> str:
    return f"${value:.6f}"


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def list_models(base_url: str) -> set[str]:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=30) as response:
        names = {item["name"] for item in json.load(response)["models"]}
    return {name.removesuffix(":latest") for name in names}


def ask(url: str, model: str, messages: list[dict], timeout: int) -> dict:
    """Один запрос: возвращает ответ, время генерации, токены и стоимость."""
    began = time.time()
    try:
        reply = chat(model, messages, url=f"{url}/api/chat", timeout=timeout)
        answer = reply["content"]
        total_s = (reply.get("total_duration") or 0) / 1e9
        load_s = (reply.get("load_duration") or 0) / 1e9
        # Загрузка модели — разовая стоимость, не время запроса.
        latency = max(total_s - load_s, 0.0)
        eval_s = (reply.get("eval_duration") or 0) / 1e9
        prompt_tokens = reply.get("prompt_eval_count") or 0
        output_tokens = reply.get("eval_count") or 0
    except urllib.error.URLError as exc:
        print(f"  ошибка {model}: {exc}", file=sys.stderr)
        answer, latency, eval_s, prompt_tokens, output_tokens = "", time.time() - began, 0.0, 0, 0

    return {
        "output": answer,
        "latency_s": latency,
        "eval_s": eval_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="base,distilled,golden,teacher")
    parser.add_argument("--splits", default="eval,holdout", help="eval,holdout")
    parser.add_argument("--input", default=None, help="свой JSONL вместо --splits")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=0, help="первые N запросов; 0 — все")
    parser.add_argument("--only", help="номера запросов через запятую, например 1,3,17")
    parser.add_argument("--category", default=None, help="simple/multi/invalid/...")
    parser.add_argument("--only-diff", action="store_true", help="только вопросы, где ответы различаются")
    parser.add_argument("--show-system", action="store_true", help="печатать system целиком")
    parser.add_argument("--max-chars", type=int, default=0, help="обрезать ответ; 0 — полностью")
    parser.add_argument("--timeout", type=int, default=900, help="таймаут одного запроса, с")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="куда сохранить JSON")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in MODELS]
    if unknown:
        print(f"неизвестные варианты: {unknown}; знаю {list(MODELS)}", file=sys.stderr)
        return 1

    if args.input:
        rows = load_jsonl(Path(args.input))
        for number, row in enumerate(rows, 1):
            row["_n"], row["_split"] = number, Path(args.input).stem
    else:
        rows = []
        for split in (s.strip() for s in args.splits.split(",") if s.strip()):
            for row in load_jsonl(WATCH_DATA / f"{split}.jsonl"):
                rows.append({**row, "_split": split})
        for number, row in enumerate(rows, 1):
            row["_n"] = number

    if args.only:
        wanted = {int(n) for n in args.only.split(",") if n.strip()}
        rows = [row for row in rows if row["_n"] in wanted]
    if args.category:
        rows = [row for row in rows if row["meta"]["category"] == args.category]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Нет запросов под указанный фильтр.", file=sys.stderr)
        return 1

    try:
        available = list_models(args.url)
    except urllib.error.URLError as exc:
        print(f"Ollama недоступна по {args.url}: {exc}", file=sys.stderr)
        return 1
    missing = [MODELS[v] for v in variants if MODELS[v].removesuffix(":latest") not in available]
    if missing:
        print("В Ollama нет моделей: " + ", ".join(missing), file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ сбор
    # Проход по моделям: каждая прогоняет набор подряд, поэтому скорость честная.
    collected: dict[str, list[dict]] = {v: [] for v in variants}
    started = time.time()
    for variant in variants:
        model = MODELS[variant]
        print(f"== {LABELS[variant]} ({model}): {len(rows)} запросов ==", flush=True)
        for index, row in enumerate(rows, 1):
            reply = ask(args.url, model, build_messages(row["meta"]["user"]), args.timeout)
            metrics = score_row(row["meta"]["expected"], reply["output"])
            eval_s = reply["eval_s"]
            tps = reply["output_tokens"] / eval_s if eval_s > 0 else None
            collected[variant].append(
                {
                    "output": reply["output"],
                    "exact": bool(metrics["exact"]),
                    "schema_ok": bool(metrics["schema_ok"]),
                    "error_acc": metrics.get("error_acc"),
                    "latency_s": round(reply["latency_s"], 3),
                    "tokens_per_sec": round(tps, 2) if tps is not None else None,
                    "prompt_tokens": reply["prompt_tokens"],
                    "output_tokens": reply["output_tokens"],
                    "cost_usd": cost_usd(model, reply["prompt_tokens"], reply["output_tokens"]),
                }
            )
            if index % 10 == 0 or index == len(rows):
                print(f"  {index}/{len(rows)} ({round(time.time() - started)} с)", flush=True)
    print(f"\nсбор завершён за {round(time.time() - started)} с. Печатаю сравнение по вопросам.\n")

    # --------------------------------------------------------------- вывод
    out: list[dict] = []
    for index, row in enumerate(rows, 1):
        meta = row["meta"]
        user = meta["user"]
        expected = meta["expected"]
        items = {variant: collected[variant][index - 1] for variant in variants}

        out.append(
            {
                "n": row["_n"],
                "split": row["_split"],
                "category": meta["category"],
                "user": user,
                "expected": expected,
                **{variant: items[variant] for variant in variants},
            }
        )

        if args.only_diff and len({item["output"] for item in items.values()}) == 1:
            continue

        system = SYSTEM_PROMPT
        print(f"\n{'=' * 90}")
        print(f"[{index}/{len(rows)}] {row['_split']} №{row['_n']} · {meta['category']}")
        print(f"SYSTEM: {system if args.show_system else short(system, 160)}")
        print(f"USER:   {user}")
        print(f"ЭТАЛОН: {json.dumps(expected, ensure_ascii=False)}")

        for variant in variants:
            item = items[variant]
            mark = "✔" if item["exact"] else "✘"
            print(
                f"--- {LABELS[variant]}: {MODELS[variant]} — {item['latency_s']:.2f} с, "
                f"{item['output_tokens']} ток, {fmt_money(item['cost_usd'])}, exact {mark} ---"
            )
            print(short(item["output"], args.max_chars) if item["output"] else "(пусто)")

        teacher_ok = items.get("teacher", {}).get("exact")
        verdicts = []
        for variant in variants:
            ok = items[variant]["exact"]
            extra = ""
            # Зону «замена / лучше учителя / ошибка» показываем только для учеников после тюна.
            if teacher_ok is not None and variant in ("distilled", "golden"):
                if ok and teacher_ok:
                    extra = " (замена)"
                elif ok and not teacher_ok:
                    extra = " (лучше учителя)"
                elif not ok and teacher_ok:
                    extra = " (ошибка)"
            verdicts.append(f"{LABELS[variant]} {'✔' if ok else '✘'}{extra}")
        print("вердикт: " + " | ".join(verdicts))

    # ------------------------------------------------------------------ итог
    total = len(rows)
    stats = {}
    for variant in variants:
        items = collected[variant]
        expected_errors = [row for row, item in zip(rows, items, strict=True) if not row["meta"]["expected"].get("ok")]
        errors_ok = sum(1 for row, item in zip(rows, items, strict=True) if not row["meta"]["expected"].get("ok") and item["error_acc"])
        stats[variant] = {
            "exact": sum(item["exact"] for item in items) / total,
            "schema_ok": sum(item["schema_ok"] for item in items) / total,
            "error_acc": errors_ok / len(expected_errors) if expected_errors else None,
            "median_latency_s": median([item["latency_s"] for item in items]),
            "mean_tokens_per_sec": (
                sum(item["tokens_per_sec"] for item in items if item["tokens_per_sec"] is not None)
                / len([item for item in items if item["tokens_per_sec"] is not None])
                if any(item["tokens_per_sec"] is not None for item in items)
                else None
            ),
            "cost_usd_per_request": sum(item["cost_usd"] for item in items) / total,
        }

    print(f"\n{'=' * 90}")
    print(f"ИТОГ: {total} запросов за {round(time.time() - started)} с")
    print(
        f"{'вариант':<10} {'модель':<30} {'exact':>6} {'schema':>7} {'error':>6} "
        f"{'медиана':>8} {'ток/с':>7} {'$/запрос':>10}"
    )
    print("-" * 90)
    for variant in variants:
        st = stats[variant]
        err = f"{st['error_acc']:.0%}" if st["error_acc"] is not None else "—"
        tps = f"{st['mean_tokens_per_sec']:.1f}" if st["mean_tokens_per_sec"] is not None else "—"
        print(
            f"{LABELS[variant]:<10} {MODELS[variant]:<30} {st['exact']:>6.0%} {st['schema_ok']:>7.0%} "
            f"{err:>6} {st['median_latency_s']:>8.2f} {tps:>7} {fmt_money(st['cost_usd_per_request']):>10}"
        )

    if "teacher" in variants:
        print("\nЗоны (ученик vs учитель):")
        for variant in variants:
            if variant == "teacher":
                continue
            replace = student_err = both_err = better = 0
            for row in out:
                s_ok, t_ok = row[variant]["exact"], row["teacher"]["exact"]
                if s_ok and t_ok:
                    replace += 1
                elif not s_ok and t_ok:
                    student_err += 1
                elif not s_ok and not t_ok:
                    both_err += 1
                else:
                    better += 1
            print(
                f"  {LABELS[variant]:<10} заменяет {replace}/{total} ({replace / total:.0%}) · "
                f"ещё ошибается {student_err}/{total} ({student_err / total:.0%}) · "
                f"общая ошибка {both_err}/{total} ({both_err / total:.0%}) · "
                f"ученик лучше {better}/{total}"
            )

        print("\nЭкономия (на запрос):")
        teacher = stats["teacher"]
        for variant in variants:
            if variant == "teacher":
                continue
            st = stats[variant]
            speedup = teacher["median_latency_s"] / st["median_latency_s"] if st["median_latency_s"] else 1.0
            cheaper = st["cost_usd_per_request"] / teacher["cost_usd_per_request"] if teacher["cost_usd_per_request"] else 1.0
            out_ratio = (
                USD_PER_1M[MODEL_PRICE_CLASS[MODELS["teacher"]]]["output"]
                / USD_PER_1M[MODEL_PRICE_CLASS[MODELS[variant]]]["output"]
            )
            print(
                f"  {LABELS[variant]:<10} vs учитель: в {speedup:.1f}× быстрее, "
                f"стоимость {cheaper:.0%} от учителя (выходные токены {out_ratio:.1f}× дешевле)"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "variants": variants,
                "models": {v: MODELS[v] for v in variants},
                "summary": stats,
                "results": out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
