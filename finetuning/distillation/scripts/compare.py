"""Сводное сравнение трёх вариантов + анализ «зон замены».

Читает ``results/<split>/<variant>.json`` (от ``eval_three.py`` и ``eval_hf.py``)
и собирает:

- таблицу качество/скорость/стоимость по вариантам (объединённо eval+holdout);
- попарный анализ ошибок ученика против учителя: где ученик уже заменяет учителя,
  где ещё ошибается, где обе ошибаются одинаково (унаследованные ошибки).

Пишет ``results/compare.md`` (для отчёта) и ``results/compare.json`` (машинно).

Запуск:
    python scripts/compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_WATCH_SCRIPTS = Path(__file__).resolve().parents[2] / "watch_parser" / "scripts"
if str(_WATCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_WATCH_SCRIPTS))

from pricing import MODEL_PRICE_CLASS, USD_PER_1M, cost_usd  # noqa: E402  isort: skip
from spec import normalize_text  # noqa: E402  isort: skip

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VARIANTS = ["base", "distilled", "golden", "teacher"]
SPLITS = ["eval", "holdout"]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def combined_rows(payloads: dict[str, dict]) -> list[dict]:
    rows = []
    for split in SPLITS:
        for row in payloads[split]["results"]:
            rows.append({**row, "_split": split})
    return rows


def quality_over(rows: list[dict]) -> dict:
    quality = {"count": len(rows)}
    for key in ("exact", "schema_ok"):
        quality[key] = round(sum(row[key] for row in rows) / len(rows), 3)
    for key in ("ticker_f1", "poll_acc", "report_acc", "error_acc"):
        values = [row[key] for row in rows if key in row]
        quality[key] = round(sum(values) / len(values), 3) if values else None
    return quality


def speed_over(rows: list[dict]) -> dict:
    latencies = [row["latency_s"] for row in rows]
    tps = [row["tokens_per_sec"] for row in rows if row["tokens_per_sec"] is not None]
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    return {
        "requests": len(rows),
        "median_latency_s": round(sorted(latencies)[len(latencies) // 2], 3),
        "mean_tokens_per_sec": round(sum(tps) / len(tps), 2) if tps else None,
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
    }


def cost_over(rows: list[dict], model: str) -> dict:
    total = sum(cost_usd(model, row["prompt_tokens"], row["output_tokens"]) for row in rows)
    price_class = MODEL_PRICE_CLASS.get(model, "qwen2.5-1.5b")
    rates = USD_PER_1M[price_class]
    return {
        "cost_usd_per_request": round(total / len(rows), 6),
        "output_usd_per_1m": rates["output"],
        "input_usd_per_1m": rates["input"],
    }


def pairwise_vs_teacher(student_rows: list[dict], teacher_rows: list[dict]) -> dict:
    """Классифицирует каждый пример: заменяет / ещё ошибается / общая ошибка / ученик лучше."""
    teacher_by_user = {normalize_text(row["user"]): row for row in teacher_rows}
    buckets = {"replace": 0, "student_err": 0, "both_err": 0, "student_better": 0}
    by_category: dict[str, dict] = {}
    for row in student_rows:
        teacher = teacher_by_user.get(normalize_text(row["user"]))
        if teacher is None:
            continue
        s_ok, t_ok = row["exact"], teacher["exact"]
        if s_ok and t_ok:
            label = "replace"
        elif not s_ok and t_ok:
            label = "student_err"
        elif not s_ok and not t_ok:
            label = "both_err"
        else:
            label = "student_better"
        buckets[label] += 1
        cat = row["category"]
        by_category.setdefault(cat, {"count": 0, "replace": 0, "student_err": 0, "both_err": 0, "student_better": 0})
        by_category[cat]["count"] += 1
        by_category[cat][label] += 1
    return {"total": len(student_rows), "buckets": buckets, "by_category": by_category}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    payloads: dict[str, dict[str, dict]] = {}
    for variant in variants:
        per_split = {split: load(Path(args.results) / split / f"{variant}.json") for split in SPLITS}
        if any(v is None for v in per_split.values()):
            continue
        payloads[variant] = per_split

    if "teacher" not in payloads:
        raise SystemExit("нет результатов учителя — сначала прогоните eval_three.py --variants base,teacher")

    # Объединённая таблица
    table = []
    for variant, per_split in payloads.items():
        rows = combined_rows(per_split)
        model = per_split["eval"]["model"]
        table.append(
            {
                "variant": variant,
                "model": model,
                "quality": quality_over(rows),
                "speed": speed_over(rows),
                "cost": cost_over(rows, model),
            }
        )

    teacher_rows = combined_rows(payloads["teacher"])
    analyses = {
        variant: pairwise_vs_teacher(combined_rows(per_split), teacher_rows)
        for variant, per_split in payloads.items()
        if variant != "teacher"
    }

    compare = {"variants": table, "pairwise_vs_teacher": analyses}
    (Path(args.results) / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _render_markdown(table, analyses)
    (Path(args.results) / "compare.md").write_text(md, encoding="utf-8")
    print(md)


def _render_markdown(table: list[dict], analyses: dict) -> str:
    lines: list[str] = ["# Дистилляция: сравнение вариантов\n"]
    lines.append("Объединённый тест: eval (60) + holdout (30) = 90 примеров. Жадная генерация.\n")
    lines.append("| Вариант | exact | schema | error_acc | медиана, с | ток/с | $/запрос |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in table:
        q, s, c = row["quality"], row["speed"], row["cost"]
        lines.append(
            f"| {row['variant']} ({row['model']}) | {q['exact']:.0%} | {q['schema_ok']:.0%} | "
            f"{_pct(q['error_acc'])} | {s['median_latency_s']} | {s['mean_tokens_per_sec']} | "
            f"{c['cost_usd_per_request']} |"
        )
    lines.append("")
    lines.append("Стоимость по справочным облачным тарифам `scripts/pricing.py` ($/1M токенов, вход/выход):")
    seen_classes: list[str] = []
    for row in table:
        price_class = MODEL_PRICE_CLASS.get(row["model"], "qwen2.5-1.5b")
        if price_class not in seen_classes:
            seen_classes.append(price_class)
            rates = USD_PER_1M[price_class]
            lines.append(f"- {price_class}: ${rates['input']}/1M вход, ${rates['output']}/1M выход")
    lines.append("\n")

    for variant, analysis in analyses.items():
        b = analysis["buckets"]
        lines.append(f"## {variant} против учителя\n")
        lines.append(
            f"- заменяет учителя (оба верно): {b['replace']}/{analysis['total']}"
            f" ({b['replace'] / analysis['total']:.0%})"
        )
        lines.append(
            f"- ещё ошибается (учитель прав, ученик нет): {b['student_err']}/{analysis['total']}"
            f" ({b['student_err'] / analysis['total']:.0%})"
        )
        lines.append(
            f"- общая ошибка (обе неверно): {b['both_err']}/{analysis['total']}"
            f" ({b['both_err'] / analysis['total']:.0%})"
        )
        lines.append(
            f"- ученик лучше (ученик прав, учитель нет): {b['student_better']}/{analysis['total']}"
        )
        lines.append("\n| Категория | всего | заменяет | ошибка ученика | общая | ученик лучше |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cat, block in sorted(analysis["by_category"].items()):
            lines.append(
                f"| {cat} | {block['count']} | {block['replace']} | {block['student_err']} | "
                f"{block['both_err']} | {block['student_better']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "—"


if __name__ == "__main__":
    main()
