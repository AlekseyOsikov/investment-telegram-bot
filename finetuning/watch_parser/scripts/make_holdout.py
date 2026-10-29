"""Свежий holdout из 30 запросов, не участвовавших в train/valid/eval.

Нужен, чтобы проверить, что v3+guard не просто подогнан под 60 dev-запросов.
"""

from __future__ import annotations

import json
from pathlib import Path

from spec import error, example_messages, success

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CASES: list[tuple[str, str, dict]] = [
    ("simple", "Следи за GAZP каждые 20 минут, сводку присылай раз в 4 часа.", success(["GAZP"], "20m", "4h")),
    ("simple", "По LKOH: проверка каждые 15 минут, отчёт раз в 12 часов.", success(["LKOH"], "15m", "12h")),
    ("multi", "Мониторь SBER и ROSN каждые 10 минут, итоги раз в 6 часов.", success(["SBER", "ROSN"], "10m", "6h")),
    ("simple", "Хочу получать сводки по VTBR раз в 2 дня, а цены проверять каждые 30 минут.", success(["VTBR"], "30m", "2d")),
    ("multi", "За MOEX, ALRS и CHMF наблюдай каждые 45 минут, сводка раз в 3 дня.", success(["MOEX", "ALRS", "CHMF"], "45m", "3d")),
    ("multi", "Добавь в отслеживание: Сбер, Газпром. Проверка каждые 5 минут, отчёт раз в час.", success(["SBER", "GAZP"], "5m", "1h")),
    ("simple", "NVTK — каждые 2 часа, сводку присылай раз в 7 дней.", success(["NVTK"], "2h", "7d")),
    ("multi", "По PLZL и YDEX: опрос каждые 90 минут, сводка раз в сутки.", success(["PLZL", "YDEX"], "90m", "1d")),

    ("colloquial", "Следи за Сбером каждые полчаса, сводку присылай раз в два часа.", success(["SBER"], "30m", "2h")),
    ("colloquial", "Газпром проверяй каждые четверть часа, отчёт раз в 12 часов.", success(["GAZP"], "15m", "12h")),
    ("colloquial", "Лукойл мониторь каждые 45 минут, итоги присылай дважды в сутки.", success(["LKOH"], "45m", "12h")),
    ("colloquial", "Роснефть и Татнефть — каждые 10 минут, сводка раз в трое суток.", success(["ROSN", "TATN"], "10m", "3d")),
    ("colloquial", "Мосбиржу проверяй каждые 3 часа, отчёт раз в неделю.", success(["MOEX"], "3h", "7d")),
    ("colloquial", "Сургутнефтегаз — каждые 90 минут, сводку раз в полсуток.", success(["SNGS"], "90m", "12h")),

    ("invalid", "Следи каждые 30 минут, сводку раз в 2 часа.", error("missing_tickers")),
    ("invalid", "Следи за SBER и GAZP каждые 15 минут.", error("missing_report_interval")),
    ("invalid", "По LKOH: сводку присылай раз в 3 часа.", error("missing_poll_interval")),
    ("invalid", "Следи за ROSN.", error("missing_intervals")),
    ("invalid", "Следи за SBER каждые 90 секунд, сводку раз в час.", error("bad_interval")),
    ("invalid", "Следи за Апельсином каждые 20 минут, сводку раз в 2 часа.", error("bad_ticker")),
    ("invalid", "Следи за GAZP и Бананом каждые 10 минут, сводку раз в 3 часа.", error("bad_ticker")),
    ("invalid", "Следи за SBER: опрос 20 минут и 40 минут, сводка раз в 2 часа.", error("ambiguous")),

    ("correction", "Следи за SBER каждые 10 минут, сводку раз в час. Нет, не 10, а 20 минут.", success(["SBER"], "20m", "1h")),
    ("correction", "По GAZP: сводка раз в 3 часа, опрос каждые 30 минут. Хотя нет, сводку лучше раз в 6 часов.", success(["GAZP"], "30m", "6h")),
    ("role_order", "Сводка по LKOH раз в 2 часа, проверка каждые 15 минут.", success(["LKOH"], "15m", "2h")),
    ("role_order", "Отчёт по MOEX раз в сутки, опрос каждые 45 минут.", success(["MOEX"], "45m", "1d")),

    ("out_of_scope", "", error("empty")),
    ("out_of_scope", "   ", error("empty")),
    ("out_of_scope", "Что такое дивидендная доходность?", error("not_watch")),
    ("out_of_scope", "Что выбрать: Сбер или ВТБ?", error("not_watch")),
]


def main() -> None:
    rows = []
    for category, user, expected in CASES:
        rows.append(
            {
                "messages": example_messages(user, expected),
                "meta": {"category": category, "user": user, "expected": expected},
            }
        )
    if len(rows) != 30:
        raise RuntimeError(f"ожидали 30, получили {len(rows)}")
    path = DATA / "holdout.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"записали {len(rows)} в {path}")


if __name__ == "__main__":
    main()
