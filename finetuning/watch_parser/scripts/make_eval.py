"""Eval-набор из 60 запросов, написанных вручную.

Он создаётся ДО обучения и не пересекается с train/valid. Формулировки
намеренно отличаются от шаблонов build_dataset.py: другие слова, порядок
ролей, падежи, исправления и «плохие» запросы. Это проверка обобщения,
а не запоминания шаблонов.
"""

from __future__ import annotations

import json
from pathlib import Path

from spec import canonical_json, error, example_messages, success

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# (категория, запрос, эталон)
CASES: list[tuple[str, str, dict]] = [
    # --- простые и составные, формальные периоды (10) ---
    ("simple", "Следи за SBER каждые 15 минут, сводку присылай раз в час.", success(["SBER"], "15m", "1h")),
    ("simple", "GAZP: проверять цену каждые 30 минут, сводку присылать раз в 90 минут.", success(["GAZP"], "30m", "90m")),
    ("simple", "Хочу получать сводки по LKOH раз в день, а цены проверять каждые 10 минут.", success(["LKOH"], "10m", "1d")),
    ("simple", "Мониторь ROSN с интервалом 45 минут, итоги присылай раз в 3 часа.", success(["ROSN"], "45m", "3h")),
    ("simple", "VTBR нужно проверять каждые 5 минут, а сводку присылать раз в 6 часов.", success(["VTBR"], "5m", "6h")),
    ("simple", "За MOEX наблюдай каждые 2 часа; сводка — раз в 12 часов.", success(["MOEX"], "2h", "12h")),
    ("simple", "Следи за ALRS. Опрос — 90 минут, сводка — 1 день.", success(["ALRS"], "90m", "1d")),
    ("simple", "По CHMF: каждые 15 минут смотри цены, раз в 2 дня присылай отчёт.", success(["CHMF"], "15m", "2d")),
    ("simple", "Настрой /watch на HYDR: проверять каждые 60 минут, отчитываться раз в 7 дней.", success(["HYDR"], "60m", "7d")),
    ("simple", "Следи за PHOR каждые 2 часа, сводку присылай раз в 3 дня.", success(["PHOR"], "2h", "3d")),

    # --- несколько тикеров (10) ---
    ("multi", "Следи за SBER и GAZP каждые 15 минут, сводку раз в час.", success(["SBER", "GAZP"], "15m", "1h")),
    ("multi", "Мониторь LKOH, ROSN и VTBR каждые 30 минут, отчёт раз в 2 часа.", success(["LKOH", "ROSN", "VTBR"], "30m", "2h")),
    ("multi", "Добавь в отслеживание: Сбер, Газпром, Лукойл. Проверка каждые 10 минут, сводка раз в 3 часа.", success(["SBER", "GAZP", "LKOH"], "10m", "3h")),
    ("multi", "По SBER, GAZP и MOEX: опрос каждые 5 минут, сводка раз в 6 часов.", success(["SBER", "GAZP", "MOEX"], "5m", "6h")),
    ("multi", "Следи за ALRS и CHMF каждые 2 часа, итоги раз в 12 часов.", success(["ALRS", "CHMF"], "2h", "12h")),
    ("multi", "Наблюдай за HYDR, PHOR, SIBN и SNGS: проверка 45 минут, сводка 1 день.", success(["HYDR", "PHOR", "SIBN", "SNGS"], "45m", "1d")),
    ("multi", "SBER, SBER и GAZP — следи каждые 15 минут, сводку раз в час.", success(["SBER", "GAZP"], "15m", "1h")),
    ("multi", "Следи за TATN и MGNT каждые 30 минут, сводку раз в 90 минут.", success(["TATN", "MGNT"], "30m", "90m")),
    ("multi", "MOEX + VTBR: опрос 10 минут, отчёт 1 час.", success(["MOEX", "VTBR"], "10m", "1h")),
    ("multi", "За NVTK, PLZL, YDEX, AFLT и IRAO следи каждые 15 минут, сводка раз в 3 дня.", success(["NVTK", "PLZL", "YDEX", "AFLT", "IRAO"], "15m", "3d")),

    # --- разговорные периоды и падежи (15) ---
    ("colloquial", "Следи за Сбером каждые четверть часа, сводку присылай раз в час.", success(["SBER"], "15m", "1h")),
    ("colloquial", "Газпром проверяй каждые полчаса, отчёт раз в два часа.", success(["GAZP"], "30m", "2h")),
    ("colloquial", "Лукойл — каждые 45 минут, сводка раз в 60 минут.", success(["LKOH"], "45m", "60m")),
    ("colloquial", "Роснефть мониторь каждые 10 минут, итоги присылай дважды в сутки.", success(["ROSN"], "10m", "12h")),
    ("colloquial", "За ВТБ следи каждые полчаса, сводку — раз в шесть часов.", success(["VTBR"], "30m", "6h")),
    ("colloquial", "Мосбиржу проверяй каждые 3 часа, отчёт раз в 6 часов.", success(["MOEX"], "3h", "6h")),
    ("colloquial", "Алросу и Северсталь смотри каждые 20 минут, сводку раз в трое суток.", success(["ALRS", "CHMF"], "20m", "3d")),
    ("colloquial", "РусГидро, ФосАгро и Сибур — опрос каждые четверть часа, сводка раз в 12 часов.", success(["HYDR", "PHOR", "SIBN"], "15m", "12h")),
    ("colloquial", "Сургутнефтегаз проверяй каждые 30 минут, отчёт раз в два дня.", success(["SNGS"], "30m", "2d")),
    ("colloquial", "Татнефть и Магнит — каждые 90 минут, сводка раз в 7 дней.", success(["TATN", "MGNT"], "90m", "7d")),
    ("colloquial", "МТС мониторь каждые 15 минут, итоги раз в сутки.", success(["MTSS"], "15m", "1d")),
    ("colloquial", "Новатэк — каждые 2 часа, сводка раз в 3 дня.", success(["NVTK"], "2h", "3d")),
    ("colloquial", "Полюс и Яндекс проверяй каждые 30 минут, отчёт раз в 12 часов.", success(["PLZL", "YDEX"], "30m", "12h")),
    ("colloquial", "Аэрофлот — каждые 45 минут, сводку раз в два часа.", success(["AFLT"], "45m", "2h")),
    ("colloquial", "Интер РАО мониторь каждые 5 минут, сводка раз в час.", success(["IRAO"], "5m", "1h")),

    # --- ошибки (10) ---
    ("invalid", "Следи каждые 15 минут, сводку раз в час.", error("missing_tickers")),
    ("invalid", "Следи за SBER и GAZP.", error("missing_intervals")),
    ("invalid", "Следи за SBER, сводку присылай раз в час.", error("missing_poll_interval")),
    ("invalid", "Следи за SBER каждые 15 минут.", error("missing_report_interval")),
    ("invalid", "Следи за SBER каждые 30 секунд, сводку раз в час.", error("bad_interval")),
    ("invalid", "Следи за Ромашкой каждые 15 минут, сводку раз в час.", error("bad_ticker")),
    ("invalid", "Следи за SBER и Ромашкой каждые 15 минут, сводку раз в час.", error("bad_ticker")),
    ("invalid", "Следи за SBER каждые 15 минут, 30 минут и сводку раз в час.", error("ambiguous")),
    ("invalid", "Следи за SBER: опрос 15 минут и 30 минут, сводка раз в час.", error("ambiguous")),
    ("invalid", "Какая погода в Москве?", error("not_watch")),

    # --- исправления и краевые случаи (5) ---
    ("correction", "Следи за SBER каждые 15 минут, сводку раз в час. Нет, не 15, а 30 минут.", success(["SBER"], "30m", "1h")),
    ("correction", "По GAZP: сводка раз в час, опрос каждые 15 минут. Хотя нет, сводку лучше раз в 2 часа.", success(["GAZP"], "15m", "2h")),
    ("correction", "Следи за sber, GAZP и sber каждые 15 минут, сводку раз в час.", success(["SBER", "GAZP"], "15m", "1h")),
    ("correction", "Следи за SBER и SBER каждые 15 минут, сводку раз в час.", success(["SBER"], "15m", "1h")),
    ("correction", "Сводка по LKOH раз в час, опрос каждые 15 минут.", success(["LKOH"], "15m", "1h")),

    # --- обратный порядок ролей (5) ---
    ("role_order", "По ROSN: сводку присылай раз в 2 часа, цены проверяй каждые 30 минут.", success(["ROSN"], "30m", "2h")),
    ("role_order", "Раз в час присылай сводку по VTBR, а каждые 10 минут проверяй цену.", success(["VTBR"], "10m", "1h")),
    ("role_order", "Каждые 20 минут смотри MOEX, раз в 6 часов присылай отчёт.", success(["MOEX"], "20m", "6h")),
    ("role_order", "Отчёт по ALRS раз в день, опрос каждые 45 минут.", success(["ALRS"], "45m", "1d")),
    ("role_order", "Сводка по CHMF раз в 12 часов, проверка каждые 5 минут.", success(["CHMF"], "5m", "12h")),

    # --- вне задачи и пустой ввод (5) ---
    ("out_of_scope", "", error("empty")),
    ("out_of_scope", "   ", error("empty")),
    ("out_of_scope", "Привет, как настроение?", error("not_watch")),
    ("out_of_scope", "Сколько стоит доллар?", error("not_watch")),
    ("out_of_scope", "Расскажи, что такое облигация.", error("not_watch")),
]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, user, expected in CASES:
        rows.append(
            {
                "messages": example_messages(user, expected),
                "meta": {
                    "category": category,
                    "user": user,
                    "expected": expected,
                },
            }
        )
    if len(rows) != 60:
        raise RuntimeError(f"ожидали 60 eval-примеров, получили {len(rows)}")

    path = DATA / "eval.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"записали {len(rows)} примеров в {path}")
    for row in rows[:3]:
        print(row["meta"]["category"], "|", row["meta"]["user"])
        print("  →", canonical_json(row["meta"]["expected"]))


if __name__ == "__main__":
    main()
