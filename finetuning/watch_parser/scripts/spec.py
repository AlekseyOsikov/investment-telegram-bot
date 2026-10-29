"""Спецификация задачи «свободная просьба → JSON для /watch».

Один источник правды для генератора датасета, eval-набора, проверок и обучения:
словарь тикеров, системный промпт, коды ошибок и валидатор ответа.

Задача узкая: на вход — русская фраза, на выход — ровно один JSON-объект.
Успех:
    {"ok":true,"secids":["SBER","GAZP"],"poll_interval":"15m","report_interval":"1h"}
Ошибка:
    {"ok":false,"error":"missing_report_interval"}
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Тикеры и падежные формы
# ---------------------------------------------------------------------------
# Латинский тикер входит во все падежи: так шаблоны можно строить без ошибок
# согласования, а модель видит и русское название, и тикер.
TICKER_FORMS: dict[str, dict[str, list[str]]] = {
    "SBER": {
        "nom": ["Сбер", "Сбербанк", "SBER"],
        "gen": ["Сбера", "Сбербанка", "SBER"],
        "dat": ["Сберу", "Сбербанку", "SBER"],
        "inst": ["Сбером", "Сбербанком", "SBER"],
        "acc": ["Сбер", "Сбербанк", "SBER"],
    },
    "GAZP": {
        "nom": ["Газпром", "GAZP"],
        "gen": ["Газпрома", "GAZP"],
        "dat": ["Газпрому", "GAZP"],
        "inst": ["Газпромом", "GAZP"],
        "acc": ["Газпром", "GAZP"],
    },
    "LKOH": {
        "nom": ["Лукойл", "LKOH"],
        "gen": ["Лукойла", "LKOH"],
        "dat": ["Лукойлу", "LKOH"],
        "inst": ["Лукойлом", "LKOH"],
        "acc": ["Лукойл", "LKOH"],
    },
    "ROSN": {
        "nom": ["Роснефть", "ROSN"],
        "gen": ["Роснефти", "ROSN"],
        "dat": ["Роснефти", "ROSN"],
        "inst": ["Роснефтью", "ROSN"],
        "acc": ["Роснефть", "ROSN"],
    },
    "VTBR": {
        "nom": ["ВТБ", "VTBR"],
        "gen": ["ВТБ", "VTBR"],
        "dat": ["ВТБ", "VTBR"],
        "inst": ["ВТБ", "VTBR"],
        "acc": ["ВТБ", "VTBR"],
    },
    "MOEX": {
        "nom": ["Московская биржа", "Мосбиржа", "MOEX"],
        "gen": ["Московской биржи", "Мосбиржи", "MOEX"],
        "dat": ["Московской бирже", "Мосбирже", "MOEX"],
        "inst": ["Московской биржей", "Мосбиржей", "MOEX"],
        "acc": ["Московскую биржу", "Мосбиржу", "MOEX"],
    },
    "ALRS": {
        "nom": ["Алроса", "ALRS"],
        "gen": ["Алросы", "ALRS"],
        "dat": ["Алросе", "ALRS"],
        "inst": ["Алросой", "ALRS"],
        "acc": ["Алросу", "ALRS"],
    },
    "CHMF": {
        "nom": ["Северсталь", "CHMF"],
        "gen": ["Северстали", "CHMF"],
        "dat": ["Северстали", "CHMF"],
        "inst": ["Северсталью", "CHMF"],
        "acc": ["Северсталь", "CHMF"],
    },
    "HYDR": {
        "nom": ["РусГидро", "HYDR"],
        "gen": ["РусГидро", "HYDR"],
        "dat": ["РусГидро", "HYDR"],
        "inst": ["РусГидро", "HYDR"],
        "acc": ["РусГидро", "HYDR"],
    },
    "PHOR": {
        "nom": ["ФосАгро", "PHOR"],
        "gen": ["ФосАгро", "PHOR"],
        "dat": ["ФосАгро", "PHOR"],
        "inst": ["ФосАгро", "PHOR"],
        "acc": ["ФосАгро", "PHOR"],
    },
    "SIBN": {
        "nom": ["Сибур", "SIBN"],
        "gen": ["Сибура", "SIBN"],
        "dat": ["Сибуру", "SIBN"],
        "inst": ["Сибуром", "SIBN"],
        "acc": ["Сибур", "SIBN"],
    },
    "SNGS": {
        "nom": ["Сургутнефтегаз", "SNGS"],
        "gen": ["Сургутнефтегаза", "SNGS"],
        "dat": ["Сургутнефтегазу", "SNGS"],
        "inst": ["Сургутнефтегазом", "SNGS"],
        "acc": ["Сургутнефтегаз", "SNGS"],
    },
    "TATN": {
        "nom": ["Татнефть", "TATN"],
        "gen": ["Татнефти", "TATN"],
        "dat": ["Татнефти", "TATN"],
        "inst": ["Татнефтью", "TATN"],
        "acc": ["Татнефть", "TATN"],
    },
    "MGNT": {
        "nom": ["Магнит", "MGNT"],
        "gen": ["Магнита", "MGNT"],
        "dat": ["Магниту", "MGNT"],
        "inst": ["Магнитом", "MGNT"],
        "acc": ["Магнит", "MGNT"],
    },
    "MTSS": {
        "nom": ["МТС", "MTSS"],
        "gen": ["МТС", "MTSS"],
        "dat": ["МТС", "MTSS"],
        "inst": ["МТС", "MTSS"],
        "acc": ["МТС", "MTSS"],
    },
    "NVTK": {
        "nom": ["Новатэк", "NVTK"],
        "gen": ["Новатэка", "NVTK"],
        "dat": ["Новатэку", "NVTK"],
        "inst": ["Новатэком", "NVTK"],
        "acc": ["Новатэк", "NVTK"],
    },
    "PLZL": {
        "nom": ["Полюс", "PLZL"],
        "gen": ["Полюса", "PLZL"],
        "dat": ["Полюсу", "PLZL"],
        "inst": ["Полюсом", "PLZL"],
        "acc": ["Полюс", "PLZL"],
    },
    "YDEX": {
        "nom": ["Яндекс", "YDEX"],
        "gen": ["Яндекса", "YDEX"],
        "dat": ["Яндексу", "YDEX"],
        "inst": ["Яндексом", "YDEX"],
        "acc": ["Яндекс", "YDEX"],
    },
    "AFLT": {
        "nom": ["Аэрофлот", "AFLT"],
        "gen": ["Аэрофлота", "AFLT"],
        "dat": ["Аэрофлоту", "AFLT"],
        "inst": ["Аэрофлотом", "AFLT"],
        "acc": ["Аэрофлот", "AFLT"],
    },
    "IRAO": {
        "nom": ["Интер РАО", "IRAO"],
        "gen": ["Интер РАО", "IRAO"],
        "dat": ["Интер РАО", "IRAO"],
        "inst": ["Интер РАО", "IRAO"],
        "acc": ["Интер РАО", "IRAO"],
    },
}

# Как тикер выглядит в системном промпте: канонический + основные названия.
PROMPT_NAMES: dict[str, str] = {
    "SBER": "Сбер/Сбербанк",
    "GAZP": "Газпром",
    "LKOH": "Лукойл",
    "ROSN": "Роснефть",
    "VTBR": "ВТБ",
    "MOEX": "Мосбиржа/Московская биржа",
    "ALRS": "Алроса",
    "CHMF": "Северсталь",
    "HYDR": "РусГидро",
    "PHOR": "ФосАгро",
    "SIBN": "Сибур",
    "SNGS": "Сургутнефтегаз",
    "TATN": "Татнефть",
    "MGNT": "Магнит",
    "MTSS": "МТС",
    "NVTK": "Новатэк",
    "PLZL": "Полюс",
    "YDEX": "Яндекс",
    "AFLT": "Аэрофлот",
    "IRAO": "Интер РАО",
}

ALIAS_TO_TICKER: dict[str, str] = {}
for _ticker, _cases in TICKER_FORMS.items():
    ALIAS_TO_TICKER[_ticker.lower()] = _ticker
    for _forms in _cases.values():
        for _form in _forms:
            ALIAS_TO_TICKER[_form.lower()] = _ticker

TICKERS: list[str] = sorted(TICKER_FORMS)

# ---------------------------------------------------------------------------
# Коды ошибок
# ---------------------------------------------------------------------------
ERROR_CODES: tuple[str, ...] = (
    "empty",
    "not_watch",
    "missing_tickers",
    "missing_intervals",
    "missing_poll_interval",
    "missing_report_interval",
    "bad_interval",
    "bad_ticker",
    "ambiguous",
)

# ---------------------------------------------------------------------------
# Системный промпт
# ---------------------------------------------------------------------------
TICKER_MAP = "; ".join(f"{t}={PROMPT_NAMES[t]}" for t in TICKERS)

SYSTEM_PROMPT = f"""Ты преобразуешь просьбу пользователя в JSON для команды /watch.
Отвечай ТОЛЬКО одним JSON-объектом, без markdown и без пояснений.

Успех:
{{"ok":true,"secids":["SBER","GAZP"],"poll_interval":"15m","report_interval":"1h"}}
Ошибка:
{{"ok":false,"error":"missing_report_interval"}}

Правила:
- secids — тикеры заглавными буквами, без повторов, по алфавиту.
- poll_interval — период опроса цен, report_interval — период сводки.
- Период: целое число и единица m/h/d (минуты/часы/дни): 15m, 2h, 1d.
  Единицы не переводи: 60 минут → 60m, 90 минут → 90m, 24 часа → 24h.
  Допустимо: минуты 1–1440, часы 1–24, дни 1–7.
- Если роли названы явно («опрос/проверка/следи/мониторь» — poll,
  «сводка/отчёт/итоги» — report), используй их.
- Если роли не названы, первый период — poll, второй — report.
- Если период один и роль не названа — это poll, верни missing_report_interval.
  Если период один и назван «сводка/отчёт» — missing_poll_interval.
- Если периодов больше двух или один период назван дважды с разными значениями —
  ambiguous.
- Если тикера нет в списке — bad_ticker. Если тикеров нет — missing_tickers.
- Если период вне диапазона или единица не m/h/d — bad_interval.
- Если просьба не про отслеживание цен — not_watch.
- Пустой текст — empty.
- Исправления («нет, не 15, а 30») — бери исправленное значение.

Тикеры: {TICKER_MAP}
"""

# ---------------------------------------------------------------------------
# Валидатор
# ---------------------------------------------------------------------------
_INTERVAL_RE = re.compile(r"^([0-9]+)([mhd])$")
_RANGES = {"m": (1, 1440), "h": (1, 24), "d": (1, 7)}


def parse_interval(value: Any) -> tuple[int, str] | None:
    """'15m' → (15, 'm'); None, если строка не является периодом."""
    if not isinstance(value, str):
        return None
    match = _INTERVAL_RE.match(value)
    if match is None:
        return None
    number, unit = int(match.group(1)), match.group(2)
    low, high = _RANGES[unit]
    if not low <= number <= high:
        return None
    return number, unit


def validate_output(obj: Any) -> list[str]:
    """Возвращает список проблем; пустой список — ответ корректен."""
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["ответ не JSON-объект"]
    if obj.get("ok") is True:
        if set(obj) != {"ok", "secids", "poll_interval", "report_interval"}:
            problems.append("лишние или недостающие поля успеха")
        secids = obj.get("secids")
        if not isinstance(secids, list) or not secids:
            problems.append("secids не непустой список")
        else:
            if any(not isinstance(t, str) for t in secids):
                problems.append("secids содержит не строку")
            if len(set(secids)) != len(secids):
                problems.append("secids содержит повторы")
            if secids != sorted(secids):
                problems.append("secids не отсортированы")
            for ticker in secids:
                if ticker not in TICKER_FORMS:
                    problems.append(f"неизвестный тикер {ticker}")
                elif ticker != ticker.upper():
                    problems.append(f"тикер не в верхнем регистре {ticker}")
        if parse_interval(obj.get("poll_interval")) is None:
            problems.append("poll_interval не период")
        if parse_interval(obj.get("report_interval")) is None:
            problems.append("report_interval не период")
    elif obj.get("ok") is False:
        if set(obj) != {"ok", "error"}:
            problems.append("лишние или недостающие поля ошибки")
        if obj.get("error") not in ERROR_CODES:
            problems.append(f"неизвестный код ошибки {obj.get('error')!r}")
    else:
        problems.append("поле ok не true/false")
    return problems


def success(tickers: list[str], poll: str, report: str) -> dict[str, Any]:
    return {
        "ok": True,
        "secids": sorted(set(tickers)),
        "poll_interval": poll,
        "report_interval": report,
    }


def error(code: str) -> dict[str, Any]:
    assert code in ERROR_CODES, code
    return {"ok": False, "error": code}


def canonical_json(obj: dict[str, Any]) -> str:
    """Компактный JSON в стабильном порядке полей — эталон для обучения."""
    if obj.get("ok") is True:
        ordered = {
            "ok": True,
            "secids": obj["secids"],
            "poll_interval": obj["poll_interval"],
            "report_interval": obj["report_interval"],
        }
    else:
        ordered = {"ok": False, "error": obj["error"]}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


def example_messages(user: str, assistant: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": canonical_json(assistant)},
    ]


# Два примера для few-shot режима базовой модели (не пересекаются с eval).
FEWSHOT = [
    (
        "Следи за Сбером каждые 10 минут, сводку присылай раз в 3 часа.",
        success(["SBER"], "10m", "3h"),
    ),
    (
        "По GAZP: сводка раз в час, опрос каждые 30 минут.",
        success(["GAZP"], "30m", "1h"),
    ),
]
