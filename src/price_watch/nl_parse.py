"""Разбор свободного текста /watch локальной моделью-парсером.

Самодостаточная копия промпта, словаря тикеров, кодов ошибок, валидатора и guard из
`finetuning/watch_parser/scripts/{spec,guard,metrics}.py` — перенесена сюда, чтобы `src/` не
зависел от учебного подпроекта `finetuning/` (см. finetuning/README.md). Правки там должны
переноситься сюда вручную.

Модель (`qwen25-watch-v3`, Qwen2.5-1.5B-Instruct + LoRA) обучена превращать свободную русскую
просьбу в JSON команды /watch. Качество замерено на нативном эндпоинте Ollama `/api/chat` со
строгими жадными параметрами и поверх ответа модели — детерминированным guard; здесь
воспроизводятся те же параметры, чтобы получить то же качество.

Этот модуль чистых функций не импортирует config и не создаёт клиентов при импорте: имя модели,
адрес сервера и тайм-аут приходят аргументами (вызывающий код берёт их из config).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .parse import WatchRequest

# ---------------------------------------------------------------------------
# Тикеры и падежные формы (перенос из finetuning spec.py)
# ---------------------------------------------------------------------------
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


def canonical_json(obj: dict[str, Any]) -> str:
    """Компактный JSON в стабильном порядке полей — как в обучающем наборе."""
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


# Два примера для few-shot режима (не пересекаются с eval).
FEWSHOT: list[tuple[str, dict[str, Any]]] = [
    (
        "Следи за Сбером каждые 10 минут, сводку присылай раз в 3 часа.",
        success(["SBER"], "10m", "3h"),
    ),
    (
        "По GAZP: сводка раз в час, опрос каждые 30 минут.",
        success(["GAZP"], "30m", "1h"),
    ),
]

# ---------------------------------------------------------------------------
# Валидатор (перенос из finetuning spec.py)
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


# ---------------------------------------------------------------------------
# Guard (перенос из finetuning guard.py)
# ---------------------------------------------------------------------------
WATCH_WORDS = re.compile(
    r"\b(следи|следить|мониторь|мониторить|проверяй|проверять|наблюдай|наблюдать|"
    r"опрос|сводк|отч[её]т|итог)\w*",
    re.IGNORECASE,
)
REPORT_WORDS = re.compile(r"сводк|отч[её]т|итог", re.IGNORECASE)
NOT_WATCH_WORDS = re.compile(
    r"(выбрать|сравни|лучше|что думаешь|дивиденд|прогноз|расскажи|объясни|"
    r"сколько стоит|какие акции|посоветуй)",
    re.IGNORECASE,
)
PERIOD_RE = re.compile(
    r"(?:\d+\s*(?:минут\w*|час\w*|дн\w*|день|дня|дней|сут\w*|недел\w*|месяц\w*|год\w*|секунд\w*)"
    r"|полчас\w*|четверть часа|полсут\w*|двое суток|трое суток|два часа|три часа|"
    r"шесть часов|двенадцать часов|два дня|три дня|семь дней|час|сутки|неделю|день|дня|дней)",
    re.IGNORECASE,
)
CORRECTION_RE = re.compile(
    r"(нет,\s*не|хотя\s+нет|а\s+не\s+\d|лучше\s+(?:раз|сводк))",
    re.IGNORECASE,
)


def find_tickers(text: str) -> set[str]:
    lowered = text.lower().replace("ё", "е")
    found: set[str] = set()
    for alias, ticker in ALIAS_TO_TICKER.items():
        pattern = rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])"
        if re.search(pattern, lowered):
            found.add(ticker)
    return found


def count_periods(text: str) -> int:
    return len(PERIOD_RE.findall(text))


def ticker_mentions(text: str) -> list[str]:
    """Все упоминания тикеров по порядку, включая повторы."""
    lowered = text.lower().replace("ё", "е")
    spans: list[tuple[int, int, str]] = []
    for alias, ticker in ALIAS_TO_TICKER.items():
        pattern = rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])"
        for match in re.finditer(pattern, lowered):
            spans.append((match.start(), match.end(), ticker))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[str] = []
    last_end = -1
    for start, end, ticker in spans:
        if start >= last_end:
            result.append(ticker)
            last_end = end
    return result


def period_value(text: str) -> str | None:
    value = text.lower().replace("ё", "е").strip()
    special = {
        "полчаса": "30m",
        "четверть часа": "15m",
        "полсуток": "12h",
        "двое суток": "2d",
        "трое суток": "3d",
        "два часа": "2h",
        "три часа": "3h",
        "шесть часов": "6h",
        "двенадцать часов": "12h",
        "два дня": "2d",
        "три дня": "3d",
        "семь дней": "7d",
        "час": "1h",
        "сутки": "1d",
        "день": "1d",
        "дня": "1d",
        "дней": "1d",
        "неделю": "7d",
    }
    if value in special:
        return special[value]
    match = re.match(
        r"(\d+)\s*(минут\w*|час\w*|дн\w*|день|дня|дней|сут\w*|недел\w*|месяц\w*|год\w*|секунд\w*)",
        value,
    )
    if not match:
        return None
    number = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("минут"):
        return f"{number}m"
    if unit.startswith("час"):
        return f"{number}h"
    if unit.startswith(("дн", "день", "дня", "дней", "сут")):
        return f"{number}d"
    return None


def period_values(text: str) -> list[str] | None:
    values = [period_value(match) for match in PERIOD_RE.findall(text)]
    if any(value is None for value in values):
        return None
    return values


def duplicate_ticker_success(user: str, output: dict) -> dict:
    """Если ambiguous вызван только дублем тикера, собираем успех по двум периодам."""
    if output.get("error") != "ambiguous":
        return output
    mentions = ticker_mentions(user)
    if len(mentions) < 2 or len(set(mentions)) == len(mentions):
        return output
    if len(REPORT_WORDS.findall(user)) != 1:
        return output
    periods = period_values(user)
    if periods is None or len(periods) != 2:
        return output
    matches = list(PERIOD_RE.finditer(user))
    report_match = REPORT_WORDS.search(user)
    if report_match is None:
        return output
    if report_match.start() <= matches[0].start():
        report, poll = periods[0], periods[1]
    elif report_match.start() <= matches[1].start():
        report, poll = periods[1], periods[0]
    else:
        return output
    return success(sorted(set(mentions)), poll, report)


def unknown_ticker_candidate(text: str) -> str | None:
    """Ищет название после «за»/«и»/запятой, которого нет в словаре тикеров."""
    cleaned = text
    for alias in sorted(ALIAS_TO_TICKER, key=len, reverse=True):
        cleaned = re.sub(
            rf"(?<![A-Za-zА-Яа-я0-9]){re.escape(alias)}(?![A-Za-zА-Яа-я0-9])",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    match = re.search(r"(?:за|и|,)\s+([А-ЯЁA-Z][\w-]*)", cleaned)
    return match.group(1) if match else None


def apply_guard(user: str, output: dict | None) -> dict:
    if not user.strip():
        return error("empty")

    lowered = user.lower().replace("ё", "е")
    tickers = find_tickers(lowered)
    periods = count_periods(lowered)
    has_watch = bool(WATCH_WORDS.search(lowered))

    if unknown_ticker_candidate(user):
        return error("bad_ticker")

    if has_watch and not tickers and periods and "за " not in lowered:
        return error("missing_tickers")
    if tickers and periods == 0 and not has_watch and NOT_WATCH_WORDS.search(lowered):
        return error("not_watch")
    if has_watch and tickers and periods == 0:
        return error("missing_intervals")
    if has_watch and tickers and periods == 1:
        if REPORT_WORDS.search(lowered):
            return error("missing_poll_interval")
        return error("missing_report_interval")
    if periods > 2 and not CORRECTION_RE.search(lowered):
        return error("ambiguous")

    if not isinstance(output, dict):
        return error("empty")

    if output.get("ok") is not True:
        output = duplicate_ticker_success(user, output)

    # Согласованность тикеров: в успешном ответе должны быть ровно те тикеры,
    # которые явно названы в запросе. Это чинит и пропуск («SBER, SBER, GAZP»),
    # и лишний выдуманный тикер («Магнит» → MGNT, а не MTSS).
    if output.get("ok") is True and tickers:
        corrected = sorted(tickers)
        if output.get("secids") != corrected:
            output = {**output, "secids": corrected}

    return output


# ---------------------------------------------------------------------------
# Сборка сообщений и разбор ответа
# ---------------------------------------------------------------------------
def build_messages(user_text: str) -> list[dict[str, str]]:
    """system + два few-shot + вопрос пользователя — как в прогоне eval модели."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in FEWSHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": canonical_json(assistant)})
    messages.append({"role": "user", "content": user_text})
    return messages


def extract_json(text: str) -> dict | None:
    """Вытаскивает JSON-объект из ответа модели; None, если не удалось."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def to_watch_request(obj: dict) -> WatchRequest | str:
    """Ответ модели (после guard) → WatchRequest; иначе строка-код ошибки."""
    if obj.get("ok") is True:
        secids = obj.get("secids")
        poll = obj.get("poll_interval")
        report = obj.get("report_interval")
        if (
            isinstance(secids, list)
            and secids
            and isinstance(poll, str)
            and isinstance(report, str)
        ):
            return WatchRequest(secids=list(secids), poll_interval=poll, report_interval=report)
        return "empty"
    code = obj.get("error")
    return code if code in ERROR_CODES else "empty"


# ---------------------------------------------------------------------------
# Вызов Ollama (нативный /api/chat)
# ---------------------------------------------------------------------------
# OpenAI-совместимый /v1 не пробрасывает repeat_penalty/top_k/num_ctx, а именно эти
# параметры дают замеренное качество — поэтому обращаемся к нативному эндпоинту.
_NATIVE_OPTIONS = {
    "temperature": 0,
    "num_predict": 128,
    "num_ctx": 4096,
    "seed": 42,
    "repeat_penalty": 1.0,
    "top_k": 1,
    "top_p": 1.0,
}


def native_chat_url(base_url: str) -> str:
    """`http://host:11434/v1` → `http://host:11434/api/chat`."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/api/chat"


def _chat(model: str, messages: list[dict], base_url: str, timeout: float) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": dict(_NATIVE_OPTIONS),
    }
    with httpx.Client(trust_env=False) as client:
        response = client.post(native_chat_url(base_url), json=body, timeout=timeout)
    response.raise_for_status()
    return (response.json().get("message", {}).get("content") or "").strip()


def parse_free_text(
    text: str, *, model: str, base_url: str, timeout: float
) -> WatchRequest | str | None:
    """Полный синхронный разбор свободного текста.

    Возвращает `WatchRequest` при успехе, строку-код ошибки при ошибке разбора и `None`,
    если модель недоступна или ответ не разобрать (вызывающий код показывает подсказку).
    """
    try:
        raw = _chat(model, build_messages(text), base_url, timeout)
    except httpx.HTTPError:
        return None
    obj = extract_json(raw)
    if obj is None:
        return None
    return to_watch_request(apply_guard(text, obj))
