"""Сессия с MCP-серверами рыночных данных (mcp-moex, mcp-bybit, atomno-mcp-cbr-rates)
для основного ответа /smart_agent.

В отличие от client.py (диагностический цикл «подключиться → перечислить → закрыть»),
здесь сессия остаётся открытой на весь ВОПРОС пользователя: схемы инструментов нужны
до первого обращения к LLM, а сами вызовы происходят по ходу цикла tool-calls (см.
agents/market_tools.py). Процесс сервера один на вопрос и не живёт между вопросами —
см. design.md изменения add-smart-agent-moex-tools, решение 3.

Источников может быть НЕСКОЛЬКО (сейчас MOEX, Bybit и Банк России) — каждый со своей
готовой командой запуска (`MarketSource.params`), перечнем разрешённых инструментов и
требованием (или его отсутствием) к пометке `read_only_hint` (design.md изменения
add-smart-agent-cbr-tools). Способ запуска у источников РАЗНЫЙ: MOEX/Bybit
устанавливаются из локального каталога проекта (`build_server_params`), Банк России —
из опубликованного пакета через `uvx`, без каталога (`build_uvx_server_params`) —
`MarketSource`/`open_market_tools` не знают, какой из способов использован, только
готовую `StdioServerParameters`. `MarketTools` (одна сессия, один источник) не знает
о существовании других источников и не префиксует свои имена — этим её поведение и
покрывающие её тесты не меняются по сравнению с однo-источниковой версией.
Объединение нескольких источников в один фасад для модели (префиксация имён,
диспетчеризация вызова по префиксу, частичная доступность) — отдельный слой,
`MultiMarketTools`/`open_multi_market_tools()`, ниже.

Граница ответственности такая же, как у client.py/tools_command.py: здесь только
механика MCP, без Telegram и без знания о том, как агент строит контекст. Всё, что
не требует сети (фильтр read-only, схема для OpenAI, разбор аргументов, текст
результата, усечение), — чистые функции, покрытые tests/test_market_tools.py без
запуска процессов.

Два класса неудач разведены намеренно:
- неудача ЗАПУСКА сервера (нет `uv`, нет каталога, тайм-аут рукопожатия, нарушение
  протокола) — исключение из open_market_tools(), его перехватывает вызывающий код и
  переводит вопрос в режим «без инструментов» (для одного источника) или отмечает
  этот источник недоступным, не трогая остальные (для нескольких, см.
  open_multi_market_tools());
- неудача ВЫЗОВА инструмента (неизвестное имя, невалидные аргументы, ошибка сервера,
  обрыв обмена, тайм-аут вызова) — исключением НЕ становится: MarketTools.call()
  возвращает текст для модели, чтобы вопрос не остался без ответа.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Инструменты сервера, которые вообще можно отдать МОДЕЛИ. Это второй рубеж защиты
# после пометки read-only: сервер для ответа smart-агента запускается БЕЗ режима
# расписания (build_server_params ниже), поэтому инструментов watch_* в этой сессии
# нет вовсе, а перечень страхует от ошибки конфигурации и от новых read-only
# инструментов сервера, которые модели пока отдавать не решено. Новый инструмент для
# модели требует добавить его сюда явно — это желаемое трение. См. design.md
# изменения add-price-watch-commands, решение 6. Три последних имени — инструменты
# анализа истории (метрики, сравнение с индексом, сравнение двух бумаг): они чистые
# функции без сети и без побочных эффектов (изменение add-smart-agent-analytics-chain).
MODEL_TOOL_ALLOWLIST = frozenset(
    {
        "search_securities",
        "get_current_price",
        "get_price_history",
        "compute_price_metrics",
        "compare_with_benchmark",
        "compare_securities",
    }
)

# Тот же принцип для второго источника (mcp-bybit, design.md изменения
# add-smart-agent-bybit-tools) — свой перечень, потому что имя поиска у Bybit другое
# (search_symbols, а не search_securities), а остальные пять имён совпадают буквально
# с MOEX и без префикса источника (см. MultiMarketTools ниже) были бы неразличимы.
BYBIT_MODEL_TOOL_ALLOWLIST = frozenset(
    {
        "search_symbols",
        "get_current_price",
        "get_price_history",
        "compute_price_metrics",
        "compare_with_benchmark",
        "compare_securities",
    }
)

# Третий источник — Банк России (пакет PyPI atomno-mcp-cbr-rates, design.md изменения
# add-smart-agent-cbr-tools). Проверено вручную подключением к живому серверу: пять
# инструментов, ни у одного из них НЕТ пометки read_only_hint, хотя по сигнатуре и
# описанию каждый — параметризованное чтение публичных данных cbr.ru без побочных
# эффектов (код валюты, даты, диапазон лет — ничего похожего на запись). Перечень
# разрешённых имён здесь остаётся ОБЯЗАТЕЛЬНЫМ, как и у MOEX/Bybit, — снимается
# только проверка пометки, см. REQUIRE_READ_ONLY_EXCEPTIONS ниже.
CBR_MODEL_TOOL_ALLOWLIST = frozenset(
    {
        "get_rate",
        "history_rates",
        "key_rate",
        "inflation",
        "statistics",
    }
)
CBR_PACKAGE = "atomno-mcp-cbr-rates"

# Идентификаторы источников — используются как префикс имени инструмента для модели
# (см. MultiMarketTools) и как ключ в статусах/конфигурации (agents/smart_agent.py,
# agents/market_tools.py). Разделитель префикса — "__": имена инструментов всех
# серверов используют одиночное подчёркивание (snake_case), поэтому первое "__" в
# имени всегда однозначно отделяет источник от собственного имени инструмента.
SOURCE_MOEX = "moex"
SOURCE_BYBIT = "bybit"
SOURCE_CBR = "cbr"
SOURCE_PREFIX_SEPARATOR = "__"

# Инструмент, результаты которого получают имена-ссылки (r1, r2, ...), и параметры
# инструментов анализа, принимающие от модели ТОЛЬКО такую ссылку. Единственное место,
# где сказано, какие параметры несут историю: по нему строятся схема для модели,
# проверка и подстановка ссылок. См. design.md изменения add-smart-agent-analytics-chain.
HISTORY_TOOL = "get_price_history"
REF_PARAMS: dict[str, tuple[str, ...]] = {
    "compute_price_metrics": ("history",),
    "compare_with_benchmark": ("history", "benchmark_history"),
    "compare_securities": ("history_a", "history_b"),
}

REF_PARAM_DESCRIPTION = (
    'Ссылка на результат get_price_history: имя из пометки «Результат сохранён как rN» '
    '(например, {"ref":"r1"}). Сами данные передавать нельзя.'
)
REF_TOOL_DESCRIPTION_SUFFIX = (
    'В параметры с историей передавай не данные, а ссылку {"ref":"rN"} на результат '
    "get_price_history: имя указано в пометке к результату."
)

# Пометка в конце усечённого результата. Обрезка идёт с головы (там заголовок и
# сводка), а хронологические данные (свечи) лежат в конце — поэтому пометка прямо
# направляет модель к более крупному интервалу или меньшему периоду.
TRUNCATION_NOTE = (
    "\n[РЕЗУЛЬТАТ ОБРЕЗАН: показано только начало. Запроси меньший период или "
    "более крупный interval.]"
)


@dataclass(frozen=True)
class ToolOutcome:
    """Итог одного вызова инструмента — то, что уходит модели сообщением `tool`.

    text — содержимое сообщения. is_error — вызов не дал данных (любая причина);
    transport_failure — сбой обмена с сервером (обрыв, тайм-аут): только он
    заставляет показать пользователю предупреждение о недоступности данных биржи,
    обычная ошибка вроде неизвестного тикера — рабочая ситуация, о которой модель
    сама скажет в ответе.
    """

    text: str
    is_error: bool = False
    transport_failure: bool = False


# --------------------------------------------------------------------------- #
# Чистые функции (без сети) — покрыты tests/test_market_tools.py
# --------------------------------------------------------------------------- #


def is_read_only(tool) -> bool:
    """Инструмент предназначен только для чтения — по аннотации, которую ставит сам
    сервер. Отсутствие аннотации — НЕ read-only: в протоколе инструмент по умолчанию
    считается потенциально изменяющим, а этот фильтр — защита на случай, если в
    сервере когда-нибудь появится пишущий инструмент (design.md, решение 5)."""
    annotations = getattr(tool, "annotations", None)
    return getattr(annotations, "read_only_hint", None) is True


def is_model_tool(
    tool, allowlist: frozenset[str] = MODEL_TOOL_ALLOWLIST, require_read_only: bool = True
) -> bool:
    """Инструмент можно отдать модели: он в явном перечне разрешённых И (если для его
    источника это требуется) помечен сервером как только читающий. `allowlist` по
    умолчанию — перечень MOEX (обратная совместимость для однo-источникового вызова);
    у Bybit/CBR свои перечни, передаваемые явно.

    `require_read_only=False` снимает ТОЛЬКО проверку пометки `read_only_hint` — вхождение
    в `allowlist` остаётся обязательным всегда. Это узкое, явное исключение для
    ОДНОГО заранее решённого источника (сейчас — Банк России, design.md изменения
    add-smart-agent-cbr-tools, решение 2), а не общее правило: значение задаёт
    вызывающий код по идентификатору источника, а не что-либо в самом инструменте —
    его нельзя получить, просто не выставив аннотацию на сервере."""
    if tool.name not in allowlist:
        return False
    return True if not require_read_only else is_read_only(tool)


def build_server_params(directory: str, program: str = "mcp-moex") -> StdioServerParameters:
    """Параметры запуска сервера рыночных данных, устанавливаемого из ЛОКАЛЬНОГО
    каталога проекта (MOEX/Bybit), для ответа smart-агента.

    Команда собирается списком аргументов без shell, поэтому значение directory не
    может внедрить команду. Флага `--watch-db` здесь НЕТ намеренно (и mcp-bybit его
    вообще не поддерживает): в режиме по умолчанию сервер публикует только
    инструменты чтения, а инструменты расписания (принимают chat_id от клиента)
    модель не видит вообще. `program` — имя программы источника (`mcp-moex` по
    умолчанию для обратной совместимости, `mcp-bybit` для второго источника). Для
    источника без локального каталога (Банк России) см. build_uvx_server_params().
    """
    return StdioServerParameters(
        command="uv", args=["run", "--directory", directory, program]
    )


def build_uvx_server_params(package: str) -> StdioServerParameters:
    """Параметры запуска сервера рыночных данных, устанавливаемого и запускаемого
    автоматически из ОПУБЛИКОВАННОГО пакета (Банк России, `uvx <package>`), без
    локального каталога проекта (design.md изменения add-smart-agent-cbr-tools,
    решение 1). `package` — имя пакета, известное системе (константа в коде, см.
    CBR_PACKAGE), а не значение, которое задаёт оператор: единственный операторский
    параметр источника такого типа — булев флаг включения, поэтому здесь даже
    теоретически нет строки, куда можно было бы внедрить произвольную команду."""
    return StdioServerParameters(command="uvx", args=[package])


def describe_launch_failure(exc: BaseException) -> str:
    """Короткая причина, почему сервер источника не запустился, — для
    /smart_agent_show (подробности пишутся в журнал). SDK часто заворачивает исходную
    ошибку в ExceptionGroup (anyio), поэтому разворачиваем её до первой конкретной.
    Проверка по атрибуту `exceptions`, а не по BaseExceptionGroup: тот появился
    только в Python 3.11, а проект заявляет 3.10+ (pyproject.toml)."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    if isinstance(exc, FileNotFoundError):
        return "не найден uv (проверь PATH)"
    if isinstance(exc, TimeoutError):
        return "сервер не ответил за отведённое время"
    return f"сбой запуска ({type(exc).__name__})"


def to_openai_tool(tool) -> dict:
    """Описание инструмента MCP в формате `tools` OpenAI-совместимого API."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


def parse_arguments(raw: str | None) -> tuple[dict | None, str | None]:
    """Разбирает `arguments` вызова от модели: (аргументы, None) либо (None, текст
    ошибки для модели). Пустая строка/None — вызов без аргументов: часть моделей так
    вызывает инструменты, у которых нет обязательных параметров."""
    if raw is None or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, (
            f"Аргументы вызова не разобраны как JSON ({exc.msg}). "
            "Передай аргументы одним JSON-объектом."
        )
    if not isinstance(parsed, dict):
        return None, "Аргументы вызова должны быть JSON-объектом, а не другим типом значения."
    return parsed, None


def result_to_text(result) -> str:
    """Результат вызова инструмента → текст для модели.

    is_error — текст сервера как есть (он уже написан для LLM: что случилось и что
    делать). Иначе — structured_content в КОМПАКТНОЙ сериализации (pretty-print
    сервера в замере был на треть длиннее без единой лишней смысловой единицы), а
    если его нет — склеенные текстовые блоки контента.
    """
    text_blocks = "".join(getattr(block, "text", "") or "" for block in (result.content or []))
    if getattr(result, "is_error", False):
        return text_blocks or "Инструмент сообщил об ошибке без пояснения."

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        try:
            return json.dumps(structured, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            logger.warning("structured_content не сериализуется в JSON — беру текст блоков.")
    return text_blocks or "Инструмент вернул пустой результат."


def truncate_result(text: str, max_chars: int) -> str:
    """Усекает результат до max_chars (вместе с пометкой), сохраняя НАЧАЛО. Пометка
    влезает в предел, а не добавляется сверху: предел — это то, сколько модель
    получит в сообщении."""
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(TRUNCATION_NOTE)
    if budget <= 0:
        return text[:max_chars]
    return text[:budget] + TRUNCATION_NOTE


# --------------------------------------------------------------------------- #
# Ссылки на результаты истории (чистые функции)
#
# Модель называет результат предыдущего шага ({"ref": "r1"}), а данные подставляет
# код: перепечатывая сотни свечей, модель их искажает и тратит на это токены. Серверу
# данные уходят целиком «по значению» и дословно в том виде, в каком он их выдал.
# --------------------------------------------------------------------------- #


def ref_note(name: str) -> str:
    """Пометка в конце сообщения с результатом истории: под каким именем он сохранён."""
    return (
        f"\n[Результат сохранён как {name}: чтобы передать его в инструмент анализа, "
        f'укажи {{"ref":"{name}"}}.]'
    )


def extract_structured(result) -> dict | None:
    """Полный результат вызова как словарь: `structured_content`, а если его нет —
    JSON из текстовых блоков. None — разобрать не удалось (ссылка тогда не выдаётся)."""
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    text_blocks = "".join(getattr(block, "text", "") or "" for block in (result.content or []))
    try:
        parsed = json.loads(text_blocks)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _contains_schema_ref(node) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_contains_schema_ref(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_schema_ref(item) for item in node)
    return False


def with_ref_parameters(schema: dict, params: tuple[str, ...]) -> dict:
    """Копия схемы инструмента, где параметры `params` описаны как ссылка
    `{"ref": "rN"}`, а не как структура истории. `$defs` убирается, если на него больше
    никто не ссылается (иначе в схеме остались бы мёртвые определения свечи и сводки).
    Исходная схема не изменяется."""
    result = copy.deepcopy(schema)
    properties = result.get("properties", {})
    for param in params:
        if param in properties:
            properties[param] = {
                "type": "object",
                "description": REF_PARAM_DESCRIPTION,
                "properties": {
                    "ref": {"type": "string", "description": "Имя результата, например r1."}
                },
                "required": ["ref"],
                "additionalProperties": False,
            }
    if "$defs" in result and not _contains_schema_ref(
        {key: value for key, value in result.items() if key != "$defs"}
    ):
        del result["$defs"]
    return result


def _is_ref_value(value) -> bool:
    return isinstance(value, dict) and set(value) == {"ref"} and isinstance(value["ref"], str)


def _available_refs_hint(store: dict[str, dict]) -> str:
    if not store:
        return "Ссылок пока нет: сначала вызови get_price_history для нужной бумаги."
    return f"Доступные ссылки: {', '.join(store)}."


def resolve_refs(
    tool_name: str, arguments: dict, store: dict[str, dict]
) -> tuple[dict | None, str | None]:
    """Заменяет ссылки в аргументах вызова сохранёнными результатами: (аргументы, None)
    либо (None, текст ошибки для модели) — тогда до сервера вызов не доходит.

    В параметрах из REF_PARAMS допустима ТОЛЬКО ссылка (объект с единственным строковым
    полем `ref` на существующее имя): история вместо ссылки, значение иного вида, лишние
    поля и неизвестное имя — ошибки. В любом другом параметре любого инструмента ссылка
    тоже отклоняется. Отсутствие обязательного параметра здесь не проверяется — это
    делает сервер, и его текст ошибки идёт модели как есть. Подставляется глубокая копия:
    результат можно использовать повторно, а библиотека вправе менять уходящий словарь.
    """
    ref_params = REF_PARAMS.get(tool_name, ())
    resolved = dict(arguments)
    for param in ref_params:
        if param not in arguments:
            continue
        value = arguments[param]
        if not _is_ref_value(value):
            return None, (
                f"Параметр «{param}» принимает только ссылку вида "
                '{"ref":"rN"} на результат get_price_history, а не данные. '
                + _available_refs_hint(store)
            )
        name = value["ref"]
        if name not in store:
            return None, (
                f"Параметр «{param}»: ссылки «{name}» нет среди результатов этого вопроса. "
                + _available_refs_hint(store)
            )
        resolved[param] = copy.deepcopy(store[name])
    for param, value in arguments.items():
        if param not in ref_params and isinstance(value, dict) and "ref" in value:
            return None, (
                f"Параметр «{param}» инструмента «{tool_name}» не принимает ссылок: "
                "ссылки {\"ref\":\"rN\"} допустимы только в параметрах с историей цен "
                "у инструментов анализа."
            )
    return resolved, None


# --------------------------------------------------------------------------- #
# Сессия
# --------------------------------------------------------------------------- #


class MarketTools:
    """Открытая сессия с сервером: инструменты для модели и выполнение вызовов.
    Создаётся open_market_tools(), самостоятельно не конструируется."""

    def __init__(
        self,
        session: ClientSession,
        tools: list,
        timeout: float,
        max_result_chars: int,
        allowlist: frozenset[str] = MODEL_TOOL_ALLOWLIST,
        require_read_only: bool = True,
    ) -> None:
        self._session = session
        self._timeout = timeout
        self._max_result_chars = max_result_chars
        allowed = [tool for tool in tools if is_model_tool(tool, allowlist, require_read_only)]
        skipped = [
            tool.name for tool in tools if not is_model_tool(tool, allowlist, require_read_only)
        ]
        if skipped:
            logger.warning(
                "Инструменты не отданы модели (нет пометки read-only или нет в перечне "
                "разрешённых): %s",
                ", ".join(skipped),
            )
        self._names = [tool.name for tool in allowed]
        self.openai_tools: list[dict] = [self._model_tool(tool) for tool in allowed]
        # Реестр результатов истории этого вопроса: имя -> полный результат сервера.
        # Объект живёт один вопрос, поэтому имена не переживают вопрос без очистки.
        self._results: dict[str, dict] = {}
        # Ссылки выдаются, только если модели предоставлен хотя бы один инструмент
        # анализа: со старым сервером сообщения остаются такими, как до появления ссылок.
        self._analytics = any(name in REF_PARAMS for name in self._names)

    @property
    def analytics_available(self) -> bool:
        """Среди предоставленных модели инструментов есть инструмент анализа истории."""
        return self._analytics

    @staticmethod
    def _model_tool(tool) -> dict:
        """Описание инструмента для модели: у инструментов анализа параметры с историей
        описаны как ссылка, а к описанию добавлено предложение о ссылках."""
        converted = to_openai_tool(tool)
        params = REF_PARAMS.get(tool.name)
        if params:
            function = converted["function"]
            function["parameters"] = with_ref_parameters(function["parameters"], params)
            function["description"] = (
                f"{function['description']}\n\n{REF_TOOL_DESCRIPTION_SUFFIX}".strip()
            )
        return converted

    async def call(self, name: str, raw_arguments: str | None) -> ToolOutcome:
        """Выполняет вызов и НЕ бросает исключений уровня инструмента: любая неудача
        превращается в текст для модели (см. докстринг модуля)."""
        if name not in self._names:
            available = ", ".join(self._names) or "нет"
            return ToolOutcome(
                f"Инструмента «{name}» нет среди доступных. Доступные инструменты: {available}.",
                is_error=True,
            )

        arguments, error = parse_arguments(raw_arguments)
        if error is not None:
            return ToolOutcome(error, is_error=True)
        if self._analytics:
            arguments, error = resolve_refs(name, arguments, self._results)
            if error is not None:
                return ToolOutcome(error, is_error=True)

        try:
            async with asyncio.timeout(self._timeout):
                result = await self._session.call_tool(name, arguments)
        except TimeoutError:
            logger.warning("Тайм-аут вызова MCP-инструмента %s (%s с).", name, self._timeout)
            return ToolOutcome(
                f"Инструмент «{name}» не ответил вовремя. Данные сейчас недоступны.",
                is_error=True,
                transport_failure=True,
            )
        except Exception:  # noqa: BLE001 — любой сбой обмена не должен ронять вопрос
            logger.exception("Сбой обмена с MCP-сервером при вызове %s.", name)
            return ToolOutcome(
                f"Не удалось выполнить инструмент «{name}»: сбой соединения с сервером. "
                "Данные сейчас недоступны.",
                is_error=True,
                transport_failure=True,
            )

        if not hasattr(result, "content"):
            # call_tool у SDK 2.x может вернуть и не CallToolResult (например, запрос
            # дополнительного ввода) — для read-only инструментов это не ожидается.
            logger.warning("Неожиданный тип ответа инструмента %s: %s", name, type(result).__name__)
            return ToolOutcome(
                f"Инструмент «{name}» вернул неожиданный ответ.", is_error=True
            )

        is_error = bool(getattr(result, "is_error", False))
        text = result_to_text(result)
        if name == HISTORY_TOOL and self._analytics and not is_error:
            outcome = self._remember_history(result, text)
            if outcome is not None:
                return outcome
        return ToolOutcome(truncate_result(text, self._max_result_chars), is_error=is_error)

    def _remember_history(self, result, text: str) -> ToolOutcome | None:
        """Сохраняет результат истории под именем rN и возвращает сообщение для модели с
        пометкой; None — сохранить не удалось (тогда результат идёт как обычный текст).

        Хранится полный результат, а не усечённый текст. Пометка добавляется ПОСЛЕ
        усечения, но усечение оставляет под неё место: объём сообщения вместе с
        пометками не превышает предела, и пометка не теряется при усечении с головы.
        """
        stored = extract_structured(result)
        if stored is None:
            logger.warning(
                "Результат %s не разобран как объект — ссылка не выдана.", HISTORY_TOOL
            )
            return None
        name = f"r{len(self._results) + 1}"
        note = ref_note(name)
        if len(note) > self._max_result_chars:
            logger.warning("Предел размера результата меньше пометки о ссылке — ссылка не выдана.")
            return None
        self._results[name] = stored
        return ToolOutcome(truncate_result(text, self._max_result_chars - len(note)) + note)


@asynccontextmanager
async def open_market_tools(
    params: StdioServerParameters,
    timeout: float,
    max_result_chars: int,
    allowlist: frozenset[str] = MODEL_TOOL_ALLOWLIST,
    require_read_only: bool = True,
) -> AsyncIterator[MarketTools]:
    """Запускает сервер по готовым `params`, делает рукопожатие и получает список
    инструментов; процесс завершается при выходе из блока.

    `params` собирает вызывающий код (build_server_params() — для источника с
    локальным каталогом проекта; build_uvx_server_params() — для источника,
    устанавливаемого из опубликованного пакета, design.md изменения
    add-smart-agent-cbr-tools, решение 1): эта функция сама не знает и не должна
    знать, КАК источник запускается — только что запускать. Оба builder'а собирают
    команду списком аргументов без shell, так что операторские значения (каталог,
    имя пакета) не могут внедрить произвольную команду.
    Тайм-аут охватывает рукопожатие и list_tools() — самое вероятное место зависания
    (медленный старт uv/uvx, скачивание пакета из PyPI при первом запуске); сам вызов
    инструментов ограничен тем же значением отдельно (MarketTools.call). Исключения
    запуска не перехватываются — см. докстринг модуля.
    `allowlist`/`require_read_only` по умолчанию соответствуют MOEX (обратная
    совместимость для однo-источникового вызова); другие источники передают их явно.
    """
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            async with asyncio.timeout(timeout):
                await session.initialize()
                listed = await session.list_tools()
            yield MarketTools(
                session, list(listed.tools), timeout, max_result_chars, allowlist, require_read_only
            )


# --------------------------------------------------------------------------- #
# Несколько источников за один вопрос (design.md изменения
# add-smart-agent-bybit-tools, решения 1-3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MarketSource:
    """Постоянные данные одного настроенного источника — то, что нужно, чтобы его
    открыть (open_market_tools) и опознать в статусах/сообщениях слоя. `label` —
    человекочитаемое имя для текстов (agents/market_tools.py), а не для протокола.

    `params` — уже готовая команда запуска (build_server_params() для источника с
    локальным каталогом, build_uvx_server_params() для источника-пакета без каталога)
    — MarketSource и всё, что его использует, не знают, КАК источник запускается
    (design.md изменения add-smart-agent-cbr-tools, решение 1). `require_read_only`
    по умолчанию `True` (поведение MOEX/Bybit не меняется); `False` — узкое, явное
    исключение для конкретного источника (см. is_model_tool())."""

    id: str
    label: str
    params: StdioServerParameters
    allowlist: frozenset[str]
    require_read_only: bool = True


class MultiMarketTools:
    """Фасад над несколькими открытыми `MarketTools` (по одной на источник) для
    модели: объединённый список инструментов с именами вида
    `<источник>__<имя>` и диспетчеризация вызова по этому префиксу к нужному
    источнику (design.md, решения 1 и 4). Каждый `MarketTools` внутри не меняется и
    не знает о существовании других — префикс существует только на границе с
    моделью, поэтому реестр ссылок на историю (rN) у каждого источника свой и
    естественно изолирован: диспетчеризация уже направляет вызов инструмента анализа
    к правильному источнику ДО того, как тот разрешает свои ссылки."""

    def __init__(self, sources: dict[str, MarketTools]) -> None:
        self._sources = sources
        self.openai_tools: list[dict] = []
        for source_id, tools in sources.items():
            for tool in tools.openai_tools:
                prefixed = copy.deepcopy(tool)
                prefixed["function"]["name"] = (
                    f"{source_id}{SOURCE_PREFIX_SEPARATOR}{tool['function']['name']}"
                )
                self.openai_tools.append(prefixed)

    def source_ids(self) -> list[str]:
        """Источники, сессию с которыми удалось открыть в этом вопросе (в порядке
        добавления)."""
        return list(self._sources)

    def analytics_available_for(self, source_id: str) -> bool:
        tools = self._sources.get(source_id)
        return tools.analytics_available if tools is not None else False

    async def call(self, name: str, raw_arguments: str | None) -> ToolOutcome:
        """Разбирает признак источника (первое вхождение SOURCE_PREFIX_SEPARATOR) и
        делегирует вызов его MarketTools.call(). Неизвестный или отсутствующий
        признак — ошибка текстом для модели, без обращения к какому-либо серверу."""
        source_id, _, bare_name = name.partition(SOURCE_PREFIX_SEPARATOR)
        if not bare_name:
            return ToolOutcome(
                f"Имя инструмента «{name}» не несёт признака источника "
                f"(ожидался вид «источник{SOURCE_PREFIX_SEPARATOR}имя»).",
                is_error=True,
            )
        tools = self._sources.get(source_id)
        if tools is None:
            available = ", ".join(self._sources) or "нет"
            return ToolOutcome(
                f"Источник «{source_id}» недоступен в этом вопросе. Доступные "
                f"источники: {available}.",
                is_error=True,
            )
        return await tools.call(bare_name, raw_arguments)


@asynccontextmanager
async def open_multi_market_tools(
    sources: list[MarketSource], timeout: float, max_result_chars: int
) -> AsyncIterator[tuple[MultiMarketTools, dict[str, str]]]:
    """Открывает сессию с каждым источником из `sources` ПОСЛЕДОВАТЕЛЬНО, в одном и
    том же asyncio-таске: источник, который не удалось поднять (сбой запуска,
    тайм-аут рукопожатия), не мешает использовать остальные (design.md, решение 3) —
    открытие остальных просто продолжается дальше по списку. Каждая сессия
    закрывается по выходу из блока независимо от исхода другой (через общий
    AsyncExitStack).

    НЕ открывает сессии конкурентно (asyncio.gather/отдельные таски) — это было
    первым вариантом (design.md, решение 2), но открытие сессии в одном таске и её
    использование/закрытие в другом ломает anyio: `stdio_client`/`ClientSession`
    держат внутренний task group, чей cancel scope обязан закрываться в ТОМ ЖЕ
    таске, где был открыт (`RuntimeError: Attempted to exit cancel scope in a
    different task than it was entered in` — воспроизведено вручную при попытке
    открыть источники через asyncio.gather с общим AsyncExitStack, закрываемым в
    таске-вызывающем). Последовательное открытие в одном таске избегает этого риска
    полностью: у каждого источника — на несколько сотен миллисекунд больше задержка
    (~0,75 с на источник, значит ~1,5 с на вопрос с двумя настроенными источниками
    вместо ~0,75 с при гипотетической параллели), но корректность важнее экономии
    доли секунды. См. design.md, обновлённое решение 2.

    Возвращает (агрегатор доступных источников, {source_id: причина неудачи} для
    источников, которые поднять не удалось). Пустой `sources` — агрегатор без
    инструментов и пустой словарь неудач; вызывающий код такой случай не должен
    создавать (список настроенных источников формируется им самим), но он не
    считается ошибкой.
    """
    async with AsyncExitStack() as stack:
        opened: dict[str, MarketTools] = {}
        failures: dict[str, str] = {}

        for source in sources:
            try:
                tools = await stack.enter_async_context(
                    open_market_tools(
                        source.params,
                        timeout,
                        max_result_chars,
                        source.allowlist,
                        source.require_read_only,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — сбой ОДНОГО источника не должен мешать другим
                logger.warning(
                    "Не удалось запустить источник рыночных данных %r — отвечаю без "
                    "его инструментов.",
                    source.id,
                    exc_info=True,
                )
                failures[source.id] = describe_launch_failure(exc)
                continue
            if not tools.openai_tools:
                # Сессия открылась, но сервер не предоставил ни одного инструмента
                # только для чтения из перечня — источник считается недоступным, как
                # и при сбое запуска (та же семантика STATUS_UNAVAILABLE, что и до
                # появления второго источника).
                failures[source.id] = "сервер не предоставил инструментов только для чтения"
                continue
            opened[source.id] = tools

        yield MultiMarketTools(opened), failures
