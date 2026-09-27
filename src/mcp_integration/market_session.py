"""Сессия с MCP-сервером рыночных данных (mcp-moex) для основного ответа /smart_agent.

В отличие от client.py (диагностический цикл «подключиться → перечислить → закрыть»),
здесь сессия остаётся открытой на весь ВОПРОС пользователя: схемы инструментов нужны
до первого обращения к LLM, а сами вызовы происходят по ходу цикла tool-calls (см.
agents/market_tools.py). Процесс сервера один на вопрос и не живёт между вопросами —
см. design.md изменения add-smart-agent-moex-tools, решение 3.

Граница ответственности такая же, как у client.py/tools_command.py: здесь только
механика MCP, без Telegram и без знания о том, как агент строит контекст. Всё, что
не требует сети (фильтр read-only, схема для OpenAI, разбор аргументов, текст
результата, усечение), — чистые функции, покрытые tests/test_market_tools.py без
запуска процессов.

Два класса неудач разведены намеренно:
- неудача ЗАПУСКА сервера (нет `uv`, нет каталога, тайм-аут рукопожатия, нарушение
  протокола) — исключение из open_market_tools(), его перехватывает вызывающий код и
  переводит вопрос в режим «без инструментов»;
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
from contextlib import asynccontextmanager
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


def is_model_tool(tool) -> bool:
    """Инструмент можно отдать модели: он в явном перечне разрешённых И помечен
    сервером как только читающий. Оба условия обязательны — см. MODEL_TOOL_ALLOWLIST."""
    return tool.name in MODEL_TOOL_ALLOWLIST and is_read_only(tool)


def build_server_params(directory: str) -> StdioServerParameters:
    """Параметры запуска сервера рыночных данных для ответа smart-агента.

    Команда собирается списком аргументов без shell, поэтому значение directory не
    может внедрить команду. Флага `--watch-db` здесь НЕТ намеренно: в режиме по
    умолчанию сервер публикует только три инструмента чтения, а инструменты
    расписания (принимают chat_id от клиента) модель не видит вообще.
    """
    return StdioServerParameters(
        command="uv", args=["run", "--directory", directory, "mcp-moex"]
    )


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
    ) -> None:
        self._session = session
        self._timeout = timeout
        self._max_result_chars = max_result_chars
        allowed = [tool for tool in tools if is_model_tool(tool)]
        skipped = [tool.name for tool in tools if not is_model_tool(tool)]
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
    directory: str, timeout: float, max_result_chars: int
) -> AsyncIterator[MarketTools]:
    """Запускает `uv run --directory <directory> mcp-moex`, делает рукопожатие и
    получает список инструментов; процесс завершается при выходе из блока.

    Команда собирается списком аргументов и запускается без shell
    (StdioServerParameters), поэтому значение directory не может внедрить команду.
    Тайм-аут охватывает рукопожатие и list_tools() — самое вероятное место зависания
    (медленный старт uv); сам вызов инструментов ограничен тем же значением отдельно
    (MarketTools.call). Исключения запуска не перехватываются — см. докстринг модуля.
    """
    params = build_server_params(directory)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            async with asyncio.timeout(timeout):
                await session.initialize()
                listed = await session.list_tools()
            yield MarketTools(session, list(listed.tools), timeout, max_result_chars)
