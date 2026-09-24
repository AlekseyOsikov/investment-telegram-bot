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
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

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
        readonly = [tool for tool in tools if is_read_only(tool)]
        skipped = [tool.name for tool in tools if not is_read_only(tool)]
        if skipped:
            logger.warning(
                "Инструменты без пометки read-only не отданы модели: %s", ", ".join(skipped)
            )
        self._names = [tool.name for tool in readonly]
        self.openai_tools: list[dict] = [to_openai_tool(tool) for tool in readonly]

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

        text = truncate_result(result_to_text(result), self._max_result_chars)
        return ToolOutcome(text, is_error=bool(getattr(result, "is_error", False)))


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
    params = StdioServerParameters(
        command="uv", args=["run", "--directory", directory, "mcp-moex"]
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            async with asyncio.timeout(timeout):
                await session.initialize()
                listed = await session.list_tools()
            yield MarketTools(session, list(listed.tools), timeout, max_result_chars)
