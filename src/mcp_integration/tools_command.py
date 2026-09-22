"""Команда /mcp_tools — подключается к MCP-серверу и присылает список его инструментов.

Технический/диагностический режим (регистрируется и упоминается в /help только при
RESEARCH_ENABLED, как /research_* — см. config.py и CLAUDE.md), а не часть основного
инвестиционного сценария бота. Не входит в ConversationHandler: ни ввода вопроса, ни
кнопок, ни состояния у команды нет — один CommandHandler, как у /agent_history.

Слой подключения (mcp_integration/client.py) не перехватывает исключения — их
перевод в сообщение на русском сделан здесь, отдельными except по тому же паттерну,
что handle_message в main.py: FileNotFoundError (команда запуска сервера не найдена
на хосте — например, не установлен Node.js/npx), TimeoutError (не уложились в
MCP_TIMEOUT_SECONDS), mcp.MCPError/протокольные и прочие ошибки сервера — общим
fallback с logger.exception. Общий research/_shared.api_error_to_message не
переиспользуется: он про исключения OpenAI SDK, здесь исключения другого рода.

Аргументы, переданные пользователем вместе с командой, игнорируются — сервер задаёт
только оператор через MCP_SERVER_COMMAND/MCP_SERVER_ARGS (config.py), пользователь
чата не может его подменить (см. «Ограничения безопасности» в CLAUDE.md).
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CommandHandler, ContextTypes

from config import TELEGRAM_MESSAGE_LIMIT

from .client import connect_and_list_tools, format_tools_result

logger = logging.getLogger(__name__)


async def mcp_tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /mcp_tools — см. докстринг модуля."""
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    # Первый запуск npx может скачивать пакет — предупреждаем, чтобы молчание бота
    # не выглядело зависанием.
    await update.message.reply_text("🔌 Подключаюсь к MCP-серверу…")

    try:
        result = await connect_and_list_tools()
    except FileNotFoundError:
        logger.error("Команда запуска MCP-сервера не найдена на хосте.")
        await update.message.reply_text(
            "❌ Не удалось запустить MCP-сервер: команда запуска не найдена на хосте бота. "
            "Администратору нужно проверить, что окружение (например, Node.js и npx) установлено."
        )
        return
    except TimeoutError:
        logger.warning("Тайм-аут подключения к MCP-серверу.")
        await update.message.reply_text(
            "⏳ MCP-сервер не ответил вовремя. Попробуй ещё раз чуть позже — "
            "первый запуск может занимать больше времени, пока сервер устанавливается."
        )
        return
    except Exception:  # noqa: BLE001 — последний рубеж для протокольных и прочих ошибок сервера
        logger.exception("Не удалось получить список инструментов MCP-сервера.")
        await update.message.reply_text(
            "⚠️ Не удалось установить соединение с MCP-сервером. Попробуй позже."
        )
        return

    text = format_tools_result(result)
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MESSAGE_LIMIT])


def build_mcp_tools_handler() -> CommandHandler:
    """Собирает CommandHandler команды /mcp_tools для регистрации в main.py."""
    return CommandHandler("mcp_tools", mcp_tools_command)
