"""Командный слой режима «ассистент поддержки»: `/support`, вопросы обычными сообщениями, выход
`/cancel` (design.md изменения add-support-assistant, решение 9). Тонкая обёртка над
`core.SupportAssistant`; Telegram знает только этот модуль.

Доступ: любой пользователь личного чата (списка id нет). Инварианты, не убирать:
- идентификатор для CRM берётся из `update.effective_user.id` (пользователь, не чат); сообщение без
  определяемого отправителя отклоняется без поиска, модели и CRM;
- режим показан в `/help` (строка в `main.py`), вне флага `RESEARCH`.

Взаимоисключение с `/agent`, `/agent_compare`, `/smart_agent`, `/dev`: команды первых трёх внутри
диалога отклоняются подсказкой выйти (`/cancel`), а `/support` не открывается, пока чат в одном из
них (`active_mode`; `/dev` отмечается собственным ConversationHandler-ом, который отклоняет
`/support` внутри диалога). Режим отмечает чат в `active_mode` (`SUPPORT_MODE`), и `/dev` не
открывается поверх него. ConversationHandler регистрируется в `main.py` ПОСЛЕ `/dev` и ДО
`/agent*` и обработчика обычного текста; код существующих режимов не меняется."""

from __future__ import annotations

import functools
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from agents.active_mode import SUPPORT_MODE, clear_active_mode, get_active_mode, set_active_mode
from config import (
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    MCP_MAX_TOOL_STEPS,
    MCP_SUPPORT_ARGS,
    MCP_SUPPORT_COMMAND,
    MCP_TIMEOUT_SECONDS,
    MCP_TOOL_RESULT_MAX_CHARS,
    RAG_SEARCH_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SUPPORT_CRM_MOCK,
    SUPPORT_DEMO_USER_ID,
    SUPPORT_HISTORY_TURNS,
    SUPPORT_INDEX_DIR,
    SUPPORT_MIN_SCORE,
    SUPPORT_TICKETS_LIMIT,
    SUPPORT_TOP_K,
    SYSTEM_PROMPT,
    TELEGRAM_MESSAGE_LIMIT,
)
from llm_queue.commands import cancel_chat_questions, enqueue_job, saved_notice
from llm_queue.gate import JobContext
from llm_queue.rules import KIND_SUPPORT, AskCancelled
from providers.main_client import main_client
from research._shared import api_error_to_message

from . import rules, search
from .core import SupportAnswer, SupportAssistant
from .crm_session import build_crm_server_params, open_crm_tools

logger = logging.getLogger(__name__)

WAITING_QUESTION = 0

# Команды других диалоговых режимов: внутри /support они отклоняются (см. докстринг модуля).
# `/dev` сюда не входит: его вход проверяет `active_mode` (для не-операторов команды как бы нет).
_OTHER_MODE_COMMANDS = ("agent", "agent_compare", "smart_agent")

OTHER_MODE_ACTIVE_TEXT = (
    "⚠️ Сейчас активен другой режим диалога. Сначала выйди из него (/cancel), потом заходи в "
    "/support."
)
LEAVE_FIRST_TEXT = "⚠️ Сначала выйди из режима поддержки (/cancel)."
NO_SENDER_TEXT = "⚠️ Не удалось определить отправителя сообщения, поэтому ответить не могу."
TOO_LONG_TEXT = "⚠️ Вопрос слишком длинный ({length} символов). Максимум — {limit}."
UNEXPECTED_ERROR_TEXT = "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."


def build_open_crm():
    """Фабрика сессии CRM по настройкам оператора или None, если CRM не настроена (ни внешней
    команды, ни встроенного мока). Параметры запуска собираются один раз."""
    params = build_crm_server_params(
        MCP_SUPPORT_COMMAND, MCP_SUPPORT_ARGS, SUPPORT_CRM_MOCK, SUPPORT_DEMO_USER_ID
    )
    if params is None:
        return None

    def open_crm(user_id: int):
        return open_crm_tools(params, user_id, MCP_TIMEOUT_SECONDS, MCP_TOOL_RESULT_MAX_CHARS)

    return open_crm


def build_assistant() -> SupportAssistant:
    """Ассистент с настройками оператора (`config`); один на процесс, история — по чатам."""
    return SupportAssistant(
        client=main_client,
        model=MAIN_MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        search_config=search.SupportSearchConfig(
            index_dir=SUPPORT_INDEX_DIR,
            top_k=SUPPORT_TOP_K,
            min_score=SUPPORT_MIN_SCORE,
            search_timeout=RAG_SEARCH_TIMEOUT_SECONDS,
        ),
        history_turns=SUPPORT_HISTORY_TURNS,
        tickets_limit=SUPPORT_TICKETS_LIMIT,
        max_tool_steps=MCP_MAX_TOOL_STEPS,
        open_crm=build_open_crm(),
    )


def split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Нарезка длинного текста по лимиту Telegram (с запасом от 4096)."""
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def format_service_message(answer: SupportAnswer) -> str:
    """Служебное сообщение после ответа: источники (строит код), вызовы инструментов CRM,
    предупреждения и расход токенов."""
    parts: list[str] = []
    if answer.sources:
        parts.append(answer.sources)
    if answer.tool_lines:
        parts.append("\n".join(answer.tool_lines))
    if answer.warnings:
        parts.append("\n".join(answer.warnings))
    if answer.answered_by_model and answer.prompt_tokens is not None:
        parts.append(
            f"📈 запросов к модели: {answer.llm_calls}, токенов: {answer.prompt_tokens} + "
            f"{answer.completion_tokens if answer.completion_tokens is not None else 'н/д'}"
        )
    return "\n\n".join(parts)


def sender_id(update: Update) -> int | None:
    """Id ПОЛЬЗОВАТЕЛЯ-отправителя или None (анонимный администратор, канал)."""
    user_id = getattr(update.effective_user, "id", None)
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        return None
    return user_id


async def _answer_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    ctx: JobContext,
    assistant: SupportAssistant,
    user_id: int,
    question: str,
) -> None:
    """Задача очереди: `ask()` в потоке под слотом с флагом отмены, перевод ошибок SDK в
    русские сообщения (`research._shared.api_error_to_message`), ответ и служебное сообщение."""
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        answer = await ctx.run_thread(
            functools.partial(
                assistant.ask, chat_id, user_id, question, should_cancel=ctx.should_cancel
            ),
            late=saved_notice(context, chat_id),
        )
    except AskCancelled:
        return  # вопрос снят до записи в историю: состояние не менялось
    except Exception as exc:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        message = api_error_to_message(exc, MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR)
        if message is None:
            logger.exception(
                "Непредвиденная ошибка режима поддержки (%s).", MAIN_CLIENT_LABEL
            )
            message = UNEXPECTED_ERROR_TEXT
        await update.message.reply_text(message)
        return

    for chunk in split_for_telegram(answer.text):
        await update.message.reply_text(chunk)
    service = format_service_message(answer)
    if service:
        await update.message.reply_text(service)


def build_support_handlers(assistant: SupportAssistant | None = None) -> list[ConversationHandler]:
    """Обработчики режима. `assistant` по умолчанию собирается из `config`; параметр — для
    проверки сборки."""
    assistant = assistant or build_assistant()

    private = filters.ChatType.PRIVATE
    text_filter = filters.TEXT & ~filters.COMMAND & private

    async def enter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if update.message is None:
            return ConversationHandler.END
        if sender_id(update) is None:
            await update.message.reply_text(NO_SENDER_TEXT)
            return ConversationHandler.END
        chat_id = update.effective_chat.id
        if get_active_mode(chat_id) is not None:
            await update.message.reply_text(OTHER_MODE_ACTIVE_TEXT)
            return ConversationHandler.END
        set_active_mode(chat_id, SUPPORT_MODE)
        await update.message.reply_text(rules.ENTER_TEXT)
        return WAITING_QUESTION

    async def receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if update.message is None:
            return ConversationHandler.END
        user_id = sender_id(update)
        if user_id is None:
            await update.message.reply_text(NO_SENDER_TEXT)  # без поиска, модели и CRM
            return WAITING_QUESTION
        question = update.message.text
        if not question or not question.strip():
            await update.message.reply_text(rules.EMPTY_QUESTION_TEXT)
            return WAITING_QUESTION
        if len(question) > MAX_INPUT_CHARS:
            await update.message.reply_text(
                TOO_LONG_TEXT.format(length=len(question), limit=MAX_INPUT_CHARS)
            )
            return WAITING_QUESTION

        async def run(ctx: JobContext) -> None:
            await _answer_question(update, context, ctx, assistant, user_id, question)

        await enqueue_job(update, context, KIND_SUPPORT, question, run)
        return WAITING_QUESTION

    async def leave_first(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if update.message is not None:
            await update.message.reply_text(LEAVE_FIRST_TEXT)
        return WAITING_QUESTION

    async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if update.message is None:
            return ConversationHandler.END
        chat_id = update.effective_chat.id
        assistant.clear(chat_id)
        clear_active_mode(chat_id)
        cancel_chat_questions(context, chat_id)
        await update.message.reply_text(rules.EXIT_TEXT)
        return ConversationHandler.END

    return [
        ConversationHandler(
            entry_points=[CommandHandler(rules.ENTER_COMMAND, enter, filters=private)],
            states={
                WAITING_QUESTION: [
                    MessageHandler(text_filter, receive_question),
                    CommandHandler(list(_OTHER_MODE_COMMANDS), leave_first, filters=private),
                ],
            },
            fallbacks=[CommandHandler("cancel", cancel, filters=private)],
        )
    ]
