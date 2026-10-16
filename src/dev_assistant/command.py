"""Командный слой режима «ассистент разработчика»: `/dev`, вопросы обычными сообщениями, выход
`/cancel` (design.md изменения add-dev-assistant-mode, решения 2, 8, 9). Тонкая обёртка над
`core.DevAssistant`; Telegram знает только этот модуль.

Доступ (инварианты, не убирать):
- пустой список `DEV_ASSISTANT_USER_IDS` — `build_dev_assistant_handlers()` возвращает `[]`, команда
  ведёт себя как неизвестная;
- точка входа, обработчик вопросов и `/cancel` собраны с `filters.User(user_id=ids)` — сообщение
  постороннего вообще не доходит до обработчика; поверх этого каждый колбэк явно вызывает
  `rules.is_authorized(update.effective_user, ids)` (защита от ошибки регистрации). Отказ —
  МОЛЧА, как для неизвестной команды: режим не раскрывается; проверка — по id пользователя, не чата;
- режим НЕ упоминается в `/help` и не меняет `main.py: help_command`.

Взаимоисключение с `/agent`, `/agent_compare`, `/smart_agent`: входящие команды этих режимов внутри
диалога отклоняются подсказкой выйти (`/cancel`), а `/dev` не открывается, пока чат в одном из них
(`active_mode`). Тот же ConversationHandler регистрируется в `main.py` ДО них и до обработчика
обычного текста; код существующих режимов не меняется."""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from agents.active_mode import get_active_mode
from config import (
    DEV_ASSISTANT_HISTORY_TURNS,
    DEV_ASSISTANT_INDEX_DIR,
    DEV_ASSISTANT_MIN_SCORE,
    DEV_ASSISTANT_TOP_K,
    DEV_ASSISTANT_USER_IDS,
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    MCP_GIT_DIR,
    MCP_MAX_TOOL_STEPS,
    MCP_TIMEOUT_SECONDS,
    MCP_TOOL_RESULT_MAX_CHARS,
    RAG_SEARCH_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    TELEGRAM_MESSAGE_LIMIT,
)
from llm_queue.commands import cancel_chat_questions, enqueue_job, saved_notice
from llm_queue.gate import JobContext
from llm_queue.rules import KIND_DEV_ASSISTANT, AskCancelled
from providers.main_client import main_client
from research._shared import api_error_to_message

from . import rules, search
from .core import DevAnswer, DevAssistant
from .sources import PROJECT_ROOT

logger = logging.getLogger(__name__)

WAITING_QUESTION = 0

# Команды других диалоговых режимов: внутри /dev они отклоняются (см. докстринг модуля).
_OTHER_MODE_COMMANDS = ("agent", "agent_compare", "smart_agent", "support")

OTHER_MODE_ACTIVE_TEXT = (
    "⚠️ Сейчас активен другой режим диалога. Сначала выйди из него (/cancel), потом заходи в /dev."
)
LEAVE_FIRST_TEXT = "⚠️ Сначала выйди из режима ассистента разработчика (/cancel)."
TOO_LONG_TEXT = "⚠️ Вопрос слишком длинный ({length} символов). Максимум — {limit}."
UNEXPECTED_ERROR_TEXT = "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."
PROFILE_CHANGED_TEXT = "⚠️ Вопрос снят, обмен не записан."


def build_assistant() -> DevAssistant:
    """Ассистент с настройками оператора (`config`); один на процесс, история — по чатам."""
    return DevAssistant(
        client=main_client,
        model=MAIN_MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        search_config=search.DevSearchConfig(
            index_dir=DEV_ASSISTANT_INDEX_DIR,
            top_k=DEV_ASSISTANT_TOP_K,
            min_score=DEV_ASSISTANT_MIN_SCORE,
            search_timeout=RAG_SEARCH_TIMEOUT_SECONDS,
        ),
        history_turns=DEV_ASSISTANT_HISTORY_TURNS,
        git_directory=MCP_GIT_DIR,
        git_root=PROJECT_ROOT,
        mcp_timeout=MCP_TIMEOUT_SECONDS,
        max_tool_steps=MCP_MAX_TOOL_STEPS,
        tool_result_max_chars=MCP_TOOL_RESULT_MAX_CHARS,
    )


def split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Нарезка длинного текста по лимиту Telegram (с запасом от 4096)."""
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def format_service_message(answer: DevAnswer) -> str:
    """Служебное сообщение после ответа: источники (строит код), вызовы git-инструментов,
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


async def _answer_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    ctx: JobContext,
    assistant: DevAssistant,
    question: str,
) -> None:
    """Задача очереди: `ask()` в потоке под слотом с флагом отмены, перевод ошибок SDK в
    русские сообщения (`research._shared.api_error_to_message`), ответ и служебное сообщение."""
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        answer = await ctx.run_thread(
            functools.partial(assistant.ask, chat_id, question, should_cancel=ctx.should_cancel),
            late=saved_notice(context, chat_id),
        )
    except AskCancelled:
        return  # вопрос снят до записи в историю: состояние не менялось
    except Exception as exc:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        message = api_error_to_message(exc, MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR)
        if message is None:
            logger.exception(
                "Непредвиденная ошибка режима ассистента разработчика (%s).", MAIN_CLIENT_LABEL
            )
            message = UNEXPECTED_ERROR_TEXT
        await update.message.reply_text(message)
        return

    for chunk in split_for_telegram(answer.text):
        await update.message.reply_text(chunk)
    service = format_service_message(answer)
    if service:
        await update.message.reply_text(service)


def build_dev_assistant_handlers(
    user_ids: Sequence[int] | None = None, assistant: DevAssistant | None = None
) -> list[ConversationHandler]:
    """Обработчики режима. Пустой список id — `[]` (режим выключен, команда неизвестна).
    `user_ids`/`assistant` по умолчанию берутся из `config`; параметры — для проверки сборки."""
    ids = tuple(DEV_ASSISTANT_USER_IDS if user_ids is None else user_ids)
    if not ids:
        return []
    assistant = assistant or build_assistant()

    allowed = filters.User(user_id=list(ids))
    private = filters.ChatType.PRIVATE
    text_filter = filters.TEXT & ~filters.COMMAND & private & allowed

    def authorized(update: Update) -> bool:
        return rules.is_authorized(update.effective_user, ids)

    async def enter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not authorized(update) or update.message is None:
            return ConversationHandler.END  # молча: режим не раскрывается
        if get_active_mode(update.effective_chat.id) is not None:
            await update.message.reply_text(OTHER_MODE_ACTIVE_TEXT)
            return ConversationHandler.END
        await update.message.reply_text(rules.ENTER_TEXT)
        return WAITING_QUESTION

    async def receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not authorized(update) or update.message is None:
            return ConversationHandler.END  # доступ отозван/ошибка регистрации: без обращений
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
            await _answer_question(update, context, ctx, assistant, question)

        await enqueue_job(update, context, KIND_DEV_ASSISTANT, question, run)
        return WAITING_QUESTION

    async def leave_first(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if authorized(update) and update.message is not None:
            await update.message.reply_text(LEAVE_FIRST_TEXT)
        return WAITING_QUESTION

    async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not authorized(update) or update.message is None:
            return ConversationHandler.END
        chat_id = update.effective_chat.id
        assistant.clear(chat_id)
        cancel_chat_questions(context, chat_id)
        await update.message.reply_text(rules.EXIT_TEXT)
        return ConversationHandler.END

    return [
        ConversationHandler(
            entry_points=[CommandHandler(rules.ENTER_COMMAND, enter, filters=private & allowed)],
            states={
                WAITING_QUESTION: [
                    MessageHandler(text_filter, receive_question),
                    CommandHandler(
                        list(_OTHER_MODE_COMMANDS), leave_first, filters=private & allowed
                    ),
                ],
            },
            fallbacks=[CommandHandler("cancel", cancel, filters=private & allowed)],
        )
    ]
