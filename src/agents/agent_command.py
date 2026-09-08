"""Команда /agent — простой LLM-агент, оформленный как отдельная сущность.

В отличие от handle_message (main.py), где вызов LLM выполняется инлайн внутри
Telegram-обработчика, здесь весь цикл «принять вопрос -> вызвать API -> разобрать
ответ» инкапсулирован в классе SimpleAgent (agents/simple_agent.py). Обработчики
этого модуля отвечают только за Telegram-часть: получить текст вопроса, показать
индикатор набора, отправить готовый ответ агента и превратить возможную ошибку API в
сообщение на русском — по тому же принципу, что и handle_message, с той же
параметризацией через MAIN_CLIENT_LABEL/MAIN_API_KEY_ENV_VAR (см. «Архитектура» в
CLAUDE.md).

Команда работает как диалог (ConversationHandler): после /agent пользователь может
задавать вопросы один за другим, каждый — независимый вызов агента (без истории
переписки, как и в основном потоке), пока не выйдет из режима. Выход — кнопка на
постоянной клавиатуре под полем ввода (ReplyKeyboardMarkup, а не inline-кнопка под
конкретным сообщением, как в research-режимах) или, как и раньше, команда /cancel —
оба пути ведут в один и тот же _exit_agent_mode. Клавиатура с кнопкой выхода
прикрепляется к каждому ответу бота, пока пользователь остаётся в состоянии
WAITING_QUESTION (а не только один раз к приветствию) — так кнопка гарантированно не
пропадает из чата ни после ошибок валидации/API, ни после ответа модели, а исчезает
только через ReplyKeyboardRemove в _exit_agent_mode.

main.py подключает команду через build_agent_conversation_handler() — единственную
точку интеграции с остальным приложением.
"""

from __future__ import annotations

import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from telegram import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ChatAction
from telegram.ext import (
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import (
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAX_INPUT_CHARS,
    TELEGRAM_MESSAGE_LIMIT,
)

from .simple_agent import SimpleAgent

logger = logging.getLogger(__name__)

WAITING_QUESTION = 0

# Текст кнопки одновременно и подпись на клавиатуре, и «команда» — MessageHandler
# сравнивает с ним обычный текст сообщения, который Telegram отправляет при нажатии
# кнопки ReplyKeyboardMarkup (в отличие от inline-кнопок, у них нет callback_data).
EXIT_BUTTON_TEXT = "🚪 Выйти из режима агента"

AGENT_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(EXIT_BUTTON_TEXT)]], resize_keyboard=True
)

AGENT_INTRO_TEXT = (
    "🤖 Режим простого агента.\n\n"
    f"Задавай вопросы — каждый передаётся в {MAIN_CLIENT_LABEL} как независимый запрос "
    "(без истории переписки).\n\n"
    "👉 Введи вопрос. Чтобы выйти, нажми кнопку внизу (или отправь /cancel)."
)

# Один экземпляр агента на процесс — он stateless (см. докстринг SimpleAgent), поэтому
# безопасно переиспользовать между запросами разных пользователей.
agent = SimpleAgent()


async def _exit_agent_mode(update: Update) -> int:
    """Общий выход из режима агента — по кнопке и по /cancel (см. build_..._handler)."""
    await update.message.reply_text(
        "Режим агента завершён. Возвращаюсь в обычный режим.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим агента (/agent)."""
    await update.message.reply_text(AGENT_INTRO_TEXT, reply_markup=AGENT_KEYBOARD)
    return WAITING_QUESTION


async def agent_receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает вопрос пользователя, вызывает агента и отправляет ответ обратно."""
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if user_text == EXIT_BUTTON_TEXT:
        return await _exit_agent_mode(update)

    if not user_text or not user_text.strip():
        await update.message.reply_text(
            "👉 Пожалуйста, отправь текстовый вопрос.", reply_markup=AGENT_KEYBOARD
        )
        return WAITING_QUESTION

    if len(user_text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Вопрос слишком длинный ({len(user_text)} символов). Максимум — {MAX_INPUT_CHARS}.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        answer = agent.ask(user_text)
    except AuthenticationError:
        logger.error(
            "Ошибка аутентификации %s API — проверьте %s.", MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR
        )
        await update.message.reply_text(
            f"❌ Ошибка авторизации на сервере {MAIN_CLIENT_LABEL}. "
            "Администратору бота нужно проверить API-ключ.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except RateLimitError:
        logger.warning("Превышен лимит запросов к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ Сервис {MAIN_CLIENT_LABEL} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ {MAIN_CLIENT_LABEL} не ответил вовремя. Попробуй отправить вопрос ещё раз.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except APIConnectionError:
        logger.error("Не удалось подключиться к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"🌐 Не получилось подключиться к серверу {MAIN_CLIENT_LABEL}. "
            "Проверь соединение и попробуй позже.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except APIStatusError as exc:
        logger.error("%s API вернул ошибку: %s", MAIN_CLIENT_LABEL, exc)
        await update.message.reply_text(
            f"⚠️ Сервер {MAIN_CLIENT_LABEL} вернул ошибку. Попробуй позже.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к %s API (агент).", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION

    # Клавиатуру достаточно прикрепить к последнему куску ответа — Telegram и так
    # держит её показанной до следующего reply_markup, но делаем это явно на каждом
    # сообщении бота в этом состоянии (см. докстринг модуля), а не только на первом.
    chunks = [answer[i : i + TELEGRAM_MESSAGE_LIMIT] for i in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT)]
    for chunk in chunks:
        await update.message.reply_text(chunk, reply_markup=AGENT_KEYBOARD)

    return WAITING_QUESTION


async def agent_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — тот же выход из режима агента, что и по кнопке (см. _exit_agent_mode)."""
    return await _exit_agent_mode(update)


def build_agent_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler команды /agent для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("agent", agent_command)],
        states={WAITING_QUESTION: [MessageHandler(text_filter, agent_receive_question)]},
        fallbacks=[CommandHandler("cancel", agent_cancel)],
    )
