"""Команда /agent — LLM-агент с памятью диалога, оформленный как отдельная сущность.

В отличие от handle_message (main.py), где вызов LLM выполняется инлайн внутри
Telegram-обработчика, здесь весь цикл «принять вопрос -> вызвать API -> разобрать
ответ» инкапсулирован в классе Agent (agents/agent.py). Обработчики этого модуля
отвечают только за Telegram-часть: получить текст вопроса, показать индикатор
набора, отправить готовый ответ агента и превратить возможную ошибку API в
сообщение на русском — по тому же принципу, что и handle_message, с той же
параметризацией через MAIN_CLIENT_LABEL/MAIN_API_KEY_ENV_VAR (см. «Архитектура» в
CLAUDE.md).

Команда работает как диалог (ConversationHandler): после /agent пользователь может
задавать вопросы один за другим, пока не выйдет из режима. В отличие от основного
потока и от прежней stateless-версии этого агента, каждый вызов видит всю
предыдущую переписку этого чата с агентом — история хранится в JSON-файле на диске
(см. докстринг Agent) и переживает и повторный вход в /agent, и перезапуск всего
бота. Экземпляры Agent кэшируются по chat_id в _agents (см. _get_agent) — один на
чат за всё время жизни процесса, а не общий на всех пользователей, как раньше.
Команда /agent_reset (agent_reset_command) отдельно от диалога полностью очищает
историю чата, а /agent_history (agent_history_command) печатает её как есть —
обе работают и внутри режима агента, и вне его, не входят в ConversationHandler и
не меняют его состояние. Выход из режима самого
диалога (не путать с очисткой истории) — кнопка на постоянной клавиатуре под полем
ввода (ReplyKeyboardMarkup, а не inline-кнопка под конкретным сообщением, как в
research-режимах) или, как и раньше, команда /cancel — оба пути ведут в один и тот
же _exit_agent_mode; история диалога при этом не очищается, чтобы следующий /agent
мог его продолжить. Клавиатура с кнопкой выхода прикрепляется к каждому ответу
бота, пока пользователь остаётся в состоянии WAITING_QUESTION (а не только один раз
к приветствию) — так кнопка гарантированно не пропадает из чата ни после ошибок
валидации/API, ни после ответа модели, а исчезает только через ReplyKeyboardRemove в
_exit_agent_mode.

main.py подключает команду через build_agent_conversation_handler(),
build_agent_reset_handler() и build_agent_history_handler() — единственные точки
интеграции с остальным приложением.
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

from .agent import Agent

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
    "🤖 Режим агента.\n\n"
    f"Задавай вопросы — агент помнит историю этого диалога и передаёт её в "
    f"{MAIN_CLIENT_LABEL} при каждом следующем вопросе, даже после перезапуска бота.\n\n"
    "👉 Введи вопрос. Чтобы выйти, нажми кнопку внизу (или отправь /cancel). "
    "Команда /agent_reset в любой момент очищает историю."
)

# Экземпляр Agent хранит историю диалога конкретного чата (см. докстринг Agent),
# поэтому, в отличие от прежней stateless-версии, не может быть один на все чаты —
# кэшируем по chat_id и переиспользуем в пределах жизни процесса; при первом
# обращении после перезапуска процесса Agent сам подхватит историю с диска.
_agents: dict[int, Agent] = {}


def _get_agent(chat_id: int) -> Agent:
    agent = _agents.get(chat_id)
    if agent is None:
        agent = Agent(chat_id)
        _agents[chat_id] = agent
    return agent


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
        answer = _get_agent(chat_id).ask(user_text)
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


async def agent_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_reset — очищает историю диалога агента для этого чата.

    Работает независимо от того, находится ли пользователь сейчас в режиме /agent
    (см. докстринг модуля) — не входит в ConversationHandler и не меняет его
    состояние.
    """
    chat_id = update.effective_chat.id
    _get_agent(chat_id).reset()
    await update.message.reply_text(
        "🗑 История диалога с агентом очищена. Следующий вопрос агент увидит как первый."
    )


async def agent_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_history — печатает сохранённую историю диалога агента для этого чата.

    Как и /agent_reset, работает независимо от того, находится ли пользователь
    сейчас в режиме /agent — не входит в ConversationHandler и не меняет его
    состояние. История не ограничена по длине (см. докстринг Agent), поэтому
    результат режется на части по TELEGRAM_MESSAGE_LIMIT так же, как в
    handle_message (main.py).
    """
    chat_id = update.effective_chat.id
    history = _get_agent(chat_id).get_history()

    if not history:
        await update.message.reply_text(
            "📭 История диалога с агентом пуста. Используй /agent, чтобы задать вопрос."
        )
        return

    lines = [f"📜 История диалога с агентом ({len(history) // 2} вопрос(ов)):"]
    for i in range(0, len(history), 2):
        pair_number = i // 2 + 1
        lines.append(f"\n{pair_number}. 🙋 {history[i]['content']}")
        if i + 1 < len(history):
            lines.append(f"🤖 {history[i + 1]['content']}")

    text = "\n".join(lines)
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MESSAGE_LIMIT])


def build_agent_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler команды /agent для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("agent", agent_command)],
        states={WAITING_QUESTION: [MessageHandler(text_filter, agent_receive_question)]},
        fallbacks=[CommandHandler("cancel", agent_cancel)],
    )


def build_agent_reset_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_reset для регистрации в main.py."""
    return CommandHandler("agent_reset", agent_reset_command)


def build_agent_history_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_history для регистрации в main.py."""
    return CommandHandler("agent_history", agent_history_command)
