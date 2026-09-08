"""Telegram-бот — прокси между пользователями и LLM-провайдером.

Архитектура нарочно простая (stateless):
- Каждое сообщение пользователя — независимый запрос к выбранному провайдеру.
- История диалога не хранится ни в памяти, ни на диске.
- Никаких админ-команд и никакого способа для пользователя сменить модель
  или системный промпт — это единственная защита от злоупотреблений,
  доступная без дополнительной инфраструктуры (rate-limit, whitelist и т.д.).

Провайдер и модель основного потока настраиваются переменными окружения MAIN_CLIENT
("deepseek" или "kimi") и MAIN_MODEL (config.py) — сам клиент собирается в
providers/main_client.py. Общая конфигурация вынесена в config.py, подключения к
конкретным провайдерам — в providers/deepseek_client.py и providers/kimi_client.py,
режим исследования ограничений API (/research_constraints) — в
research/constraints.py, режим исследования способов рассуждения
(/research_reasoning) — в research/reasoning.py, режим исследования влияния
temperature (/research_temperature) — в research/temperature.py, режим исследования
моделей (/research_models) — в research/models.py. Команда /agent — простой LLM-агент,
оформленный как отдельная сущность (класс SimpleAgent), — в agents/simple_agent.py и
agents/agent_command.py.
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
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import (
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_MESSAGE_LIMIT,
)
from agents.agent_command import build_agent_conversation_handler
from providers.main_client import main_client
from research.constraints import build_constraints_conversation_handler
from research.models import build_models_conversation_handler
from research.reasoning import build_reasoning_conversation_handler
from research.temperature import build_temperature_conversation_handler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Обработчики команд
# --------------------------------------------------------------------------- #

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    welcome_text = (
        f"👋 Привет! Я бот-ассистент по инвестициям и личным финансам на базе {MAIN_CLIENT_LABEL} AI "
        f"(модель {MAIN_MODEL}).\n\n"
        "Просто напиши свой вопрос об инвестициях, накоплениях или личных финансах — "
        f"я перешлю его в {MAIN_CLIENT_LABEL} и пришлю тебе ответ.\n\n"
        "⚠️ Важно: я не являюсь лицензированным финансовым советником, а мои ответы — "
        "не индивидуальная инвестиционная рекомендация. Перед принятием решений "
        "проконсультируйся с лицензированным финансовым консультантом.\n\n"
        "🔒 Я не запоминаю историю переписки: каждое сообщение обрабатывается независимо "
        "от предыдущих. Не присылай, пожалуйста, номера счетов, карт и другие "
        "чувствительные данные.\n\n"
        "Используй /help, чтобы посмотреть список команд."
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    help_text = (
        "ℹ️ <b>Как пользоваться ботом</b>\n\n"
        "Отправь текстовый вопрос об инвестициях или личных финансах — получишь ответ "
        f"от {MAIN_CLIENT_LABEL}.\n\n"
        "<b>Команды:</b>\n"
        "/start — приветственное сообщение\n"
        "/help — эта справка\n"
        f"/research_constraints — режим исследования влияния ограничений {MAIN_CLIENT_LABEL} API "
        "(включая формат ответа) на ответ (технический эксперимент, не для обычных вопросов)\n"
        f"/research_reasoning — режим исследования способов рассуждения {MAIN_CLIENT_LABEL} API "
        "(технический эксперимент, не для обычных вопросов)\n"
        f"/research_temperature — режим исследования влияния temperature на ответ {MAIN_CLIENT_LABEL} API "
        "(технический эксперимент, не для обычных вопросов)\n"
        "/research_models — режим исследования разных моделей API "
        "(технический эксперимент, не для обычных вопросов)\n"
        "/agent — простой LLM-агент: задавай вопросы один за другим, "
        "пока не отправишь /cancel\n\n"
        "<b>Ограничения:</b>\n"
        f"— максимальная длина запроса: {MAX_INPUT_CHARS} символов\n"
        "— бот не хранит историю диалога (каждый вопрос — новый контекст)\n"
        "— бот работает только с текстом (без файлов, фото и голосовых)\n\n"
        "<b>⚠️ Дисклеймер:</b>\n"
        "Бот не является лицензированным финансовым советником, а его ответы не являются "
        "индивидуальной инвестиционной рекомендацией и не гарантируют доходность. Перед "
        "принятием финансовых решений проконсультируйся с лицензированным специалистом. "
        "Не присылай боту номера счетов, карт и другие чувствительные персональные данные."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


# --------------------------------------------------------------------------- #
# Основной обработчик сообщений
# --------------------------------------------------------------------------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Пересылает текст пользователя в основного провайдера (MAIN_CLIENT) и отправляет ответ обратно."""
    user_text = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else "unknown"

    if not user_text or not user_text.strip():
        await update.message.reply_text("Пожалуйста, отправь текстовое сообщение.")
        return

    if len(user_text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            "⚠️ Сообщение слишком длинное "
            f"({len(user_text)} символов). Максимум — {MAX_INPUT_CHARS}."
        )
        return

    logger.info("Запрос от пользователя %s (%d символов)", user_id, len(user_text))

    # Показываем статус "печатает..." на время ожидания ответа от провайдера.
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        response = main_client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        answer = response.choices[0].message.content or (
            f"{MAIN_CLIENT_LABEL} вернул пустой ответ. Попробуй переформулировать запрос."
        )
    except AuthenticationError:
        logger.error(
            "Ошибка аутентификации %s API — проверьте %s.", MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR
        )
        await update.message.reply_text(
            f"❌ Ошибка авторизации на сервере {MAIN_CLIENT_LABEL}. "
            "Администратору бота нужно проверить API-ключ."
        )
        return
    except RateLimitError:
        logger.warning("Превышен лимит запросов к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ Сервис {MAIN_CLIENT_LABEL} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        )
        return
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ {MAIN_CLIENT_LABEL} не ответил вовремя. Попробуй отправить сообщение ещё раз."
        )
        return
    except APIConnectionError:
        logger.error("Не удалось подключиться к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"🌐 Не получилось подключиться к серверу {MAIN_CLIENT_LABEL}. "
            "Проверь соединение и попробуй позже."
        )
        return
    except APIStatusError as exc:
        logger.error("%s API вернул ошибку: %s", MAIN_CLIENT_LABEL, exc)
        await update.message.reply_text(
            f"⚠️ Сервер {MAIN_CLIENT_LABEL} вернул ошибку. Попробуй позже."
        )
        return
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."
        )
        return

    # Telegram ограничивает длину сообщения — режем длинный ответ на части.
    for i in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(answer[i : i + TELEGRAM_MESSAGE_LIMIT])


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок python-telegram-bot."""
    logger.error("Необработанное исключение при обработке обновления", exc_info=context.error)


# --------------------------------------------------------------------------- #
# Точка входа
# --------------------------------------------------------------------------- #

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(build_constraints_conversation_handler())
    application.add_handler(build_reasoning_conversation_handler())
    application.add_handler(build_temperature_conversation_handler())
    application.add_handler(build_models_conversation_handler())
    application.add_handler(build_agent_conversation_handler())
    # Только личные чаты и только текст — никаких групп, файлов, команд извне списка выше.
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, handle_message)
    )
    application.add_error_handler(error_handler)

    logger.info("Бот запущен. Провайдер: %s, модель: %s", MAIN_CLIENT_LABEL, MAIN_MODEL)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
