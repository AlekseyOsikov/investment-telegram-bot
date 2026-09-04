"""Telegram-бот — прокси между пользователями и DeepSeek API.

Архитектура нарочно простая (stateless):
- Каждое сообщение пользователя — независимый запрос к DeepSeek.
- История диалога не хранится ни в памяти, ни на диске.
- Никаких админ-команд и никакого способа для пользователя сменить модель
  или системный промпт — это единственная защита от злоупотреблений,
  доступная без дополнительной инфраструктуры (rate-limit, whitelist и т.д.).
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
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

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Разумные лимиты, чтобы не улететь по токенам/времени на один запрос.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))

# Telegram режет сообщения по 4096 символов — оставляем запас.
TELEGRAM_MESSAGE_LIMIT = 4000

SYSTEM_PROMPT = (
    "Ты — полезный ассистент, отвечающий пользователю Telegram-бота. "
    "Отвечай кратко и по существу."
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Библиотека httpx (используется и telegram, и openai) логирует каждый запрос —
# приглушаем, чтобы не засорять логи.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _validate_config() -> None:
    """Проверяет наличие обязательных переменных окружения перед стартом."""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")

    if missing:
        logger.error(
            "Отсутствуют обязательные переменные окружения: %s. "
            "Скопируйте .env.example в .env и заполните значения.",
            ", ".join(missing),
        )
        sys.exit(1)


_validate_config()

# Клиент OpenAI SDK, направленный на DeepSeek (OpenAI-совместимый API).
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# --------------------------------------------------------------------------- #
# Обработчики команд
# --------------------------------------------------------------------------- #

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    welcome_text = (
        "👋 Привет! Я бот-прокси к DeepSeek AI.\n\n"
        "Просто напиши мне любое сообщение — я перешлю его в DeepSeek "
        f"(модель {DEEPSEEK_MODEL}) и пришлю тебе ответ.\n\n"
        "⚠️ Я не запоминаю историю переписки: каждое сообщение "
        "обрабатывается независимо от предыдущих.\n\n"
        "Используй /help, чтобы посмотреть список команд."
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    help_text = (
        "ℹ️ <b>Как пользоваться ботом</b>\n\n"
        "Просто отправь текстовое сообщение — получишь ответ от DeepSeek.\n\n"
        "<b>Команды:</b>\n"
        "/start — приветственное сообщение\n"
        "/help — эта справка\n\n"
        "<b>Ограничения:</b>\n"
        f"— максимальная длина запроса: {MAX_INPUT_CHARS} символов\n"
        "— бот не хранит историю диалога (каждый вопрос — новый контекст)\n"
        "— бот работает только с текстом (без файлов, фото и голосовых)"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


# --------------------------------------------------------------------------- #
# Основной обработчик сообщений
# --------------------------------------------------------------------------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Пересылает текст пользователя в DeepSeek и отправляет ответ обратно."""
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

    # Показываем статус "печатает..." на время ожидания ответа от DeepSeek.
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        answer = response.choices[0].message.content or (
            "DeepSeek вернул пустой ответ. Попробуй переформулировать запрос."
        )
    except AuthenticationError:
        logger.error("Ошибка аутентификации DeepSeek API — проверьте DEEPSEEK_API_KEY.")
        await update.message.reply_text(
            "❌ Ошибка авторизации на сервере DeepSeek. "
            "Администратору бота нужно проверить API-ключ."
        )
        return
    except RateLimitError:
        logger.warning("Превышен лимит запросов к DeepSeek API.")
        await update.message.reply_text(
            "⏳ Сервис DeepSeek временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        )
        return
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к DeepSeek API.")
        await update.message.reply_text(
            "⏳ DeepSeek не ответил вовремя. Попробуй отправить сообщение ещё раз."
        )
        return
    except APIConnectionError:
        logger.error("Не удалось подключиться к DeepSeek API.")
        await update.message.reply_text(
            "🌐 Не получилось подключиться к серверу DeepSeek. "
            "Проверь соединение и попробуй позже."
        )
        return
    except APIStatusError as exc:
        logger.error("DeepSeek API вернул ошибку: %s", exc)
        await update.message.reply_text(
            "⚠️ Сервер DeepSeek вернул ошибку. Попробуй позже."
        )
        return
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к DeepSeek API.")
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
    # Только личные чаты и только текст — никаких групп, файлов, команд извне списка выше.
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, handle_message)
    )
    application.add_error_handler(error_handler)

    logger.info("Бот запущен. Модель DeepSeek: %s", DEEPSEEK_MODEL)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
