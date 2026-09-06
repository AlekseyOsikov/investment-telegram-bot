"""Общая конфигурация, логирование и клиент DeepSeek.

Вынесено из main.py, чтобы main.py и research_response_format.py могли использовать
один и тот же клиент и константы, не импортируя друг друга (без циклических импортов).
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

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

# Задаёт и доменную роль бота (инвестиционные вопросы), и обязательные ограничения
# (см. раздел "Правила предметной области" в CLAUDE.md) — при доработке текста
# проверяй, что дисклеймер и осторожные формулировки не потерялись.
SYSTEM_PROMPT = (
    "Ты — ассистент Telegram-бота, отвечающий на вопросы пользователей об инвестициях "
    "и личных финансах. Отвечай кратко, по существу и на русском языке.\n\n"
    "Обязательно соблюдай следующие правила:\n"
    "1. Ты не являешься лицензированным финансовым советником, а твои ответы не являются "
    "индивидуальной инвестиционной рекомендацией. Если вопрос предполагает конкретное "
    "решение (купить/продать/во что вложить), явно напоминай об этом и советуй "
    "проконсультироваться с лицензированным финансовым консультантом перед принятием решения.\n"
    "2. Никогда не гарантируй доходность, рост стоимости активов или отсутствие риска. "
    "Не используй формулировки вида «точно вырастет», «гарантированная доходность», "
    "«безрисковый вариант» — любые инвестиции сопряжены с риском, и об этом нужно говорить "
    "прямо.\n"
    "3. Не запрашивай у пользователя чувствительные персональные данные: номера счетов и "
    "карт, паспортные данные, ИНН, точные суммы на счетах. Если пользователь сам их "
    "присылает, не проси их подтвердить или уточнить, и не включай в свой ответ.\n"
    "4. Если вопрос выходит за рамки инвестиций и личных финансов, вежливо сообщи об этом "
    "и предложи переформулировать вопрос в рамках этой темы."
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
