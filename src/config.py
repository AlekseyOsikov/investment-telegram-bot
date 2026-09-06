"""Общая конфигурация, логирование и настройки, не привязанные к конкретному провайдеру.

Подключения к LLM-провайдерам (API-ключ, базовый URL, клиент openai.OpenAI, идентификаторы
моделей) вынесены в отдельные модули — deepseek_client.py и kimi_client.py. Какой из них
обслуживает основной поток бота, определяет MAIN_CLIENT (см. ниже) — от него зависит,
какой из двух API-ключей обязателен для старта, а какой нужен только техническому режиму
/research_models и не должен блокировать запуск остального бота (эту проверку выполняет
каждый клиентский модуль сам, см. их докстринги). Сам выбор клиента для основного потока
(объект main_client) собирается в main_client.py — не здесь, чтобы не создавать цикл
импорта: deepseek_client.py и kimi_client.py импортируют config.py ради побочного эффекта
(load_dotenv() и logging.basicConfig() должны отработать раньше, чем они читают переменные
окружения и логируют), поэтому только config.py вызывает load_dotenv(), а сам он не может
импортировать их в ответ.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Провайдер, обслуживающий основной поток бота (main.py) и research-режимы, которые сами
# не выбирают конкретную модель (research_reasoning.py, research_temperature.py,
# research_response_format.py) — все они используют main_client.py. research_models.py
# не зависит от MAIN_CLIENT: он всегда сравнивает обе пары моделей обоих провайдеров.
_SUPPORTED_MAIN_CLIENTS = ("deepseek", "kimi")
MAIN_CLIENT = os.getenv("MAIN_CLIENT", "deepseek").strip().lower()

# Модель основного потока — единственная переменная, а не своя на каждого провайдера,
# т.к. одновременно обслуживает только один из них (тот, что выбран в MAIN_CLIENT).
# Дефолт зависит от MAIN_CLIENT, чтобы работать "из коробки" при любом выборе провайдера.
_DEFAULT_MAIN_MODEL_BY_CLIENT = {"deepseek": "deepseek-v4-flash", "kimi": "kimi-k3"}
MAIN_MODEL = os.getenv(
    "MAIN_MODEL", _DEFAULT_MAIN_MODEL_BY_CLIENT.get(MAIN_CLIENT, "deepseek-v4-flash")
)

# Только для сообщений пользователю/логов основного потока (main.py и три research-режима
# из комментария выше) — чтобы они называли реально выбранного провайдера, а не всегда
# "DeepSeek", и подсказывали правильную переменную окружения при ошибке авторизации.
MAIN_CLIENT_LABEL = "DeepSeek" if MAIN_CLIENT == "deepseek" else "Kimi"
MAIN_API_KEY_ENV_VAR = "DEEPSEEK_API_KEY" if MAIN_CLIENT == "deepseek" else "KIMI_API_KEY"

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
    """Проверяет наличие обязательных переменных окружения перед стартом.

    Обязательность DEEPSEEK_API_KEY/KIMI_API_KEY зависит от MAIN_CLIENT и проверяется
    отдельно в deepseek_client.py/kimi_client.py (модуль выбранного провайдера сам
    завершает процесс, если его ключа нет) — здесь только TELEGRAM_BOT_TOKEN и сам
    MAIN_CLIENT, т.к. это переменные, за которые отвечает именно этот модуль.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.error(
            "Отсутствует обязательная переменная окружения TELEGRAM_BOT_TOKEN. "
            "Скопируйте .env.example в .env и заполните значение."
        )
        sys.exit(1)

    if MAIN_CLIENT not in _SUPPORTED_MAIN_CLIENTS:
        logger.error(
            "Недопустимое значение MAIN_CLIENT=%r. Допустимые значения: %s.",
            MAIN_CLIENT,
            ", ".join(_SUPPORTED_MAIN_CLIENTS),
        )
        sys.exit(1)


_validate_config()
