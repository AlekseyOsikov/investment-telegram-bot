"""Общая конфигурация, логирование и настройки, не привязанные к конкретному провайдеру.

Подключения к LLM-провайдерам (API-ключ, базовый URL, клиент openai.OpenAI, идентификаторы
моделей) вынесены в отдельные модули — providers/deepseek_client.py и
providers/kimi_client.py. Какой из них обслуживает основной поток бота, определяет
MAIN_CLIENT (см. ниже) — от него зависит, какой из двух API-ключей обязателен для
старта, а какой нужен только техническому режиму /research_models и не должен блокировать
запуск остального бота (эту проверку выполняет каждый клиентский модуль сам, см. их
докстринги). Сам выбор клиента для основного потока (объект main_client) собирается в
providers/main_client.py — не здесь, чтобы не создавать цикл импорта:
providers/deepseek_client.py и providers/kimi_client.py импортируют config.py ради
побочного эффекта (load_dotenv() и logging.basicConfig() должны отработать раньше, чем
они читают переменные окружения и логируют), поэтому только config.py вызывает
load_dotenv(), а сам он не может импортировать их в ответ.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

from dotenv import load_dotenv
from telegram.warnings import PTBUserWarning

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Провайдер, обслуживающий основной поток бота (main.py) и research-режимы, которые сами
# не выбирают конкретную модель (research/reasoning.py, research/temperature.py,
# research/constraints.py) — все они используют providers/main_client.py.
# research/models.py не зависит от MAIN_CLIENT: он всегда сравнивает обе пары моделей
# обоих провайдеров.
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

# Каталог для JSON-файлов истории диалога агента (agents/agent.py), один файл на
# chat_id. Единственное место в проекте, где история диалога сохраняется на диск —
# см. «Ограничения безопасности» в CLAUDE.md про то, почему это осознанное
# исключение, а не общее правило для всего бота.
AGENT_HISTORY_DIR = os.getenv("AGENT_HISTORY_DIR", "data/agent_history")

# Управление контекстом агента (agents/agent.py): вместо передачи в LLM всей
# сохранённой истории целиком, более старая её часть периодически сворачивается в
# текстовую сводку (summary), а «свежий хвост» (минимум AGENT_CONTEXT_RECENT_PAIRS
# последних пар вопрос-ответ, плюс ещё до AGENT_SUMMARY_CHUNK_PAIRS-1 пар, ещё не
# набравших полный блок на свёртку) передаётся как есть. Подробности алгоритма — в
# докстринге Agent._maybe_update_summary (agents/agent.py).
AGENT_CONTEXT_RECENT_PAIRS = int(os.getenv("AGENT_CONTEXT_RECENT_PAIRS", "10"))
AGENT_SUMMARY_CHUNK_PAIRS = int(os.getenv("AGENT_SUMMARY_CHUNK_PAIRS", "10"))
AGENT_SUMMARY_MAX_TOKENS = int(os.getenv("AGENT_SUMMARY_MAX_TOKENS", "500"))

# Системный промпт для сворачивания истории агента в сводку — отдельный от основного
# SYSTEM_PROMPT (та роль — отвечать пользователю на инвестиционные вопросы, эта —
# техническая задача сжатия диалога без искажения фактов). Не инвестиционный совет,
# поэтому дисклеймеры из SYSTEM_PROMPT здесь не нужны.
AGENT_SUMMARY_SYSTEM_PROMPT = (
    "Ты сжимаешь историю диалога Telegram-бота с пользователем в краткую сводку для "
    "внутреннего использования другой моделью — сама сводка пользователю не показывается.\n\n"
    "Тебе дают текущую сводку более ранней части диалога (может быть пустой, если это "
    "первое сворачивание) и следующий по времени фрагмент диалога. Верни ОДНУ новую "
    "сводку, объединяющую и старую сводку, и новый фрагмент.\n\n"
    "Правила:\n"
    "1. Сохраняй все факты, значимые для продолжения диалога: о чём спрашивал "
    "пользователь, какие темы/активы/цели упоминались, какие ответы/рекомендации "
    "уже были даны.\n"
    "2. Не придумывай ничего, чего не было в диалоге.\n"
    "3. Пиши кратко и по-русски, без вступлений вида «вот сводка» — сразу текст сводки.\n"
    "4. Не включай в сводку номера счетов/карт, паспортные данные и другие "
    "чувствительные персональные данные, даже если пользователь их упоминал."
)

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

# python-telegram-bot предупреждает про каждый ConversationHandler, где в состояниях
# есть CallbackQueryHandler, а per_message=False (по умолчанию) — это ожидаемо и
# безопасно для наших research-режимов (research/constraints.py, reasoning.py,
# temperature.py, models.py): их состояния вперемешку принимают и текст (вопрос,
# max_tokens, стоп-слова), и нажатия inline-кнопок, а per_message=True требует, чтобы
# *все* обработчики (entry_points/states/fallbacks) были CallbackQueryHandler — что
# ломает ввод текста в этих режимах. Каждый чат ведёт только один активный диалог, так
# что отслеживание разговора по конкретному сообщению с кнопками (то, что даёт
# per_message=True) здесь не нужно — предупреждение только шумит в логах при старте.
warnings.filterwarnings(
    "ignore",
    message=r"If 'per_message=False', 'CallbackQueryHandler' will not be tracked for every message\.",
    category=PTBUserWarning,
)

logger = logging.getLogger(__name__)


def _validate_config() -> None:
    """Проверяет наличие обязательных переменных окружения перед стартом.

    Обязательность DEEPSEEK_API_KEY/KIMI_API_KEY зависит от MAIN_CLIENT и проверяется
    отдельно в providers/deepseek_client.py/providers/kimi_client.py (модуль выбранного провайдера сам
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
