"""Подключение к DeepSeek API (OpenAI-совместимый).

Вынесено из config.py в отдельный файл, т.к. подключение к Kimi (см. kimi_client.py)
устроено иначе: свой API-ключ, свой базовый URL. config.py (на уровень выше, вне пакета
providers) остаётся источником общих для бота настроек (Telegram-токен,
MAIN_CLIENT/MAIN_MODEL, лимиты, системный промпт, логирование) и импортируется здесь
исключительно ради побочного эффекта — load_dotenv() и logging.basicConfig() должны
отработать раньше, чем этот модуль прочитает переменные окружения и что-либо залогирует.

Обязательность DEEPSEEK_API_KEY зависит от MAIN_CLIENT (config.py): если основной поток
бота работает через DeepSeek (MAIN_CLIENT=deepseek, значение по умолчанию), ключ
обязателен и его отсутствие завершает процесс при старте — как и раньше. Если основной
поток работает через Kimi (MAIN_CLIENT=kimi), DeepSeek нужен только сценариям
/research_models на его моделях, поэтому отсутствие ключа — не фатально, только
предупреждение.
"""

from __future__ import annotations

import logging
import os
import sys

from openai import OpenAI

from config import MAIN_CLIENT

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# DEEPSEEK_MODEL_PRO/DEEPSEEK_MODEL_FLASH — модели DeepSeek, сравниваемые в
# /research_models (research/models.py). MAIN_MODEL (модель основного потока) не
# определяется здесь — он общий для обоих провайдеров и живёт в config.py, т.к. в
# конкретный момент используется только с одним из них (см. main_client.py).
DEEPSEEK_MODEL_PRO = os.getenv("DEEPSEEK_MODEL_PRO", "deepseek-v4-pro")
DEEPSEEK_MODEL_FLASH = os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash")

if MAIN_CLIENT == "deepseek" and not DEEPSEEK_API_KEY:
    logger.error(
        "Отсутствует обязательная переменная окружения DEEPSEEK_API_KEY "
        "(MAIN_CLIENT=deepseek). Скопируйте .env.example в .env и заполните значение."
    )
    sys.exit(1)
elif not DEEPSEEK_API_KEY:
    logger.warning(
        "DEEPSEEK_API_KEY не задан — сценарии /research_models на моделях DeepSeek будут "
        "возвращать ошибку авторизации при вызове. Основной поток бота работает через "
        "Kimi (MAIN_CLIENT=kimi), поэтому это не блокирует запуск."
    )

# Плейсхолдер вместо пустого ключа — см. тот же приём и его обоснование в kimi_client.py:
# конструктор OpenAI() поднимает ошибку сразу при отсутствующем api_key, а не только при
# первом реальном запросе, что свело бы на нет необязательность ключа при MAIN_CLIENT=kimi.
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY or "not-configured", base_url=DEEPSEEK_BASE_URL)
