"""Подключение к Kimi (Moonshot AI) API — OpenAI-совместимый, но отдельный от DeepSeek:
свой ключ и свой базовый URL (см. также deepseek_client.py).

Обязательность KIMI_API_KEY зависит от MAIN_CLIENT (config.py): если основной поток бота
работает через Kimi (MAIN_CLIENT=kimi), ключ обязателен и его отсутствие завершает
процесс при старте. Если основной поток работает через DeepSeek (MAIN_CLIENT=deepseek,
значение по умолчанию), Kimi нужен только сценариям /research_models на его моделях,
поэтому отсутствие ключа — не фатально: такие сценарии просто вернут ошибку авторизации
при вызове API, что уже перехватывается общим блоком обработки ошибок в
research_models.py, как и любая другая ошибка API.
"""

from __future__ import annotations

import logging
import os
import sys

from openai import OpenAI

from config import MAIN_CLIENT

logger = logging.getLogger(__name__)

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")

# Модели Kimi, сравниваемые в /research_models (research_models.py).
KIMI_MODEL_K3 = os.getenv("KIMI_MODEL_K3", "kimi-k3")
KIMI_MODEL_K2_6 = os.getenv("KIMI_MODEL_K2_6", "kimi-k2.6")

if MAIN_CLIENT == "kimi" and not KIMI_API_KEY:
    logger.error(
        "Отсутствует обязательная переменная окружения KIMI_API_KEY "
        "(MAIN_CLIENT=kimi). Скопируйте .env.example в .env и заполните значение."
    )
    sys.exit(1)
elif not KIMI_API_KEY:
    logger.warning(
        "KIMI_API_KEY не задан — сценарии /research_models на моделях Kimi будут "
        "возвращать ошибку авторизации при вызове. Остальной бот продолжит работать."
    )

# Плейсхолдер вместо пустого ключа: конструктор OpenAI() поднимает ошибку сразу же при
# отсутствующем api_key (ещё до первого реального запроса) — это свело бы на нет
# опциональность KIMI_API_KEY при MAIN_CLIENT=deepseek. С плейсхолдером клиент создаётся
# без ошибок, а настоящая ошибка авторизации придёт от сервера Kimi при первом вызове.
kimi_client = OpenAI(api_key=KIMI_API_KEY or "not-configured", base_url=KIMI_BASE_URL)
