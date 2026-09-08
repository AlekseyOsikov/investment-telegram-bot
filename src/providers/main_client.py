"""Клиент основного потока бота — выбирается переменной окружения MAIN_CLIENT (config.py).

main.py и research-режимы, которые сами не выбирают конкретную модель
(research/reasoning.py, research/temperature.py, research/constraints.py) —
импортируют main_client отсюда вместо того, чтобы обращаться к deepseek_client.py или
kimi_client.py напрямую. Благодаря этому смена MAIN_CLIENT в .env не требует правок в
этих модулях: main_client всегда указывает на клиента, соответствующего MAIN_CLIENT.
(research/models.py — исключение: он всегда сравнивает обе пары моделей обоих
провайдеров явно и от MAIN_CLIENT не зависит.)

Сборка main_client вынесена в отдельный модуль, а не в config.py, чтобы не создавать
цикл импорта: deepseek_client.py и kimi_client.py импортируют config.py (ради
load_dotenv()/logging.basicConfig() и переменной MAIN_CLIENT), поэтому config.py не
может в ответ импортировать их.
"""

from __future__ import annotations

from config import MAIN_CLIENT

from .deepseek_client import deepseek_client
from .kimi_client import kimi_client

main_client = deepseek_client if MAIN_CLIENT == "deepseek" else kimi_client
