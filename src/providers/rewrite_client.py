"""Клиент и модель переписывания вопроса — выбираются REWRITE_PROVIDER/REWRITE_MODEL (config.py;
design.md изменения add-rag-rerank-and-rewrite, решения 2 и 5).

Устроен по образцу providers/main_client.py и вынесен из config.py по той же причине (цикл
импорта: клиенты импортируют config.py). `rewrite_backend` — `None`, если переписывание не
настроено или недоступно, и тогда SmartAgent ищет по вопросу как есть:

- REWRITE_PROVIDER пуст — переписывание выключено (значение по умолчанию);
- `deepseek`/`kimi` без API-ключа — предупреждение в журнале и `None`: переписывание —
  необязательная функция, отсутствие ключа не должно останавливать бот (в отличие от ключа
  провайдера основного потока, см. deepseek_client.py/kimi_client.py);
- `ollama` — ключ не нужен; недоступность сервера обнаруживается при вызове, а не при запуске.

Выбор провайдера — настройка оператора (.env, перезапуск); пользователь чата его не меняет.
Модель по умолчанию для deepseek/kimi — быстрая модель провайдера, а не MAIN_MODEL:
переписывание должно быть дешёвым и не зависеть от провайдера основного потока.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from config import REWRITE_MODEL, REWRITE_PROVIDER

from .deepseek_client import DEEPSEEK_API_KEY, DEEPSEEK_MODEL_FLASH, deepseek_client
from .kimi_client import KIMI_API_KEY, KIMI_MODEL_K2_6, kimi_client
from .ollama_client import ollama_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteBackend:
    """Клиент OpenAI-совместимого API, модель и подпись провайдера (для журнала и показа
    состояния)."""

    client: OpenAI
    model: str
    provider: str


def build_rewrite_backend(
    provider: str,
    model: str,
    deepseek_key: str | None,
    kimi_key: str | None,
) -> RewriteBackend | None:
    """Выбор клиента переписывания по настройкам. Чистая по отношению к окружению: ключи и
    настройки приходят аргументами (проверяется без сети)."""
    if not provider:
        return None
    if provider == "ollama":
        return RewriteBackend(ollama_client, model, "ollama")
    if provider == "deepseek":
        if not deepseek_key:
            logger.warning(
                "REWRITE_PROVIDER=deepseek, но DEEPSEEK_API_KEY не задан — переписывание "
                "вопроса пропускается, поиск идёт по вопросу как есть."
            )
            return None
        return RewriteBackend(deepseek_client, model or DEEPSEEK_MODEL_FLASH, "deepseek")
    if provider == "kimi":
        if not kimi_key:
            logger.warning(
                "REWRITE_PROVIDER=kimi, но KIMI_API_KEY не задан — переписывание вопроса "
                "пропускается, поиск идёт по вопросу как есть."
            )
            return None
        return RewriteBackend(kimi_client, model or KIMI_MODEL_K2_6, "kimi")
    # Недопустимое значение config._validate_config() отсекает ещё до импорта этого модуля.
    return None


rewrite_backend = build_rewrite_backend(
    REWRITE_PROVIDER, REWRITE_MODEL, DEEPSEEK_API_KEY, KIMI_API_KEY
)
