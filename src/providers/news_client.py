"""Клиент и модель разбора новостей — выбираются NEWS_PROVIDER/NEWS_MODEL (config.py; design.md
изменения add-news-digest, решение 5).

Устроен по образцу providers/rewrite_client.py. `news_backend` — `None`, если разбор недоступен, и
тогда дайджест присылает новости без разбора (заголовок, источник, ссылка):

- `ollama` без модели — `None` (дефолта модели нет, сервер Ollama общий с `MAIN_CLIENT`);
- `deepseek`/`kimi` без API-ключа — предупреждение в журнале и `None`: разбор необязателен, его
  отсутствие не должно останавливать бот;
- модель `deepseek`/`kimi` по умолчанию — быстрая модель провайдера, не MAIN_MODEL.

Выбор провайдера — настройка оператора (.env, перезапуск); пользователь чата его не меняет.
`is_local` нужен вызывающему коду: облачные параметры (`extra_body`) локальному серверу не
отправляются.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from config import NEWS_MODEL, NEWS_PROVIDER

from .deepseek_client import DEEPSEEK_API_KEY, DEEPSEEK_MODEL_FLASH, deepseek_client
from .kimi_client import KIMI_API_KEY, KIMI_MODEL_K2_6, kimi_client
from .ollama_client import ollama_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsBackend:
    """Клиент OpenAI-совместимого API, модель и подпись провайдера (для журнала)."""

    client: OpenAI
    model: str
    provider: str

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"


def build_news_backend(
    provider: str,
    model: str,
    deepseek_key: str | None,
    kimi_key: str | None,
) -> NewsBackend | None:
    """Выбор клиента разбора по настройкам. Чистая по отношению к окружению: ключи и настройки
    приходят аргументами (проверяется без сети)."""
    if provider == "ollama":
        if not model:
            logger.warning(
                "NEWS_PROVIDER=ollama требует модель в NEWS_MODEL — новости будут приходить "
                "без разбора."
            )
            return None
        return NewsBackend(ollama_client, model, "ollama")
    if provider == "deepseek":
        if not deepseek_key:
            logger.warning(
                "NEWS_PROVIDER=deepseek, но DEEPSEEK_API_KEY не задан — новости будут "
                "приходить без разбора."
            )
            return None
        return NewsBackend(deepseek_client, model or DEEPSEEK_MODEL_FLASH, "deepseek")
    if provider == "kimi":
        if not kimi_key:
            logger.warning(
                "NEWS_PROVIDER=kimi, но KIMI_API_KEY не задан — новости будут приходить "
                "без разбора."
            )
            return None
        return NewsBackend(kimi_client, model or KIMI_MODEL_K2_6, "kimi")
    # Недопустимое значение config._validate_config() отсекает ещё до импорта этого модуля.
    return None


news_backend = build_news_backend(NEWS_PROVIDER, NEWS_MODEL, DEEPSEEK_API_KEY, KIMI_API_KEY)
