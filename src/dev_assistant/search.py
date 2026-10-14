"""Поиск по индексу документации проекта для режима ассистента разработчика (design.md изменения
add-dev-assistant-mode, решение 4): эмбеддинг вопроса с бюджетом → поиск в ОТДЕЛЬНОМ индексе →
отбор по порогу и `top_k`. Без переписывания вопроса и второго этапа отбора слоя `rag`.

Внешние вызовы (эмбеддинг, поиск, проверка индекса) принимаются параметрами — порядок шагов
проверяется без сети (tests/test_dev_assistant_search.py). Исходы:
- `FOUND` — есть фрагменты;
- `NOT_FOUND` — поиск состоялся, подходящих нет;
- `NO_INDEX` — индекс не собран;
- `FAILED` — эмбеддинг/чтение индекса не удались (причина в `failure`)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

STRATEGY = "markdown"

FOUND = "found"
NOT_FOUND = "not_found"
NO_INDEX = "no_index"
FAILED = "failed"


@dataclass(frozen=True)
class DevSearchConfig:
    """Настройки поиска (config.DEV_ASSISTANT_*, RAG_SEARCH_TIMEOUT_SECONDS) — решение оператора."""

    index_dir: str
    top_k: int
    min_score: float
    search_timeout: float
    strategy: str = STRATEGY


@dataclass(frozen=True)
class DevRetrieval:
    outcome: str
    chunks: tuple[Any, ...] = ()
    failure: str | None = None


def select_chunks(results: Sequence[Any], min_score: float, top_k: int) -> list[Any]:
    """Отбор: порог по `score` (`min_score <= 0` — шаг отключён) и не более `top_k` с
    сохранением порядка близости, который вернул индекс."""
    kept = [item for item in results if min_score <= 0 or item.score >= min_score]
    return kept[: max(top_k, 0)]


def retrieve(
    config: DevSearchConfig,
    question: str,
    *,
    embed: Callable[..., list[list[float]]] | None = None,
    search: Callable[..., Sequence[Any]] | None = None,
    index_exists: Callable[[str, str], bool] | None = None,
) -> DevRetrieval:
    """Один поиск на вопрос. Не бросает исключений: сбой — исход `FAILED`. В эмбеддинг уходит
    только текст вопроса. `embed`/`search`/`index_exists` по умолчанию — `embed_texts` и
    `rag.index_store` (импортируются лениво, чтобы модуль не создавал клиент эмбеддингов и не
    грузил faiss при импорте)."""
    if embed is None or search is None or index_exists is None:
        from providers.embeddings_client import embed_texts
        from rag import index_store

        embed = embed or embed_texts
        search = search or index_store.search
        index_exists = index_exists or index_store.index_exists

    try:
        if not index_exists(config.strategy, config.index_dir):
            return DevRetrieval(NO_INDEX)
        vectors = embed([question], budget_seconds=config.search_timeout)
        results = search(config.strategy, config.index_dir, vectors[0], config.top_k)
    except Exception as exc:  # noqa: BLE001 — сбой поиска не роняет режим, пользователю — текст
        logger.warning("Сбой поиска по документации проекта.", exc_info=True)
        return DevRetrieval(FAILED, failure=f"{type(exc).__name__}: {exc}".strip()[:200])

    chunks = select_chunks(results, config.min_score, config.top_k)
    if not chunks:
        return DevRetrieval(NOT_FOUND)
    return DevRetrieval(FOUND, chunks=tuple(chunks))
