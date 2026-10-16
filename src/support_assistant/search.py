"""Поиск по индексу справки поддержки (design.md изменения add-support-assistant, решение 6):
эмбеддинг вопроса с бюджетом → поиск в ОТДЕЛЬНОМ индексе → порог и `top_k`. Механику (порядок шагов,
исходы, неперехват исключений) делит с `/dev` (`dev_assistant.search`): здесь только своя
конфигурация. Без переписывания вопроса и второго этапа отбора слоя `rag`.

Исходы: `FOUND`, `NOT_FOUND`, `NO_INDEX`, `FAILED` (причина в `failure`). Внешние вызовы
(`embed`, `search`, `index_exists`) подставляются параметрами
(tests/test_support_assistant_search.py)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from dev_assistant import search as _shared
from dev_assistant.search import (
    FAILED,
    FOUND,
    NO_INDEX,
    NOT_FOUND,
    STRATEGY,
    select_chunks,
)
from dev_assistant.search import DevRetrieval as Retrieval

__all__ = [
    "FAILED",
    "FOUND",
    "NOT_FOUND",
    "NO_INDEX",
    "STRATEGY",
    "Retrieval",
    "SupportSearchConfig",
    "retrieve",
    "select_chunks",
]


@dataclass(frozen=True)
class SupportSearchConfig:
    """Настройки поиска (config.SUPPORT_*, RAG_SEARCH_TIMEOUT_SECONDS) — решение оператора."""

    index_dir: str
    top_k: int
    min_score: float
    search_timeout: float
    strategy: str = STRATEGY


def retrieve(
    config: SupportSearchConfig,
    question: str,
    *,
    embed: Callable[..., list[list[float]]] | None = None,
    search: Callable[..., Sequence[Any]] | None = None,
    index_exists: Callable[[str, str], bool] | None = None,
) -> Retrieval:
    """Один поиск на вопрос. Не бросает исключений: сбой — исход `FAILED`. В эмбеддинг уходит
    только текст вопроса."""
    return _shared.retrieve(
        config, question, embed=embed, search=search, index_exists=index_exists
    )
