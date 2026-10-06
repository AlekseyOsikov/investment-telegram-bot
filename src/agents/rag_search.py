"""Поиск справочных материалов для слоя `rag` у `/smart_agent`: переписывание вопроса в поисковый
запрос, эмбеддинг, поиск кандидатов в индексе и второй этап отбора (design.md изменений
add-smart-agent-rag, решения 2, 6, 7, 10, и add-rag-rerank-and-rewrite, решения 1-3).

Отделён от agents/smart_agent.py: здесь нет состояния чата — только конфигурация и результат
одного поиска. Состояние (`_last_rag_*`, краткосрочная память для истории переписывания)
остаётся в SmartAgent, который передаёт сюда готовые значения и сам записывает итог. Внешние
вызовы (эмбеддинг, поиск в индексе, проверка индекса) принимаются параметрами, поэтому
порядок шагов проверяется без сети (tests/test_rag_search.py)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import rag_context, rag_rewrite

if TYPE_CHECKING:
    from providers.rewrite_client import RewriteBackend

logger = logging.getLogger(__name__)

# Предел токенов ответа модели переписывания: запрос короткий (rag_rewrite.MAX_QUERY_CHARS), а
# «размышления» на моделях с thinking отключаются отдельно (см. request_rewrite).
_REWRITE_MAX_TOKENS = 120

REWRITE_MODE_BOTH = "both"


@dataclass(frozen=True)
class RagSearchConfig:
    """Настройки поиска (config.RAG_* и REWRITE_*) — решение оператора, не пользователя."""

    strategy: str
    index_dir: str
    top_k: int
    min_score: float
    search_timeout: float
    candidates: int
    relative_margin: float
    min_chunk_chars: int
    max_per_doc: int
    rewrite_backend: RewriteBackend | None
    rewrite_timeout: float
    rewrite_search_mode: str
    rewrite_history_questions: int

    def selection_settings(self) -> rag_context.SelectionSettings:
        return rag_context.SelectionSettings(
            top_k=self.top_k,
            min_score=self.min_score,
            relative_margin=self.relative_margin,
            min_chunk_chars=self.min_chunk_chars,
            max_per_doc=self.max_per_doc,
        )


@dataclass(frozen=True)
class Retrieval:
    """Итог одного поиска: материалы, сведения об отборе (None — поиска не было или он не
    удался) и причина сбоя поиска (None — сбоя не было)."""

    materials: rag_context.Materials
    search_info: rag_context.SearchInfo | None = None
    failure: str | None = None


def call_with_deadline(fn: Callable[[], str | None], seconds: float) -> str | None:
    """Выполняет `fn()` в потоке-демоне и ждёт результата не дольше `seconds` ПО ЧАСАМ.
    Таймаут HTTP-клиента — это пауза между чтениями, а не предел общего времени: провайдер,
    держащий соединение пустыми строками keep-alive, «отвечает» через минуты (случай из
    живого прогона /research_rag_compare, см. research/rag_compare_eval.call_with_deadline).
    По истечении времени бросает TimeoutError; поток-сирота доживает сам, его результат
    отбрасывается. Исключение, брошенное `fn`, пробрасывается."""
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — передаётся вызывающему
            box["error"] = exc

    thread = threading.Thread(target=runner, name="rag-rewrite", daemon=True)
    thread.start()
    thread.join(max(seconds, 0.001))
    if thread.is_alive():
        raise TimeoutError(f"нет ответа за {seconds:g} с")
    if "error" in box:
        raise box["error"]
    return box.get("value")



def request_rewrite(
    config: RagSearchConfig,
    backend: RewriteBackend,
    user_text: str,
    history: list[str] | tuple[str, ...] = (),
) -> str | None:
    """Один вызов модели переписывания (в потоке, см. rewrite_query). Без автоповторов
    SDK (повторы умножали бы ожидание), температура 0 (воспроизводимость сравнения
    режимов). «thinking» отключается у deepseek/kimi, как у оценщика
    research/rag_compare.py: короткому запросу не нужны скрытые размышления, а на моделях с
    рассуждениями весь лимит токенов уходил на них и content оставался пустым."""
    extra_body = None if backend.provider == "ollama" else {"thinking": {"type": "disabled"}}
    response = backend.client.with_options(max_retries=0).chat.completions.create(
        model=backend.model,
        messages=rag_rewrite.build_messages(user_text, history),
        max_tokens=_REWRITE_MAX_TOKENS,
        timeout=config.rewrite_timeout,
        temperature=0,
        extra_body=extra_body,
    )
    return response.choices[0].message.content


def rewrite_query(
    config: RagSearchConfig, user_text: str, history: list[str] | tuple[str, ...] = ()
) -> tuple[str | None, str | None]:
    """Переписывает вопрос в поисковый запрос (design.md изменения
    add-rag-rerank-and-rewrite, решения 2, 3, 9). Возвращает (запрос, причина сбоя):
    (None, None) — переписывание не настроено; (None, причина) — вызов не удался
    (недоступный сервер, превышение REWRITE_TIMEOUT_SECONDS, ошибка SDK, пустой или
    непригодный ответ). Не бросает исключений: любой сбой означает «искать по исходному
    вопросу» (БЕЗ вопросов истории), а пользователь о необязательной оптимизации не
    уведомляется. Модель получает текст вопроса и — только если передана `history` (прошлые
    вопросы пользователя, см. SmartAgent._history_for_rewrite) — их тексты; ответы модели,
    память, профиль, инварианты и фрагменты ей не передаются."""
    backend = config.rewrite_backend
    if backend is None:
        return None, None
    try:
        raw = call_with_deadline(
            lambda: request_rewrite(config, backend, user_text, history), config.rewrite_timeout
        )
    except Exception as exc:  # noqa: BLE001 — сбой переписывания не должен ронять ответ
        logger.warning("Сбой переписывания вопроса (%s).", backend.provider, exc_info=True)
        return None, f"{type(exc).__name__}: {exc}".strip()[:200]
    query = rag_rewrite.normalize_response(raw)
    if query is None:
        return None, "модель вернула пустой или непригодный запрос"
    return query, None


def rewrite_status(rewritten: str | None, rewrite_failure: str | None) -> str:
    """Состояние переписывания для SearchInfo.rewrite_status."""
    if rewritten is not None:
        return rag_context.REWRITE_OK
    if rewrite_failure is not None:
        return rag_context.REWRITE_FAILED
    return rag_context.REWRITE_OFF


def search_texts(user_text: str, rewritten: str | None, mode: str) -> tuple[list[str], bool]:
    """Тексты, по которым идёт поиск, и признак «искали по обоим» (режим `both`). Без
    переписанного запроса — только исходный вопрос."""
    if rewritten is None:
        return [user_text], False
    if mode == REWRITE_MODE_BOTH:
        return [user_text, rewritten], True
    return [rewritten], False


def retrieve(
    config: RagSearchConfig,
    user_text: str,
    rewrite: tuple[str | None, str | None] | None = None,
    history: list[str] | None = None,
    *,
    default_history: Callable[[], list[str]] = list,
    embed: Callable | None = None,
    search: Callable | None = None,
    index_exists: Callable[[str, str], bool] | None = None,
) -> Retrieval:
    """Поиск справочных материалов для ОДНОГО вопроса. Пустой результат без предупреждения —
    индекс выбранной стратегии не построен (это не сбой: иначе окружение без индекса
    сопровождалось бы служебным шумом на каждом вопросе). Порядок: переписывание вопроса (если
    настроено; сбой — поиск по исходному вопросу) → эмбеддинг поискового текста → поиск
    `candidates` кандидатов → второй этап отбора (порог и эвристики, не более `top_k`). Любой
    сбой самого поиска — сервер эмбеддингов недоступен или не уложился в бюджет
    `search_timeout`, повреждённые файлы индекса — не роняет ответ: результат пустой, с
    предупреждением и причиной сбоя. В эмбеддинг уходит только поисковый текст (исходный вопрос
    и/или его переписанная форма), а модели переписывания — текст вопроса и, если оператор
    включил историю, прошлые вопросы пользователя (но не ответы модели, не память, не профиль).

    `rewrite` — готовый результат переписывания `(запрос, причина сбоя)` вместо вызова модели:
    так сравнение режимов (research/rag_compare.py) переписывает вопрос один раз и
    переиспользует результат между режимами. `history` — прошлые вопросы для модели
    переписывания: None — взять `default_history()` (рабочий путь: краткосрочная память
    активного профиля), список (в том числе пустой) — использовать как есть. Слой включён —
    проверяет вызывающий. `embed`, `search`, `index_exists` — подмена внешних вызовов для тестов
    (по умолчанию embed_texts и rag.index_store; импортируются лениво, чтобы модуль не создавал
    клиент эмбеддингов и не грузил faiss при импорте)."""
    if embed is None or search is None or index_exists is None:
        from providers.embeddings_client import embed_texts
        from rag import index_store

        embed = embed or embed_texts
        search = search or index_store.search
        index_exists = index_exists or index_store.index_exists
    try:
        if not index_exists(config.strategy, config.index_dir):
            return Retrieval(rag_context.Materials())
        if rewrite is not None:
            rewritten, rewrite_failure = rewrite
            used_history = list(history or [])
        else:
            used_history = list(history) if history is not None else default_history()
            rewritten, rewrite_failure = rewrite_query(config, user_text, used_history)
        # Прошлые вопросы реально повлияли на запрос только если вызов удался: при сбое
        # поиск идёт по исходному вопросу без истории (спека rag-query-rewrite).
        history_used = len(used_history) if rewritten is not None else 0
        texts, both = search_texts(user_text, rewritten, config.rewrite_search_mode)
        # Один запрос к серверу эмбеддингов на все тексты поиска (бюджет ожидания — общий).
        vectors = embed(texts, budget_seconds=config.search_timeout)
        limit = max(config.candidates, config.top_k)
        found = rag_rewrite.merge_candidates(
            *(search(config.strategy, config.index_dir, vector, limit) for vector in vectors)
        )
    except Exception as exc:  # noqa: BLE001 — сбой поиска не должен ронять ответ
        logger.warning("Сбой поиска справочных материалов.", exc_info=True)
        reason = f"{type(exc).__name__}: {exc}".strip()
        return Retrieval(
            rag_context.Materials(warning=rag_context.FAILURE_WARNING), failure=reason[:200]
        )
    selection = rag_context.select_chunks(found, config.selection_settings())
    info = rag_context.SearchInfo(
        query=rewritten,
        rewrite_status=rewrite_status(rewritten, rewrite_failure),
        rewrite_reason=rewrite_failure,
        both=both,
        candidates=selection.candidates,
        selected=len(selection.chunks),
        dropped=selection.dropped,
        history_used=history_used,
    )
    return Retrieval(rag_context.Materials(chunks=selection.chunks, searched=True), info)
