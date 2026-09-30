"""Команда /research_chunking_compare — сравнение результатов поиска по запросу
между двумя стратегиями чанкинга (rag/) на уже построенном индексе. Технический
режим (регистрируется и упоминается в /help только при RESEARCH_ENABLED, как
остальные /research_* — см. config.py и CLAUDE.md), не часть основного
инвестиционного сценария бота.

ConversationHandler с ОДНИМ состоянием WAITING_QUERY, зацикленным до /cancel
(design.md изменения add-rag-indexing-pipeline, раздел «Структура кода») — сравнение
нескольких запросов подряд в одной сессии ожидаемо, поэтому не одноразовый мастер, как
research/constraints.py/research/temperature.py.

Ollama вызывается через тот же SDK `openai`, что DeepSeek/Kimi
(providers/embeddings_client.py), поэтому переиспользуем
research/_shared.api_error_to_message для перевода сбоя эндпоинта эмбеддингов в
сообщение на русском (Requirement «Команда бота для сравнения стратегий чанкинга»,
сценарий «Эндпоинт эмбеддингов недоступен во время запроса»).
"""

from __future__ import annotations

from telegram import Update
from telegram.ext import (
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import RAG_COMPARE_TOP_K, RAG_INDEX_DIR, TELEGRAM_MESSAGE_LIMIT
from providers.embeddings_client import embed_texts
from rag.index_store import SearchResult, index_exists, search

from ._shared import api_error_to_message, build_cancel_handler

WAITING_QUERY = 0

# Порядок и подписи стратегий — те же идентификаторы, что в rag/index_store.py и
# rag/cli.py, тот же порядок вывода, что в research/chunking_stats.py.
_STRATEGIES = (("fixed", "Фиксированный размер"), ("structural", "Структурная"))

# api_error_to_message() поддерживает MAIN_CLIENT_LABEL/MAIN_API_KEY_ENV_VAR по
# умолчанию — здесь свои: сбой относится к серверу эмбеддингов, а не к MAIN_CLIENT.
_EMBEDDINGS_LABEL = "Ollama"
_EMBEDDINGS_CONFIG_HINT = "EMBEDDINGS_BASE_URL/EMBEDDINGS_MODEL"

_INTRO_TEXT = (
    "🔬 Режим сравнения двух стратегий чанкинга.\n\n"
    "Это техническое исследование, а не инвестиционный сервис. Отправь произвольный "
    "текстовый запрос — покажу, что находит поиск по каждой из двух стратегий "
    "чанкинга. Можно отправлять запросы один за другим, пока не отправишь /cancel."
)


def _format_metadata(result: SearchResult) -> str:
    """Строка метаданных найденного чанка — источник, автор/дата (rag/filenames.py),
    номер чанка внутри документа и его chunk_id в индексе (rag/index_store.py)."""
    author = result.author or "—"
    date = result.date or "—"
    return (
        f"    источник: {result.source} | автор: {author} | дата: {date} | "
        f"чанк №{result.chunk_index} | chunk_id: {result.chunk_id}"
    )


# Один и тот же разделитель используется дважды: отделяет метаданные чанка от его
# текста (внутри одного результата) и отделяет один результат от следующего (перед
# номером и заголовком очередного чанка) — так результаты не сливаются визуально в
# одну простыню текста.
_SEPARATOR = "--------"


def _format_result(result: SearchResult, position: int) -> str:
    snippet = result.text if len(result.text) <= 200 else result.text[:200] + "…"
    return (
        f"{position}. [{result.score:.3f}] {result.title}\n"
        f"{_format_metadata(result)}\n"
        f"{_SEPARATOR}\n"
        f"{snippet}"
    )


def _format_strategy_block(label: str, results: list[SearchResult]) -> str:
    if not results:
        return f"📚 {label}: ничего не найдено."
    lines = [_format_result(result, i + 1) for i, result in enumerate(results)]
    return f"📚 {label}:\n" + f"\n{_SEPARATOR}\n".join(lines)


async def chunking_compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим сравнения (/research_chunking_compare)."""
    await update.message.reply_text(_INTRO_TEXT)
    return WAITING_QUERY


async def chunking_compare_receive_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Обрабатывает один запрос: эмбеддинг запроса + поиск по обеим стратегиям +
    ответ; затем остаётся в WAITING_QUERY для следующего запроса."""
    query = update.message.text
    if not query or not query.strip():
        await update.message.reply_text("👉 Пожалуйста, отправь текст запроса.")
        return WAITING_QUERY

    missing_labels = [
        label for strategy, label in _STRATEGIES if not index_exists(strategy, RAG_INDEX_DIR)
    ]
    if missing_labels:
        await update.message.reply_text(
            "⚠️ Индекс ещё не построен (" + ", ".join(missing_labels) + "). Сначала "
            "запусти индексацию из командной строки: `make index` (или `python -m rag.cli`)."
        )
        return ConversationHandler.END

    try:
        query_vector = embed_texts([query])[0]
    except Exception as exc:  # noqa: BLE001 — последний рубеж, переводим в сообщение ниже
        message = api_error_to_message(exc, _EMBEDDINGS_LABEL, _EMBEDDINGS_CONFIG_HINT)
        if message is None:
            message = "❌ Не удалось получить эмбеддинг запроса (непредвиденная ошибка)."
        await update.message.reply_text(message)
        # Остаёмся в режиме, а не завершаем диалог: embed_texts() уже сама повторила
        # запрос один раз (providers/embeddings_client.py) — если сбой всё равно
        # произошёл, это может быть как разовая проблема (например, Ollama как раз
        # подгружает модель после простоя), так и более долгий сбой; в обоих случаях
        # заставлять пользователя заново вызывать /research_chunking_compare ради
        # повторной попытки — лишнее трение для инструмента, специально рассчитанного
        # на несколько запросов подряд.
        await update.message.reply_text("👉 Можешь попробовать ещё раз или отправить /cancel.")
        return WAITING_QUERY

    # Каждая стратегия — отдельным сообщением (а не одним общим), чтобы результаты
    # двух стратегий не сливались визуально в одну простыню текста; длинный блок
    # одной стратегии при этом всё равно режется по TELEGRAM_MESSAGE_LIMIT, как и
    # обычные ответы (main.py:handle_message).
    for strategy, label in _STRATEGIES:
        results = search(strategy, RAG_INDEX_DIR, query_vector, RAG_COMPARE_TOP_K)
        block = _format_strategy_block(label, results)
        for i in range(0, len(block), TELEGRAM_MESSAGE_LIMIT):
            await update.message.reply_text(block[i : i + TELEGRAM_MESSAGE_LIMIT])

    await update.message.reply_text("👉 Можешь отправить следующий запрос или /cancel.")
    return WAITING_QUERY


chunking_compare_cancel = build_cancel_handler("chunking_compare_active")


def build_chunking_compare_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler команды /research_chunking_compare для main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("research_chunking_compare", chunking_compare_command)],
        states={
            WAITING_QUERY: [MessageHandler(text_filter, chunking_compare_receive_query)],
        },
        fallbacks=[CommandHandler("cancel", chunking_compare_cancel)],
    )
