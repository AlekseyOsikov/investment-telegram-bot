"""Команда /research_chunking_stats — статистика по количеству и размеру чанков обеих
стратегий чанкинга (rag/) из уже построенного индекса, без повторного запуска
пайплайна. Технический/диагностический режим (регистрируется и упоминается в /help
только при RESEARCH_ENABLED, как остальные /research_* — см. config.py и CLAUDE.md),
не часть основного инвестиционного сценария бота.

Как и /mcp_tools (mcp_integration/tools_command.py), не входит в ConversationHandler —
ни ввода, ни состояния у команды нет, один CommandHandler. Читает только
rag/index_store.py (SQLite) — не вызывает эмбеддинги и не обращается к Ollama, поэтому
единственная ошибка, которую нужно обрабатывать, — «индекс ещё не построен» (Requirement
«Команда бота со статистикой чанкинга», сценарий «Индекс ещё не построен»).
"""

from __future__ import annotations

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

from config import RAG_INDEX_DIR
from rag.index_store import IndexStats, get_stats

# Порядок стратегий в выводе — тот же, что и в /research_chunking_compare, и
# используемые в rag/index_store.py/rag/cli.py идентификаторы стратегий.
_STRATEGIES = (("fixed", "Фиксированный размер"), ("structural", "Структурная"))


def _format_stats(label: str, stats: IndexStats | None) -> str:
    if stats is None:
        return f"📊 {label}: индекс не построен."
    if stats.total_chunks == 0:
        return f"📊 {label}: индекс построен, но чанков нет (пустой корпус)."
    return (
        f"📊 {label}: чанков — {stats.total_chunks}, "
        f"средний размер — {stats.avg_chars:.0f} символов, "
        f"мин — {stats.min_chars}, макс — {stats.max_chars}."
    )


async def chunking_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /research_chunking_stats — см. докстринг модуля."""
    all_missing = True
    lines: list[str] = []
    for strategy, label in _STRATEGIES:
        stats = get_stats(strategy, RAG_INDEX_DIR)
        if stats is not None:
            all_missing = False
        lines.append(_format_stats(label, stats))

    if all_missing:
        await update.message.reply_text(
            "⚠️ Индекс ещё не построен ни для одной стратегии. Сначала запусти "
            "индексацию из командной строки: `make index` (или `python -m rag.cli`)."
        )
        return

    await update.message.reply_text("\n\n".join(lines))


def build_chunking_stats_handler() -> CommandHandler:
    """Собирает CommandHandler команды /research_chunking_stats для main.py."""
    return CommandHandler("research_chunking_stats", chunking_stats_command)
