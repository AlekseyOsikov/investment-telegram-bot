"""Тесты чистого форматирования /research_chunking_stats (research/chunking_stats.py).

Без Telegram и индекса. Не покрыто (нужны Telegram и индекс на диске): chunking_stats_command,
build_chunking_stats_handler.
"""

from rag.index_store import IndexStats
from research import chunking_stats as cs


def test_stats_line_has_count_average_min_and_max():
    stats = IndexStats(total_chunks=120, avg_chars=845.6, min_chars=40, max_chars=1500)
    assert cs._format_stats("Структурная", stats) == (
        "📊 Структурная: чанков — 120, средний размер — 846 символов, мин — 40, макс — 1500."
    )


def test_missing_index_is_reported_as_not_built():
    assert cs._format_stats("Фиксированный размер", None) == (
        "📊 Фиксированный размер: индекс не построен."
    )


def test_empty_index_is_reported_as_empty_corpus():
    stats = IndexStats(total_chunks=0, avg_chars=0.0, min_chars=0, max_chars=0)
    text = cs._format_stats("Структурная", stats)
    assert "индекс построен, но чанков нет" in text
    assert "средний размер" not in text


def test_single_chunk_is_formatted_like_any_other():
    stats = IndexStats(total_chunks=1, avg_chars=10.0, min_chars=10, max_chars=10)
    assert "чанков — 1, средний размер — 10 символов, мин — 10, макс — 10" in cs._format_stats(
        "X", stats
    )


def test_strategies_order_matches_compare_command():
    from research import chunking_compare

    assert cs._STRATEGIES == chunking_compare._STRATEGIES
