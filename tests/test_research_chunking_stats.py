"""Тесты чистого форматирования /research_chunking_stats (research/chunking_stats.py).

Форматирование — без Telegram и индекса; обработчик — на реальном индексе во временном
каталоге (синтетические векторы) и заглушке Update.
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


# --- обработчик команды и сборка: реальный индекс во временном каталоге, заглушка Update ---

import asyncio  # noqa: E402

from rag.index_store import IndexRecord, build_index  # noqa: E402


class _FakeMessage:
    def __init__(self):
        self.replies: list[str] = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class _FakeUpdate:
    def __init__(self):
        self.message = _FakeMessage()


def make_records(sizes):
    return [
        IndexRecord(
            source="doc.txt",
            title="Документ",
            author=None,
            date=None,
            chunk_index=i,
            text="а" * size,
            embedding=[1.0, float(i + 1)],
        )
        for i, size in enumerate(sizes)
    ]


def run_command(monkeypatch, index_dir):
    monkeypatch.setattr(cs, "RAG_INDEX_DIR", str(index_dir))
    update = _FakeUpdate()
    asyncio.run(cs.chunking_stats_command(update, None))
    return update.message.replies


def test_command_without_any_index_asks_to_build_it(monkeypatch, tmp_path):
    replies = run_command(monkeypatch, tmp_path)
    assert len(replies) == 1
    assert "Индекс ещё не построен ни для одной стратегии" in replies[0]
    assert "make index" in replies[0]
    assert "📊" not in replies[0]


def test_command_with_one_index_reports_both_strategies_in_order(monkeypatch, tmp_path):
    build_index("structural", make_records([10, 30]), str(tmp_path))
    replies = run_command(monkeypatch, tmp_path)
    assert len(replies) == 1
    first, second = replies[0].split("\n\n")
    assert first == "📊 Фиксированный размер: индекс не построен."
    assert second == (
        "📊 Структурная: чанков — 2, средний размер — 20 символов, мин — 10, макс — 30."
    )


def test_command_with_both_indexes_reports_each_stats(monkeypatch, tmp_path):
    build_index("fixed", make_records([5]), str(tmp_path))
    build_index("structural", make_records([10, 20, 30]), str(tmp_path))
    text = run_command(monkeypatch, tmp_path)[0]
    assert "Фиксированный размер: чанков — 1" in text
    assert "Структурная: чанков — 3, средний размер — 20 символов" in text
    assert "не построен" not in text


def test_handler_is_command_handler_for_research_chunking_stats():
    handler = cs.build_chunking_stats_handler()
    assert "research_chunking_stats" in handler.commands
    assert handler.callback is cs.chunking_stats_command
