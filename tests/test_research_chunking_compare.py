"""Тесты чистого форматирования /research_chunking_compare (research/chunking_compare.py):
строка метаданных чанка, результат с усечением текста, блок стратегии.

Без сети, Telegram и индекса. Не покрыто (нужны Telegram, эмбеддинги и индекс):
chunking_compare_command, chunking_compare_receive_query,
build_chunking_compare_conversation_handler.
"""

from rag.index_store import SearchResult
from research import chunking_compare as cc


def _result(**overrides) -> SearchResult:
    values = {
        "chunk_id": "doc1-3",
        "source": "doc1.txt",
        "title": "Заголовок документа",
        "author": "Автор",
        "date": "2026-01-02",
        "chunk_index": 3,
        "text": "Короткий текст чанка.",
        "score": 0.8123,
    }
    values.update(overrides)
    return SearchResult(**values)


def test_metadata_has_source_author_date_chunk_number_and_id():
    assert cc._format_metadata(_result()) == (
        "    источник: doc1.txt | автор: Автор | дата: 2026-01-02 | чанк №3 | chunk_id: doc1-3"
    )


def test_metadata_without_author_or_date_shows_dashes():
    line = cc._format_metadata(_result(author=None, date=None))
    assert "автор: — | дата: —" in line
    line = cc._format_metadata(_result(author="", date=""))
    assert "автор: — | дата: —" in line


def test_result_has_position_score_title_metadata_separator_and_text():
    text = cc._format_result(_result(), 2)
    lines = text.split("\n")
    assert lines[0] == "2. [0.812] Заголовок документа"
    assert lines[1] == cc._format_metadata(_result())
    assert lines[2] == cc._SEPARATOR
    assert lines[3] == "Короткий текст чанка."
    assert len(lines) == 4


def test_result_keeps_text_of_exactly_200_chars_whole():
    text = cc._format_result(_result(text="а" * 200), 1)
    assert text.endswith("а" * 200)
    assert "…" not in text


def test_result_truncates_text_over_200_chars_with_ellipsis():
    text = cc._format_result(_result(text="б" * 201), 1)
    assert text.endswith("б" * 200 + "…")
    assert "б" * 201 not in text


def test_result_score_is_shown_with_three_decimals():
    assert "[0.500]" in cc._format_result(_result(score=0.5), 1)
    assert "[1.000]" in cc._format_result(_result(score=1), 1)


def test_empty_strategy_block_says_nothing_found():
    assert cc._format_strategy_block("Структурная", []) == "📚 Структурная: ничего не найдено."


def test_strategy_block_numbers_results_and_separates_them():
    results = [_result(title="Первый"), _result(title="Второй", score=0.5)]
    block = cc._format_strategy_block("Фиксированный размер", results)
    assert block.startswith("📚 Фиксированный размер:\n1. [0.812] Первый")
    assert "\n2. [0.500] Второй" in block
    assert f"\n{cc._SEPARATOR}\n2. " in block
    assert block.count("2. [") == 1


def test_strategies_are_fixed_and_structural_in_order():
    assert [key for key, _ in cc._STRATEGIES] == ["fixed", "structural"]
    assert all(label for _, label in cc._STRATEGIES)
