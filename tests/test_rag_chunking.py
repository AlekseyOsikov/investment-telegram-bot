"""Тесты двух стратегий чанкинга (rag/chunking.py) — чистые функции на синтетическом
тексте, по тому же принципу, что test_task_state.py/test_market_tools.py: без сети,
без subprocess/pdftotext и без файловой системы."""

from rag.chunking import chunk_fixed_size, chunk_structural
from rag.extraction import clean_text


# --------------------------------------------------------------------------- #
# Фиксированный размер
# --------------------------------------------------------------------------- #


def test_fixed_size_chunk_length_bounded():
    text = "0123456789" * 5  # 50 символов
    chunks = chunk_fixed_size(text, chunk_chars=10, overlap=3)
    assert chunks  # хоть один чанк получен
    assert all(len(chunk.text) <= 10 for chunk in chunks)


def test_fixed_size_consecutive_chunks_overlap():
    text = "".join(str(i % 10) for i in range(30))  # "012345678901234567890123456789"
    chunks = chunk_fixed_size(text, chunk_chars=10, overlap=3)
    # первые два чанка — оба полной длины (не последний укороченный), поэтому у них
    # должен быть ТОЧНО overlap=3 общих символов на стыке.
    assert len(chunks[0].text) == 10
    assert len(chunks[1].text) == 10
    assert chunks[0].text[-3:] == chunks[1].text[:3]


def test_fixed_size_short_text_single_chunk():
    text = "короткий текст"
    chunks = chunk_fixed_size(text, chunk_chars=1200, overlap=200)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_fixed_size_empty_text_no_chunks():
    assert chunk_fixed_size("", chunk_chars=100, overlap=10) == []


# --------------------------------------------------------------------------- #
# Структурная стратегия
# --------------------------------------------------------------------------- #


def test_structural_heading_starts_new_section():
    raw = (
        "Заголовок один\n\n"
        "Первый абзац текста под первым заголовком.\n\n"
        "Заголовок два\n\n"
        "Второй абзац текста под вторым заголовком."
    )
    chunks = chunk_structural(raw)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Заголовок один")
    assert chunks[1].text.startswith("Заголовок два")
    assert "Первый абзац" in chunks[0].text
    assert "Второй абзац" in chunks[1].text


def test_structural_no_heading_falls_back_to_one_chunk():
    raw = (
        "Обычный абзац без заголовков, оканчивающийся точкой.\n\n"
        "Ещё один обычный абзац текста, тоже с точкой в конце."
    )
    chunks = chunk_structural(raw)
    assert len(chunks) == 1
    assert clean_text(raw) == chunks[0].text


def test_structural_trailing_heading_without_following_paragraph_not_split():
    # Последний абзац похож на заголовок, но озаглавливать ему уже нечего — не
    # должен начинать новую (пустую) секцию.
    raw = "Заголовок\n\nАбзац текста.\n\nЕщё короткая строка без точки в конце"
    chunks = chunk_structural(raw)
    assert len(chunks) == 1
    assert "Ещё короткая строка" in chunks[0].text


def test_structural_page_split_on_form_feed():
    raw = (
        "Заголовок страницы 1\n\n"
        "Текст первой страницы.\n"
        "\x0c"
        "Заголовок страницы 2\n\n"
        "Текст второй страницы."
    )
    chunks = chunk_structural(raw)
    assert len(chunks) == 2
    assert "первой страницы" in chunks[0].text
    assert "второй страницы" in chunks[1].text
    # ни один чанк не должен содержать текст с другой страницы
    assert "второй" not in chunks[0].text
    assert "первой" not in chunks[1].text


def test_structural_page_with_only_heading_and_body_on_same_page():
    # Одностраничный документ без form-feed — вся эвристика должна работать так же,
    # как для txt-файла (страница = весь документ).
    raw = "Диверсификация\n\nДиверсификация - это распределение вложений между активами."
    chunks = chunk_structural(raw)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Диверсификация")


def test_structural_empty_page_after_trailing_form_feed_skipped():
    raw = "Заголовок\n\nТекст.\x0c"
    chunks = chunk_structural(raw)
    assert len(chunks) == 1
