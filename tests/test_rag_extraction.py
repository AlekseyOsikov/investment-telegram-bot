"""Тесты очистки текста (rag/extraction.py:clean_text) — чистая функция, без
subprocess/pdftotext и без файловой системы (вызов pdftotext проверяется вручную,
см. tasks.md, задача 3.2)."""

from rag.extraction import clean_text


def test_collapses_repeated_blank_lines_to_one():
    text = "Абзац один.\n\n\n\nАбзац два."
    assert clean_text(text) == "Абзац один.\n\nАбзац два."


def test_strips_leading_and_trailing_blank_lines():
    text = "\n\n  \nАбзац.\n\n  \n"
    assert clean_text(text) == "Абзац."


def test_whitespace_only_input_returns_empty_string():
    assert clean_text("\n\n   \n\t\n") == ""
    assert clean_text("") == ""


def test_no_result_line_is_whitespace_only():
    text = "Строка 1.\n   \n\nСтрока 2.\n\n\n\t\n\nСтрока 3."
    result = clean_text(text)
    assert all(line != "" or True for line in result.split("\n"))
    # ни один "абзац" (непустая строка) не состоит только из пробелов
    for line in result.split("\n"):
        if line:
            assert line.strip() == line and line != ""


def test_single_blank_line_between_paragraphs_preserved():
    text = "Абзац один.\nещё строка.\n\nАбзац два."
    assert clean_text(text) == "Абзац один.\nещё строка.\n\nАбзац два."


def test_trims_trailing_whitespace_on_each_line():
    text = "Строка с пробелом в конце.   \nВторая строка.\t"
    assert clean_text(text) == "Строка с пробелом в конце.\nВторая строка."
