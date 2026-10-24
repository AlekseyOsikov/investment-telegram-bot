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


# --- Дополнение: чтение txt и выбор способа извлечения ------------------------------------------
# extract_pdf_text запускает системную `pdftotext` (процесс) и проверяется вручную; здесь
# проверяется только чтение txt и то, какой способ выбирает extract_text.

from rag import extraction  # noqa: E402


def test_extract_txt_text_returns_file_content_unchanged(tmp_path):
    path = tmp_path / "документ.txt"
    path.write_text("  Строка один.\n\n\nСтрока два.  \n", encoding="utf-8")
    assert extraction.extract_txt_text(str(path)) == "  Строка один.\n\n\nСтрока два.  \n"


def test_extract_txt_text_keeps_form_feed_between_pages(tmp_path):
    path = tmp_path / "страницы.txt"
    path.write_text("Страница 1.\fСтраница 2.", encoding="utf-8")
    assert extraction.extract_txt_text(str(path)).split("\f") == ["Страница 1.", "Страница 2."]


def test_extract_txt_text_of_empty_file_is_empty_string(tmp_path):
    path = tmp_path / "пусто.txt"
    path.write_text("", encoding="utf-8")
    assert extraction.extract_txt_text(str(path)) == ""


def test_extract_txt_text_of_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        extraction.extract_txt_text(str(tmp_path / "нет.txt"))


def test_extract_text_reads_txt_directly(tmp_path):
    path = tmp_path / "заметка.txt"
    path.write_text("Текст заметки.", encoding="utf-8")
    assert extraction.extract_text(str(path)) == "Текст заметки."


def test_extract_text_of_unknown_extension_is_read_as_text(tmp_path):
    path = tmp_path / "заметка.md"
    path.write_text("Текст.", encoding="utf-8")
    assert extraction.extract_text(str(path)) == "Текст."


def test_extract_text_sends_pdf_of_any_case_to_pdf_extractor(monkeypatch):
    seen = []
    monkeypatch.setattr(extraction, "extract_pdf_text", lambda path: seen.append(path) or "PDF")
    assert extraction.extract_text("книга.pdf") == "PDF"
    assert extraction.extract_text("КНИГА.PDF") == "PDF"
    assert seen == ["книга.pdf", "КНИГА.PDF"]


def test_extract_text_does_not_treat_pdf_in_the_middle_of_name_as_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "extract_pdf_text", lambda path: "PDF")
    path = tmp_path / "pdf-заметки.txt"
    path.write_text("Обычный текст.", encoding="utf-8")
    assert extraction.extract_text(str(path)) == "Обычный текст."
