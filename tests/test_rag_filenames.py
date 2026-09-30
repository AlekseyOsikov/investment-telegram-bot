"""Тесты разбора метаданных из имени файла (rag/filenames.py) — чистая функция, без
файловой системы и без сети, по тому же принципу, что test_task_state.py."""

from rag.filenames import FileMetadata, parse_filename


def test_author_and_date_present():
    result = parse_filename("Обзор рынка облигаций, часть 1 (author_name, 22.07.2026).txt")
    assert result == FileMetadata(
        title="Обзор рынка облигаций, часть 1",
        author="author_name",
        date="22.07.2026",
    )


def test_only_date_present():
    result = parse_filename("Разбор портфеля - пример с комментариями (23.12.2026).txt")
    assert result == FileMetadata(
        title="Разбор портфеля - пример с комментариями",
        author=None,
        date="23.12.2026",
    )


def test_only_author_present():
    result = parse_filename("Основы диверсификации (Иванов).pdf")
    assert result == FileMetadata(title="Основы диверсификации", author="Иванов", date=None)


def test_no_parenthetical_suffix():
    result = parse_filename("Просто заголовок без скобок.pdf")
    assert result == FileMetadata(
        title="Просто заголовок без скобок", author=None, date=None
    )


def test_typical_pdf_filenames():
    """Типичные имена PDF: длинный заголовок с точками и номером части, суффикс — только автор."""
    cases = [
        ("Диверсификация. Часть 1. Что это такое и как её применять (Иванов).pdf",
         "Диверсификация. Часть 1. Что это такое и как её применять", "Иванов", None),
        ("Обзор фондов и особенности работы с ними (Иванов).pdf",
         "Обзор фондов и особенности работы с ними", "Иванов", None),
    ]
    for filename, expected_title, expected_author, expected_date in cases:
        result = parse_filename(filename)
        assert result.title == expected_title
        assert result.author == expected_author
        assert result.date == expected_date


def test_typical_txt_filenames():
    """Типичные имена txt: автор — ник из латиницы с подчёркиванием, затем дата."""
    result = parse_filename("Короткая заметка о фондах (author_two, 04.08.2026).txt")
    assert result == FileMetadata(
        title="Короткая заметка о фондах", author="author_two", date="04.08.2026"
    )

    result = parse_filename("Заметка про стоимостной подход (Broker_Name, 18.08.2026).txt")
    assert result == FileMetadata(
        title="Заметка про стоимостной подход",
        author="Broker_Name",
        date="18.08.2026",
    )
