"""Извлечение текста из исходных файлов (rag/cli.py) — PDF через системную утилиту
`pdftotext` (поставляется пакетом poppler-utils), txt — обычным чтением файла, и общая
очистка пустых строк/абзацев перед чанкингом (rag/chunking.py).

`extract_pdf_text`/`extract_txt_text`/`extract_text` возвращают СЫРОЙ текст, без
очистки — PDF-файлы этого корпуса сохраняют между страницами form-feed-символ (`\\f`),
который структурная стратегия чанкинга (rag/chunking.py:chunk_structural) использует
для разбиения по страницам ДО очистки каждой страницы отдельно (расщепление на строки
через str.splitlines() уже само по себе трактует `\\f` как границу строки — экспериментально
проверено, что clean_text() на СЫРОМ многостраничном тексте превращает `\\f` в
одну пустую строку-разделитель абзаца, что верно для стратегии фиксированного размера,
но недостаточно для структурной — ей нужна явная граница страницы, а не просто «здесь
была одна пустая строка», неотличимая от настоящего пустого абзаца в исходном
документе). См. design.md изменения add-rag-indexing-pipeline, раздел Context/решение
«Извлечение текста из PDF».
"""

from __future__ import annotations

import subprocess


def clean_text(text: str) -> str:
    """Убирает пустые строки и пустые абзацы: обрезает пробелы по краям каждой строки,
    схлопывает любое число подряд идущих пустых строк в ОДНУ (граница абзаца), убирает
    пустые строки по краям всего текста. Строка из одних пробелов считается пустой.

    Пустой/из одних пробелов вход даёт пустую строку на выходе.
    """
    result: list[str] = []
    blank_pending = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if result:
                blank_pending = True
            continue
        if blank_pending:
            result.append("")
            blank_pending = False
        result.append(line)
    return "\n".join(result)


def extract_pdf_text(path: str) -> str:
    """Извлекает СЫРОЙ (неочищенный) текст PDF через `pdftotext -layout` — сохраняет
    form-feed-символы между страницами для chunk_structural (rag/chunking.py)."""
    result = subprocess.run(
        ["pdftotext", "-layout", path, "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def extract_txt_text(path: str) -> str:
    """Читает txt-файл как есть, без очистки — её делает clean_text отдельно."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(path: str) -> str:
    """Извлекает сырой текст документа независимо от типа файла — PDF через
    extract_pdf_text, всё остальное (txt) через extract_txt_text."""
    if path.lower().endswith(".pdf"):
        return extract_pdf_text(path)
    return extract_txt_text(path)
