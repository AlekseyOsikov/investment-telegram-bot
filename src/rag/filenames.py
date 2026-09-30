"""Разбор метаданных из имени файла для пайплайна индексации (rag/cli.py).

Формат имени: `<Title>(<author>, <date>).<ext>`, где скобочный суффикс целиком, а
также `author` и `date` внутри него по отдельности — необязательны (см.
Requirement «Разбор метаданных из имени файла» в
openspec/changes/add-rag-indexing-pipeline/specs/rag-indexing/spec.md). Разбор
работает только со строкой имени файла — сам файл на диске не читается и не
изменяется (см. design.md, решение «Исходные файлы — данные только для чтения»).

Правило для скобочного суффикса с ОДНИМ значением (без запятой): если значение
похоже на дату (`ДД.ММ.ГГГГ`), это `date`, иначе — `author`. Наблюдалось на реальном
корпусе: PDF-файлы используют суффикс вида `(Иванов)` (только автор), часть
txt-файлов — `(23.12.2026)` (только дата), часть — `(author_name, 22.07.2026)`
(автор и дата через запятую).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Скобочный суффикс — последняя пара "(...)" в конце имени (без расширения), без
# вложенных скобок внутри. Жадный `.+` для title сам находит ПОСЛЕДНЮЮ такую пару за
# счёт бэктрекинга, даже если в названии встречаются другие символы "(", ")".
_PARENTHETICAL_RE = re.compile(r"^(?P<title>.+)\((?P<inner>[^()]*)\)\s*$")

# Единственный формат даты, встречающийся в реальных именах файлов: ДД.ММ.ГГГГ
# (день/месяц — одна или две цифры, год — четыре).
_DATE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")


@dataclass(frozen=True)
class FileMetadata:
    """Метаданные, извлечённые из имени файла. `author`/`date` — `None`, если их не
    было в имени файла (а не пустая строка)."""

    title: str
    author: str | None
    date: str | None


def parse_filename(filename: str) -> FileMetadata:
    """Разбирает `<Title>(<author>, <date>).<ext>` в `FileMetadata`.

    `filename` — имя файла (можно с расширением или без — расширение, если есть,
    просто отбрасывается по последней точке).
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename

    match = _PARENTHETICAL_RE.match(stem)
    if not match:
        return FileMetadata(title=stem.strip(), author=None, date=None)

    title = match.group("title").strip()
    inner = match.group("inner").strip()
    if not inner:
        return FileMetadata(title=title, author=None, date=None)

    if "," in inner:
        author_part, date_part = inner.split(",", 1)
        author = author_part.strip() or None
        date = date_part.strip() or None
        return FileMetadata(title=title, author=author, date=date)

    if _DATE_RE.match(inner):
        return FileMetadata(title=title, author=None, date=inner)
    return FileMetadata(title=title, author=inner, date=None)
