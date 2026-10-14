"""Чанкер Markdown для индекса документации проекта (design.md изменения add-dev-assistant-mode,
решение 3). Делит файл по заголовкам `#`…`######`, игнорируя `#` внутри ограждённых блоков кода;
каждому фрагменту даёт «хлебные крошки» (`файл › H1 › H2`), которые идут в `title` записи и в
текст эмбеддинга. `chunk_structural` из `rag/` не годится: он заточен под PDF-слайды.

Чистые функции без файловой системы и сети (tests/test_dev_assistant_chunking.py). Константы —
параметры чанкера, предмет кода, а не эксплуатации (не в `.env`)."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 200
BREADCRUMB_SEPARATOR = " › "

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class MarkdownChunk:
    """Фрагмент документа: `title` — хлебные крошки раздела, `text` — тело с крошками в
    первой строке (так они попадают в эмбеддинг), `chunk_index` — порядковый номер в файле."""

    chunk_index: int
    title: str
    text: str


@dataclass
class _Section:
    path: list[str]
    lines: list[str]


def _split_sections(text: str) -> list[_Section]:
    """Разбор на разделы по заголовкам; `#` внутри ограждённого блока заголовком не считается.
    Текст до первого заголовка — раздел с пустым путём заголовков."""
    sections: list[_Section] = [_Section(path=[], lines=[])]
    stack: list[tuple[int, str]] = []
    fence: str | None = None

    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            sections[-1].lines.append(line)
            continue
        heading = _HEADING_RE.match(line) if fence is None else None
        if heading is None:
            sections[-1].lines.append(line)
            continue
        level, title = len(heading.group(1)), heading.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(_Section(path=[name for _, name in stack], lines=[line]))
    return sections


def _split_long(body: str, limit: int) -> list[str]:
    """Режет длинный раздел по абзацам (пустые строки), не разрывая абзац без нужды; абзац
    длиннее предела режется по строкам, строка длиннее — жёстко по символам. Ограждённые блоки
    кода по возможности остаются целыми (абзацем считается и блок кода с пустыми строками
    внутри — только пока он укладывается в предел)."""
    paragraphs: list[str] = []
    current: list[str] = []
    fence = False
    for line in body.split("\n"):
        if _FENCE_RE.match(line):
            fence = not fence
        if not line.strip() and not fence:
            if current:
                paragraphs.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append("\n".join(current))

    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= limit:
            pieces.append(paragraph)
            continue
        buffer = ""
        for line in paragraph.split("\n"):
            while len(line) > limit:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.append(line[:limit])
                line = line[limit:]
            candidate = f"{buffer}\n{line}" if buffer else line
            if len(candidate) > limit and buffer:
                pieces.append(buffer)
                buffer = line
            else:
                buffer = candidate
        if buffer:
            pieces.append(buffer)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}\n\n{piece}" if buffer else piece
        if len(candidate) > limit and buffer:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_markdown(
    text: str,
    rel_path: str,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[MarkdownChunk]:
    """Делит Markdown-файл на фрагменты по заголовкам.

    - заголовок раздела несёт крошки `rel_path › H1 › H2` (в `title` и первой строкой `text`);
    - раздел длиннее `max_chars` режется по абзацам, крошки повторяются в каждом куске;
    - раздел короче `min_chars` присоединяется к следующему (хвостовой — к предыдущему), чтобы не
      плодить фрагменты из одного заголовка; присоединённый текст сохраняет собственный заголовок
      внутри тела, крошки — у ведущего раздела;
    - пустой файл (только пробелы) даёт пустой список.
    """
    if not text.strip():
        return []

    sections = [
        (section.path, "\n".join(section.lines).strip())
        for section in _split_sections(text)
    ]
    sections = [(path, body) for path, body in sections if body]

    # Склейка коротких разделов: короткий присоединяется к следующему.
    merged: list[tuple[list[str], str]] = []
    carry_path: list[str] | None = None
    carry_body = ""
    for path, body in sections:
        if carry_body:
            body = f"{carry_body}\n\n{body}"
            path = carry_path if carry_path is not None else path
            carry_path, carry_body = None, ""
        if len(body) < min_chars:
            carry_path, carry_body = path, body
            continue
        merged.append((path, body))
    if carry_body:
        if merged:
            path, body = merged[-1]
            merged[-1] = (path, f"{body}\n\n{carry_body}")
        else:
            merged.append((carry_path or [], carry_body))

    chunks: list[MarkdownChunk] = []
    for path, body in merged:
        title = BREADCRUMB_SEPARATOR.join([rel_path, *path])
        parts = [body] if len(body) <= max_chars else _split_long(body, max_chars)
        for part in parts:
            chunks.append(
                MarkdownChunk(
                    chunk_index=len(chunks),
                    title=title,
                    text=f"{title}\n\n{part}",
                )
            )
    return chunks
