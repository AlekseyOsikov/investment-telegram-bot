"""Две независимые стратегии чанкинга (rag/cli.py) — см. Requirement «Стратегия
чанкинга по фиксированному размеру» и «Стратегия чанкинга по структуре» в
openspec/changes/add-rag-indexing-pipeline/specs/rag-indexing/spec.md, и design.md
того же изменения, раздел «Структурный чанкинг: эвристика заголовков + откат по
страницам PDF».

`chunk_fixed_size` ожидает УЖЕ очищенный текст (rag/extraction.clean_text) — весь
документ единым текстом, независимо от того, PDF это было или txt.

`chunk_structural` ожидает СЫРОЙ извлечённый текст (rag/extraction.extract_text,
БЕЗ очистки) — он сам разбивает его по form-feed-символам (`\\f`, границы страниц
PDF; для txt их никогда не бывает, поэтому весь документ — одна «страница») и
очищает каждую страницу ОТДЕЛЬНО через clean_text, до применения эвристики
заголовков — если бы очистка произошла ДО разбиения на страницы, `\\f` слился бы с
обычной пустой строкой-разделителем абзаца (см. докстринг rag/extraction.py) и
границу страницы нечем было бы отличить от настоящего пустого абзаца в исходнике.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extraction import clean_text

# «Короткая самостоятельная строка» эвристики заголовков (design.md, решение
# «Структурный чанкинг») — подобраны по реальным заголовкам корпуса (например,
# «Лайфхак № 1. Начинайте с анализа базовых данных компании» — 58 символов); не
# вынесены в config.py/.env — это деталь реализации эвристики, а не настройка,
# которую имеет смысл менять оператору.
_HEADING_MAX_CHARS = 100
_HEADING_TERMINAL_PUNCTUATION = ".!?:"


@dataclass(frozen=True)
class Chunk:
    """Один чанк документа. `chunk_index` — позиция чанка в пределах ОДНОГО вызова
    стратегии (т.е. в пределах одного документа) начиная с 0; sквозной `chunk_id`
    (источник + стратегия + этот индекс) собирает rag/index_store.py."""

    text: str
    chunk_index: int


def chunk_fixed_size(cleaned_text: str, *, chunk_chars: int, overlap: int) -> list[Chunk]:
    """Режет очищенный текст документа на перекрывающиеся окна по `chunk_chars`
    символов, с перекрытием `overlap` символов между соседними чанками (`overlap`
    должен быть меньше `chunk_chars` — проверяется в config._validate_config()).

    Реализация — простая посимвольная нарезка без учёта границ слов: каждый чанк
    гарантированно не длиннее `chunk_chars` (в отличие от подхода с бережным
    сохранением целых слов, здесь никогда не возникает более длинного «неразбиваемого»
    фрагмента — предусмотренное спекой исключение из этого ограничения просто ни разу
    не срабатывает, что спеке не противоречит).
    """
    if not cleaned_text:
        return []

    step = chunk_chars - overlap
    length = len(cleaned_text)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < length:
        end = min(start + chunk_chars, length)
        piece = cleaned_text[start:end]
        if piece.strip():
            chunks.append(Chunk(text=piece, chunk_index=index))
            index += 1
        if end >= length:
            break
        start += step
    return chunks


def _split_paragraphs(page: str) -> list[str]:
    """Делит УЖЕ очищенную страницу на абзацы по границе в одну пустую строку —
    clean_text гарантирует, что между абзацами ровно одна пустая строка, поэтому
    `"\\n\\n"` однозначно определяет границу абзаца."""
    return [paragraph for paragraph in page.split("\n\n") if paragraph]


def _is_heading_paragraph(paragraph: str) -> bool:
    """Строка-заголовок (design.md): самостоятельная (без внутренних переносов
    строки — иначе это уже не «короткая строка», а многострочный абзац), короткая, и
    не заканчивающаяся знаком препинания конца предложения."""
    if "\n" in paragraph:
        return False
    stripped = paragraph.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    return stripped[-1] not in _HEADING_TERMINAL_PUNCTUATION


def _split_page_into_sections(page: str) -> list[str]:
    """Делит одну (уже очищенную) страницу на секции по эвристике заголовков.

    Секция начинается на абзаце-заголовке (кроме, возможно, самой первой секции,
    если страница начинается не с заголовка) и включает все следующие абзацы до
    следующего заголовка. Последний абзац страницы заголовком не считается, даже
    если формально им выглядит — «заголовок» обязан предшествовать хотя бы одному
    абзацу, который он озаглавливает (Requirement «Стратегия чанкинга по
    структуре», сценарий «Заголовок не найден — откат к одному чанку»). Если на
    странице заголовков не нашлось вовсе, возвращается ОДНА секция — вся страница
    целиком.
    """
    paragraphs = _split_paragraphs(page)
    sections: list[str] = []
    current: list[str] = []
    for i, paragraph in enumerate(paragraphs):
        is_heading = _is_heading_paragraph(paragraph) and i + 1 < len(paragraphs)
        if is_heading and current:
            sections.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        sections.append("\n\n".join(current))
    return sections or ([page] if page else [])


def chunk_structural(raw_text: str) -> list[Chunk]:
    """Разбивает СЫРОЙ извлечённый текст документа на чанки по структуре: сначала по
    страницам (form-feed `\\f` — для txt их нет, весь документ — одна страница), затем
    внутри каждой страницы — по эвристике заголовков (см. докстринг модуля и
    _split_page_into_sections). Пустые после очистки страницы (например, хвостовой
    пустой сегмент после последнего `\\f` в PDF) пропускаются.
    """
    chunks: list[Chunk] = []
    index = 0
    for raw_page in raw_text.split("\f"):
        page = clean_text(raw_page)
        if not page:
            continue
        for section in _split_page_into_sections(page):
            if section.strip():
                chunks.append(Chunk(text=section, chunk_index=index))
                index += 1
    return chunks
