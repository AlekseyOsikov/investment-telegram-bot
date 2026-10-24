"""Тесты Markdown-чанкера индекса документации проекта (dev_assistant/chunking.py)."""

from dev_assistant.chunking import BREADCRUMB_SEPARATOR, chunk_markdown

PATH = "docs/guide.md"
BODY = "Достаточно длинный абзац для самостоятельного фрагмента. " * 6  # > 200 символов


def titles(chunks):
    return [chunk.title for chunk in chunks]


def test_empty_file_gives_no_chunks():
    assert chunk_markdown("", PATH) == []
    assert chunk_markdown("  \n\n ", PATH) == []


def test_file_without_headings_is_one_chunk_with_path_title():
    chunks = chunk_markdown(BODY, PATH)
    assert len(chunks) == 1
    assert chunks[0].title == PATH
    assert chunks[0].text.startswith(PATH)
    assert chunks[0].chunk_index == 0


def test_nested_headings_make_breadcrumbs():
    text = f"# Один\n\n{BODY}\n\n## Два\n\n{BODY}\n\n### Три\n\n{BODY}\n\n## Другой\n\n{BODY}\n"
    chunks = chunk_markdown(text, PATH)
    sep = BREADCRUMB_SEPARATOR
    assert titles(chunks) == [
        f"{PATH}{sep}Один",
        f"{PATH}{sep}Один{sep}Два",
        f"{PATH}{sep}Один{sep}Два{sep}Три",
        f"{PATH}{sep}Один{sep}Другой",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]


def test_hash_inside_fenced_block_is_not_heading():
    code = "```bash\n# это комментарий, а не заголовок\nmake test\n```"
    text = f"# Команды\n\n{BODY}\n\n{code}\n"
    chunks = chunk_markdown(text, PATH)
    assert len(chunks) == 1
    assert "# это комментарий" in chunks[0].text
    assert titles(chunks) == [f"{PATH}{BREADCRUMB_SEPARATOR}Команды"]


def test_tilde_fence_also_protects_hashes():
    text = f"# A\n\n{BODY}\n\n~~~\n## не заголовок\n~~~\n"
    assert len(chunk_markdown(text, PATH)) == 1


def test_long_section_is_split_by_paragraphs_with_same_breadcrumbs():
    paragraphs = "\n\n".join(f"Абзац {i}. " + "слово " * 60 for i in range(8))
    chunks = chunk_markdown(f"# Длинный\n\n{paragraphs}\n", PATH, max_chars=700)
    assert len(chunks) > 1
    assert len(set(titles(chunks))) == 1
    assert all(len(chunk.text) <= 700 + len(chunk.title) + 2 for chunk in chunks)
    joined = "\n".join(chunk.text for chunk in chunks)
    for i in range(8):
        assert f"Абзац {i}." in joined


def test_overlong_single_line_is_hard_cut():
    chunks = chunk_markdown("# X\n\n" + "а" * 5000, PATH, max_chars=1000, min_chars=10)
    assert len(chunks) >= 5


def test_short_section_merges_into_next():
    text = f"# Короткий\n\nмало\n\n# Длинный\n\n{BODY}\n"
    chunks = chunk_markdown(text, PATH)
    assert len(chunks) == 1
    assert "мало" in chunks[0].text and "Длинный" in chunks[0].text
    assert chunks[0].title == f"{PATH}{BREADCRUMB_SEPARATOR}Короткий"


def test_trailing_short_section_merges_into_previous():
    text = f"# Длинный\n\n{BODY}\n\n# Хвост\n\nмало\n"
    chunks = chunk_markdown(text, PATH)
    assert len(chunks) == 1
    assert "мало" in chunks[0].text


def test_only_short_content_still_yields_chunk():
    chunks = chunk_markdown("# Заголовок\n\nкоротко\n", PATH)
    assert len(chunks) == 1
    assert "коротко" in chunks[0].text


# --- Дополнение: разрезание длинного раздела (_split_long) ----------------------------------------

from dev_assistant.chunking import _split_long  # noqa: E402


def test_split_long_short_body_is_one_piece():
    assert _split_long("Абзац один.\n\nАбзац два.", 100) == ["Абзац один.\n\nАбзац два."]


def test_split_long_packs_paragraphs_up_to_limit_and_loses_nothing():
    paragraphs = [f"Абзац номер {i} " + "х" * 20 for i in range(1, 7)]
    chunks = _split_long("\n\n".join(paragraphs), 60)
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert len(chunks) > 1
    assert "\n\n".join(chunks) == "\n\n".join(paragraphs)


def test_split_long_keeps_paragraph_boundaries_when_paragraphs_fit():
    first, second = "а" * 50, "б" * 50
    assert _split_long(f"{first}\n\n{second}", 60) == [first, second]


def test_split_long_cuts_overlong_paragraph_by_lines():
    lines = [f"строка {i} " + "я" * 20 for i in range(1, 6)]
    chunks = _split_long("\n".join(lines), 60)
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert "\n".join(chunks) == "\n".join(lines)


def test_split_long_flushes_short_line_before_hard_cutting_a_longer_one():
    chunks = _split_long("коротко\n" + "ж" * 150, 60)
    assert chunks[0] == "коротко"
    assert [len(chunk) for chunk in chunks[1:]] == [60, 60, 30]
    assert "".join(chunks[1:]) == "ж" * 150


def test_split_long_keeps_fenced_code_block_with_blank_lines_in_one_piece():
    code = "```\nпервая\n\nвторая\n```"
    chunks = _split_long(f"Вступление.\n\n{code}\n\nЗаключение.", 40)
    assert any(code in chunk for chunk in chunks)
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_split_long_cuts_fenced_block_only_when_it_exceeds_limit():
    code = "```\n" + "\n\n".join("к" * 30 for _ in range(4)) + "\n```"
    chunks = _split_long(code, 50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert len(chunks) > 1
    assert "".join(chunks).replace("\n", "") == code.replace("\n", "")


def test_split_long_of_empty_body_is_empty():
    assert _split_long("", 100) == []
    assert _split_long("\n\n  \n", 100) == []
