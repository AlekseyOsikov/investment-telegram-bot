"""Тесты чистых правил оснастки «локальная модель как код-ассистент»
(scripts/ollama/local_llm_lib.py).

Только чистые функции — ни сети, ни Ollama, ни диска (запись и чтение подставляются словарём).
Сами скрипты (цикл агента, замеры) проверяются вручную. Пути, файлы и модули — вымышленные.
"""

import re

import local_llm_lib as lib
import pytest

# --------------------------------------------------------------------------- #
# Границы доступа
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel", ["/etc/passwd", "../x", "src/../../x", "a/../.."])
def test_split_rel_path_rejects_absolute_and_parent_paths(rel):
    assert lib.split_rel_path(rel) is None


def test_split_rel_path_normalizes_dots_and_empty_parts():
    assert lib.split_rel_path("./src//price_watch/./parse.py") == ("src", "price_watch", "parse.py")
    assert lib.split_rel_path(".") == ()


@pytest.mark.parametrize(
    "rel",
    [".env", ".env.local", ".env.example", "src/.env", "data/x.json", ".git/config",
     ".venv/bin/python", "reports/a.md", "src/__pycache__/m.pyc"],
)
def test_denied_paths_are_not_given_to_the_model(rel):
    assert lib.is_denied(lib.split_rel_path(rel))


@pytest.mark.parametrize(
    "rel", ["src/main.py", "tests/test_a.py", "docs/local-llm.md", "README.md"])
def test_ordinary_paths_are_allowed(rel):
    assert not lib.is_denied(lib.split_rel_path(rel))


def test_writable_only_inside_listed_top_level_dirs():
    writable = ("src", "tests")
    assert lib.is_writable(("src", "a.py"), writable)
    assert lib.is_writable(("tests", "t.py"), writable)
    assert not lib.is_writable(("README.md",), writable)
    assert not lib.is_writable(("docs", "x.md"), writable)
    assert not lib.is_writable((), writable)
    assert not lib.is_writable(("src", "a.py"), ())  # профиль research: записи нет вовсе


# --------------------------------------------------------------------------- #
# Чтение и поиск
# --------------------------------------------------------------------------- #


def numbered(n):
    return "\n".join(f"строка {i}" for i in range(1, n + 1))


def test_read_report_numbers_lines_and_names_the_range():
    text = lib.read_report("a.py", numbered(10), 3, 5)
    assert text.startswith("[a.py, строки 3-5 из 10]")
    assert "3: строка 3" in text and "5: строка 5" in text and "6: строка 6" not in text


def test_read_report_caps_lines_per_call():
    text = lib.read_report("a.py", numbered(1000), 1, 1000)
    assert f"строки 1-{lib.MAX_READ_LINES} из 1000" in text.splitlines()[0]


def test_read_report_clamps_to_file_and_total_size():
    assert "строки 1-3 из 3" in lib.read_report("a.py", numbered(3), -5, 99)
    long_text = "\n".join("x" * 500 for _ in range(100))
    assert len(lib.read_report("a.py", long_text, 1, 100)) <= lib.MAX_READ_CHARS


def test_grep_text_returns_path_line_and_trimmed_text():
    hits = lib.grep_text("src/a.py", "x = 1\n  def f():   \nreturn 0\n", re.compile(r"def "))
    assert hits == ["src/a.py:2:def f():"]


def test_grep_text_cuts_very_long_lines():
    (hit,) = lib.grep_text("a.py", "needle " + "z" * 1000, re.compile("needle"))
    assert len(hit) <= len("a.py:1:") + lib.MAX_GREP_LINE


def test_grep_report_says_when_nothing_found_and_caps_hits():
    assert lib.grep_report([]) == "совпадений нет"
    many = [f"a.py:{i}:x" for i in range(lib.MAX_GREP_HITS + 5)]
    report = lib.grep_report(many)
    assert report.count("\n") == lib.MAX_GREP_HITS  # 60 строк + пометка
    assert "показано" in report.splitlines()[-1]


# --------------------------------------------------------------------------- #
# Правки
# --------------------------------------------------------------------------- #


def test_replace_once_replaces_a_unique_fragment():
    assert lib.replace_once("a b c", "b", "X") == ("a X c", 1)


def test_replace_once_refuses_missing_and_ambiguous_fragments():
    assert lib.replace_once("a b c", "z", "X") == (None, 0)
    assert lib.replace_once("b b", "b", "X") == (None, 2)
    assert lib.replace_once("abc", "", "X") == (None, 0)  # пустой old_string ничего не значит


# --------------------------------------------------------------------------- #
# Обрезка контекста
# --------------------------------------------------------------------------- #


def tool_msg(text):
    return {"role": "tool", "tool_name": "grep", "content": text}


def test_trim_keeps_last_results_and_stubs_old_ones():
    messages = [{"role": "user", "content": "q"}, tool_msg("a" * 10), tool_msg("b" * 20),
                tool_msg("c"), tool_msg("d"), tool_msg("e")]
    lib.trim_tool_results(messages, keep=3)
    assert messages[1]["content"].startswith(lib.TRIMMED_MARK) and "10" in messages[1]["content"]
    assert messages[2]["content"].startswith(lib.TRIMMED_MARK)
    assert [m["content"] for m in messages[3:]] == ["c", "d", "e"]
    assert messages[0]["content"] == "q"


def test_trim_is_idempotent_and_ignores_non_tool_messages():
    messages = [tool_msg("a"), tool_msg("b"), tool_msg("c"), tool_msg("d")]
    lib.trim_tool_results(messages, keep=2)
    once = [m["content"] for m in messages]
    lib.trim_tool_results(messages, keep=2)
    assert [m["content"] for m in messages] == once
    short = [tool_msg("x")]
    lib.trim_tool_results(short, keep=3)
    assert short[0]["content"] == "x"


# --------------------------------------------------------------------------- #
# Ответ модели блоками
# --------------------------------------------------------------------------- #

ANSWER = """Текст до.
=== FILE: src/terms/rules.py ===
```python
X = 1
```
=== END ===
=== EDIT: src/main.py ===
<<<<<<< SEARCH
import a
=======
import a
import b
>>>>>>> REPLACE
<<<<<<< SEARCH
call()
=======
call()
call2()
>>>>>>> REPLACE
=== END ===
"""


def test_strip_fence_removes_one_outer_fence_only():
    assert lib.strip_fence("```python\nX = 1\n```") == "X = 1"
    assert lib.strip_fence("X = 1\nY = 2") == "X = 1\nY = 2"
    assert lib.strip_fence("```\ninner ``` text\n```") == "inner ``` text"


def test_parse_answer_extracts_files_and_every_search_block():
    files, edits = lib.parse_answer(ANSWER)
    assert [f.path for f in files] == ["src/terms/rules.py"]
    assert [(e.path, e.search) for e in edits] == [("src/main.py", "import a"),
                                                   ("src/main.py", "call()")]
    assert edits[0].replace == "import a\nimport b"


def test_apply_answer_writes_unfenced_files_and_applies_edits():
    store = {"src/main.py": "import a\n\ncall()\n"}
    problems = lib.apply_answer(ANSWER, store.get, store.__setitem__)
    assert problems == []
    assert store["src/terms/rules.py"] == "X = 1\n"
    assert store["src/main.py"] == "import a\nimport b\n\ncall()\ncall2()\n"


def test_apply_answer_reports_edits_whose_search_does_not_match():
    store = {"src/main.py": "import other\n"}
    problems = lib.apply_answer(ANSWER, store.get, store.__setitem__)
    assert len(problems) == 2 and all("SEARCH найден 0 раз" in p for p in problems)
    assert store["src/main.py"] == "import other\n"  # неудавшаяся правка файл не меняет


def test_apply_answer_reports_edits_of_missing_files():
    store = {}
    problems = lib.apply_answer(ANSWER, store.get, store.__setitem__)
    assert len(problems) == 2 and "src/main.py" not in store


def test_apply_answer_without_blocks_does_nothing():
    store = {}
    assert lib.apply_answer("просто текст", store.get, store.__setitem__) == []
    assert store == {}


# --------------------------------------------------------------------------- #
# Проверка ответа
# --------------------------------------------------------------------------- #


def test_find_refs_extracts_file_line_pairs():
    text = "см. `src/a/b.py:42` и CLAUDE.md:7, но не `src/a/b.py` и не ver:3"
    assert lib.find_refs(text) == [("src/a/b.py", 42), ("CLAUDE.md", 7)]


@pytest.mark.parametrize(("line", "total", "ok"), [(1, 10, True), (10, 10, True), (11, 10, False),
                                                   (0, 10, False)])
def test_ref_is_valid(line, total, ok):
    assert lib.ref_is_valid(line, total) is ok


TESTS_TEXT = """
from alpha.core import run
import alpha.util
from beta import helper
from gamma import (
    first,
    second,
)
"""


@pytest.mark.parametrize(
    ("dotted", "expected"),
    [("alpha.core", True), ("alpha.util", True), ("beta.helper", True), ("gamma.second", True),
     ("alpha.cli", False), ("beta.other", False), ("delta", False), ("alpha.cor", False)],
)
def test_is_imported(dotted, expected):
    assert lib.is_imported(dotted, TESTS_TEXT) is expected


def test_uncovered_modules_lists_non_imported_and_skips_init():
    rel = ["alpha/core.py", "alpha/cli.py", "alpha/__init__.py", "beta/helper.py", "main.py"]
    assert lib.uncovered_modules(rel, TESTS_TEXT) == ["alpha/cli.py", "main.py"]


# --------------------------------------------------------------------------- #
# Параметры запросов
# --------------------------------------------------------------------------- #


def test_rate_handles_zero_duration():
    assert lib.rate(100, 2_000_000_000) == 50.0
    assert lib.rate(100, 0) is None


@pytest.mark.parametrize(
    ("model", "expected"),
    [("gpt-oss:20b", "low"), ("qwen3:30b-a3b", False), ("qwen3-coder:30b", None),
     ("qwen2.5-coder:3b", None)],
)
def test_default_think(model, expected):
    assert lib.default_think(model) == expected


def test_profile_body_strips_frontmatter_only():
    assert lib.profile_body("---\nname: x\n---\n\nТело\n---\nещё") == "Тело\n---\nещё"
    assert lib.profile_body("Без шапки\n") == "Без шапки"
