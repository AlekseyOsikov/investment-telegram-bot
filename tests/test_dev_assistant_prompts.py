"""Тесты сообщений модели и источников режима ассистента разработчика (dev_assistant/prompts.py)."""

from dataclasses import dataclass

from dev_assistant import prompts


@dataclass
class Chunk:
    title: str
    text: str


def test_assistant_message_is_separate_system_and_keeps_safeguards():
    message = prompts.assistant_message()
    assert message["role"] == "system"
    text = message["content"]
    assert "НЕ ослабляет" in text and "основной системный промпт" in text
    assert "лицензированным финансовым советником" in text
    assert "не даёшь инвестиционных рекомендаций" in text
    assert "ТОЛЬКО на вопросы об этом проекте" in text
    assert "данные" in text.lower()


def test_available_git_message_denies_diff_and_file_contents():
    text = prompts.git_layer_message(prompts.GIT_AVAILABLE)["content"]
    assert "diff" in text and "недоступн" in text
    assert "get_diff" not in text and "list_roots" not in text
    assert "truncated" in text


def test_unavailable_and_not_configured_forbid_guessing():
    unavailable = prompts.git_layer_message(prompts.GIT_UNAVAILABLE)["content"]
    not_configured = prompts.git_layer_message(prompts.GIT_NOT_CONFIGURED)["content"]
    assert "НЕДОСТУПНЫ" in unavailable and "Не называй ветку" in unavailable
    assert "не настроен" in not_configured and "Не обещай" in not_configured


def test_user_message_with_chunks_puts_fragments_before_question():
    chunks = [Chunk("README.md › Запуск", "make run"), Chunk("CLAUDE.md › Очередь", "слоты")]
    content = prompts.build_user_message("как запустить?", chunks)["content"]
    assert content.index("make run") < content.index("как запустить?")
    assert 'n="1"' in content and 'n="2"' in content
    assert "README.md › Запуск" in content


def test_user_message_without_chunks_marks_nothing_found_only_when_asked():
    assert prompts.build_user_message("в?", [])["content"] == "в?"
    assert prompts.NO_DOCS_NOTE in prompts.build_user_message("в?", [], no_docs=True)["content"]


def test_sources_come_from_actual_chunks_deduplicated_in_order():
    chunks = [Chunk("A › 1", "x"), Chunk("B › 2", "y"), Chunk("A › 1", "z")]
    assert prompts.source_lines(chunks) == ["A › 1", "B › 2"]
    block = prompts.format_sources(chunks, "2026-10-14T10:00:00+00:00")
    assert block.startswith("Источники:") and "A › 1" in block and "Индекс собран:" in block


def test_no_sources_block_without_chunks():
    assert prompts.format_sources([], "2026-10-14") == ""


def test_build_messages_order_and_unchanged_main_prompt():
    chunks = [Chunk("README.md › Запуск", "make run")]
    messages = prompts.build_messages(
        "ОСНОВНОЙ ПРОМПТ", prompts.GIT_AVAILABLE, [("q1", "a1")], "q2", chunks
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "system", "system", "user", "assistant", "user"]
    assert messages[0]["content"] == "ОСНОВНОЙ ПРОМПТ"
    assert messages[1] == prompts.assistant_message()
    assert messages[2] == prompts.git_layer_message(prompts.GIT_AVAILABLE)
    assert messages[3]["content"] == "q1" and messages[4]["content"] == "a1"
    assert "make run" in messages[-1]["content"] and "q2" in messages[-1]["content"]


def test_history_holds_only_plain_question_and_answer():
    messages = prompts.history_messages([("вопрос", "ответ")])
    assert messages == [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
