"""Чистые правила выбора провайдера основного потока (main_client_settings)."""

from main_client_settings import (
    connection_error_text,
    default_main_model,
    disable_thinking_extra_body,
    main_api_key_env_var,
    main_client_label,
    validate_main_settings,
)


def test_cloud_providers_valid_with_default_model():
    for client in ("deepseek", "kimi"):
        assert default_main_model(client)
        assert validate_main_settings(client, default_main_model(client)) is None


def test_labels_and_env_vars():
    assert main_client_label("deepseek") == "DeepSeek"
    assert main_client_label("kimi") == "Kimi"
    assert main_client_label("ollama") == "Ollama"
    assert main_api_key_env_var("ollama") == "OLLAMA_BASE_URL"
    assert main_api_key_env_var("kimi") == "KIMI_API_KEY"


def test_ollama_requires_model():
    assert default_main_model("ollama") == ""
    assert validate_main_settings("ollama", "") is not None
    assert validate_main_settings("ollama", "  ") is not None
    assert validate_main_settings("ollama", "qwen3:8b") is None


def test_unsupported_client_lists_all_values():
    error = validate_main_settings("foo", "x")
    assert error is not None
    for name in ("deepseek", "kimi", "ollama"):
        assert name in error


def test_connection_error_text_for_ollama_has_no_key_hint():
    text = connection_error_text("Ollama")
    assert "Ollama" in text
    assert "api-ключ" not in text.lower()
    assert "загружена" in text


def test_connection_error_text_cloud_unchanged():
    assert "DeepSeek" in connection_error_text("DeepSeek")


def test_thinking_not_sent_to_ollama():
    assert disable_thinking_extra_body("ollama") is None
    assert disable_thinking_extra_body("deepseek") == {"thinking": {"type": "disabled"}}
