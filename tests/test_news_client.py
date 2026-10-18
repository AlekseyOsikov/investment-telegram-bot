"""Тесты выбора клиента разбора новостей (providers/news_client.py) без сети.

Импорт модуля тянет config.py (он требует токен бота и ключ провайдера основного потока), поэтому
окружение для импорта задаётся здесь; вызовов сети нет — проверяется только сборка по настройкам.
"""

import importlib
import logging

import pytest

_PROXY_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


@pytest.fixture
def news_client(monkeypatch):
    """Модуль импортируется внутри теста: клиенты OpenAI создаются при импорте и падают на
    нестандартных прокси из окружения разработчика (socks://), которые здесь не нужны."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("MAIN_CLIENT", "ollama")
    monkeypatch.setenv("MAIN_MODEL", "test-model")
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    return importlib.import_module("providers.news_client")


def test_empty_provider_settings_give_no_backend(news_client, caplog):
    with caplog.at_level(logging.WARNING):
        assert news_client.build_news_backend("ollama", "", None, None) is None
    assert "NEWS_MODEL" in caplog.text


def test_ollama_with_model_is_local(news_client):
    backend = news_client.build_news_backend("ollama", "qwen-test", None, None)
    assert backend is not None and backend.is_local
    assert backend.model == "qwen-test" and backend.provider == "ollama"


def test_cloud_provider_without_key_is_disabled_with_warning(news_client, caplog):
    with caplog.at_level(logging.WARNING):
        assert news_client.build_news_backend("deepseek", "", None, "k") is None
        assert news_client.build_news_backend("kimi", "", "k", None) is None
    assert "DEEPSEEK_API_KEY" in caplog.text and "KIMI_API_KEY" in caplog.text


def test_cloud_provider_with_key_gets_default_model_and_is_not_local(news_client):
    deepseek = news_client.build_news_backend("deepseek", "", "key", None)
    kimi = news_client.build_news_backend("kimi", "custom-model", None, "key")
    assert deepseek is not None and deepseek.model and not deepseek.is_local
    assert kimi is not None and kimi.model == "custom-model"


def test_unknown_provider_gives_no_backend(news_client):
    assert news_client.build_news_backend("other", "m", "k", "k") is None
