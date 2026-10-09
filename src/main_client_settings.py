"""Чистые правила выбора провайдера основного потока (без окружения и побочных эффектов).

Вынесены из config.py, чтобы их можно было проверять тестами: сам config.py при импорте
читает окружение и может завершить процесс. Здесь — допустимые значения MAIN_CLIENT,
модель по умолчанию, подписи провайдера и проверка настроек.
"""

from __future__ import annotations

SUPPORTED_MAIN_CLIENTS = ("deepseek", "kimi", "ollama")

# Для ollama дефолта нет: набор локальных моделей зависит от оператора, MAIN_MODEL обязателен.
_DEFAULT_MODEL = {"deepseek": "deepseek-v4-flash", "kimi": "kimi-k3", "ollama": ""}
_LABEL = {"deepseek": "DeepSeek", "kimi": "Kimi", "ollama": "Ollama"}
# Что проверять оператору при ошибке доступа: ключ облачного провайдера или адрес Ollama.
_API_KEY_ENV_VAR = {
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "KIMI_API_KEY",
    "ollama": "OLLAMA_BASE_URL",
}


def default_main_model(client: str) -> str:
    return _DEFAULT_MODEL.get(client, _DEFAULT_MODEL["deepseek"])


def main_client_label(client: str) -> str:
    return _LABEL.get(client, _LABEL["deepseek"])


def main_api_key_env_var(client: str) -> str:
    return _API_KEY_ENV_VAR.get(client, _API_KEY_ENV_VAR["deepseek"])


def connection_error_text(label: str) -> str:
    """Сообщение пользователю при APIConnectionError; для Ollama — без подсказки про ключ."""
    if label == _LABEL["ollama"]:
        return (
            f"🌐 Не получилось подключиться к серверу {label}. "
            "Администратору бота нужно проверить, что Ollama запущен и модель загружена."
        )
    return f"🌐 Не получилось подключиться к серверу {label}. Проверь соединение и попробуй позже."


def disable_thinking_extra_body(client: str) -> dict | None:
    """extra_body для отключения «thinking» — только облачным провайдерам; Ollama его не получает."""
    if client == "ollama":
        return None
    return {"thinking": {"type": "disabled"}}


def validate_main_settings(client: str, model: str) -> str | None:
    """Возвращает текст ошибки конфигурации или None, если настройки допустимы."""
    if client not in SUPPORTED_MAIN_CLIENTS:
        return (
            f"Недопустимое значение MAIN_CLIENT={client!r}. "
            f"Допустимые значения: {', '.join(SUPPORTED_MAIN_CLIENTS)}."
        )
    if client == "ollama" and not model.strip():
        return (
            "При MAIN_CLIENT=ollama обязательно задать MAIN_MODEL — имя локальной модели "
            "Ollama (например, из `ollama list`)."
        )
    return None
