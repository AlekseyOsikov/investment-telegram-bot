"""Настройки LLM Gateway.

Читаются из окружения (и `.env` проекта, если он есть). Имена переменных
начинаются с `LLM_GATEWAY_`, чтобы не пересекаться с настройками бота. Для
удобства есть фолбэки на уже существующие `OLLAMA_BASE_URL`, `DEEPSEEK_*`,
`ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv уже есть в зависимостях бота
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - окружение без dotenv
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = ROOT / "reports" / "llm-gateway" / "audit.jsonl"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    """Конфигурация гейтвея. Все поля можно переопределить в тестах/демо."""

    # openai — любой OpenAI-совместимый backend (OpenAI, DeepSeek, Ollama);
    # anthropic — нативный /v1/messages с трансляцией форматов.
    provider: str = "openai"
    backend_url: str = "http://localhost:11434/v1"
    backend_api_key: str = "ollama"
    default_model: str = "qwen3:30b-a3b"

    # block — найденный секрет блокирует запрос; mask — заменяет на [REDACTED_*].
    guard_mode: str = "block"
    rate_limit_per_minute: int = 60
    # Эвристика «секрет без известного префикса» (длинный токен с высокой энтропией).
    # По умолчанию выключена: возможны ложные срабатывания на длинных строках/хэшах.
    entropy: bool = False
    # Доверенные reverse proxy (IP или CIDR): только их X-Forwarded-For используется для
    # ключа rate-limit. Пусто — XFF не доверяем, ключ = адрес непосредственного клиента.
    trusted_proxies: tuple[str, ...] = ()

    audit_log_path: Path = DEFAULT_AUDIT_PATH
    # Писать ли в аудит запрос/ответ (уже замаскированные) и до какой длины.
    log_prompts: bool = True
    audit_max_chars: int = 4000

    # Случайный маркер, который подмешивается в system prompt. Если модель его
    # повторила — значит, произошла утечка системного промпта.
    canary: str = field(default_factory=lambda: "GW-CANARY-" + secrets.token_hex(8))
    request_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> Settings:
        provider = (_env("LLM_GATEWAY_PROVIDER", "openai") or "openai").lower()
        if provider == "deepseek":
            backend_url = _env(
                "LLM_GATEWAY_BACKEND_URL", _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            )
            backend_api_key = _env(
                "LLM_GATEWAY_BACKEND_API_KEY", _env("DEEPSEEK_API_KEY", "")
            )
            default_model = _env("LLM_GATEWAY_MODEL", _env("MAIN_MODEL", "deepseek-chat"))
            provider = "openai"
        elif provider == "anthropic":
            backend_url = _env(
                "LLM_GATEWAY_BACKEND_URL", "https://api.anthropic.com/v1"
            )
            backend_api_key = _env(
                "LLM_GATEWAY_BACKEND_API_KEY", _env("ANTHROPIC_API_KEY", "")
            )
            default_model = _env(
                "LLM_GATEWAY_MODEL", "claude-3-5-sonnet-latest"
            )
        else:
            backend_url = _env(
                "LLM_GATEWAY_BACKEND_URL",
                _env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            )
            backend_api_key = _env("LLM_GATEWAY_BACKEND_API_KEY", "ollama")
            default_model = _env("LLM_GATEWAY_MODEL", "qwen3:30b-a3b")

        guard_mode = (_env("LLM_GATEWAY_GUARD_MODE", "block") or "block").lower()
        if guard_mode not in ("block", "mask"):
            guard_mode = "block"

        rate_limit = int(_env("LLM_GATEWAY_RATE_LIMIT", "60") or "60")
        entropy = (_env("LLM_GATEWAY_ENTROPY", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        trusted_proxies = tuple(
            part.strip()
            for part in (_env("LLM_GATEWAY_TRUSTED_PROXIES", "") or "").split(",")
            if part.strip()
        )
        audit_path = Path(
            _env("LLM_GATEWAY_AUDIT_LOG", str(DEFAULT_AUDIT_PATH)) or DEFAULT_AUDIT_PATH
        )
        canary = _env("LLM_GATEWAY_CANARY") or ("GW-CANARY-" + secrets.token_hex(8))
        request_timeout = float(_env("LLM_GATEWAY_REQUEST_TIMEOUT", "120") or "120")

        return cls(
            provider=provider,
            backend_url=backend_url or "http://localhost:11434/v1",
            backend_api_key=backend_api_key or "ollama",
            default_model=default_model or "qwen3:30b-a3b",
            guard_mode=guard_mode,
            rate_limit_per_minute=rate_limit,
            entropy=entropy,
            trusted_proxies=trusted_proxies,
            audit_log_path=audit_path,
            canary=canary,
            request_timeout=request_timeout,
        )
