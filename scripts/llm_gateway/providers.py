"""Адаптеры backend'а: OpenAI-совместимый и Anthropic.

Гейтвей работает в OpenAI-формате (``/v1/chat/completions``), а адаптер
переводит запрос/ответ в формат конкретного провайдера. Это позволяет одним и
тем же guard'ам проксировать OpenAI, DeepSeek, Ollama и Anthropic.
"""

from __future__ import annotations

from typing import Any

import httpx


class ProviderError(Exception):
    """Ошибка backend'а (сеть, HTTP-статус, не-JSON ответ)."""


class OpenAICompatibleProvider:
    """Любой backend с OpenAI-совместимым ``/chat/completions``."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"backend request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"backend returned {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError("backend returned non-JSON response") from exc


class AnthropicProvider:
    """Нативный Anthropic Messages API с трансляцией в OpenAI-формат."""

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in payload.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                system_parts.append(_as_text(content))
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        body: dict[str, Any] = {
            "model": payload.get("model") or "claude-3-5-sonnet-latest",
            "max_tokens": int(payload.get("max_tokens") or 1024),
            "messages": messages,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if payload.get("temperature") is not None:
            body["temperature"] = payload["temperature"]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(
                    f"{self.base_url}/messages", json=body, headers=headers
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"anthropic returned {response.status_code}: {response.text[:300]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError("anthropic returned non-JSON response") from exc

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        return {
            "id": data.get("id", "chatcmpl-anthropic"),
            "object": "chat.completion",
            "model": data.get("model", body["model"]),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": data.get("stop_reason") or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def build_provider(settings: Any) -> OpenAICompatibleProvider | AnthropicProvider:
    if settings.provider == "anthropic":
        return AnthropicProvider(
            base_url=settings.backend_url,
            api_key=settings.backend_api_key,
            timeout=settings.request_timeout,
        )
    return OpenAICompatibleProvider(
        base_url=settings.backend_url,
        api_key=settings.backend_api_key,
        timeout=settings.request_timeout,
    )


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")
