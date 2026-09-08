"""SimpleAgent — простой LLM-агент как отдельная сущность, а не голый вызов API.

Инкапсулирует полный цикл обращения к LLM-провайдеру: сборку сообщений (системный
промпт + вопрос пользователя), сам вызов OpenAI-совместимого API и разбор ответа
(включая fallback на случай пустого content). По умолчанию использует
main_client/MAIN_MODEL — того же провайдера и модель, что и основной поток бота
(main.py), поэтому не требует собственного выбора провайдера и не нарушает
ограничение «никакого runtime-переключения модели» (см. «Ограничения безопасности» в
CLAUDE.md): провайдер и модель агента по-прежнему настраиваются только через
MAIN_CLIENT/MAIN_MODEL в .env, а не параметром, доступным пользователю чата.

Ошибки OpenAI SDK (аутентификация, лимиты, таймауты и т.д.) намеренно не
перехватываются здесь — агент отвечает только за построение запроса и разбор ответа,
а перевод ошибки API в сообщение пользователю на русском — забота вызывающего
Telegram-обработчика (agents/agent_command.py), по тому же принципу, что и
handle_message в main.py.
"""

from __future__ import annotations

from config import MAIN_MODEL, MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS, SYSTEM_PROMPT
from providers.main_client import main_client


class SimpleAgent:
    """Stateless-агент: один вопрос пользователя -> один запрос к LLM -> один ответ.

    Не хранит историю между вызовами ask() — каждый вызов независим, как и в основном
    потоке бота (см. «Архитектура» в CLAUDE.md).
    """

    def __init__(
        self,
        client=main_client,
        model: str = MAIN_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout

    def ask(self, user_text: str) -> str:
        """Отправляет вопрос пользователя в LLM и возвращает текст ответа.

        Может выбросить исключение OpenAI SDK (сетевые ошибки, ошибки API и т.д.) —
        перехват и перевод в сообщение пользователю на русском выполняет вызывающий
        код, см. докстринг модуля.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=self._max_output_tokens,
            timeout=self._timeout,
        )
        content = response.choices[0].message.content
        return content or "Модель вернула пустой ответ. Попробуй переформулировать вопрос."
