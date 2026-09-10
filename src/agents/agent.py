"""Agent — LLM-агент как отдельная сущность, а не голый вызов API.

Инкапсулирует полный цикл обращения к LLM-провайдеру: сборку сообщений (системный
промпт + история диалога + вопрос пользователя), сам вызов OpenAI-совместимого API и
разбор ответа (включая fallback на случай пустого content). По умолчанию использует
main_client/MAIN_MODEL — того же провайдера и модель, что и основной поток бота
(main.py), поэтому не требует собственного выбора провайдера и не нарушает
ограничение «никакого runtime-переключения модели» (см. «Ограничения безопасности» в
CLAUDE.md): провайдер и модель агента по-прежнему настраиваются только через
MAIN_CLIENT/MAIN_MODEL в .env, а не параметром, доступным пользователю чата.

В отличие от основного потока бота (main.py) и от прежней stateless-версии этого
агента, Agent хранит историю диалога — по явному запросу пользователя (см.
CLAUDE.md, раздел «Ограничения безопасности», про то, что это осознанное исключение
из правила «никакой памяти без явного запроса», а не тихое нарушение). История
хранится в JSON-файле на чат (AGENT_HISTORY_DIR/<chat_id>.json, см. config.py) и
переживает и перезапуск процесса бота, и повторный вход в /agent — конструктор
загружает файл, если он есть, а ask() дописывает в него очередную пару
"вопрос-ответ" после каждого успешного обращения к API. Один экземпляр Agent
обслуживает один чат (см. agents/agent_command.py, где экземпляры кэшируются по
chat_id) — это не тот же объект, что раньше переиспользовался на все чаты сразу,
т.к. теперь у каждого чата собственное состояние (история).

Ошибки OpenAI SDK (аутентификация, лимиты, таймауты и т.д.) намеренно не
перехватываются здесь — агент отвечает только за построение запроса и разбор ответа,
а перевод ошибки API в сообщение пользователю на русском — забота вызывающего
Telegram-обработчика (agents/agent_command.py), по тому же принципу, что и
handle_message в main.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import (
    AGENT_HISTORY_DIR,
    MAIN_MODEL,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from providers.main_client import main_client

logger = logging.getLogger(__name__)


class Agent:
    """Агент с историей диалога, сохраняемой в JSON-файле на чат (см. докстринг модуля).

    История не ограничена по длине (по явному решению — см. CLAUDE.md): каждый
    вызов ask() передаёт в LLM весь сохранённый диалог этого чата целиком.
    """

    def __init__(
        self,
        chat_id: int,
        client=main_client,
        model: str = MAIN_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        history_dir: str = AGENT_HISTORY_DIR,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._history_path = Path(history_dir) / f"{chat_id}.json"
        self._history: list[dict[str, str]] = self._load_history()

    def _load_history(self) -> list[dict[str, str]]:
        """Читает историю чата из JSON-файла; отсутствие/повреждение файла — пустая история."""
        try:
            raw = self._history_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []

        try:
            history = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Повреждённый файл истории агента %s — начинаю с пустой истории.",
                self._history_path,
            )
            return []

        return history if isinstance(history, list) else []

    def _save_history(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.write_text(
            json.dumps(self._history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        """Полностью очищает историю диалога этого чата (см. /agent_reset в agent_command.py)."""
        self._history = []
        self._history_path.unlink(missing_ok=True)

    def get_history(self) -> list[dict[str, str]]:
        """Возвращает копию сохранённой истории (см. /agent_history в agent_command.py).

        Копия, а не сама внутренняя история — чтобы вызывающий код не мог случайно
        исказить состояние агента через возвращённый список.
        """
        return list(self._history)

    def ask(self, user_text: str) -> str:
        """Отправляет вопрос пользователя в LLM вместе со всей историей чата и возвращает ответ.

        Может выбросить исключение OpenAI SDK (сетевые ошибки, ошибки API и т.д.) —
        перехват и перевод в сообщение пользователю на русском выполняет вызывающий
        код, см. докстринг модуля. История дописывается только после успешного
        ответа API — неудачный вызов не искажает сохранённый диалог.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_output_tokens,
            timeout=self._timeout,
        )
        content = response.choices[0].message.content
        answer = content or "Модель вернула пустой ответ. Попробуй переформулировать вопрос."

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": answer})
        self._save_history()

        return answer
