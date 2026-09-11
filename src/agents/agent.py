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

ask() дополнительно считает и возвращает (в AgentAnswer) статистику по токенам —
DeepSeek/Kimi API не даёт токены текущего вопроса и ответа по отдельности, только
usage.prompt_tokens (весь промпт: системный промпт + история + вопрос целиком) и
usage.completion_tokens (ответ), поэтому:
- "токены ответа" — это usage.completion_tokens как есть;
- "токены всей истории" — это usage.prompt_tokens как есть (весь промпт, отправленный
  в этом вызове);
- "токены нового вопроса" оцениваются двумя способами: приблизительно, по длине
  текста вопроса (len // 2, без обращения к API — делитель подобран эмпирически под
  русский текст: кириллица в BPE-токенайзерах вроде cl100k обычно кодируется куда
  менее эффективно, чем латиница, ближе к ~2 символам на токен, а не к ~4, как для
  английского), и через разницу — из текущего
  usage.prompt_tokens вычитается last_context_tokens ПРЕДЫДУЩЕГО вызова этого чата
  (сохраняется в JSON-файле истории). last_context_tokens — это prompt_tokens +
  completion_tokens предыдущего вызова МИНУС его reasoning_tokens (если API их
  возвращает, см. _extract_usage) — вычитать нужно именно то, что реально попало в
  content и, значит, в сохранённую историю, а не весь completion_tokens: у моделей
  с рассуждениями заметная часть completion_tokens уходит на скрытые размышления,
  которые в историю не попадают, и без поправки на них "разница" на следующем шаге
  уходит в минус. Точность дополнительно ограничена служебными токенами на разметку сообщений чата
  (несколько токенов на сообщение) — они не в счёт. Для первого вопроса в чате (или
  сразу после reset()) базы для сравнения ещё нет — разница возвращается как None.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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


@dataclass
class AgentAnswer:
    """Результат ask() — текст ответа плюс статистика по токенам (см. докстринг модуля)."""

    text: str
    request_tokens_diff: int | None
    request_tokens_approx: int
    history_tokens: int | None
    response_tokens: int | None


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
        self._history, self._last_context_tokens = self._load_state()

    def _load_state(self) -> tuple[list[dict[str, str]], int | None]:
        """Читает историю чата и last_context_tokens из JSON-файла.

        Поддерживает два формата файла: старый — голый список сообщений (до
        добавления статистики по токенам, last_context_tokens в этом случае
        неизвестен — None) и новый — объект {"messages": [...],
        "last_context_tokens": ...}. Отсутствие/повреждение файла — пустая история.
        """
        try:
            raw = self._history_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Повреждённый файл истории агента %s — начинаю с пустой истории.",
                self._history_path,
            )
            return [], None

        if isinstance(data, list):
            return data, None

        if isinstance(data, dict):
            messages = data.get("messages")
            last_context_tokens = data.get("last_context_tokens")
            return (
                messages if isinstance(messages, list) else [],
                last_context_tokens if isinstance(last_context_tokens, int) else None,
            )

        return [], None

    def _save_history(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"messages": self._history, "last_context_tokens": self._last_context_tokens}
        self._history_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        """Полностью очищает историю диалога этого чата (см. /agent_reset в agent_command.py).

        Вместе с историей сбрасывается и last_context_tokens — следующий вопрос
        снова станет "первым" для расчёта разницы токенов (см. докстринг модуля).
        """
        self._history = []
        self._last_context_tokens = None
        self._history_path.unlink(missing_ok=True)

    def get_history(self) -> list[dict[str, str]]:
        """Возвращает копию сохранённой истории (см. /agent_history в agent_command.py).

        Копия, а не сама внутренняя история — чтобы вызывающий код не мог случайно
        исказить состояние агента через возвращённый список.
        """
        return list(self._history)

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        """Достаёт prompt_tokens/completion_tokens (и, если есть, reasoning_tokens) из
        ответа API. reasoning_tokens (usage.completion_tokens_details.reasoning_tokens,
        как в OpenAI-совместимой схеме reasoning-моделей) — часть completion_tokens,
        которая ушла на скрытые размышления и НЕ попадает в content, а значит и в
        сохранённую историю; без вычитания этой части last_context_tokens в ask()
        завышается и уводит следующий расчёт "по разнице" в минус. Если поле
        отсутствует у конкретного провайдера/модели — считаем reasoning_tokens = 0
        (поведение как раньше).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return None
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) or 0
        logger.debug("Usage от API (агент): %r", usage)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    def ask(self, user_text: str) -> AgentAnswer:
        """Отправляет вопрос пользователя в LLM вместе со всей историей чата и возвращает ответ.

        Может выбросить исключение OpenAI SDK (сетевые ошибки, ошибки API и т.д.) —
        перехват и перевод в сообщение пользователю на русском выполняет вызывающий
        код, см. докстринг модуля. История дописывается только после успешного
        ответа API — неудачный вызов не искажает сохранённый диалог. Статистика по
        токенам в возвращённом AgentAnswer — см. докстринг модуля про то, откуда
        берётся каждое из четырёх значений.
        """
        # Делитель 2, а не общепринятые для английского языка 4 символа/токен — см.
        # докстринг модуля про то, почему кириллица в BPE-токенайзерах менее эффективна.
        request_tokens_approx = len(user_text) // 2

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

        usage = self._extract_usage(response)
        history_tokens = usage["prompt_tokens"] if usage else None
        response_tokens = usage["completion_tokens"] if usage else None
        request_tokens_diff = (
            usage["prompt_tokens"] - self._last_context_tokens
            if usage and self._last_context_tokens is not None
            else None
        )
        # В историю попадает только видимый content, а не completion_tokens целиком —
        # вычитаем reasoning_tokens (см. докстринг _extract_usage), иначе база для
        # следующего diff завышается и результат уходит в минус.
        self._last_context_tokens = (
            usage["prompt_tokens"] + usage["completion_tokens"] - usage["reasoning_tokens"]
            if usage
            else None
        )

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": answer})
        self._save_history()

        return AgentAnswer(
            text=answer,
            request_tokens_diff=request_tokens_diff,
            request_tokens_approx=request_tokens_approx,
            history_tokens=history_tokens,
            response_tokens=response_tokens,
        )
