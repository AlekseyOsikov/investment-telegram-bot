"""SmartAgent — LLM-агент с явно разделённой моделью памяти (три независимых слоя),
в отличие от Agent (agents/agent.py), который переключает ОДНУ стратегию управления
контекстом поверх единой истории диалога. Отдельная, независимая от /agent сущность
(см. «Управление памятью smart-агента» в CLAUDE.md про то, почему это новый агент, а
не 5-я стратегия Agent) — свой класс, свой файл истории, свои команды /smart_agent_*.

Три слоя, каждый хранится отдельно и пишется по-разному:
- short_term (краткосрочная, текущий диалог) — сырые сообщения, пишутся
  АВТОМАТИЧЕСКИ на каждый вызов ask(), как обычная история чата.
- working (рабочая, данные текущей задачи) — одна активная задача на чат
  ({"goal", "status", "data", "created_at"} или None), пишется ТОЛЬКО явно
  (start_task/set_task_data/finish_task), никогда автоматически.
- long_term (долговременная, факты) — список текстовых фактов, пишется ТОЛЬКО явно
  (remember/forget), дословно, без LLM-классификации: ответственность за то, чтобы не
  сохранять туда чувствительные данные (номера счетов/карт, паспортные данные и т.п.,
  см. «Правила предметной области» в CLAUDE.md), остаётся на пользователе, как и для
  working — эта же осторожность применима к обоим explicit-слоям, не только к
  long_term.

Каждый слой можно независимо включать/выключать в СБОРКЕ контекста (enabled_layers,
переключается /smart_agent_toggle) без удаления самих данных — это и есть проверка "что
попадает в каждый слой и как это влияет на ответы", см. get_last_context_messages().

Использует main_client/MAIN_MODEL — того же провайдера и модель, что и основной поток
бота, по тому же принципу, что и Agent (см. докстринг agents/agent.py про то, почему
это не нарушает запрет на runtime-переключение модели/провайдера/системного промпта
из «Ограничений безопасности» в CLAUDE.md).

Как и Agent.ask(), SmartAgent.ask() не перехватывает исключения OpenAI SDK — перевод в
сообщение пользователю на русском делает вызывающий Telegram-обработчик
(agents/smart_agent_command.py), по тому же принципу, что и handle_message (main.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import (
    AGENT_MEMORY_DIR,
    AGENT_MEMORY_LONG_TERM_MAX_FACTS,
    AGENT_MEMORY_SHORT_TERM_PAIRS,
    MAIN_MODEL,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from providers.main_client import main_client

logger = logging.getLogger(__name__)

LAYER_SHORT_TERM = "short_term"
LAYER_WORKING = "working"
LAYER_LONG_TERM = "long_term"
ALL_LAYERS = (LAYER_SHORT_TERM, LAYER_WORKING, LAYER_LONG_TERM)


@dataclass
class SmartAgentAnswer:
    """Результат ask() — текст ответа плюс статистика по токенам, по тому же принципу,
    что AgentAnswer в agents/agent.py (см. его докстринг про то, откуда берётся каждое
    из значений — здесь ровно то же самое, только контекст строится из явных слоёв, а
    не стратегией)."""

    text: str
    request_tokens_approx: int
    context_tokens: int | None
    response_tokens: int | None


class SmartAgent:
    """Один экземпляр на чат (см. agents/smart_agent_command.py, кэш по chat_id, как
    _agents в agent_command.py). Состояние — три слоя памяти плюс enabled_layers —
    хранится в JSON-файле AGENT_MEMORY_DIR/<chat_id>.json, отдельном от файла истории
    /agent того же чата (AGENT_HISTORY_DIR/<chat_id>.json)."""

    def __init__(
        self,
        chat_id: int | str,
        client=main_client,
        model: str = MAIN_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        memory_dir: str = AGENT_MEMORY_DIR,
        short_term_pairs: int = AGENT_MEMORY_SHORT_TERM_PAIRS,
        long_term_max_facts: int = AGENT_MEMORY_LONG_TERM_MAX_FACTS,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._memory_path = Path(memory_dir) / f"{chat_id}.json"
        self._short_term_pairs = short_term_pairs
        self._long_term_max_facts = long_term_max_facts
        (
            self._short_term,
            self._working,
            self._long_term,
            self._enabled_layers,
        ) = self._load_state()
        # То, что реально ушло в LLM на последний ask() — см. get_last_context_messages().
        self._last_context_messages: list[dict[str, str]] = []

    @staticmethod
    def _default_enabled_layers() -> dict[str, bool]:
        return {layer: True for layer in ALL_LAYERS}

    def _load_state(
        self,
    ) -> tuple[list[dict[str, str]], dict | None, list[str], dict[str, bool]]:
        try:
            raw = self._memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], None, [], self._default_enabled_layers()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Повреждённый файл памяти smart-агента %s — начинаю с пустой памяти.",
                self._memory_path,
            )
            return [], None, [], self._default_enabled_layers()

        if not isinstance(data, dict):
            return [], None, [], self._default_enabled_layers()

        short_term = data.get("short_term")
        short_term = short_term if isinstance(short_term, list) else []

        working = data.get("working")
        working = working if isinstance(working, dict) else None

        long_term = data.get("long_term")
        long_term = long_term if isinstance(long_term, list) else []

        enabled_layers = self._default_enabled_layers()
        raw_enabled_layers = data.get("enabled_layers")
        if isinstance(raw_enabled_layers, dict):
            enabled_layers.update(
                {
                    layer: value
                    for layer, value in raw_enabled_layers.items()
                    if layer in ALL_LAYERS and isinstance(value, bool)
                }
            )

        return short_term, working, long_term, enabled_layers

    def _save_state(self) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "short_term": self._short_term,
            "working": self._working,
            "long_term": self._long_term,
            "enabled_layers": self._enabled_layers,
        }
        self._memory_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Наблюдаемость: что реально лежит в каждом слое и что ушло в LLM --- #

    def get_short_term(self) -> list[dict[str, str]]:
        return list(self._short_term)

    def get_working(self) -> dict | None:
        return dict(self._working) if self._working is not None else None

    def get_long_term_facts(self) -> list[str]:
        return list(self._long_term)

    def get_enabled_layers(self) -> dict[str, bool]:
        return dict(self._enabled_layers)

    def get_last_context_messages(self) -> list[dict[str, str]]:
        """Системные сообщения, реально собранные из включённых слоёв на последний
        вызов ask() (см. /smart_agent_show) — так можно проверить, что именно из
        каждого слоя попало в конкретный запрос к LLM."""
        return list(self._last_context_messages)

    # --- Переключение слоёв в СБОРКЕ контекста (данные слоя не удаляются) --- #

    def set_layer_enabled(self, layer: str, enabled: bool) -> None:
        self._enabled_layers[layer] = enabled
        self._save_state()

    # --- Долговременная память: только явная запись, дословно --- #

    def remember(self, fact: str) -> None:
        """Добавляет факт в долговременную память ДОСЛОВНО, без LLM-классификации —
        единственный способ туда что-то попадает (см. докстринг класса). Если список
        превышает long_term_max_facts, вытесняется САМЫЙ СТАРЫЙ факт — простой лимит
        по количеству, а не по токенам (в отличие от AGENT_FACTS_MAX_TOKENS у Agent):
        здесь нет вызова LLM, который нужно было бы защищать от обрезки ответа."""
        self._long_term.append(fact)
        if len(self._long_term) > self._long_term_max_facts:
            self._long_term = self._long_term[-self._long_term_max_facts :]
        self._save_state()

    def forget(self, index: int) -> bool:
        """Удаляет факт по 1-based номеру (как показывает /smart_agent_long_show).
        Возвращает False, если номер вне диапазона — вызывающий код превращает это в
        сообщение пользователю."""
        if index < 1 or index > len(self._long_term):
            return False
        del self._long_term[index - 1]
        self._save_state()
        return True

    # --- Рабочая память: только явная запись, одна задача на чат за раз --- #

    def start_task(self, goal: str) -> None:
        """Начинает новую рабочую задачу — ЗАМЕНЯЕТ предыдущую (одна активная задача
        на чат, см. докстринг класса), а не копит несколько параллельно."""
        self._working = {
            "goal": goal,
            "status": "active",
            "data": {},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_state()

    def set_task_data(self, key: str, value: str) -> bool:
        """Явно кладёт пару ключ-значение в данные текущей задачи. Возвращает False,
        если задача не начата — вызывающий код подсказывает /smart_agent_task_start."""
        if self._working is None:
            return False
        self._working["data"][key] = value
        self._save_state()
        return True

    def finish_task(self) -> bool:
        """Завершает и очищает текущую задачу. Возвращает False, если задачи и так не
        было — вызывающий код превращает это в соответствующее сообщение."""
        if self._working is None:
            return False
        self._working = None
        self._save_state()
        return True

    # --- Полная очистка (не трогает enabled_layers — это настройка режима, не данные,
    # тот же принцип, что Agent.reset() не трогает выбранную стратегию) --- #

    def reset_all(self) -> None:
        self._short_term = []
        self._working = None
        self._long_term = []
        self._save_state()

    # --- Сборка контекста и вызов LLM --- #

    def _recent_short_term(self) -> list[dict[str, str]]:
        window_size = 2 * self._short_term_pairs
        return self._short_term[-window_size:] if window_size > 0 else []

    def _build_context_messages(self) -> list[dict[str, str]]:
        """Собирает контекст LLM только из явно ВКЛЮЧЁННЫХ слоёв (self._enabled_layers)
        — в отличие от Agent, здесь нет автоматического выбора одной стратегии:
        пользователь сам решает и что сохранять (см. remember/start_task), и какие
        слои участвуют в конкретном запросе (см. set_layer_enabled). Порядок слоёв в
        сообщении — от самого общего/стабильного контекста к самому свежему:
        долговременная память -> рабочая задача -> краткосрочный диалог.
        """
        messages = [{"role": "system", "content": self._system_prompt}]

        if self._enabled_layers[LAYER_LONG_TERM] and self._long_term:
            facts_text = "\n".join(f"- {fact}" for fact in self._long_term)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Долговременная память о пользователе (сохранена им явно "
                        f"командой /smart_agent_remember):\n{facts_text}"
                    ),
                }
            )

        if self._enabled_layers[LAYER_WORKING] and self._working is not None:
            data_text = (
                "\n".join(f"- {key}: {value}" for key, value in self._working["data"].items())
                or "(пока нет дополнительных данных)"
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Текущая рабочая задача пользователя: {self._working['goal']}.\n"
                        f"Данные задачи:\n{data_text}"
                    ),
                }
            )

        if self._enabled_layers[LAYER_SHORT_TERM]:
            messages.extend(self._recent_short_term())

        return messages

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return None
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    def ask(self, user_text: str) -> SmartAgentAnswer:
        """Отправляет вопрос в LLM вместе с контекстом, собранным из явно включённых
        слоёв (см. _build_context_messages), и возвращает ответ. Может выбросить
        исключение OpenAI SDK — см. докстринг класса про то, что перехват делает
        вызывающий код. История дописывается в short_term только после успешного
        ответа API — неудачный вызов не искажает сохранённый диалог.
        """
        # Делитель 2 (а не общепринятые для английского 4 символа/токен) — как в
        # Agent.ask(), см. докстринг agents/agent.py про кириллицу в BPE-токенайзерах.
        request_tokens_approx = len(user_text) // 2

        messages = self._build_context_messages()
        self._last_context_messages = list(messages)
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
        context_tokens = usage["prompt_tokens"] if usage else None
        response_tokens = usage["completion_tokens"] if usage else None

        self._short_term.append({"role": "user", "content": user_text})
        self._short_term.append({"role": "assistant", "content": answer})
        self._save_state()

        return SmartAgentAnswer(
            text=answer,
            request_tokens_approx=request_tokens_approx,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
        )
