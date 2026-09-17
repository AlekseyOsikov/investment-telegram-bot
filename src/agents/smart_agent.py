"""SmartAgent — LLM-агент с явно разделённой моделью памяти (три независимых слоя),
в отличие от Agent (agents/agent.py), который переключает ОДНУ стратегию управления
контекстом поверх единой истории диалога. Отдельная, независимая от /agent сущность
(см. «Управление памятью smart-агента» в CLAUDE.md про то, почему это новый агент, а
не 5-я стратегия Agent) — свой класс, свой файл истории, свои команды /smart_agent_*.

Три слоя памяти, каждый хранится отдельно и пишется по-разному:
- short_term (краткосрочная, текущий диалог) — сырые сообщения, пишутся
  АВТОМАТИЧЕСКИ на каждый вызов ask(), как обычная история чата.
- working (рабочая, данные текущей задачи) — одна активная задача на профиль
  ({"goal", "status", "data", "created_at"} или None), пишется ТОЛЬКО явно
  (start_task/set_task_data/finish_task), никогда автоматически.
- long_term (долговременная, факты) — список текстовых фактов, пишется ТОЛЬКО явно
  (remember/forget), дословно, без LLM-классификации: ответственность за то, чтобы не
  сохранять туда чувствительные данные (номера счетов/карт, паспортные данные и т.п.,
  см. «Правила предметной области» в CLAUDE.md), остаётся на пользователе, как и для
  working — эта же осторожность применима к обоим explicit-слоям, не только к
  long_term.

Поверх этих трёх слоёв — ПРОФИЛИ ПОЛЬЗОВАТЕЛЯ (персонализация): на чат может быть
заведено несколько именованных профилей (например, «Консервативный»/«Агрессивный»),
и ВСЕ ТРИ СЛОЯ ПАМЯТИ хранятся В РАЗРЕЗЕ АКТИВНОГО ПРОФИЛЯ — переключение профиля
переключает и диалог, и задачу, и факты целиком, как переключение ветки в Branching-
стратегии Agent (agents/agent.py), только здесь это применено к независимой модели
памяти SmartAgent, а не к переключаемым стратегиям контекста. Профиль дополнительно
хранит "meta" — явно заданные пользователем предпочтения персонализации (стиль,
уровень опыта, формат ответа, отношение к риску, горизонт, интересы, что не
затрагивать, см. PROFILE_FIELDS) — они подключаются к каждому запросу отдельным
системным сообщением (см. _profile_meta_message), но влияют ТОЛЬКО на стиль/формат
ответа, а не на обязательные предупреждения основного SYSTEM_PROMPT (см. «Правила
предметной области» в CLAUDE.md). Как и working/long_term, meta пишется ТОЛЬКО явно
пользователем (анкета при создании профиля или /smart_agent_profile_set) — никакой
LLM-эвристики, решающей за пользователя, что положить в профиль.

Если у чата нет активного профиля (новый чат или профиль был удалён), работать с
short_term/working/long_term невозможно — методы этих слоёв возвращают "пусто"/False
для чтения и явно сигнализируют вызывающему Telegram-коду (agents/smart_agent_command.py),
что нужно сначала предложить пользователю выбрать или создать профиль
(get_active_profile_name() is None). ask() в этой ситуации не вызывается: сам вход в
диалог /smart_agent сначала проводит через выбор/создание профиля.

Каждый слой (включая профиль-как-контекст) можно независимо включать/выключать в
СБОРКЕ контекста (enabled_layers, переключается /smart_agent_toggle) без удаления
самих данных — это и есть проверка "что попадает в каждый слой и как это влияет на
ответы", см. get_last_context_messages(). enabled_layers — общая настройка на ВЕСЬ
ЧАТ (не per-profile) — она про то, как вообще собирается контекст, а не про то, чьи
данные в нём участвуют (это решает активный профиль).

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

LAYER_PROFILE = "profile"
LAYER_SHORT_TERM = "short_term"
LAYER_WORKING = "working"
LAYER_LONG_TERM = "long_term"
ALL_LAYERS = (LAYER_PROFILE, LAYER_SHORT_TERM, LAYER_WORKING, LAYER_LONG_TERM)

# Поля профиля персонализации (анкета при создании, /smart_agent_profile_set для
# точечной правки, /smart_agent_profile_show для просмотра) — только качественные
# предпочтения стиля общения, НИКОГДА не финансовые/личные данные (см. докстринг
# модуля и «Правила предметной области» в CLAUDE.md).
PROFILE_FIELD_STYLE = "style"
PROFILE_FIELD_EXPERIENCE = "experience_level"
PROFILE_FIELD_FORMAT = "format"
PROFILE_FIELD_RISK = "risk_tolerance"
PROFILE_FIELD_HORIZON = "horizon"
PROFILE_FIELD_INTERESTS = "interests"
PROFILE_FIELD_EXCLUDED = "excluded_topics"
PROFILE_FIELDS = (
    PROFILE_FIELD_STYLE,
    PROFILE_FIELD_EXPERIENCE,
    PROFILE_FIELD_FORMAT,
    PROFILE_FIELD_RISK,
    PROFILE_FIELD_HORIZON,
    PROFILE_FIELD_INTERESTS,
    PROFILE_FIELD_EXCLUDED,
)

PROFILE_FIELD_LABELS = {
    PROFILE_FIELD_STYLE: "Стиль общения",
    PROFILE_FIELD_EXPERIENCE: "Уровень опыта",
    PROFILE_FIELD_FORMAT: "Формат ответа",
    PROFILE_FIELD_RISK: "Отношение к риску",
    PROFILE_FIELD_HORIZON: "Горизонт интересов",
    PROFILE_FIELD_INTERESTS: "Интересующие темы/активы",
    PROFILE_FIELD_EXCLUDED: "Что не затрагивать",
}

# Имя профиля, в который оборачиваются данные чата, сохранённые ДО появления
# профилей (плоский формат файла памяти) — см. SmartAgent._load_state(). Становится
# активным автоматически, чтобы уже работающие чаты не прерывались выбором профиля
# на пустом месте после обновления бота.
_MIGRATED_PROFILE_NAME = "Профиль по умолчанию"


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


def _empty_profile() -> dict:
    return {
        "meta": {field: "" for field in PROFILE_FIELDS},
        "short_term": [],
        "working": None,
        "long_term": [],
    }


class SmartAgent:
    """Один экземпляр на чат (см. agents/smart_agent_command.py, кэш по chat_id, как
    _agents в agent_command.py). Состояние — именованные профили (каждый со своими
    meta/short_term/working/long_term), указатель активного профиля и общий на чат
    enabled_layers — хранится в JSON-файле AGENT_MEMORY_DIR/<chat_id>.json, отдельном
    от файла истории /agent того же чата (AGENT_HISTORY_DIR/<chat_id>.json)."""

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
            self._profiles,
            self._active_profile,
            self._enabled_layers,
        ) = self._load_state()
        # То, что реально ушло в LLM на последний ask() — см. get_last_context_messages().
        self._last_context_messages: list[dict[str, str]] = []

    @staticmethod
    def _default_enabled_layers() -> dict[str, bool]:
        return {layer: True for layer in ALL_LAYERS}

    def _load_state(
        self,
    ) -> tuple[dict[str, dict], str | None, dict[str, bool]]:
        try:
            raw = self._memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}, None, self._default_enabled_layers()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Повреждённый файл памяти smart-агента %s — начинаю с пустой памяти.",
                self._memory_path,
            )
            return {}, None, self._default_enabled_layers()

        if not isinstance(data, dict):
            return {}, None, self._default_enabled_layers()

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

        raw_profiles = data.get("profiles")
        if isinstance(raw_profiles, dict):
            profiles = {
                name: self._sanitize_profile(entry)
                for name, entry in raw_profiles.items()
                if isinstance(entry, dict)
            }
            active_profile = data.get("active_profile")
            if not isinstance(active_profile, str) or active_profile not in profiles:
                active_profile = None
            return profiles, active_profile, enabled_layers

        # Старый плоский формат (до появления профилей) — оборачиваем то, что уже
        # было накоплено, в один профиль и делаем его сразу активным, чтобы уже
        # работающие чаты не прерывались выбором профиля на пустом месте (см.
        # докстринг модуля и _MIGRATED_PROFILE_NAME выше).
        legacy_short_term = data.get("short_term")
        legacy_short_term = legacy_short_term if isinstance(legacy_short_term, list) else []
        legacy_working = data.get("working")
        legacy_working = legacy_working if isinstance(legacy_working, dict) else None
        legacy_long_term = data.get("long_term")
        legacy_long_term = legacy_long_term if isinstance(legacy_long_term, list) else []

        if not (legacy_short_term or legacy_working or legacy_long_term):
            return {}, None, enabled_layers

        migrated = _empty_profile()
        migrated["short_term"] = legacy_short_term
        migrated["working"] = legacy_working
        migrated["long_term"] = legacy_long_term
        return {_MIGRATED_PROFILE_NAME: migrated}, _MIGRATED_PROFILE_NAME, enabled_layers

    @staticmethod
    def _sanitize_profile(entry: dict) -> dict:
        raw_meta = entry.get("meta")
        raw_meta = raw_meta if isinstance(raw_meta, dict) else {}
        meta = {field: str(raw_meta.get(field, "") or "") for field in PROFILE_FIELDS}

        short_term = entry.get("short_term")
        short_term = short_term if isinstance(short_term, list) else []

        working = entry.get("working")
        working = working if isinstance(working, dict) else None

        long_term = entry.get("long_term")
        long_term = long_term if isinstance(long_term, list) else []

        return {
            "meta": meta,
            "short_term": short_term,
            "working": working,
            "long_term": long_term,
        }

    def _save_state(self) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_profile": self._active_profile,
            "enabled_layers": self._enabled_layers,
            "profiles": self._profiles,
        }
        self._memory_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Профили: CRUD + переключение (см. докстринг класса про то, почему все три
    # слоя памяти живут внутри профиля, а не отдельно от него) --- #

    def get_active_profile_name(self) -> str | None:
        return self._active_profile

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def profile_exists(self, name: str) -> bool:
        return name in self._profiles

    def get_profile_meta(self, name: str) -> dict[str, str] | None:
        entry = self._profiles.get(name)
        return dict(entry["meta"]) if entry is not None else None

    def create_profile(self, name: str, meta: dict[str, str]) -> None:
        """Создаёт новый профиль с ПУСТЫМИ short_term/working/long_term (профили не
        наследуют память друг от друга — тот же принцип, что новая ветка Branching у
        Agent стартует без facts/summary ветки-источника) и сразу делает его
        активным."""
        profile = _empty_profile()
        profile["meta"] = {field: meta.get(field, "") for field in PROFILE_FIELDS}
        self._profiles[name] = profile
        self._active_profile = name
        self._save_state()

    def switch_profile(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        self._active_profile = name
        self._save_state()
        return True

    def delete_profile(self, name: str) -> bool:
        """Удаляет профиль целиком (вместе со всей его памятью). Если он был
        активным, активный профиль становится None ЯВНО — вызывающий код
        (agents/smart_agent_command.py) обязан в этом случае сразу показать выбор
        профиля, а не оставлять чат в подвешенном состоянии."""
        if name not in self._profiles:
            return False
        del self._profiles[name]
        if self._active_profile == name:
            self._active_profile = None
        self._save_state()
        return True

    def update_profile_field(self, key: str, value: str) -> bool:
        """Точечно правит одно поле meta активного профиля (см. /smart_agent_profile_set)
        без прохождения анкеты заново. Возвращает False, если нет активного профиля
        или ключ не входит в PROFILE_FIELDS."""
        if self._active_profile is None or key not in PROFILE_FIELDS:
            return False
        self._profiles[self._active_profile]["meta"][key] = value
        self._save_state()
        return True

    # --- Наблюдаемость: что реально лежит в каждом слое активного профиля и что
    # ушло в LLM --- #

    def get_short_term(self) -> list[dict[str, str]]:
        if self._active_profile is None:
            return []
        return list(self._profiles[self._active_profile]["short_term"])

    def get_working(self) -> dict | None:
        if self._active_profile is None:
            return None
        working = self._profiles[self._active_profile]["working"]
        return dict(working) if working is not None else None

    def get_long_term_facts(self) -> list[str]:
        if self._active_profile is None:
            return []
        return list(self._profiles[self._active_profile]["long_term"])

    def get_enabled_layers(self) -> dict[str, bool]:
        return dict(self._enabled_layers)

    def get_last_context_messages(self) -> list[dict[str, str]]:
        """Системные сообщения, реально собранные из включённых слоёв на последний
        вызов ask() (см. /smart_agent_show) — так можно проверить, что именно из
        каждого слоя попало в конкретный запрос к LLM."""
        return list(self._last_context_messages)

    # --- Переключение слоёв в СБОРКЕ контекста (данные слоя не удаляются, общая
    # настройка на весь чат, а не per-profile) --- #

    def set_layer_enabled(self, layer: str, enabled: bool) -> None:
        self._enabled_layers[layer] = enabled
        self._save_state()

    # --- Долговременная память активного профиля: только явная запись, дословно --- #

    def remember(self, fact: str) -> bool:
        """Добавляет факт в долговременную память АКТИВНОГО ПРОФИЛЯ дословно, без
        LLM-классификации (см. докстринг класса). Возвращает False, если активного
        профиля нет. Если список превышает long_term_max_facts, вытесняется САМЫЙ
        СТАРЫЙ факт — простой лимит по количеству, а не по токенам (в отличие от
        AGENT_FACTS_MAX_TOKENS у Agent): здесь нет вызова LLM, который нужно было бы
        защищать от обрезки ответа."""
        if self._active_profile is None:
            return False
        long_term = self._profiles[self._active_profile]["long_term"]
        long_term.append(fact)
        if len(long_term) > self._long_term_max_facts:
            del long_term[: len(long_term) - self._long_term_max_facts]
        self._save_state()
        return True

    def forget(self, index: int) -> bool:
        """Удаляет факт по 1-based номеру (как показывает /smart_agent_long_show) из
        долговременной памяти АКТИВНОГО ПРОФИЛЯ. Возвращает False, если активного
        профиля нет или номер вне диапазона — вызывающий код превращает это в
        сообщение пользователю."""
        if self._active_profile is None:
            return False
        long_term = self._profiles[self._active_profile]["long_term"]
        if index < 1 or index > len(long_term):
            return False
        del long_term[index - 1]
        self._save_state()
        return True

    # --- Рабочая память активного профиля: только явная запись, одна задача за раз --- #

    def start_task(self, goal: str) -> bool:
        """Начинает новую рабочую задачу в АКТИВНОМ ПРОФИЛЕ — ЗАМЕНЯЕТ предыдущую
        задачу этого профиля (одна активная задача на профиль, см. докстринг класса),
        а не копит несколько параллельно. Возвращает False, если активного профиля нет."""
        if self._active_profile is None:
            return False
        self._profiles[self._active_profile]["working"] = {
            "goal": goal,
            "status": "active",
            "data": {},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_state()
        return True

    def set_task_data(self, key: str, value: str) -> bool:
        """Явно кладёт пару ключ-значение в данные текущей задачи активного профиля.
        Возвращает False, если нет активного профиля или в нём не начата задача —
        вызывающий код подсказывает /smart_agent_task_start."""
        if self._active_profile is None:
            return False
        working = self._profiles[self._active_profile]["working"]
        if working is None:
            return False
        working["data"][key] = value
        self._save_state()
        return True

    def finish_task(self) -> bool:
        """Завершает и очищает текущую задачу активного профиля. Возвращает False,
        если нет активного профиля или задачи и так не было."""
        if self._active_profile is None:
            return False
        if self._profiles[self._active_profile]["working"] is None:
            return False
        self._profiles[self._active_profile]["working"] = None
        self._save_state()
        return True

    # --- Полная очистка ТРЁХ СЛОЁВ АКТИВНОГО ПРОФИЛЯ (не трогает сам профиль,
    # meta, другие профили и enabled_layers — тот же принцип, что Agent.reset() не
    # трогает выбранную стратегию; удаление профиля целиком — отдельная явная
    # команда, delete_profile) --- #

    def reset_all(self) -> bool:
        """Возвращает False, если нет активного профиля — вызывающий код подсказывает
        /smart_agent_profile."""
        if self._active_profile is None:
            return False
        profile = self._profiles[self._active_profile]
        profile["short_term"] = []
        profile["working"] = None
        profile["long_term"] = []
        self._save_state()
        return True

    # --- Сборка контекста и вызов LLM --- #

    def _profile_meta_message(self, meta: dict[str, str]) -> dict[str, str] | None:
        """Системное сообщение с профилем персонализации активного чата — влияет
        ТОЛЬКО на стиль/формат/объём ответа, о чём сообщение явно предупреждает
        модель, а не отменяет обязательные предупреждения основного system_prompt
        (см. докстринг класса и «Правила предметной области» в CLAUDE.md). Возвращает
        None, если ни одно поле профиля не заполнено — пустое сообщение не нужно."""
        lines = [
            f"- {PROFILE_FIELD_LABELS[field]}: {meta[field]}"
            for field in PROFILE_FIELDS
            if meta.get(field)
        ]
        if not lines:
            return None
        return {
            "role": "system",
            "content": (
                "Профиль пользователя (сохранён им явно). Учитывай эти предпочтения "
                "ТОЛЬКО для стиля, формата и объёма ответа — они не отменяют "
                "обязательные предупреждения и осторожные формулировки из основной "
                "инструкции:\n" + "\n".join(lines)
            ),
        }

    def _build_context_messages(self) -> list[dict[str, str]]:
        """Собирает контекст LLM из явно ВКЛЮЧЁННЫХ слоёв (self._enabled_layers)
        активного профиля — в отличие от Agent, здесь нет автоматического выбора
        одной стратегии: пользователь сам решает и что сохранять (см. remember/
        start_task), и какие слои участвуют в конкретном запросе (см.
        set_layer_enabled). Порядок слоёв в сообщении — от самого общего/стабильного
        контекста к самому свежему: профиль -> долговременная память -> рабочая
        задача -> краткосрочный диалог.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if self._active_profile is None:
            return messages
        profile = self._profiles[self._active_profile]

        if self._enabled_layers[LAYER_PROFILE]:
            meta_message = self._profile_meta_message(profile["meta"])
            if meta_message:
                messages.append(meta_message)

        if self._enabled_layers[LAYER_LONG_TERM] and profile["long_term"]:
            facts_text = "\n".join(f"- {fact}" for fact in profile["long_term"])
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Долговременная память о пользователе (сохранена им явно "
                        f"командой /smart_agent_remember):\n{facts_text}"
                    ),
                }
            )

        if self._enabled_layers[LAYER_WORKING] and profile["working"] is not None:
            working = profile["working"]
            data_text = (
                "\n".join(f"- {key}: {value}" for key, value in working["data"].items())
                or "(пока нет дополнительных данных)"
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Текущая рабочая задача пользователя: {working['goal']}.\n"
                        f"Данные задачи:\n{data_text}"
                    ),
                }
            )

        if self._enabled_layers[LAYER_SHORT_TERM]:
            window_size = 2 * self._short_term_pairs
            short_term = profile["short_term"]
            messages.extend(short_term[-window_size:] if window_size > 0 else [])

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
        """Отправляет вопрос в LLM вместе с контекстом активного профиля, собранным
        из явно включённых слоёв (см. _build_context_messages), и возвращает ответ.
        Требует уже выбранный активный профиль — вызывающий Telegram-код
        (agents/smart_agent_command.py) проводит пользователя через выбор/создание
        профиля ДО входа в цикл вопросов, поэтому здесь отсутствие активного профиля
        — программная ошибка вызывающего кода, а не пользовательский сценарий.
        Может выбросить исключение OpenAI SDK — см. докстринг класса про то, что
        перехват делает вызывающий код. История дописывается в short_term активного
        профиля только после успешного ответа API — неудачный вызов не искажает
        сохранённый диалог.
        """
        if self._active_profile is None:
            raise RuntimeError(
                "SmartAgent.ask() вызван без активного профиля — вызывающий код "
                "должен был провести пользователя через выбор/создание профиля до "
                "входа в цикл вопросов."
            )

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

        short_term = self._profiles[self._active_profile]["short_term"]
        short_term.append({"role": "user", "content": user_text})
        short_term.append({"role": "assistant", "content": answer})
        self._save_state()

        return SmartAgentAnswer(
            text=answer,
            request_tokens_approx=request_tokens_approx,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
        )
