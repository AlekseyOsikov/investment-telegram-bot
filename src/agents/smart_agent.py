"""SmartAgent — LLM-агент с явно разделённой моделью памяти (три независимых слоя
плюс слой инвариантов), в отличие от Agent (agents/agent.py), который переключает
ОДНУ стратегию управления контекстом поверх единой истории диалога. Отдельная,
независимая от /agent сущность (см. «Управление памятью smart-агента» в CLAUDE.md про
то, почему это новый агент, а не 5-я стратегия Agent) — свой класс, свой файл истории,
свои команды /smart_agent_*.

Три слоя памяти, каждый хранится отдельно и пишется по-разному:
- short_term (краткосрочная, текущий диалог) — сырые сообщения, пишутся
  АВТОМАТИЧЕСКИ на каждый вызов ask(), как обычная история чата.
- working (рабочая, текущая задача) — одна активная задача на профиль, оформленная
  как КОНЕЧНЫЙ АВТОМАТ (этап/шаг/ожидаемое действие/пауза, см. agents/task_state.py).
  Пишется и вручную (start_task/set_task_data/finish_task/pause_task/set_task_stage),
  и АВТОМАТИЧЕСКИ — отдельным техническим вызовом LLM после каждого ответа агента
  (update_task_state), который двигает этап и заполняет данные задачи. Это осознанное
  исключение из прежнего принципа «в рабочую и долговременную память пишет только
  пользователь явной командой», сделанное по явному запросу пользователя проекта —
  см. «Управление памятью smart-агента» в CLAUDE.md.
- long_term (долговременная, факты) — список фактов {"text", "source", "created_at"},
  где source различает сохранённые пользователем дословно (/smart_agent_remember) и
  извлечённые автоматически тем же техническим вызовом. При переполнении лимита
  вытесняются СНАЧАЛА автоматические факты — иначе автоизвлечение постепенно вымыло
  бы из памяти то, что пользователь сохранил руками.

Отдельно от этих трёх слоёв ПАМЯТИ у профиля есть слой ОГРАНИЧЕНИЙ — инварианты
(agents/invariants.py): жёсткие правила пользователя («без криптовалют», «доля акций
не выше 40%»), которые агент не имеет права нарушать. Они пишутся ТОЛЬКО явными
командами (/smart_agent_invariant_add, /smart_agent_invariant_remove) — никакой
автоматической записи, в отличие от working/long_term, — уходят в контекст ПЕРВЫМ
системным сообщением, с приоритетом над профилем персонализации, и дополнительно
проверяются кодом: результат рабочей задачи прогоняется через отдельный
вызов-ревизор (_check_invariants), и при нарушении задача откатывается на доработку
(task_state.apply_invariant_violations). Лимит слоя — с вытеснением НЕ работает: при
переполнении добавление отклоняется, см. add_invariant.

Ни один слой не фильтрует содержимое на вход, поэтому ответственность за то, чтобы
не сохранять туда чувствительные данные (номера счетов/карт, паспортные данные,
денежные суммы, см. «Правила предметной области» в CLAUDE.md), остаётся на
пользователе — а в системных промптах технического вызова прямо прописан запрет
извлекать такие данные из диалога.

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

Каждый слой (включая профиль-как-контекст и инварианты) можно независимо
включать/выключать в СБОРКЕ контекста (enabled_layers, переключается
/smart_agent_toggle) без удаления самих данных — для инвариантов выключение
отключает и вызов-ревизор, см. _check_invariants. Это и есть проверка "что попадает
в каждый слой и как это влияет на ответы", см. get_last_context_messages().
enabled_layers — общая настройка на ВЕСЬ ЧАТ (не per-profile) — она про то, как
вообще собирается контекст, а не про то, чьи данные в нём участвуют (это решает
активный профиль).

Ещё один переключаемый слой — `tools` (agents/market_tools.py): доступ ОСНОВНОГО
ответа к инструментам локального MCP-сервера рыночных данных mcp-moex (поиск бумаги,
текущая цена, история цен) через function calling. Это не память и не ограничение, а
ВОЗМОЖНОСТЬ: он ничего не хранит и ничего не запрещает, только добавляет данные биржи
в ответ. Работает, если задан MCP_MOEX_DIR и слой включён; сервер поднимается один раз
на вопрос (см. ask()). Выключенный слой при настроенном сервере добавляет в контекст
короткое сообщение «данные биржи выключены» — иначе модель выдаёт цену из прежнего
ответа за текущую. Служебные вызовы LLM (автомат задачи, детектор старта, ревизор
инвариантов) инструментов НЕ получают. В short_term по-прежнему пишется только пара
«вопрос — ответ», промежуточные сообщения цикла вызовов не сохраняются.

Использует main_client/MAIN_MODEL — того же провайдера и модель, что и основной поток
бота, по тому же принципу, что и Agent (см. докстринг agents/agent.py про то, почему
это не нарушает запрет на runtime-переключение модели/провайдера/системного промпта
из «Ограничений безопасности» в CLAUDE.md).

Как и Agent.ask(), SmartAgent.ask() не перехватывает исключения OpenAI SDK — перевод в
сообщение пользователю на русском делает вызывающий Telegram-обработчик
(agents/smart_agent_command.py), по тому же принципу, что и handle_message (main.py).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import (
    AGENT_INVARIANTS_MAX_TOKENS,
    AGENT_INVARIANTS_SYSTEM_PROMPT,
    AGENT_MEMORY_DIR,
    AGENT_MEMORY_LONG_TERM_MAX_FACTS,
    AGENT_MEMORY_MAX_INVARIANTS,
    AGENT_MEMORY_SHORT_TERM_PAIRS,
    AGENT_TASK_START_MAX_TOKENS,
    AGENT_TASK_START_SYSTEM_PROMPT,
    AGENT_TASK_STATE_MAX_TOKENS,
    AGENT_TASK_STATE_SYSTEM_PROMPT,
    MAIN_MODEL,
    MAX_OUTPUT_TOKENS,
    MCP_MAX_TOOL_STEPS,
    MCP_MOEX_DIR,
    MCP_TIMEOUT_SECONDS,
    MCP_TOOL_RESULT_MAX_CHARS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from mcp_integration.market_session import open_market_tools
from providers.main_client import main_client

from . import invariants, market_tools, task_state

logger = logging.getLogger(__name__)

LAYER_PROFILE = "profile"
LAYER_INVARIANTS = "invariants"
LAYER_SHORT_TERM = "short_term"
LAYER_WORKING = "working"
LAYER_LONG_TERM = "long_term"
LAYER_TOOLS = "tools"
# Слой tools добавлен в конец, чтобы не менять порядок прежних слоёв: место сообщения
# слоя в контексте (после профиля, до долговременной памяти) определяет
# _build_context_messages, а не порядок этого кортежа. Старые файлы памяти без ключа
# "tools" читаются как «включён» (см. _load_state) — миграция не нужна.
ALL_LAYERS = (
    LAYER_PROFILE,
    LAYER_INVARIANTS,
    LAYER_SHORT_TERM,
    LAYER_WORKING,
    LAYER_LONG_TERM,
    LAYER_TOOLS,
)

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

# Происхождение факта долговременной памяти: сохранён пользователем дословно или
# извлечён техническим вызовом из диалога. От этого зависит порядок вытеснения при
# переполнении лимита (см. _append_fact) и пометка в /smart_agent_long_show.
FACT_SOURCE_USER = "user"
FACT_SOURCE_AUTO = "auto"

# Потолок на автоматическое пополнение долговременной памяти за один ход — см.
# SmartAgent._extracted_facts.
_MAX_AUTO_FACTS_PER_TURN = 3
_MAX_AUTO_FACT_CHARS = 300


@dataclass
class TaskStateUpdate:
    """Что изменилось в задаче после технического вызова — командный слой
    превращает это в сообщение пользователю (см. update_task_state). Сама строка
    состояния сюда не входит: она печатается КАЖДЫЙ ход, в том числе когда
    ничего не изменилось, и берётся отдельно через get_task_state_line()."""

    started: bool = False
    transition: str | None = None
    archived: bool = False
    # Нарушения инвариантов, из-за которых задача уехала назад на доработку
    # (см. _check_invariants). Непустой список означает, что transition — это откат,
    # а не движение вперёд, и командный слой печатает его иначе.
    violations: list[dict] = field(default_factory=list)
    # Причины, по которым автомат не сделал того, что предложила модель (переход не
    # разрешён, ключ не этого этапа, значение вне списка). Печатаются пользователю:
    # иначе отклонённая попытка выглядит как проигнорированная просьба.
    rejected: list[str] = field(default_factory=list)
    # Гейт этапа сняли отказом (сводку вводных не подтвердили) — этап тот же, но
    # работа на нём начинается заново, и командный слой сообщает об этом отдельно.
    gate_reset: bool = False


@dataclass
class SmartAgentAnswer:
    """Результат ask() — текст ответа плюс статистика по токенам, по тому же принципу,
    что AgentAnswer в agents/agent.py (см. его докстринг про то, откуда берётся каждое
    из значений — здесь ровно то же самое, только контекст строится из явных слоёв, а
    не стратегией)."""

    text: str
    request_tokens_approx: int
    # При вызовах инструментов — суммы по ВСЕМ обращениям к модели за вопрос (это
    # реальная стоимость вопроса), а не по последнему; llm_calls — сколько их было.
    context_tokens: int | None
    response_tokens: int | None
    # Слой tools: вызванные инструменты и предупреждения для пользователя
    # (недоступный сервер, сбой обмена посреди вопроса, исчерпанный лимит шагов) —
    # командный слой печатает их отдельно от ответа (agents/smart_agent_command.py).
    tool_calls: list[market_tools.ToolCallRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 1


def _empty_profile() -> dict:
    return {
        "meta": {name: "" for name in PROFILE_FIELDS},
        "invariants": [],
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
        max_invariants: int = AGENT_MEMORY_MAX_INVARIANTS,
        task_state_max_tokens: int = AGENT_TASK_STATE_MAX_TOKENS,
        task_start_max_tokens: int = AGENT_TASK_START_MAX_TOKENS,
        invariants_max_tokens: int = AGENT_INVARIANTS_MAX_TOKENS,
        mcp_moex_dir: str = MCP_MOEX_DIR,
        mcp_timeout: float = MCP_TIMEOUT_SECONDS,
        max_tool_steps: int = MCP_MAX_TOOL_STEPS,
        tool_result_max_chars: int = MCP_TOOL_RESULT_MAX_CHARS,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._memory_path = Path(memory_dir) / f"{chat_id}.json"
        self._short_term_pairs = short_term_pairs
        self._long_term_max_facts = long_term_max_facts
        self._max_invariants = max_invariants
        self._task_state_max_tokens = task_state_max_tokens
        self._task_start_max_tokens = task_start_max_tokens
        self._invariants_max_tokens = invariants_max_tokens
        self._mcp_moex_dir = mcp_moex_dir
        self._mcp_timeout = mcp_timeout
        self._max_tool_steps = max_tool_steps
        self._tool_result_max_chars = tool_result_max_chars
        (
            self._profiles,
            self._active_profile,
            self._enabled_layers,
        ) = self._load_state()
        # То, что реально ушло в LLM на последний ask() — см. get_last_context_messages().
        self._last_context_messages: list[dict[str, str]] = []
        # Слой tools: итог последнего вопроса (в памяти, на диск не пишется — это
        # диагностика для /smart_agent_show, а не настройка).
        self._tools_status = market_tools.STATUS_UNKNOWN
        self._tools_status_reason: str | None = None
        self._last_tool_calls: list[market_tools.ToolCallRecord] = []

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
        if not (data.get("short_term") or data.get("working") or data.get("long_term")):
            return {}, None, enabled_layers

        migrated = self._sanitize_profile(data)
        return {_MIGRATED_PROFILE_NAME: migrated}, _MIGRATED_PROFILE_NAME, enabled_layers

    @staticmethod
    def _sanitize_profile(entry: dict) -> dict:
        raw_meta = entry.get("meta")
        raw_meta = raw_meta if isinstance(raw_meta, dict) else {}
        meta = {name: str(raw_meta.get(name, "") or "") for name in PROFILE_FIELDS}

        # Профили, сохранённые ДО появления слоя инвариантов, ключа не содержат —
        # получают пустой список, отдельной миграции не требуется.
        invariant_items = invariants.sanitize_list(entry.get("invariants"))

        short_term = entry.get("short_term")
        short_term = short_term if isinstance(short_term, list) else []

        # Задача старого формата ({"goal", "status", "data", ...}, до появления
        # автомата) не выбрасывается, а мигрирует в сценарий по умолчанию на
        # первый этап — см. task_state.sanitize_task.
        working = task_state.sanitize_task(entry.get("working"))

        raw_long_term = entry.get("long_term")
        raw_long_term = raw_long_term if isinstance(raw_long_term, list) else []
        long_term = [
            fact
            for fact in (SmartAgent._sanitize_fact(item) for item in raw_long_term)
            if fact is not None
        ]

        return {
            "meta": meta,
            "invariants": invariant_items,
            "short_term": short_term,
            "working": working,
            "long_term": long_term,
        }

    @staticmethod
    def _sanitize_fact(item) -> dict | None:
        """Факт долговременной памяти. Строка — это старый формат (до появления
        автоматического извлечения), и она мигрирует в факт с source="user": до
        этой версии в long_term вообще ничего не попадало без явной команды
        пользователя, так что такая пометка исторически верна."""
        if isinstance(item, str):
            text = item.strip()
            return {"text": text, "source": FACT_SOURCE_USER, "created_at": ""} if text else None
        if not isinstance(item, dict):
            return None
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        source = item.get("source")
        if source not in (FACT_SOURCE_USER, FACT_SOURCE_AUTO):
            source = FACT_SOURCE_USER
        created_at = item.get("created_at")
        return {
            "text": text,
            "source": source,
            "created_at": created_at if isinstance(created_at, str) else "",
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
        profile["meta"] = {name: meta.get(name, "") for name in PROFILE_FIELDS}
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
        return copy.deepcopy(working) if working is not None else None

    def get_task_state_line(self) -> str | None:
        """Короткая строка о состоянии автомата для чата (или None, если активной
        задачи нет — тогда служебной строки быть не должно вовсе, чтобы обычные
        разовые вопросы ею не обрастали)."""
        task = self.get_working()
        return task_state.state_line(task) if task is not None else None

    def get_long_term_facts(self) -> list[dict[str, str]]:
        """Факты активного профиля как есть — каждый со своим source (см.
        FACT_SOURCE_*), чтобы /smart_agent_long_show мог показать, что пришло из
        диалога, а что сохранено пользователем."""
        if self._active_profile is None:
            return []
        return [dict(fact) for fact in self._profiles[self._active_profile]["long_term"]]

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
        """Добавляет факт в долговременную память АКТИВНОГО ПРОФИЛЯ дословно, как
        сохранённый пользователем (/smart_agent_remember). Возвращает False, если
        активного профиля нет."""
        if self._active_profile is None:
            return False
        self._append_fact(fact, FACT_SOURCE_USER)
        self._save_state()
        return True

    def _append_fact(self, text: str, source: str) -> None:
        """Общая запись факта для обоих источников (см. FACT_SOURCE_*). Требует уже
        проверенного активного профиля и НЕ сохраняет состояние на диск — это делает
        вызывающий код, чтобы пачка автоматических фактов не переписывала файл по
        разу на каждый факт.

        При переполнении лимита вытесняется самый старый АВТОМАТИЧЕСКИЙ факт, и
        только если автоматических больше нет — самый старый вообще. Так
        автоизвлечение не вымывает из памяти то, что пользователь сохранил руками,
        но и не ломает прежнее поведение FIFO, когда все факты пользовательские.
        Только что добавленный факт из кандидатов на вытеснение исключён — иначе
        при полной памяти он бы тут же и удалялся.
        """
        long_term = self._profiles[self._active_profile]["long_term"]
        long_term.append(
            {
                "text": text,
                "source": source,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        while len(long_term) > self._long_term_max_facts:
            candidates = range(len(long_term) - 1)
            victim = next(
                (i for i in candidates if long_term[i].get("source") == FACT_SOURCE_AUTO),
                0,
            )
            del long_term[victim]

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

    # --- Инварианты активного профиля: только явная запись, без вытеснения --- #

    def get_invariants(self) -> list[dict[str, str]]:
        if self._active_profile is None:
            return []
        return [dict(item) for item in self._profiles[self._active_profile]["invariants"]]

    def add_invariant(self, text: str, category: str) -> str:
        """Добавляет инвариант в АКТИВНЫЙ ПРОФИЛЬ. Возвращает код результата
        (invariants.ADD_*) — текст пользователю собирает командный слой, как и у
        остальных методов.

        При переполнении лимита НИЧЕГО не вытесняется (в отличие от долговременной
        памяти, см. _append_fact), а добавление отклоняется: молча выбросить запрет,
        который пользователь задал явно, — значит незаметно для него перестать его
        соблюдать.
        """
        if self._active_profile is None:
            return invariants.ADD_NO_PROFILE
        items = self._profiles[self._active_profile]["invariants"]
        if len(items) >= self._max_invariants:
            return invariants.ADD_LIMIT
        items.append(invariants.new_invariant(text, category))
        self._save_state()
        return invariants.ADD_OK

    def remove_invariant(self, index: int) -> dict[str, str] | None:
        """Удаляет инвариант по 1-based номеру (как показывает
        /smart_agent_invariant_show) и возвращает удалённый. Это единственный способ
        снять ограничение — ни агент, ни технический вызов инварианты не трогают."""
        if self._active_profile is None:
            return None
        items = self._profiles[self._active_profile]["invariants"]
        if index < 1 or index > len(items):
            return None
        removed = items.pop(index - 1)
        self._save_state()
        return removed

    # --- Рабочая память активного профиля: только явная запись, одна задача за раз --- #

    def start_task(self, task_type: str, goal: str) -> bool:
        """Начинает новую рабочую задачу в АКТИВНОМ ПРОФИЛЕ — ЗАМЕНЯЕТ предыдущую
        задачу этого профиля (одна активная задача на профиль, см. докстринг класса),
        а не копит несколько параллельно. Возвращает False, если активного профиля
        нет или сценарий неизвестен.

        Это явный путь пользователя (/smart_agent_task_start). Автоматический старт
        задачи по намерению в диалоге идёт через update_task_state() и срабатывает
        только тогда, когда активной задачи НЕТ — молча подменять незавершённую
        задачу автомат не должен (см. _maybe_start_task).
        """
        if self._active_profile is None or task_type not in task_state.SCENARIOS:
            return False
        profile = self._profiles[self._active_profile]
        # Уже накопленный диалог считаем учтённым: задача начинается «с этого
        # места», и разбирать ради неё всю предыдущую переписку не нужно (при
        # автоматическом старте правило другое, см. _maybe_start_task).
        profile["working"] = task_state.new_task(
            task_type, goal, processed_pairs=len(profile["short_term"]) // 2
        )
        self._save_state()
        return True

    def pause_task(self) -> bool:
        """Ставит задачу на паузу на ЛЮБОМ этапе (флаг, ортогональный этапу). Пока
        задача на паузе, технический вызов не трогает её состояние, а основная
        модель получает указание не продолжать задачу — пользователь может в это
        время спрашивать о чём угодно другом. Возвращает False, если задачи нет или
        она уже на паузе."""
        if self._active_profile is None:
            return False
        task = self._profiles[self._active_profile]["working"]
        if task is None or task.get("paused"):
            return False
        task["paused"] = True
        task["paused_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_state()
        return True

    def resume_task(self) -> bool:
        """Снимает паузу. Сводку «где остановились» вызывающий код строит из уже
        сохранённого состояния (task_state.describe_state) — без обращения к LLM и
        без переспрашивания пользователя."""
        if self._active_profile is None:
            return False
        task = self._profiles[self._active_profile]["working"]
        if task is None or not task.get("paused"):
            return False
        task["paused"] = False
        task["paused_at"] = None
        self._save_state()
        return True

    def set_task_stage(self, stage: str) -> bool:
        """Ручной перевод задачи на другой этап — предохранитель на случай, когда
        автомат ошибся или застрял. В отличие от автоматического перехода, здесь НЕ
        проверяется заполненность обычных обязательных ключей (иначе застрявшую
        задачу нельзя было бы сдвинуть вручную), но соблюдаются граф переходов и
        ГЕЙТ текущего этапа — решение человека предохранитель не заменяет (см.
        task_state.manual_transition_block; причину отказа командный слой берёт
        оттуда же)."""
        if self._active_profile is None:
            return False
        task = self._profiles[self._active_profile]["working"]
        if task is None:
            return False
        moved = task_state.manual_transition(task, stage)
        # Сохраняем и при отказе: manual_transition записала отклонённую попытку в
        # историю задачи, и терять её только потому, что переход не состоялся,
        # неправильно — она и нужна как раз для разбора таких случаев.
        self._save_state()
        return moved

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
        """Очищает три слоя памяти активного профиля (диалог, задачу, факты).

        Инварианты и meta профиля НЕ трогает — это заданный пользователем режим
        работы, а не накопленные данные диалога, по тому же принципу, по которому
        Agent.reset() не сбрасывает выбранную стратегию контекста. Снять инвариант
        можно только точечно (/smart_agent_invariant_remove).

        Возвращает False, если нет активного профиля — вызывающий код подсказывает
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
            f"- {PROFILE_FIELD_LABELS[name]}: {meta[name]}"
            for name in PROFILE_FIELDS
            if meta.get(name)
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

    def _build_context_messages(
        self, tools_message: dict[str, str] | None = None
    ) -> list[dict[str, str]]:
        """Собирает контекст LLM из явно ВКЛЮЧЁННЫХ слоёв (self._enabled_layers)
        активного профиля — в отличие от Agent, здесь нет автоматического выбора
        одной стратегии: пользователь сам решает и что сохранять (см. remember/
        start_task), и какие слои участвуют в конкретном запросе (см.
        set_layer_enabled). Порядок слоёв в сообщении — от самого жёсткого и
        стабильного к самому свежему: инварианты -> профиль -> инструменты ->
        долговременная память -> рабочая задача -> краткосрочный диалог.

        tools_message — готовое сообщение слоя tools (доступен/недоступен, см.
        agents/market_tools.py); его передаёт ask() только когда слой включён и
        сервер настроен, поэтому здесь проверка слоя не повторяется. Инварианты
        остаются первыми и прямо получают приоритет над этим сообщением.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if self._active_profile is None:
            return messages
        profile = self._profiles[self._active_profile]

        # Инварианты идут ПЕРВЫМИ, до профиля персонализации: это ограничения, а
        # профиль — предпочтения подачи, и приоритет между ними проговорён прямо в
        # тексте сообщения (см. invariants.build_context_message).
        if self._enabled_layers[LAYER_INVARIANTS] and profile["invariants"]:
            messages.append(invariants.build_context_message(profile["invariants"]))

        if self._enabled_layers[LAYER_PROFILE]:
            meta_message = self._profile_meta_message(profile["meta"])
            if meta_message:
                messages.append(meta_message)

        if tools_message is not None:
            messages.append(tools_message)

        if self._enabled_layers[LAYER_LONG_TERM] and profile["long_term"]:
            facts_text = "\n".join(f"- {fact['text']}" for fact in profile["long_term"])
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Долговременная память о пользователе (часть сохранена им "
                        "явно командой /smart_agent_remember, часть извлечена из "
                        f"диалога):\n{facts_text}"
                    ),
                }
            )

        if self._enabled_layers[LAYER_WORKING] and profile["working"] is not None:
            # Весь текст про этап/шаг/ожидание/недостающие пункты собирает сам
            # автомат (agents/task_state.py) — там же, где определены условия
            # перехода, чтобы модель и код не расходились в том, чего не хватает.
            messages.append(
                {
                    "role": "system",
                    "content": task_state.build_context_message(profile["working"]),
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

    def _chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Один вызов модели для ОСНОВНОГО ответа. Без tools (None или пусто) запрос
        идентичен прежнему — ключ `tools` в него не попадает; служебные вызовы
        (_call_json_api) идут мимо этого метода и инструментов не получают."""
        kwargs = {"tools": tools} if tools else {}
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_output_tokens,
            timeout=self._timeout,
            **kwargs,
        )

    def _plain_completion(
        self, user_text: str, tools_message: dict[str, str] | None = None
    ) -> market_tools.ToolLoopResult:
        """Обычный ответ одним вызовом модели, без инструментов. tools_message — только
        вариант «данные биржи недоступны» (см. ask()); без него это ровно прежнее
        поведение ask() до появления слоя tools."""
        messages = self._build_context_messages(tools_message)
        self._last_context_messages = list(messages)
        messages.append({"role": "user", "content": user_text})

        response = self._chat(messages)
        usage = self._extract_usage(response)
        return market_tools.ToolLoopResult(
            text=response.choices[0].message.content,
            llm_calls=1,
            prompt_tokens=usage["prompt_tokens"] if usage else None,
            completion_tokens=usage["completion_tokens"] if usage else None,
        )

    def _market_tools_mode(self) -> str:
        """STATUS_OFF — слой выключен пользователем, STATUS_NOT_CONFIGURED — не задан
        MCP_MOEX_DIR (оба — без предупреждений и без запуска процесса; при STATUS_OFF
        ask() дополнительно смотрит, настроен ли сервер, — см. там), STATUS_OK —
        стоит ПОПЫТАТЬСЯ поднять сервер (получится ли — выяснится в ask())."""
        if not self._enabled_layers[LAYER_TOOLS]:
            return market_tools.STATUS_OFF
        if not self._mcp_moex_dir:
            return market_tools.STATUS_NOT_CONFIGURED
        return market_tools.STATUS_OK

    @staticmethod
    def _describe_launch_failure(exc: BaseException) -> str:
        """Короткая причина, почему сервер не запустился, — для /smart_agent_show
        (подробности пишутся в журнал). SDK часто заворачивает исходную ошибку в
        ExceptionGroup (anyio), поэтому разворачиваем её до первой конкретной. Проверка
        по атрибуту `exceptions`, а не по BaseExceptionGroup: тот появился только в
        Python 3.11, а проект заявляет 3.10+ (pyproject.toml)."""
        while getattr(exc, "exceptions", None):
            exc = exc.exceptions[0]
        if isinstance(exc, FileNotFoundError):
            return "не найден uv (проверь PATH)"
        if isinstance(exc, TimeoutError):
            return "сервер не ответил за отведённое время"
        return f"сбой запуска ({type(exc).__name__})"

    async def _run_tool_loop(self, tools, user_text: str) -> market_tools.ToolLoopResult:
        """Цикл вызовов внутри открытой сессии MCP: контекст со слоем tools, затем
        market_tools.run_tool_loop с реальными вызовами модели и инструментов."""
        messages = self._build_context_messages(market_tools.build_context_message())
        self._last_context_messages = list(messages)
        messages.append({"role": "user", "content": user_text})

        def complete(history: list[dict], with_tools: bool):
            return self._chat(history, tools.openai_tools if with_tools else None)

        return await market_tools.run_tool_loop(
            messages, complete, tools.call, self._max_tool_steps
        )

    async def _tool_completion_async(
        self, user_text: str
    ) -> tuple[market_tools.ToolLoopResult | None, str | None]:
        """Ответ с инструментами: (результат, None) либо (None, причина), если сервер
        рыночных данных недоступен — тогда ask() отвечает без инструментов.

        Исключения основного вызова модели (OpenAI SDK) обязаны дойти до командного
        слоя КАК ЕСТЬ — он переводит их в понятные сообщения. Но контексты stdio_client/
        ClientSession построены на anyio, который может завернуть их в ExceptionGroup
        и тем самым обойти эти `except`. Поэтому ошибка тела ловится ВНУТРИ сессии,
        запоминается и возбуждается заново уже после выхода из неё. Сбой самого
        закрытия сессии, когда ответ уже получен, ответ не отменяет — только журнал.
        """
        result: market_tools.ToolLoopResult | None = None
        body_error: Exception | None = None
        opened = False
        no_tools = False
        try:
            async with open_market_tools(
                self._mcp_moex_dir, self._mcp_timeout, self._tool_result_max_chars
            ) as tools:
                opened = True
                if not tools.openai_tools:
                    no_tools = True
                else:
                    try:
                        result = await self._run_tool_loop(tools, user_text)
                    except Exception as exc:  # noqa: BLE001 — возбуждается ниже, см. докстринг
                        body_error = exc
        except Exception as exc:  # noqa: BLE001 — сюда попадают и сбой запуска, и сбой закрытия
            if not opened:
                logger.warning(
                    "Не удалось запустить MCP-сервер рыночных данных — отвечаю без "
                    "инструментов.",
                    exc_info=True,
                )
                return None, self._describe_launch_failure(exc)
            logger.warning(
                "Сбой при завершении сессии MCP-сервера рыночных данных.", exc_info=True
            )

        if body_error is not None:
            raise body_error
        if no_tools:
            return None, "сервер не предоставил инструментов только для чтения"
        if result is None:
            return None, "сессия с сервером оборвалась"
        return result, None

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
        сохранённый диалог, а промежуточные сообщения цикла вызовов инструментов туда
        не попадают вовсе: пишется ровно пара «вопрос — итоговый ответ».

        Слой tools. Если он включён и задан MCP_MOEX_DIR, ответ строится циклом вызовов
        инструментов (agents/market_tools.py) поверх одного процесса MCP-сервера на
        этот вопрос; недоступный сервер не блокирует ответ — вопрос обрабатывается без
        инструментов, с сообщением слоя «данные биржи недоступны» и предупреждением в
        SmartAgentAnswer.warnings. Незаданный MCP_MOEX_DIR — ровно прежнее поведение, без
        запуска процесса и без asyncio. Выключенный слой — тоже без процесса и без
        asyncio, но если сервер настроен, в контексте одно сообщение «данные биржи
        выключены» (market_tools.build_disabled_context_message).

        МЕТОД СИНХРОННЫЙ, но внутри путь с инструментами вызывает asyncio.run() — он
        бросит RuntimeError, если вызвать ask() в потоке с уже работающим циклом
        событий. Единственный вызывающий (agents/smart_agent_command.py) оборачивает
        вызов в asyncio.to_thread(), заодно не блокируя основной цикл на время запроса
        к LLM. Потокобезопасность держится на том, что обновления PTB обрабатываются
        последовательно (concurrent_updates не включён в main.py): состояние агента
        чата не трогают два потока сразу. Если это когда-нибудь изменится — здесь
        понадобится блокировка на чат.
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

        mode = self._market_tools_mode()
        reason: str | None = None
        warnings: list[str] = []
        if mode == market_tools.STATUS_OK:
            outcome, reason = asyncio.run(self._tool_completion_async(user_text))
            if outcome is None:
                mode = market_tools.STATUS_UNAVAILABLE
                outcome = self._plain_completion(
                    user_text, market_tools.build_unavailable_context_message()
                )
                warnings.append(market_tools.UNAVAILABLE_WARNING)
        else:
            # Слой выключен, но сервер настроен: модели нужно сказать, что данные биржи
            # выключены, иначе она выдаёт цену из прежнего ответа за текущую (см.
            # market_tools.build_disabled_context_message). Без настроенного сервера
            # сообщения нет — чат ведёт себя как до появления слоя.
            disabled_message = (
                market_tools.build_disabled_context_message()
                if mode == market_tools.STATUS_OFF and self._mcp_moex_dir
                else None
            )
            outcome = self._plain_completion(user_text, disabled_message)

        if outcome.transport_failure:
            warnings.append(market_tools.PARTIAL_DATA_WARNING)
        if outcome.step_limit_reached:
            warnings.append(market_tools.step_limit_warning(self._max_tool_steps))

        # Статус для /smart_agent_show. Выключенный слой и незаданный каталог здесь не
        # хранятся: get_tools_status() определяет их по текущей настройке, а не по
        # прошлому вопросу.
        if mode in (market_tools.STATUS_OK, market_tools.STATUS_UNAVAILABLE):
            self._tools_status, self._tools_status_reason = mode, reason
        else:
            self._tools_status, self._tools_status_reason = market_tools.STATUS_UNKNOWN, None
        self._last_tool_calls = list(outcome.calls)

        answer = outcome.text or "Модель вернула пустой ответ. Попробуй переформулировать вопрос."

        short_term = self._profiles[self._active_profile]["short_term"]
        short_term.append({"role": "user", "content": user_text})
        short_term.append({"role": "assistant", "content": answer})
        self._save_state()

        return SmartAgentAnswer(
            text=answer,
            request_tokens_approx=request_tokens_approx,
            context_tokens=outcome.prompt_tokens,
            response_tokens=outcome.completion_tokens,
            tool_calls=list(outcome.calls),
            warnings=warnings,
            llm_calls=outcome.llm_calls,
        )

    def get_tools_status(self) -> tuple[str, str | None]:
        """Статус слоя tools для /smart_agent_show: (STATUS_*, причина недоступности).
        Выключенный слой и незаданный MCP_MOEX_DIR определяются по ТЕКУЩЕЙ настройке,
        остальное — по итогу последнего вопроса с момента запуска бота."""
        if not self._enabled_layers[LAYER_TOOLS]:
            return market_tools.STATUS_OFF, None
        if not self._mcp_moex_dir:
            return market_tools.STATUS_NOT_CONFIGURED, None
        return self._tools_status, self._tools_status_reason

    def get_last_tool_calls(self) -> list[market_tools.ToolCallRecord]:
        """Инструменты, вызванные на последнем вопросе (пусто, если вызовов не было)."""
        return list(self._last_tool_calls)

    # --- Конечный автомат задачи: отдельный технический вызов после ответа --- #

    def _call_json_api(self, system_prompt: str, user_content: str, max_tokens: int) -> dict | None:
        """Один служебный вызов LLM, ожидающий строго JSON-объект
        (response_format={"type": "json_object"}, как Agent._call_facts_api и
        сценарий 2 в research/constraints.py — формат проверен на DeepSeek, для
        Kimi отдельно не проверялся). Возвращает разобранный объект, None если API
        вернул пустой content или не объект. Может выбросить json.JSONDecodeError
        (в т.ч. из-за обрезки по max_tokens) или исключение OpenAI SDK — оба
        перехватывает _call_json_with_retry."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            timeout=self._timeout,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None

    def _call_json_with_retry(
        self, system_prompt: str, user_content: str, max_tokens: int, what: str
    ) -> dict | None:
        """Обёртка с одним повтором на удвоенном лимите при обрезанном JSON — тот
        же приём, что в Agent._update_facts(). Любая ошибка гасится здесь и
        возвращает None: ответ пользователю уже отправлен, и служебный вызов не
        имеет права его испортить. Состояние при этом не меняется и не двигается
        счётчик processed_pairs, поэтому пропущенные пары разберёт следующая
        попытка (см. update_task_state)."""
        try:
            return self._call_json_api(system_prompt, user_content, max_tokens)
        except json.JSONDecodeError:
            logger.info(
                "Служебный ответ (%s, память %s) не разобрался как JSON (похоже на "
                "обрезку по max_tokens=%d) — повторяю с удвоенным лимитом.",
                what,
                self._memory_path,
                max_tokens,
            )
            try:
                return self._call_json_api(system_prompt, user_content, max_tokens * 2)
            except Exception:  # noqa: BLE001 — не должно ронять уже отправленный ответ
                logger.warning(
                    "Не удалось обновить %s даже после повтора (память %s).",
                    what,
                    self._memory_path,
                    exc_info=True,
                )
                return None
        except Exception:  # noqa: BLE001 — не должно ронять уже отправленный ответ
            logger.warning(
                "Не удалось обновить %s (память %s) — попробую после следующего вопроса.",
                what,
                self._memory_path,
                exc_info=True,
            )
            return None

    def update_task_state(self) -> TaskStateUpdate | None:
        """Двигает конечный автомат задачи по итогам последних ходов диалога.

        Вызывается командным слоем ПОСЛЕ того, как ответ агента уже отправлен
        пользователю (agents/smart_agent_command.py) — иначе пользователь ждал бы
        два последовательных вызова API, прежде чем увидеть хоть что-то.

        Если активной задачи нет — это укороченный вызов-детектор: не начинает ли
        пользователь новую задачу (см. _maybe_start_task). Если задача на паузе —
        не делается ничего вообще. Слой working выключен (/smart_agent_toggle
        working) — тоже ничего: выключенный слой не участвует ни в контексте, ни в
        записи.

        Возвращает TaskStateUpdate только если произошло СОБЫТИЕ (старт, переход,
        архивирование); обычная строка состояния печатается каждый ход и берётся
        отдельно через get_task_state_line().
        """
        if self._active_profile is None or not self._enabled_layers[LAYER_WORKING]:
            return None

        profile = self._profiles[self._active_profile]
        task = profile["working"]
        if task is None or task_state.is_done(task):
            # Завершённая задача остаётся в слоте до /smart_agent_task_done, но не
            # должна мешать начать следующую: её итог уже перенесён в
            # долговременную память, поэтому новая задача просто заменяет её.
            return self._maybe_start_task(profile)
        if task.get("paused"):
            return None

        short_term = profile["short_term"]
        unprocessed = short_term[2 * task.get("processed_pairs", 0) :]
        if not unprocessed:
            return None

        parsed = self._call_json_with_retry(
            AGENT_TASK_STATE_SYSTEM_PROMPT,
            task_state.build_state_user_content(task, unprocessed),
            self._task_state_max_tokens,
            "состояние задачи",
        )
        if parsed is None:
            return None

        was_done = task_state.is_done(task)
        change = task_state.apply_state_response(task, parsed)
        updated, transition = change.task, change.transition
        updated["processed_pairs"] = len(short_term) // 2

        # Ревизор инвариантов — как только на этапе проверки появился ещё не
        # проверенный результат: раньше проверять нечего, позже пользователь успеет
        # согласиться с вариантом, нарушающим его же ограничения. Привязка к самому
        # результату, а не к факту перехода, закрывает обход через ручной
        # /smart_agent_task_stage validation (см. task_state.needs_invariant_check).
        violations: list[dict] = []
        if task_state.needs_invariant_check(updated):
            # Отметка ставится ДО вызова: ошибка ревизора означает «нарушений нет»
            # (см. _check_invariants) и повторной проверки того же результата не
            # будет — служебная проверка не имеет права стопорить задачу, а
            # инварианты всё равно лежат в контексте основного ответа.
            task_state.mark_invariants_checked(updated)
            violations = self._check_invariants(profile, updated)
            if violations:
                updated, transition = task_state.apply_invariant_violations(
                    updated, violations
                )

        profile["working"] = updated

        archived = False
        if self._enabled_layers[LAYER_LONG_TERM]:
            if task_state.is_done(updated) and not was_done:
                # Перенос завершённой задачи в долговременную память — ОДНИМ
                # компактным фактом (см. task_state.archive_fact про то, почему не
                # по факту на каждый ключ данных).
                self._append_fact(task_state.archive_fact(updated), FACT_SOURCE_AUTO)
                archived = True
            for fact in self._extracted_facts(parsed):
                self._append_fact(fact, FACT_SOURCE_AUTO)

        self._save_state()
        return TaskStateUpdate(
            transition=transition,
            archived=archived,
            violations=violations,
            rejected=change.rejected,
            gate_reset=change.gate_reset,
        )

    def _check_invariants(self, profile: dict, task: dict) -> list[dict]:
        """Вызов-ревизор: проверяет ГОТОВЫЙ результат задачи на инварианты профиля.

        Отдельный вызов, а не ещё одно поле в техническом вызове выше: тот занят
        разбором состояния и делается после каждого хода, а проверка нужна один раз
        — когда результат появился. Диалог ревизору не передаётся, только артефакт
        (см. invariants.build_review_user_content).

        Выключенный слой инвариантов (/smart_agent_toggle invariants) отключает и
        проверку: иначе бот возвращал бы задачу на доработку, ссылаясь на
        ограничения, которых в его контексте в этот момент нет.

        Ошибка вызова или неразобранный JSON гасятся в _call_json_with_retry и
        означают «нарушений не найдено»: ответ пользователю уже отправлен, и
        служебная проверка не имеет права ни уронить его, ни застопорить задачу на
        пустом месте — инварианты при этом всё равно лежат в контексте основного
        ответа.
        """
        items = profile["invariants"]
        if not items or not self._enabled_layers[LAYER_INVARIANTS]:
            return []

        result = task_state.result_payload(task)
        if not result:
            return []

        parsed = self._call_json_with_retry(
            AGENT_INVARIANTS_SYSTEM_PROMPT,
            invariants.build_review_user_content(
                items, task_state.scenario_of(task).label, task.get("goal", ""), result
            ),
            self._invariants_max_tokens,
            "проверку инвариантов",
        )
        if parsed is None:
            return []
        return invariants.parse_review_response(parsed, items)

    def _maybe_start_task(self, profile: dict) -> TaskStateUpdate | None:
        """Укороченный вызов на случай «активной задачи нет»: решает только, начал
        ли пользователь одну из задач реестра. Отдельный дешёвый промпт и лимит
        токенов нужны потому, что этот вызов случается после КАЖДОГО обычного
        вопроса, в том числе разового («что такое ETF»).

        Автоматический старт возможен только при отсутствии активной задачи —
        подменять незавершённую задачу новой автомат не должен, для смены сценария
        есть явная команда /smart_agent_task_start.
        """
        short_term = profile["short_term"]
        if len(short_term) < 2:
            return None

        parsed = self._call_json_with_retry(
            AGENT_TASK_START_SYSTEM_PROMPT,
            task_state.build_start_user_content(short_term[-2:]),
            self._task_start_max_tokens,
            "старт задачи",
        )
        if parsed is None:
            return None

        started = task_state.parse_start_response(parsed)
        if started is None:
            return None

        task_type, goal = started
        profile["working"] = task_state.new_task(
            task_type, goal, processed_pairs=max(0, len(short_term) // 2 - 1)
        )
        self._save_state()
        return TaskStateUpdate(started=True)

    @staticmethod
    def _extracted_facts(parsed: dict) -> list[str]:
        """Факты, предложенные техническим вызовом. Ограничены и по количеству за
        ход, и по длине: долговременная память мала (AGENT_MEMORY_LONG_TERM_MAX_FACTS),
        и без потолка одна разговорчивая итерация вытеснила бы из неё всё
        остальное."""
        raw = parsed.get("new_facts")
        if not isinstance(raw, list):
            return []
        facts = []
        for item in raw[:_MAX_AUTO_FACTS_PER_TURN]:
            text = str(item).strip()
            if text:
                facts.append(text[:_MAX_AUTO_FACT_CHARS])
        return facts
