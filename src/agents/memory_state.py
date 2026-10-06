"""Состояние памяти SmartAgent на диске: константы слоёв/полей/происхождения фактов и ЧИСТЫЕ
функции разбора и сборки файла AGENT_MEMORY_DIR/<chat_id>.json (без файловой системы, LLM и
сети — покрываются tests/test_memory_state.py). Чтение файла, логирование и запись остаются в
SmartAgent (agents/smart_agent.py), который реэкспортирует эти имена."""

from __future__ import annotations

import json

from . import invariants, task_state

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
LAYER_TASK_AUTOSTART = "task_autostart"
# task_autostart — НЕ слой контекста (в отличие от шести выше он ничего не добавляет и
# не убирает из сообщений LLM), а поведенческий флаг автомата рабочей задачи: включает/
# выключает только автоматическое ОБНАРУЖЕНИЕ новой задачи (_maybe_start_task). Лежит в
# ALL_LAYERS/enabled_layers ради готового механизма хранения и переключения
# (/smart_agent_toggle) — см. design.md изменения add-smart-agent-task-autostart-toggle,
# решение 1. Добавлен в конец по тому же принципу, что и tools: не менять порядок
# прежних ключей. Старые файлы памяти без этого ключа читаются как «включён» (см.
# _load_state) — миграция не нужна.
LAYER_RAG = "rag"
# Слой rag — справочные материалы из индекса rag/ в основном ответе (design.md изменения
# add-smart-agent-rag, решение 5). Настоящий СЛОЙ контекста (добавляет сообщения),
# в отличие от task_autostart. Добавлен в конец по тому же принципу, что tools и
# task_autostart: порядок прежних ключей не меняется, а место сообщения слоя в
# контексте (после tools, до долговременной памяти) определяет _build_context_messages.
# Старые файлы памяти без ключа "rag" читаются как «включён» (см. _load_state).
ALL_LAYERS = (
    LAYER_PROFILE,
    LAYER_INVARIANTS,
    LAYER_SHORT_TERM,
    LAYER_WORKING,
    LAYER_LONG_TERM,
    LAYER_TOOLS,
    LAYER_TASK_AUTOSTART,
    LAYER_RAG,
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


def empty_profile() -> dict:
    return {
        "meta": {name: "" for name in PROFILE_FIELDS},
        "invariants": [],
        "short_term": [],
        "working": None,
        "long_term": [],
    }


def default_enabled_layers() -> dict[str, bool]:
    return {layer: True for layer in ALL_LAYERS}


def sanitize_fact(item) -> dict | None:
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


def sanitize_profile(entry: dict) -> dict:
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
    long_term = [fact for fact in (sanitize_fact(item) for item in raw_long_term) if fact]

    return {
        "meta": meta,
        "invariants": invariant_items,
        "short_term": short_term,
        "working": working,
        "long_term": long_term,
    }


def parse_state(raw: str) -> tuple[dict[str, dict], str | None, dict[str, bool], bool]:
    """Разбор текста файла памяти → (профили, активный профиль, слои, повреждён ли JSON).
    Любой непригодный вход даёт пустую память со слоями по умолчанию; признак повреждения
    нужен вызывающему только для записи в журнал."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}, None, default_enabled_layers(), True

    if not isinstance(data, dict):
        return {}, None, default_enabled_layers(), False

    enabled_layers = default_enabled_layers()
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
            name: sanitize_profile(entry)
            for name, entry in raw_profiles.items()
            if isinstance(entry, dict)
        }
        active_profile = data.get("active_profile")
        if not isinstance(active_profile, str) or active_profile not in profiles:
            active_profile = None
        return profiles, active_profile, enabled_layers, False

    # Старый плоский формат (до появления профилей) — оборачиваем то, что уже
    # было накоплено, в один профиль и делаем его сразу активным, чтобы уже
    # работающие чаты не прерывались выбором профиля на пустом месте (см.
    # _MIGRATED_PROFILE_NAME выше).
    if not (data.get("short_term") or data.get("working") or data.get("long_term")):
        return {}, None, enabled_layers, False

    migrated = sanitize_profile(data)
    return {_MIGRATED_PROFILE_NAME: migrated}, _MIGRATED_PROFILE_NAME, enabled_layers, False


def dump_state(
    profiles: dict[str, dict], active_profile: str | None, enabled_layers: dict[str, bool]
) -> str:
    return json.dumps(
        {
            "active_profile": active_profile,
            "enabled_layers": enabled_layers,
            "profiles": profiles,
        },
        ensure_ascii=False,
        indent=2,
    )

