"""Конечный автомат рабочей задачи smart-агента (/smart_agent).

Отделён от agents/smart_agent.py по тому же принципу, по которому
agents/context_strategies.py отделён от agents/agent.py: здесь живут ПРАВИЛА
автомата (реестр сценариев, этапы, условия перехода, разбор ответа модели и
тексты про состояние), а владение состоянием, его сохранение на диск и сами
вызовы LLM-клиента остаются на стороне SmartAgent. Этот модуль ничего не знает
ни про Telegram, ни про openai — только про словарь задачи.

Автомат один на все сценарии: planning -> execution -> validation -> done, с
откатом validation -> execution, если пользователь просит правки. Сценарии
(SCENARIOS) различаются только словарём ключей data и текстами инструкций, так
что новый сценарий добавляется одной записью в реестр, без изменений в самом
автомате.

Ключевое отличие от «модель сама решает, когда двигаться дальше»: условие
перехода — это СПИСОК ОБЯЗАТЕЛЬНЫХ КЛЮЧЕЙ data у текущего этапа, и его проверяет
код (see missing_keys/_validated_transition), а не модель. Тот же список
недостающих ключей уходит и в контекст основного ответа (build_context_message),
поэтому вежливый отказ на преждевременный переход всегда совпадает с реальным
состоянием автомата, а не расходится с ним.

Словарь ключей фиксирован: из ответа модели принимаются только ключи своего
сценария (см. normalize_data_updates), иначе модель на каждом ходу изобретала бы
новое имя для того же параметра («горизонт»/«horizon»/«time_horizon»), условие
перехода никогда бы не выполнилось и задача зависла бы навсегда. На РУЧНУЮ
запись (/smart_agent_task_set) это ограничение не распространяется — там ключ
задаёт человек, и произвольные ключи сохраняются как раньше.

Денежные суммы не собираются ни на одном этапе ни в одном сценарии: состав и
структура портфеля описываются только долями в процентах (см. «Правила
предметной области» в CLAUDE.md про запрет на хранение сумм и реквизитов).

Инварианты (agents/invariants.py) входят в автомат одной точкой: результат
рабочего этапа проверяется на них отдельным вызовом-ревизором, и при нарушении
задача откатывается назад тем же _roll_back, что и по просьбе пользователя
(см. apply_invariant_violations). Сам список инвариантов автомату не
принадлежит — он живёт в слое памяти SmartAgent, здесь хранятся только
претензии ревизора к текущему результату (INVARIANT_VIOLATIONS_KEY).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime

from . import invariants

logger = logging.getLogger(__name__)

STAGE_PLANNING = "planning"
STAGE_EXECUTION = "execution"
STAGE_VALIDATION = "validation"
STAGE_DONE = "done"

# Режим этапа — подсказка модели о характере работы, а не отдельная ветка кода:
# диалоговый этап собирает информацию по одному вопросу за раз, автоматический
# выдаёт результат целиком в одном ответе.
MODE_DIALOG = "dialog"
MODE_AUTO = "auto"
MODE_TERMINAL = "terminal"

# Единственный ключ с закрытым списком значений — от него зависит, куда уходит
# переход с этапа проверки (accepted -> done, changes_requested -> execution),
# поэтому произвольный текст в нём не принимается (см. normalize_data_updates).
VERDICT_KEY = "verdict"
VERDICT_ACCEPTED = "accepted"
VERDICT_CHANGES_REQUESTED = "changes_requested"

# Претензии ревизора инвариантов к результату задачи (agents/invariants.py) —
# список [{"index", "text", "why"}]. Живёт в задаче ОТДЕЛЬНО от data: data
# фильтруется по словарю сценария (normalize_data_updates), а это поле заполняет не
# модель, а код по ответу ревизора (см. apply_invariant_violations).
INVARIANT_VIOLATIONS_KEY = "invariant_violations"

# Человекочитаемые подписи ключей data — для «не хватает: горизонт инвестирования,
# ограничения» в чате и в /smart_agent_task_show. Единственное место с этими
# формулировками.
KEY_LABELS = {
    "goal_type": "цель накоплений",
    "horizon": "горизонт инвестирования",
    "risk": "отношение к риску",
    "constraints": "ограничения и исключения",
    "preferred_assets": "предпочитаемые классы активов",
    "allocation": "структура портфеля в процентах",
    "asset": "актив или идея для разбора",
    "role_in_portfolio": "роль в портфеле",
    "analysis": "разбор актива",
    "current_allocation": "текущий состав портфеля в долях",
    "concern": "что беспокоит в портфеле",
    "proposed_changes": "предлагаемые изменения долей",
    VERDICT_KEY: "решение пользователя по результату",
}

_NO_SUMS_RULE = (
    "Никогда не спрашивай и не используй денежные суммы, номера счетов и карт — "
    "только качественные параметры и доли в процентах."
)


@dataclass(frozen=True)
class TaskStage:
    """Один этап автомата. required_keys — и есть условие перехода дальше:
    пока хотя бы один из них не заполнен в data, переход отклоняется кодом."""

    name: str
    label: str
    mode: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...]
    next_stages: tuple[str, ...]
    instruction: str


@dataclass(frozen=True)
class TaskScenario:
    key: str
    label: str
    stages: tuple[TaskStage, ...]

    def stage(self, name: str) -> TaskStage | None:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def stage_index(self, name: str) -> int:
        for index, stage in enumerate(self.stages):
            if stage.name == name:
                return index
        return -1

    def known_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for stage in self.stages:
            keys.extend(stage.required_keys)
            keys.extend(stage.optional_keys)
        return tuple(keys)


def _validation_stage(result_name: str) -> TaskStage:
    """Этап проверки одинаков во всех сценариях — отличается только тем, как
    называется проверяемый результат."""
    return TaskStage(
        name=STAGE_VALIDATION,
        label="проверка",
        mode=MODE_DIALOG,
        required_keys=(VERDICT_KEY,),
        optional_keys=(),
        next_stages=(STAGE_EXECUTION, STAGE_DONE),
        instruction=(
            f"Спроси у пользователя, устраивает ли его {result_name}. Если он просит "
            "правки — уточни, какие именно, и учти их при пересчёте. Если он всем "
            "доволен — задача завершена, подведи короткий итог."
        ),
    )


_DONE_STAGE = TaskStage(
    name=STAGE_DONE,
    label="завершено",
    mode=MODE_TERMINAL,
    required_keys=(),
    optional_keys=(),
    next_stages=(),
    instruction=(
        "Задача завершена. Новых шагов по ней не предлагай — если пользователь хочет "
        "продолжить работу, ему нужно начать новую задачу."
    ),
)

SCENARIOS: dict[str, TaskScenario] = {
    "portfolio": TaskScenario(
        key="portfolio",
        label="Составление портфеля",
        stages=(
            TaskStage(
                name=STAGE_PLANNING,
                label="планирование",
                mode=MODE_DIALOG,
                required_keys=("goal_type", "horizon", "risk", "constraints"),
                optional_keys=("preferred_assets",),
                next_stages=(STAGE_EXECUTION,),
                instruction=(
                    "Собери недостающие пункты, задавая по одному уточняющему вопросу "
                    "за раз. Не предлагай структуру портфеля, пока не собраны все "
                    f"пункты. {_NO_SUMS_RULE}"
                ),
            ),
            TaskStage(
                name=STAGE_EXECUTION,
                label="формирование портфеля",
                mode=MODE_AUTO,
                required_keys=("allocation",),
                optional_keys=(),
                # Назад в планирование — если пользователь захотел поправить
                # исходные параметры уже после того, как они были собраны.
                next_stages=(STAGE_VALIDATION, STAGE_PLANNING),
                instruction=(
                    "Предложи структуру портфеля в ПРОЦЕНТАХ по классам активов "
                    "(сумма долей — 100%), с коротким обоснованием по каждому классу "
                    "исходя из собранных цели, горизонта, отношения к риску и "
                    "ограничений. Обязательно напомни о рыночной неопределённости и о "
                    f"том, что это не индивидуальная рекомендация. {_NO_SUMS_RULE}"
                ),
            ),
            _validation_stage("предложенная структура портфеля"),
            _DONE_STAGE,
        ),
    ),
    "asset": TaskScenario(
        key="asset",
        label="Разбор актива или идеи",
        stages=(
            TaskStage(
                name=STAGE_PLANNING,
                label="планирование",
                mode=MODE_DIALOG,
                required_keys=("asset", "horizon", "role_in_portfolio"),
                optional_keys=("constraints",),
                next_stages=(STAGE_EXECUTION,),
                instruction=(
                    "Уточни, какой именно актив или идею разбираем, на каком горизонте "
                    "и какую роль он должен играть в портфеле — по одному вопросу за "
                    f"раз. {_NO_SUMS_RULE}"
                ),
            ),
            TaskStage(
                name=STAGE_EXECUTION,
                label="разбор",
                mode=MODE_AUTO,
                required_keys=("analysis",),
                optional_keys=(),
                # Назад в планирование — если пользователь захотел поправить
                # исходные параметры уже после того, как они были собраны.
                next_stages=(STAGE_VALIDATION, STAGE_PLANNING),
                instruction=(
                    "Дай разбор: что это за инструмент, какие у него риски, насколько "
                    "он уместен на заявленном горизонте и в заявленной роли, какие "
                    "есть альтернативы. Без прогнозов доходности и без обещаний "
                    f"результата. {_NO_SUMS_RULE}"
                ),
            ),
            _validation_stage("разбор"),
            _DONE_STAGE,
        ),
    ),
    "review": TaskScenario(
        key="review",
        label="Ревизия портфеля",
        stages=(
            TaskStage(
                name=STAGE_PLANNING,
                label="планирование",
                mode=MODE_DIALOG,
                required_keys=("current_allocation", "concern", "constraints"),
                optional_keys=("horizon",),
                next_stages=(STAGE_EXECUTION,),
                instruction=(
                    "Уточни текущий состав портфеля ТОЛЬКО в долях или процентах, что "
                    "именно беспокоит пользователя и какие есть ограничения — по "
                    "одному вопросу за раз. Если пользователь называет суммы, не "
                    "сохраняй их и попроси перевести состав в проценты. "
                    f"{_NO_SUMS_RULE}"
                ),
            ),
            TaskStage(
                name=STAGE_EXECUTION,
                label="предложение изменений",
                mode=MODE_AUTO,
                required_keys=("proposed_changes",),
                optional_keys=(),
                # Назад в планирование — если пользователь захотел поправить
                # исходные параметры уже после того, как они были собраны.
                next_stages=(STAGE_VALIDATION, STAGE_PLANNING),
                instruction=(
                    "Предложи изменения долей: что увеличить, что уменьшить и почему, "
                    "с указанием рисков каждого изменения. Только в процентах. "
                    "Обязательно напомни о рыночной неопределённости и о том, что это "
                    f"не индивидуальная рекомендация. {_NO_SUMS_RULE}"
                ),
            ),
            _validation_stage("предложенные изменения"),
            _DONE_STAGE,
        ),
    ),
}

# Сценарий, в который мигрируют задачи, сохранённые ДО появления автомата (в них
# был только "status": "active" без типа и этапа) — см. sanitize_task.
DEFAULT_SCENARIO = "portfolio"

SCENARIO_LABELS = {key: scenario.label for key, scenario in SCENARIOS.items()}


# --------------------------------------------------------------------------- #
# Создание и загрузка задачи
# --------------------------------------------------------------------------- #


def new_task(task_type: str, goal: str, processed_pairs: int = 0) -> dict:
    """Новая задача на первом этапе сценария. step/expected_action заполнит первый
    же технический вызов после ответа агента (см. apply_state_response).

    processed_pairs — сколько пар уже накопленного диалога технический вызов
    считает УЖЕ учтёнными и разбирать не будет. При ручном старте это весь диалог
    до команды (задача начинается «с этого места»), при автоматическом — весь,
    кроме последней пары: именно в ней пользователь и заявил задачу, и там же
    обычно лежат первые её параметры.
    """
    return {
        "task_type": task_type,
        "goal": goal,
        "stage": STAGE_PLANNING,
        "step": "",
        "expected_action": "",
        "paused": False,
        "paused_at": None,
        "awaiting_correction": False,
        INVARIANT_VIOLATIONS_KEY: [],
        "data": {},
        "processed_pairs": max(0, processed_pairs),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def sanitize_task(raw) -> dict | None:
    """Приводит задачу из JSON-файла памяти к актуальной схеме. Задачи старого
    формата ({"goal", "status", "data", "created_at"}) не выбрасываются, а
    мигрируют в сценарий DEFAULT_SCENARIO на этапе planning — этап по старому
    "status" восстановить неоткуда, а planning безопасен: автомат доберёт
    недостающие ключи и сам уйдёт дальше.

    Ключи data, которых нет в словаре сценария, СОХРАНЯЮТСЯ — их мог положить
    пользователь вручную через /smart_agent_task_set (фильтрация по словарю
    применяется только к тому, что возвращает модель, см. normalize_data_updates).
    """
    if not isinstance(raw, dict):
        return None

    task_type = raw.get("task_type")
    if task_type not in SCENARIOS:
        task_type = DEFAULT_SCENARIO
    scenario = SCENARIOS[task_type]

    stage = raw.get("stage")
    if scenario.stage(stage) is None:
        stage = STAGE_PLANNING

    raw_data = raw.get("data")
    data = (
        {str(key): _as_text(value) for key, value in raw_data.items()}
        if isinstance(raw_data, dict)
        else {}
    )

    processed_pairs = raw.get("processed_pairs")
    if not isinstance(processed_pairs, int) or processed_pairs < 0:
        processed_pairs = 0

    return {
        "task_type": task_type,
        "goal": _as_text(raw.get("goal", "")),
        "stage": stage,
        "step": _as_text(raw.get("step", "")),
        "expected_action": _as_text(raw.get("expected_action", "")),
        "paused": bool(raw.get("paused", False)),
        "paused_at": raw.get("paused_at") if isinstance(raw.get("paused_at"), str) else None,
        "awaiting_correction": bool(raw.get("awaiting_correction", False)),
        INVARIANT_VIOLATIONS_KEY: _sanitize_violations(raw.get(INVARIANT_VIOLATIONS_KEY)),
        "data": data,
        "processed_pairs": processed_pairs,
        "created_at": _as_text(raw.get("created_at", "")),
    }


def _sanitize_violations(raw) -> list[dict]:
    """Претензии ревизора из файла памяти. Задачи, сохранённые ДО появления
    инвариантов, поля не содержат — получается пустой список, отдельной миграции не
    нужно (тот же приём, что invariants.sanitize_list)."""
    if not isinstance(raw, list):
        return []
    violations = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        violations.append(
            {
                "index": index,
                "text": _as_text(item.get("text", "")),
                "why": _as_text(item.get("why", "")),
            }
        )
    return violations


def _as_text(value) -> str:
    """Значения data всегда храним строками: модель может вернуть список долей или
    вложенный объект, а в контекст и в /smart_agent_task_show всё равно уходит
    текст."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# --------------------------------------------------------------------------- #
# Состояние автомата: условия перехода
# --------------------------------------------------------------------------- #


def scenario_of(task: dict) -> TaskScenario:
    return SCENARIOS.get(task.get("task_type"), SCENARIOS[DEFAULT_SCENARIO])


def current_stage(task: dict) -> TaskStage:
    scenario = scenario_of(task)
    return scenario.stage(task.get("stage")) or scenario.stages[0]


def missing_keys(task: dict) -> list[str]:
    """Обязательные ключи текущего этапа, которых ещё нет в data — то самое
    условие перехода, которое проверяет код и о котором основная модель
    рассказывает пользователю, если он просит перейти дальше раньше времени."""
    data = task.get("data", {})
    return [key for key in current_stage(task).required_keys if not str(data.get(key, "")).strip()]


def is_done(task: dict) -> bool:
    return task.get("stage") == STAGE_DONE


def is_validation(task: dict) -> bool:
    return task.get("stage") == STAGE_VALIDATION


def invariant_violations(task: dict) -> list[dict]:
    raw = task.get(INVARIANT_VIOLATIONS_KEY)
    return list(raw) if isinstance(raw, list) else []


def result_payload(task: dict) -> dict[str, str]:
    """Готовый результат задачи — то, что заполнили автоматические этапы (структура
    портфеля, разбор, предлагаемые изменения). Именно он, а не весь диалог, уходит
    ревизору инвариантов: какие ключи считаются результатом, знает реестр сценариев,
    а не SmartAgent (см. build_review_user_content в agents/invariants.py)."""
    data = task.get("data", {})
    return {
        key_label(key): data[key]
        for stage in scenario_of(task).stages
        if stage.mode == MODE_AUTO
        for key in stage.required_keys
        if data.get(key)
    }


def key_label(key: str) -> str:
    return KEY_LABELS.get(key, key)


def manual_transition(task: dict, target: str) -> bool:
    """Ручной перевод этапа (/smart_agent_task_stage) прямо в переданной задаче.

    Проверяет только граф переходов, но НЕ заполненность обязательных ключей —
    это предохранитель на случай, когда автомат ошибся, и требовать от него
    выполненных условий бессмысленно. Откат назад идёт через тот же _roll_back,
    что и автоматический, иначе переход вперёд по уже собранным данным тут же
    отменил бы ручное решение.
    """
    scenario = scenario_of(task)
    if target not in allowed_manual_stages(task):
        return False
    if scenario.stage_index(target) < scenario.stage_index(task["stage"]):
        _roll_back(scenario, task, target)
    else:
        task["stage"] = target
    return True


def manual_stage_options(task: dict) -> list[tuple[str, str]]:
    """Пары (имя этапа, подпись) для подсказки в /smart_agent_task_stage."""
    scenario = scenario_of(task)
    return [
        (name, (scenario.stage(name) or current_stage(task)).label)
        for name in allowed_manual_stages(task)
    ]


def short_summary(task: dict) -> str:
    """Одна строка о задаче для сводного /smart_agent_show."""
    scenario = scenario_of(task)
    stage = current_stage(task)
    paused = " · ⏸ на паузе" if task.get("paused") else ""
    return (
        f"{scenario.label} — «{task.get('goal') or '—'}», этап: {stage.label}, "
        f"данных: {len([v for v in task.get('data', {}).values() if v])}{paused}"
    )


def allowed_manual_stages(task: dict) -> tuple[str, ...]:
    """Этапы, на которые можно перевести задачу вручную (/smart_agent_task_stage).
    Это предохранитель на случай, когда автомат ошибся или застрял, поэтому здесь
    проверяется только сам граф переходов, но НЕ заполненность обязательных
    ключей — иначе застрявшую задачу нельзя было бы сдвинуть вручную."""
    return current_stage(task).next_stages


# --------------------------------------------------------------------------- #
# Разбор ответа технического вызова
# --------------------------------------------------------------------------- #


def normalize_data_updates(scenario: TaskScenario, updates: dict) -> dict[str, str]:
    """Оставляет только ключи из словаря сценария (см. докстринг модуля про то,
    почему это критично) и приводит значения к строкам. verdict дополнительно
    ограничен двумя допустимыми значениями: от него зависит направление перехода
    с этапа проверки, и произвольный текст вида «пользователь вроде согласен»
    сделал бы переход неоднозначным."""
    known = set(scenario.known_keys())
    normalized: dict[str, str] = {}
    for key, value in updates.items():
        key = str(key)
        if key not in known:
            logger.debug("Ключ %r не входит в словарь сценария %s — пропускаю.", key, scenario.key)
            continue
        text = _as_text(value).strip()
        if not text:
            continue
        if key == VERDICT_KEY:
            text = text.lower()
            if text not in (VERDICT_ACCEPTED, VERDICT_CHANGES_REQUESTED):
                continue
        normalized[key] = text
    return normalized


def _forward_target(scenario: TaskScenario, task: dict) -> str | None:
    """Следующий этап ВПЕРЁД, если условия текущего выполнены.

    Считается по данным, а НЕ по полю "stage" из ответа модели: иначе возникает
    рассинхрон, когда все обязательные ключи собраны (и агент уже сказал
    пользователю «готово»), а модель в служебном ответе оставила прежний этап —
    задача в этом случае навсегда зависала на проверке. Направление перехода в
    этом автомате всегда однозначно определяется данными, так что спрашивать его
    у модели просто незачем.
    """
    stage = current_stage(task)
    if missing_keys(task):
        return None
    index = scenario.stage_index(stage.name)
    forward = [name for name in stage.next_stages if scenario.stage_index(name) > index]
    if not forward:
        return None
    # С проверки вперёд (в done) уходим только при согласии пользователя; просьба
    # о правках — это откат назад, см. _backward_target.
    if stage.name == STAGE_VALIDATION and task.get("data", {}).get(VERDICT_KEY) != VERDICT_ACCEPTED:
        return None
    return forward[0]


def _backward_target(scenario: TaskScenario, task: dict, requested) -> str | None:
    """Откат назад — единственный вид перехода, который НЕ выводится из данных
    автоматически: «вернуться и переделать» может решить только пользователь. С
    этапа проверки направление задаёт вердикт (правки -> назад к работе), с
    остальных этапов — поле "stage" из ответа модели, если она указала более
    ранний допустимый этап."""
    stage = current_stage(task)
    if stage.name == STAGE_VALIDATION:
        if task.get("data", {}).get(VERDICT_KEY) == VERDICT_CHANGES_REQUESTED:
            return STAGE_EXECUTION
        return None
    if not isinstance(requested, str) or requested not in stage.next_stages:
        return None
    index = scenario.stage_index(stage.name)
    return requested if 0 <= scenario.stage_index(requested) < index else None


def _roll_back(scenario: TaskScenario, task: dict, target: str) -> None:
    """Выполняет откат назад: снимает обязательные ключи этапов ПОСЛЕ целевого и
    ставит флаг ожидания правки.

    Ключи самого целевого этапа сохраняются — возвращаются, чтобы поправить один
    параметр, а не чтобы заново отвечать на все вопросы этапа. Но тогда его
    условия сразу оказываются выполненными, и автоматический переход вперёд
    немедленно отменил бы откат — поэтому до тех пор, пока не придёт правка хотя
    бы одного ключа текущего этапа, движение вперёд заблокировано флагом
    awaiting_correction (см. apply_state_response).
    """
    start = scenario.stage_index(target)
    if start < 0:
        return
    for stage in scenario.stages[start + 1 :]:
        for key in stage.required_keys:
            task["data"].pop(key, None)
    task["stage"] = target
    task["awaiting_correction"] = True


def _last_auto_stage(scenario: TaskScenario, before: str) -> str | None:
    """Последний РАБОЧИЙ (MODE_AUTO) этап до указанного — туда возвращается задача,
    результат которой не прошёл проверку на инварианты. Считается по реестру, а не
    хардкодом STAGE_EXECUTION, чтобы сценарий с несколькими рабочими этапами не
    пришлось чинить отдельно."""
    index = scenario.stage_index(before)
    if index < 0:
        return None
    auto = [stage.name for stage in scenario.stages[:index] if stage.mode == MODE_AUTO]
    return auto[-1] if auto else None


def apply_invariant_violations(task: dict, violations: list[dict]) -> tuple[dict, str | None]:
    """Возвращает задачу на доработку, потому что её результат нарушил инварианты.

    Решение принимает КОД по ответу ревизора (agents/invariants.py), а не модель в
    свободном тексте — в этом и смысл проверки: состояние автомата реально уезжает
    назад, а не просто сопровождается укоризненной фразой в чате.

    Механика — тот же _roll_back, что и у «пользователь просит правки»: ключ с
    результатом СОХРАНЯЕТСЯ (агент правит прежний вариант, а не сочиняет с нуля), а
    флаг awaiting_correction блокирует движение вперёд до тех пор, пока не будет
    записан новый результат. Без флага автоматический переход вперёд отменил бы
    откат в тот же миг, ведь обязательный ключ этапа уже заполнен.
    """
    updated = copy.deepcopy(task)
    if not violations:
        return updated, None

    scenario = scenario_of(updated)
    stage_before = current_stage(updated)
    target = _last_auto_stage(scenario, stage_before.name)
    if target is None:
        # Нарушение на этапе, до которого рабочего этапа не было (в текущих сценариях
        # не случается) — состояние не трогаем, но претензии сохраняем: они уйдут в
        # контекст следующего ответа.
        updated[INVARIANT_VIOLATIONS_KEY] = list(violations)
        return updated, None

    _roll_back(scenario, updated, target)
    updated[INVARIANT_VIOLATIONS_KEY] = list(violations)
    return updated, f"{stage_before.label} → {current_stage(updated).label}"


def apply_state_response(task: dict, parsed: dict) -> tuple[dict, str | None]:
    """Применяет разобранный JSON технического вызова к задаче.

    Порядок: сначала данные и шаг, потом переходы. ВПЕРЁД задача двигается сама,
    как только собраны обязательные ключи этапа (_forward_target) — поле "stage"
    из ответа модели на это не влияет, иначе задача зависает на этапе, условия
    которого уже выполнены. НАЗАД — только по явному сигналу (вердикт «нужны
    правки» или более ранний этап в "stage"), и такой откат ждёт правку, прежде
    чем снова разрешить движение вперёд (см. _roll_back).

    Возвращает новую задачу и, если этап сменился, короткую подпись перехода для
    чата («планирование → формирование портфеля»).
    """
    updated = copy.deepcopy(task)
    scenario = scenario_of(updated)
    stage_before = current_stage(updated)

    raw_updates = parsed.get("data_updates")
    updates = (
        normalize_data_updates(scenario, raw_updates) if isinstance(raw_updates, dict) else {}
    )
    updated["data"].update(updates)

    for field in ("step", "expected_action"):
        value = parsed.get(field)
        if isinstance(value, str) and value.strip():
            updated[field] = value.strip()

    backward = _backward_target(scenario, updated, parsed.get("stage"))
    if backward is not None:
        _roll_back(scenario, updated, backward)
    else:
        # Правка после отката пришла — можно снова двигаться вперёд. Заодно
        # снимаются претензии ревизора инвариантов: они относились к ПРЕЖНЕМУ
        # результату, а ключ этапа только что переписан (см.
        # apply_invariant_violations).
        if any(key in current_stage(updated).required_keys for key in updates):
            updated["awaiting_correction"] = False
            updated[INVARIANT_VIOLATIONS_KEY] = []

        if not updated.get("awaiting_correction"):
            # Вперёд — столько шагов, сколько позволяют собранные данные: обычно
            # ноль или один, но если модель за один ход заполнила ключи сразу двух
            # этапов, застревать на полпути неправильно. Ограничение по числу
            # этапов — защита от зацикливания на случай кривого реестра сценария.
            for _ in range(len(scenario.stages)):
                forward = _forward_target(scenario, updated)
                if forward is None:
                    break
                updated["stage"] = forward

    if updated["stage"] == stage_before.name:
        if missing_keys(updated):
            logger.debug(
                "Задача (%s) остаётся на этапе %s, не хватает: %s.",
                scenario.key,
                stage_before.name,
                ", ".join(missing_keys(updated)),
            )
        return updated, None

    return updated, f"{stage_before.label} → {current_stage(updated).label}"


def parse_start_response(parsed: dict) -> tuple[str, str] | None:
    """Разбирает ответ вызова-детектора старта задачи. Возвращает (тип, цель) или
    None, если модель не увидела начала задачи либо назвала неизвестный сценарий."""
    task_type = parsed.get("task_type")
    if not isinstance(task_type, str) or task_type not in SCENARIOS:
        return None
    goal = parsed.get("goal")
    goal = goal.strip() if isinstance(goal, str) else ""
    return task_type, goal or SCENARIOS[task_type].label


# --------------------------------------------------------------------------- #
# Тексты: контекст для LLM, строка состояния и сводка для пользователя
# --------------------------------------------------------------------------- #


def render_exchanges(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{'Пользователь' if m.get('role') == 'user' else 'Агент'}: {m.get('content', '')}"
        for m in messages
    )


def build_context_message(task: dict) -> str:
    """Системное сообщение об активной задаче для ОСНОВНОГО вызова (не для
    технического). Содержит и инструкцию этапа, и список недостающих пунктов —
    поэтому на просьбу «переходи уже к портфелю» модель отвечает конкретикой
    («не хватает горизонта и ограничений»), а не общими словами, и это всегда
    совпадает с тем, что проверит код при переходе."""
    scenario = scenario_of(task)
    stage = current_stage(task)
    position = scenario.stage_index(stage.name) + 1
    lines = [
        f"Активная рабочая задача пользователя: {scenario.label} — «{task.get('goal', '')}».",
        f"Этап {position} из {len(scenario.stages)}: {stage.label}.",
        f"Что делать на этом этапе: {stage.instruction}",
    ]

    if task.get("paused"):
        lines.append(
            "ЗАДАЧА НА ПАУЗЕ: не продолжай её и не задавай вопросов по ней, пока "
            "пользователь не снимет паузу командой /smart_agent_task_resume. На другие "
            "вопросы отвечай как обычно."
        )
        return "\n".join(lines)

    if task.get("step"):
        lines.append(f"Текущий шаг: {task['step']}")
    if task.get("expected_action"):
        lines.append(f"Ожидаемое действие: {task['expected_action']}")

    collected = [
        f"- {key_label(key)}: {value}" for key, value in task.get("data", {}).items() if value
    ]
    if collected:
        lines.append("Уже собрано:\n" + "\n".join(collected))

    violations = invariant_violations(task)
    if violations:
        # Ровно тот же текст, что ушёл пользователю строкой об откате — модель и
        # пользователь должны видеть одну и ту же претензию.
        lines.append(
            "ПРЕДЫДУЩИЙ РЕЗУЛЬТАТ НАРУШИЛ ИНВАРИАНТЫ ПОЛЬЗОВАТЕЛЯ:\n"
            + invariants.render_violations(violations)
            + "\nПредложи новый вариант, который не нарушает ни одного инварианта, и "
            "объясни, что именно изменилось. Прежний вариант не повторяй."
        )

    missing = missing_keys(task)
    if missing:
        lines.append(
            "Не хватает для перехода к следующему этапу:\n"
            + "\n".join(f"- {key_label(key)}" for key in missing)
        )
        lines.append(
            "Не переходи к следующему этапу, пока эти пункты не собраны. Если "
            "пользователь просит перейти дальше — вежливо объясни, каких именно "
            "пунктов не хватает, и продолжи текущий этап."
        )
    elif violations:
        lines.append(
            "Этап не сменится, пока не появится новый результат, не нарушающий "
            "инварианты."
        )
    elif task.get("awaiting_correction"):
        lines.append(
            "Пользователь вернулся на этот этап, чтобы что-то поправить. Уточни, "
            "что именно нужно изменить, и зафиксируй новое значение, прежде чем "
            "двигаться дальше."
        )
    elif stage.name != STAGE_DONE:
        lines.append("Все пункты этого этапа собраны — можно переходить к следующему.")

    return "\n".join(lines)


def build_state_user_content(task: dict, messages: list[dict[str, str]]) -> str:
    """Пользовательская часть запроса технического вызова: текущее состояние,
    допустимые этапы и ключи, новые пары вопрос-ответ."""
    scenario = scenario_of(task)
    stage = current_stage(task)
    state = {
        "task_type": scenario.key,
        "goal": task.get("goal", ""),
        "stage": stage.name,
        "step": task.get("step", ""),
        "expected_action": task.get("expected_action", ""),
        "data": task.get("data", {}),
    }
    stages_overview = "\n".join(
        f"- {item.name} ({item.label}): {item.instruction}" for item in scenario.stages
    )
    allowed_keys = "\n".join(
        f"- {key}: {key_label(key)}"
        for key in (*stage.required_keys, *stage.optional_keys)
    )
    return (
        f"Сценарий задачи: {scenario.label}.\n"
        f"Этапы сценария:\n{stages_overview}\n\n"
        f"Текущее состояние задачи (JSON):\n{json.dumps(state, ensure_ascii=False)}\n\n"
        f"Текущий этап: {stage.name}. Допустимые следующие этапы: "
        f"{', '.join(stage.next_stages) or 'нет (задача завершена)'}.\n"
        f"Обязательные для перехода ключи текущего этапа: "
        f"{', '.join(stage.required_keys) or 'нет'}.\n"
        f"Ключи, которые можно заполнять на этом этапе:\n{allowed_keys or '- (нет)'}\n\n"
        f"Новые пары вопрос-ответ, ещё не учтённые в состоянии:\n"
        f"{render_exchanges(messages)}"
    )


def build_start_user_content(messages: list[dict[str, str]]) -> str:
    """Пользовательская часть запроса вызова-детектора старта (активной задачи
    нет): нужен только выбор сценария или «none»."""
    catalog = "\n".join(
        f"- {key}: {scenario.label}" for key, scenario in SCENARIOS.items()
    )
    return (
        f"Доступные сценарии задач:\n{catalog}\n\n"
        f"Последние пары вопрос-ответ:\n{render_exchanges(messages)}"
    )


def state_line(task: dict) -> str:
    """Короткая служебная строка о состоянии — печатается после каждого ответа
    агента, чтобы автомат был виден пользователю, а не двигался незаметно."""
    scenario = scenario_of(task)
    stage = current_stage(task)
    if task.get("paused"):
        return f"⏸ {scenario.label} · {stage.label} · задача на паузе"

    violations = invariant_violations(task)
    missing = missing_keys(task)
    if stage.name == STAGE_DONE:
        tail = "задача завершена, очистить — /smart_agent_task_done"
    elif violations:
        numbers = ", ".join(f"№{item['index']}" for item in violations)
        tail = f"нарушены инварианты {numbers} — нужен новый вариант"
    elif missing:
        tail = "ждём: " + ", ".join(key_label(key) for key in missing)
    elif task.get("awaiting_correction"):
        tail = "ждём правку: " + (task.get("expected_action") or "что именно изменить")
    elif task.get("expected_action"):
        tail = f"ждём: {task['expected_action']}"
    else:
        tail = "все пункты этапа собраны"
    if len(tail) > 120:
        tail = tail[:117] + "..."
    return f"📍 {scenario.label} · {stage.label} · {tail}"


def describe_state(task: dict) -> str:
    """Подробная сводка состояния — для /smart_agent_task_show и для
    возобновления после паузы. Строится ЦЕЛИКОМ из сохранённого состояния, без
    обращения к LLM: именно это и позволяет продолжить задачу после перерыва, не
    переспрашивая пользователя и не пересказывая ему диалог заново."""
    scenario = scenario_of(task)
    stage = current_stage(task)
    position = scenario.stage_index(stage.name) + 1
    lines = [
        f"📋 Задача: {scenario.label}",
        f"Цель: {task.get('goal') or '—'}",
        f"Этап: {stage.label} ({position} из {len(scenario.stages)})"
        + (" · ⏸ на паузе" if task.get("paused") else ""),
        f"Текущий шаг: {task.get('step') or '—'}",
        f"Ожидаемое действие: {task.get('expected_action') or '—'}",
    ]

    collected = [
        f"- {key_label(key)}: {value}" for key, value in task.get("data", {}).items() if value
    ]
    lines.append("Собрано:\n" + ("\n".join(collected) if collected else "— пока ничего"))

    violations = invariant_violations(task)
    if violations:
        lines.append(
            "⛔ Результат вернулся на доработку — нарушены инварианты:\n"
            + invariants.render_violations(violations)
        )

    missing = missing_keys(task)
    if missing:
        lines.append(
            "Не хватает до следующего этапа:\n"
            + "\n".join(f"- {key_label(key)}" for key in missing)
        )
    elif violations:
        lines.append("Ждём новый вариант результата, не нарушающий инварианты.")
    elif task.get("awaiting_correction"):
        lines.append(
            "Ждём правку: задача вернулась на этот этап и останется здесь, пока не "
            "будет уточнено, что именно изменить."
        )
    elif stage.name != STAGE_DONE:
        lines.append("Не хватает до следующего этапа: ничего, все пункты собраны.")

    return "\n".join(lines)


def archive_fact(task: dict) -> str:
    """ОДИН компактный факт о завершённой задаче для долговременной памяти.

    Именно один, а не по факту на каждый ключ data: долговременная память
    ограничена AGENT_MEMORY_LONG_TERM_MAX_FACTS с вытеснением, и подробный дамп
    одной задачи вытеснил бы из неё всё остальное.
    """
    scenario = scenario_of(task)
    stage_results = [
        key
        for item in scenario.stages
        if item.mode == MODE_AUTO
        for key in item.required_keys
    ]
    data = task.get("data", {})
    outcome = "; ".join(
        f"{key_label(key)}: {data[key]}" for key in stage_results if data.get(key)
    )
    finished_at = datetime.now().strftime("%Y-%m-%d")
    fact = f"{finished_at}: задача «{scenario.label}» завершена. Цель: {task.get('goal') or '—'}."
    if outcome:
        fact += f" Итог — {outcome}"
    return fact if len(fact) <= 600 else fact[:597] + "..."
