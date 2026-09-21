"""Конечный автомат рабочей задачи smart-агента (/smart_agent).

Отделён от agents/smart_agent.py по тому же принципу, по которому
agents/context_strategies.py отделён от agents/agent.py: здесь живут ПРАВИЛА
автомата (реестр сценариев, этапы, условия перехода, разбор ответа модели и
тексты про состояние), а владение состоянием, его сохранение на диск и сами
вызовы LLM-клиента остаются на стороне SmartAgent. Этот модуль ничего не знает
ни про Telegram, ни про openai — только про словарь задачи.

Автомат один на все сценарии: planning -> execution -> validation -> done, с
откатом validation -> execution, если пользователь просит правки. Сценарии
(SCENARIOS) различаются только словарём ключей data, наличием гейта и текстами
инструкций, так что новый сценарий добавляется одной записью в реестр, без
изменений в самом автомате.

Имена этапов инженерные, но означают они шаги ИНВЕСТИЦИОННОГО процесса, и путать
их с одноимёнными понятиями из других доменов нельзя:
- planning — профилирование: сбор вводных пользователя (цель, горизонт,
  отношение к риску, ограничения) и подтверждение сводки этих вводных;
- execution — ФОРМИРОВАНИЕ ПРЕДЛОЖЕНИЯ (структуры портфеля, разбора, изменений
  долей), а НЕ исполнение сделок: бот ничего не покупает и не продаёт (см.
  «Правила предметной области» в CLAUDE.md). Пользователю этап и подписан по
  смыслу — «формирование портфеля»/«разбор»/«предложение изменений»;
- validation — ПРИЁМКА результата пользователем (устраивает или нужны правки), а
  не машинная проверка. Машинная проверка в этом проекте есть, но живёт отдельно:
  ревизор инвариантов (agents/invariants.py), см. ниже;
- done — работа зафиксирована, итог перенесён в долговременную память.

Ключевое отличие от «модель сама решает, когда двигаться дальше»: условие
перехода — это СПИСОК ОБЯЗАТЕЛЬНЫХ КЛЮЧЕЙ data у текущего этапа (плюс, если у
этапа есть гейт, конкретное значение ключа-гейта), и его проверяет код
(см. missing_keys/_forward_target), а не модель. Тот же список недостающих
пунктов уходит и в контекст основного ответа (build_context_message), поэтому
вежливый отказ на преждевременный переход всегда совпадает с реальным состоянием
автомата, а не расходится с ним.

ГЕЙТ (TaskStage.gate) — пара «ключ, требуемое значение»: пока в data нет именно
этого значения, вперёд с этапа не уйти, даже если все остальные ключи собраны.
Гейтов два, и оба означают явное решение ЧЕЛОВЕКА, которое из данных не выводится:
- на planning — подтверждение сводки вводных (brief_verdict == confirmed): агент
  проговаривает, как он понял цель/горизонт/риск/ограничения, и не начинает
  работу, пока пользователь это не подтвердил. Отсюда же и требование не давать
  рекомендацию на недопонятых ответах;
- на validation — приёмка результата (verdict == accepted).
Гейт на planning есть НЕ у всех сценариев: в portfolio и review результат — это
рекомендация по структуре, и подтверждение вводных там осмысленно, а asset —
справочный разбор, где подтверждение трёх параметров было бы формальностью. Это
решает реестр, а не логика переходов.

Из ответа модели принимаются только ключи ТЕКУЩЕГО ЭТАПА (см.
normalize_data_updates) — не всего сценария. Два разных повода, и оба важны:
модель иначе изобретала бы новое имя для того же параметра
(«горизонт»/«horizon»/«time_horizon»), и условие перехода не выполнилось бы
никогда; а заполнив за один ход ключи сразу двух этапов, она проводила бы задачу
через этап, которого фактически не было — например, записала бы приёмку
(verdict) вместе с самим результатом и увела задачу в done, не задав
пользователю вопроса и не дав сработать ревизору инвариантов. На РУЧНУЮ запись
(/smart_agent_task_set) фильтр не распространяется — там ключ задаёт человек, и
произвольные ключи сохраняются как раньше.

Каждая смена этапа, как и каждая ОТКЛОНЁННАЯ попытка её добиться, пишется в
короткую историю переходов внутри задачи (transitions, см. record_transition):
иначе «автомат не пустил» выглядело бы для пользователя просто как то, что бот
проигнорировал просьбу.

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
from dataclasses import dataclass, field
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

# Ключи-ГЕЙТЫ: от них зависит направление перехода, поэтому произвольный текст в
# них не принимается — только значения из CLOSED_VALUE_KEYS ниже (см.
# normalize_data_updates).
#
# verdict — приёмка результата пользователем на этапе проверки
# (accepted -> done, changes_requested -> назад к работе над результатом).
VERDICT_KEY = "verdict"
VERDICT_ACCEPTED = "accepted"
VERDICT_CHANGES_REQUESTED = "changes_requested"

# brief/brief_verdict — сводка вводных и её подтверждение на этапе профилирования
# (confirmed -> к работе, corrected -> сводка сбрасывается, остаёмся на этапе и
# уточняем параметры). Сводка — отдельный ключ, а не просто вопрос в чате, чтобы
# после паузы её можно было показать заново из состояния, не обращаясь к LLM.
BRIEF_KEY = "brief"
BRIEF_VERDICT_KEY = "brief_verdict"
BRIEF_CONFIRMED = "confirmed"
BRIEF_CORRECTED = "corrected"

# Ключи с закрытым списком значений. Всё, что не из списка, отбрасывается: на этих
# ключах держатся переходы, и «пользователь вроде согласен» в них означало бы
# трактовку на глаз (а при попытке модели записать своё значение — молчаливый
# пропуск этапа).
CLOSED_VALUE_KEYS: dict[str, tuple[str, ...]] = {
    VERDICT_KEY: (VERDICT_ACCEPTED, VERDICT_CHANGES_REQUESTED),
    BRIEF_VERDICT_KEY: (BRIEF_CONFIRMED, BRIEF_CORRECTED),
}

# Сколько последних записей истории переходов хранится в задаче. Ровно столько,
# чтобы в /smart_agent_task_show было видно, как задача пришла в текущее
# состояние: это диагностика, а не журнал аудита, и файл памяти чата от неё
# заметно расти не должен. Не в .env: это механика автомата, а не настройка
# оператора бота.
TRANSITIONS_MAX = 10

# Кто выполнил переход — для истории (см. record_transition). REJECTED — не
# переход, а зафиксированная ПОПЫТКА, которую автомат не пропустил.
BY_AUTO = "auto"
BY_MANUAL = "manual"
BY_ROLLBACK = "rollback"
BY_INVARIANTS = "invariants"
BY_GATE_REJECTED = "gate_rejected"
BY_REJECTED = "rejected"

_BY_LABELS = {
    BY_AUTO: "автомат",
    BY_MANUAL: "вручную",
    BY_ROLLBACK: "откат",
    BY_INVARIANTS: "инварианты",
    BY_GATE_REJECTED: "сводка не подтверждена",
    BY_REJECTED: "отклонено",
}

# Претензии ревизора инвариантов к результату задачи (agents/invariants.py) —
# список [{"index", "text", "why"}]. Живёт в задаче ОТДЕЛЬНО от data: data
# фильтруется по ключам текущего этапа (normalize_data_updates), а это поле
# заполняет не модель, а код по ответу ревизора (см. apply_invariant_violations).
INVARIANT_VIOLATIONS_KEY = "invariant_violations"

# Снимок результата, который ревизор инвариантов уже проверял. Нужен, чтобы
# проверка была привязана к САМОМУ РЕЗУЛЬТАТУ, а не к факту перехода на этап
# проверки: иначе задачу, переведённую на проверку вручную
# (/smart_agent_task_stage validation), ревизор не увидел бы вовсе — перехода-то
# не было, — и её результат ушёл бы в done без проверки (см. needs_invariant_check).
INVARIANTS_CHECKED_KEY = "invariants_checked_result"

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
    BRIEF_KEY: "сводка вводных",
    BRIEF_VERDICT_KEY: "подтверждение вводных",
    VERDICT_KEY: "решение пользователя по результату",
}

_NO_SUMS_RULE = (
    "Никогда не спрашивай и не используй денежные суммы, номера счетов и карт — "
    "только качественные параметры и доли в процентах."
)


@dataclass(frozen=True)
class TaskStage:
    """Один этап автомата. required_keys — и есть условие перехода дальше:
    пока хотя бы один из них не заполнен в data, переход отклоняется кодом.

    gate — необязательная пара (ключ, требуемое значение): решение человека,
    которое из данных не выводится (подтверждение сводки вводных, приёмка
    результата). Пока в data нет именно этого значения, вперёд не уйти, даже если
    все обязательные ключи заполнены. Этап без гейта двигается вперёд просто по
    собранным данным.

    sequential_keys — ключи, которые принимаются от модели только ПОСЛЕ того, как
    заполнены все обязательные ключи, стоящие раньше них в required_keys. Нужны
    ровно для гейта: сводку вводных нельзя проговорить до того, как вводные
    собраны, а подтвердить её нельзя до того, как она проговорена — иначе модель
    закрыла бы гейт первым же ходом, и он перестал бы что-либо гарантировать.
    Порядок ключей в required_keys для таких этапов значим.
    """

    name: str
    label: str
    mode: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...]
    next_stages: tuple[str, ...]
    instruction: str
    gate: tuple[str, str] | None = None
    sequential_keys: tuple[str, ...] = ()

    def gate_satisfied(self, data: dict) -> bool:
        """Выполнено ли условие гейта. Этап без гейта — всегда да."""
        if self.gate is None:
            return True
        key, expected = self.gate
        return str(data.get(key, "")).strip().lower() == expected

    def keys_before(self, key: str) -> tuple[str, ...]:
        """Обязательные ключи этапа, стоящие в required_keys раньше указанного."""
        if key not in self.required_keys:
            return ()
        return self.required_keys[: self.required_keys.index(key)]


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
        # Приёмка результата — решение пользователя, а не вывод из данных.
        gate=(VERDICT_KEY, VERDICT_ACCEPTED),
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
                required_keys=(
                    "goal_type",
                    "horizon",
                    "risk",
                    "constraints",
                    BRIEF_KEY,
                    BRIEF_VERDICT_KEY,
                ),
                optional_keys=("preferred_assets",),
                next_stages=(STAGE_EXECUTION,),
                gate=(BRIEF_VERDICT_KEY, BRIEF_CONFIRMED),
                sequential_keys=(BRIEF_KEY, BRIEF_VERDICT_KEY),
                instruction=(
                    "Собери недостающие пункты, задавая по одному уточняющему вопросу "
                    "за раз. Когда все пункты собраны, НЕ предлагай структуру "
                    "портфеля сразу: сначала проговори короткую сводку вводных (цель, "
                    "горизонт, отношение к риску, ограничения) и спроси прямо, всё ли "
                    "верно понято. Начинай работу только после подтверждения; если "
                    "пользователь поправляет вводные — уточни их и проговори сводку "
                    f"заново. {_NO_SUMS_RULE}"
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
                # Гейта подтверждения вводных здесь намеренно НЕТ (в отличие от
                # portfolio/review): это справочный разбор, а не рекомендация по
                # структуре, и подтверждение трёх параметров было бы формальностью
                # ради автомата. См. докстринг модуля.
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
                required_keys=(
                    "current_allocation",
                    "concern",
                    "constraints",
                    BRIEF_KEY,
                    BRIEF_VERDICT_KEY,
                ),
                optional_keys=("horizon",),
                next_stages=(STAGE_EXECUTION,),
                gate=(BRIEF_VERDICT_KEY, BRIEF_CONFIRMED),
                sequential_keys=(BRIEF_KEY, BRIEF_VERDICT_KEY),
                instruction=(
                    "Уточни текущий состав портфеля ТОЛЬКО в долях или процентах, что "
                    "именно беспокоит пользователя и какие есть ограничения — по "
                    "одному вопросу за раз. Если пользователь называет суммы, не "
                    "сохраняй их и попроси перевести состав в проценты. Когда всё "
                    "собрано, проговори короткую сводку вводных и спроси, верно ли "
                    "понято, — предлагать изменения можно только после подтверждения. "
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
        INVARIANTS_CHECKED_KEY: {},
        "data": {},
        "transitions": [],
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

    checked = raw.get(INVARIANTS_CHECKED_KEY)
    task = {
        "task_type": task_type,
        "goal": _as_text(raw.get("goal", "")),
        "stage": stage,
        "step": _as_text(raw.get("step", "")),
        "expected_action": _as_text(raw.get("expected_action", "")),
        "paused": bool(raw.get("paused", False)),
        "paused_at": raw.get("paused_at") if isinstance(raw.get("paused_at"), str) else None,
        "awaiting_correction": bool(raw.get("awaiting_correction", False)),
        INVARIANT_VIOLATIONS_KEY: _sanitize_violations(raw.get(INVARIANT_VIOLATIONS_KEY)),
        INVARIANTS_CHECKED_KEY: (
            {str(key): _as_text(value) for key, value in checked.items()}
            if isinstance(checked, dict)
            else {}
        ),
        "data": data,
        "transitions": _sanitize_transitions(raw.get("transitions")),
        "processed_pairs": processed_pairs,
        "created_at": _as_text(raw.get("created_at", "")),
    }
    _migrate_gate(task)
    return task


def _sanitize_transitions(raw) -> list[dict]:
    """История переходов из файла памяти. У задач, сохранённых ДО её появления, поля
    нет — получается пустой список, отдельной миграции не нужно (тот же приём, что
    _sanitize_violations)."""
    if not isinstance(raw, list):
        return []
    items = []
    for entry in raw[-TRANSITIONS_MAX:]:
        if not isinstance(entry, dict):
            continue
        items.append(
            {
                "at": _as_text(entry.get("at", "")),
                "from": _as_text(entry.get("from", "")),
                "to": _as_text(entry.get("to", "")),
                "by": _as_text(entry.get("by", "")),
                "note": _as_text(entry.get("note", "")),
            }
        )
    return items


def _migrate_gate(task: dict) -> None:
    """Задачи, сохранённые ДО появления гейта подтверждения вводных, но уже ушедшие
    с этапа профилирования: гейт считается пройденным задним числом.

    Иначе пользователь, у которого работа уже на проверке, при откате назад к
    параметрам внезапно обнаружил бы, что от него требуют подтвердить сводку,
    которой ему никогда не показывали. Задачи, оставшиеся на профилировании,
    ничего не получают — они доберут сводку штатным путём.
    """
    scenario = SCENARIOS[task["task_type"]]
    planning = scenario.stage(STAGE_PLANNING)
    if planning is None or planning.gate is None:
        return
    if scenario.stage_index(task["stage"]) <= scenario.stage_index(STAGE_PLANNING):
        return
    data = task["data"]
    if not data.get(BRIEF_KEY):
        data[BRIEF_KEY] = "(вводные собраны до появления явного подтверждения сводки)"
    if not data.get(BRIEF_VERDICT_KEY):
        data[BRIEF_VERDICT_KEY] = BRIEF_CONFIRMED


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


def stage_label(scenario: TaskScenario, name: str) -> str:
    stage = scenario.stage(name)
    return stage.label if stage is not None else name


def record_transition(task: dict, source: str, target: str, by: str, note: str = "") -> None:
    """Пишет в историю задачи и состоявшийся переход, и ОТКЛОНЁННУЮ попытку
    (by=BY_REJECTED, source == target).

    Отклонённые попытки здесь не для полноты картины: без них «автомат не пустил»
    выглядит для пользователя так, будто бот проигнорировал просьбу, и проверить
    поведение автомата можно только по логам. Длина списка ограничена
    TRANSITIONS_MAX — это диагностика, а не журнал аудита.
    """
    history = task.setdefault("transitions", [])
    history.append(
        {
            "at": datetime.now().isoformat(timespec="seconds"),
            "from": source,
            "to": target,
            "by": by,
            "note": note,
        }
    )
    del history[:-TRANSITIONS_MAX]


def transitions(task: dict) -> list[dict]:
    raw = task.get("transitions")
    return list(raw) if isinstance(raw, list) else []


def needs_invariant_check(task: dict) -> bool:
    """Нужно ли показать результат задачи ревизору инвариантов.

    Привязано к САМОМУ РЕЗУЛЬТАТУ, а не к факту перехода на этап проверки: иначе
    задача, переведённая на проверку вручную (/smart_agent_task_stage validation),
    ревизора не проходила бы вовсе, и её результат ушёл бы в done без проверки.
    Повторно один и тот же результат не проверяется — это лишний вызов API на
    каждом ходу диалога о приёмке.
    """
    if not is_validation(task):
        return False
    result = result_payload(task)
    if not result:
        return False
    return result != task.get(INVARIANTS_CHECKED_KEY)


def mark_invariants_checked(task: dict) -> None:
    task[INVARIANTS_CHECKED_KEY] = result_payload(task)


def manual_transition_block(task: dict, target: str) -> str | None:
    """Почему ручной перевод на target невозможен — фразой на русском, или None,
    если возможен. Вынесено из manual_transition отдельно, чтобы командный слой мог
    объяснить пользователю причину теми же словами, которыми её проверяет код."""
    scenario = scenario_of(task)
    stage = current_stage(task)
    if target not in allowed_manual_stages(task):
        return (
            f"переход «{stage.label} → {stage_label(scenario, target)}» не разрешён "
            "из текущего этапа"
        )
    if not is_forward_stage(task, target):
        return None
    if stage.gate is not None and not stage.gate_satisfied(task.get("data", {})):
        key, expected = stage.gate
        return (
            f"на этапе «{stage.label}» нет решения пользователя: {key_label(key)}. "
            "Ручной перевод его не заменяет — ответь в диалоге или задай явно: "
            f"/smart_agent_task_set {key} {expected}"
        )
    return None


def manual_transition(task: dict, target: str) -> bool:
    """Ручной перевод этапа (/smart_agent_task_stage) прямо в переданной задаче.

    Проверяет граф переходов и ГЕЙТ текущего этапа, но НЕ заполненность обычных
    обязательных ключей: это предохранитель на случай, когда автомат ошибся или
    застрял на данных, и требовать от него выполненных условий бессмысленно.

    Гейт — исключение именно потому, что он не «условие по данным», а решение
    человека (подтверждение вводных, приёмка результата): предохранитель сдвигает
    застрявшую задачу, но не решает за пользователя. Закрыть гейт можно только
    назвав его явно — ответом в диалоге или /smart_agent_task_set, где ключ
    указывает сам человек (см. manual_transition_block).

    Откат назад идёт через тот же _roll_back, что и автоматический, иначе переход
    вперёд по уже собранным данным тут же отменил бы ручное решение.
    """
    scenario = scenario_of(task)
    block = manual_transition_block(task, target)
    if block is not None:
        record_transition(task, task["stage"], target, BY_REJECTED, f"ручной перевод: {block}")
        return False
    source = task["stage"]
    if scenario.stage_index(target) < scenario.stage_index(source):
        _roll_back(scenario, task, target, BY_MANUAL)
    else:
        task["stage"] = target
        record_transition(task, source, target, BY_MANUAL)
    return True


def is_forward_stage(task: dict, target: str) -> bool:
    """Идёт ли переход на target ВПЕРЁД по графу — нужно вызывающему коду ручной
    команды, чтобы предупредить, что обязательные пункты текущего этапа при этом
    остались несобранными."""
    scenario = scenario_of(task)
    return scenario.stage_index(target) > scenario.stage_index(task.get("stage", ""))


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
    """Этапы, на которые в принципе ведёт граф из текущего (/smart_agent_task_stage).
    Заполненность обязательных ключей здесь НЕ проверяется — иначе застрявшую задачу
    нельзя было бы сдвинуть вручную. Отдельно от графа проверяется только гейт, см.
    manual_transition_block."""
    return current_stage(task).next_stages


# --------------------------------------------------------------------------- #
# Разбор ответа технического вызова
# --------------------------------------------------------------------------- #


def normalize_data_updates(
    stage: TaskStage, data: dict, updates: dict
) -> tuple[dict[str, str], list[str]]:
    """Оставляет только ключи ТЕКУЩЕГО ЭТАПА и приводит значения к строкам.
    Возвращает (принятые обновления, причины отказов на русском).

    Три вида отказа, и каждый закрывает свою дыру (см. докстринг модуля):
    - ключ не этого этапа — иначе модель заполнила бы ключи сразу двух этапов и
      провела задачу через этап, которого фактически не было;
    - значение вне закрытого списка (CLOSED_VALUE_KEYS) — от таких ключей зависит
      направление перехода, трактовать их на глаз нельзя;
    - последовательный ключ раньше времени (sequential_keys) — сводку вводных
      нельзя проговорить до сбора вводных, а подтвердить её нельзя до того, как она
      проговорена, иначе гейт закрывался бы первым же ходом.

    Причины отказов не глотаются: вызывающий код показывает их пользователем
    строкой в чате и пишет в историю задачи.
    """
    allowed = (*stage.required_keys, *stage.optional_keys)
    normalized: dict[str, str] = {}
    rejected: list[str] = []
    merged = dict(data)

    # Порядок обхода — как в реестре этапа, а не как в ответе модели: предпосылки
    # последовательных ключей должны считаться с учётом значений, принятых в этом
    # же ответе (вводные и сводка могут прийти одним ходом).
    for key in allowed:
        if key not in updates:
            continue
        text = _as_text(updates[key]).strip()
        if not text:
            continue
        choices = CLOSED_VALUE_KEYS.get(key)
        if choices is not None:
            text = text.lower()
            if text not in choices:
                rejected.append(
                    f"{key_label(key)}: значение «{_as_text(updates[key]).strip()}» "
                    f"не из списка ({', '.join(choices)})"
                )
                continue
        if key in stage.sequential_keys:
            unfilled = [
                earlier
                for earlier in stage.keys_before(key)
                if not str(merged.get(earlier, "")).strip()
            ]
            if unfilled:
                rejected.append(
                    f"{key_label(key)} — рано: сначала нужно собрать "
                    + ", ".join(key_label(item) for item in unfilled)
                )
                continue
        normalized[key] = text
        merged[key] = text

    for key in updates:
        if str(key) not in allowed:
            rejected.append(f"«{key}» — не пункт этапа «{stage.label}»")

    for reason in rejected:
        logger.debug("Обновление задачи отклонено: %s.", reason)
    return normalized, rejected


def next_forward_stage(scenario: TaskScenario, stage: TaskStage) -> TaskStage | None:
    """Этап, следующий за указанным ВПЕРЁД по графу (или None у финального). Нужен и
    условию перехода, и контексту основного ответа: модель должна знать, что будет
    дальше, иначе она сочиняет пользователю собственные названия этапов."""
    index = scenario.stage_index(stage.name)
    for name in stage.next_stages:
        if scenario.stage_index(name) > index:
            return scenario.stage(name)
    return None


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
    # Гейт — единственное, что не выводится из «все ключи заполнены»: нужно именно
    # то значение, которое означает решение человека (сводка подтверждена, результат
    # принят). Другое значение ключа-гейта — это отказ, то есть движение назад или
    # сброс гейта, см. _backward_target/_reset_gate.
    if not stage.gate_satisfied(task.get("data", {})):
        return None
    forward = next_forward_stage(scenario, stage)
    return forward.name if forward is not None else None


def _gate_rejected(stage: TaskStage, data: dict) -> bool:
    """Ключ-гейт заполнен, но НЕ тем значением, которое пропускает вперёд, — то есть
    человек сказал «нет»: сводка понята неверно, результат не устраивает."""
    if stage.gate is None:
        return False
    key, expected = stage.gate
    value = str(data.get(key, "")).strip().lower()
    return bool(value) and value != expected


def _reset_gate(stage: TaskStage, task: dict) -> None:
    """Отказ на гейте, откатывать который некуда (сводку вводных не подтвердили на
    первом же этапе): этап не меняется, но гейт и то, что он подтверждал, снимаются
    — агент уточняет параметры и проговаривает сводку заново.

    Флаг awaiting_correction здесь по той же причине, что и в _roll_back: без него
    задача формально ничего не ждёт, и следующий же ход мог бы снова закрыть гейт,
    не изменив ни одного параметра.
    """
    if stage.gate is None:
        return
    for key in (*stage.sequential_keys, stage.gate[0]):
        task["data"].pop(key, None)
    task["awaiting_correction"] = True
    record_transition(
        task,
        stage.name,
        stage.name,
        BY_GATE_REJECTED,
        f"{key_label(stage.gate[0])} — отказ, собираем заново",
    )


def _rejected_stage_request(
    scenario: TaskScenario, stage_before: TaskStage, task: dict, requested
) -> list[str]:
    """Причина, по которой запрошенный моделью этап не состоялся — на русском и в
    тех же формулировках, что уходят пользователю. Пустой список, если модель ничего
    не просила или её просьба и так исполнена."""
    if not isinstance(requested, str):
        return []
    requested = requested.strip()
    if not requested or requested in (stage_before.name, task["stage"]):
        return []
    if scenario.stage(requested) is None:
        return [f"этап «{requested}» в сценарии «{scenario.label}» не существует"]

    label = stage_label(scenario, requested)
    direction = f"переход «{stage_before.label} → {label}»"
    if requested not in stage_before.next_stages:
        return [f"{direction} не разрешён из текущего этапа"]
    missing = missing_keys(task)
    if missing:
        return [
            f"{direction} — не хватает: " + ", ".join(key_label(key) for key in missing)
        ]
    return [f"{direction} — вперёд автомат двигается сам, по собранным данным"]


def _backward_target(scenario: TaskScenario, task: dict, requested) -> str | None:
    """Откат назад — единственный вид перехода, который НЕ выводится из данных
    автоматически: «вернуться и переделать» может решить только пользователь.
    Направление задаёт либо отказ на гейте этапа (результат не принят -> назад к
    работе над ним), либо поле "stage" из ответа модели, если она указала более
    ранний допустимый этап.

    Возвращает None, если отказ на гейте некуда откатывать (например, сводку
    вводных не подтвердили на первом же этапе) — такой отказ обрабатывается сбросом
    гейта на месте, см. _reset_gate."""
    stage = current_stage(task)
    if _gate_rejected(stage, task.get("data", {})):
        return _last_auto_stage(scenario, stage.name)
    if not isinstance(requested, str) or requested not in stage.next_stages:
        return None
    index = scenario.stage_index(stage.name)
    return requested if 0 <= scenario.stage_index(requested) < index else None


def _roll_back(
    scenario: TaskScenario, task: dict, target: str, by: str, note: str = ""
) -> None:
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
    source = task["stage"]
    for stage in scenario.stages[start + 1 :]:
        for key in stage.required_keys:
            task["data"].pop(key, None)
    # Гейт целевого этапа снимается, хотя остальные его ключи сохраняются: если
    # пользователь вернулся править вводные, прежнее подтверждение сводки больше
    # ничего не значит — сводку нужно проговорить и подтвердить заново, иначе
    # задача уехала бы вперёд по устаревшему согласию.
    target_stage = scenario.stages[start]
    if target_stage.gate is not None:
        for key in (*target_stage.sequential_keys, target_stage.gate[0]):
            task["data"].pop(key, None)
    task["stage"] = target
    task["awaiting_correction"] = True
    if target != source:
        record_transition(task, source, target, by, note)


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

    _roll_back(
        scenario,
        updated,
        target,
        BY_INVARIANTS,
        "нарушены инварианты " + ", ".join(f"№{item['index']}" for item in violations),
    )
    updated[INVARIANT_VIOLATIONS_KEY] = list(violations)
    return updated, f"{stage_before.label} → {current_stage(updated).label}"


@dataclass
class StateChange:
    """Итог технического вызова по одной задаче.

    transition — подпись состоявшегося перехода для чата («планирование →
    формирование портфеля») или None, если этап не сменился. rejected — причины, по
    которым автомат НЕ сделал того, что предложила модель; командный слой печатает
    их пользователю, а не глотает (см. record_transition про то, почему
    отклонённая попытка не должна выглядеть как проигнорированная просьба).
    gate_reset — гейт этапа сняли отказом (сводку вводных не подтвердили): этап тот
    же, но работа на нём начинается заново, и молча этого делать нельзя.
    """

    task: dict
    transition: str | None = None
    rejected: list[str] = field(default_factory=list)
    gate_reset: bool = False


def apply_state_response(task: dict, parsed: dict) -> StateChange:
    """Применяет разобранный JSON технического вызова к задаче.

    Порядок: сначала данные и шаг, потом переходы. ВПЕРЁД задача двигается сама,
    как только собраны обязательные ключи этапа и выполнен его гейт
    (_forward_target) — поле "stage" из ответа модели на это не влияет, иначе
    задача зависает на этапе, условия которого уже выполнены. НАЗАД — только по
    явному сигналу (отказ на гейте или более ранний допустимый этап в "stage"), и
    такой откат ждёт правку, прежде чем снова разрешить движение вперёд (см.
    _roll_back).

    Обновления данных фильтруются по ключам ТЕКУЩЕГО этапа, поэтому «перепрыгнуть»
    этап, заполнив ключи следующего, нельзя: цикл движения вперёд остаётся, но
    данных на второй шаг в одном ходу взяться уже неоткуда (кроме ручного
    /smart_agent_task_set — там ключи задаёт человек, и это его право).
    """
    updated = copy.deepcopy(task)
    scenario = scenario_of(updated)
    stage_before = current_stage(updated)

    raw_updates = parsed.get("data_updates")
    updates, rejected = (
        normalize_data_updates(stage_before, updated["data"], raw_updates)
        if isinstance(raw_updates, dict)
        else ({}, [])
    )
    updated["data"].update(updates)

    for name in ("step", "expected_action"):
        value = parsed.get(name)
        if isinstance(value, str) and value.strip():
            updated[name] = value.strip()

    requested = parsed.get("stage")
    gate_reset = False
    backward = _backward_target(scenario, updated, requested)
    if backward is not None:
        _roll_back(
            scenario,
            updated,
            backward,
            BY_ROLLBACK,
            f"{key_label(stage_before.gate[0])} — отказ"
            if _gate_rejected(stage_before, updated["data"])
            else "возврат к ранее собранному по просьбе пользователя",
        )
    elif _gate_rejected(stage_before, updated["data"]):
        _reset_gate(stage_before, updated)
        gate_reset = True
    else:
        # Правка после отката пришла — можно снова двигаться вперёд. Заодно
        # снимаются претензии ревизора инвариантов: они относились к ПРЕЖНЕМУ
        # результату, а ключ этапа только что переписан (см.
        # apply_invariant_violations).
        if any(key in stage_before.required_keys for key in updates):
            updated["awaiting_correction"] = False
            updated[INVARIANT_VIOLATIONS_KEY] = []

        if not updated.get("awaiting_correction"):
            # Вперёд — столько шагов, сколько позволяют собранные данные. Обычно
            # это ноль или один: ключи следующего этапа от модели в этом же ходу не
            # принимаются. Ограничение по числу этапов — защита от зацикливания на
            # случай кривого реестра сценария.
            for _ in range(len(scenario.stages)):
                forward = _forward_target(scenario, updated)
                if forward is None:
                    break
                source = updated["stage"]
                updated["stage"] = forward
                record_transition(updated, source, forward, BY_AUTO)

    rejected.extend(_rejected_stage_request(scenario, stage_before, updated, requested))
    for reason in rejected:
        record_transition(updated, stage_before.name, stage_before.name, BY_REJECTED, reason)

    if updated["stage"] == stage_before.name:
        if missing_keys(updated):
            logger.debug(
                "Задача (%s) остаётся на этапе %s, не хватает: %s.",
                scenario.key,
                stage_before.name,
                ", ".join(missing_keys(updated)),
            )
        return StateChange(task=updated, rejected=rejected, gate_reset=gate_reset)

    return StateChange(
        task=updated,
        transition=f"{stage_before.label} → {current_stage(updated).label}",
        rejected=rejected,
        gate_reset=gate_reset,
    )


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

    # Без этой строки модель не знает, что будет дальше, и сочиняет пользователю
    # собственные этапы («дальше — наполнение блоков»), которых в автомате нет, —
    # а заодно анонсирует как шаг работы служебные проверки бота.
    forward = next_forward_stage(scenario, stage)
    lines.append(
        (f"Следующий этап: {forward.label}." if forward is not None else "Это последний этап.")
        + " Названий этапов не выдумывай и не обещай пользователю шагов, которых нет "
        "в этом списке. Служебные проверки бота (в том числе проверку ограничений) "
        "как отдельный шаг работы не анонсируй."
    )

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
        if stage.gate is not None and missing == [stage.gate[0]]:
            # Всё, кроме решения человека, собрано. Без этой строки модель видит
            # только «не хватает подтверждения вводных» и не понимает, что от неё
            # требуется не заполнить пункт, а дождаться ответа.
            lines.append(
                f"Осталось только решение пользователя: {key_label(stage.gate[0])}. "
                "Задай прямой вопрос и дождись ответа — сам за пользователя этот "
                "пункт не решай."
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
    gate_note = ""
    if stage.gate is not None:
        gate_key, gate_value = stage.gate
        gate_note = (
            f"Условие-гейт этапа: ключ {gate_key} ({key_label(gate_key)}) должен "
            f"получить значение \"{gate_value}\" — это решение ПОЛЬЗОВАТЕЛЯ, "
            "заполняй его только по его прямому ответу.\n"
        )
    if stage.sequential_keys:
        gate_note += (
            "Ключи "
            + ", ".join(stage.sequential_keys)
            + " заполняются только после того, как собраны все предшествующие им "
            "обязательные ключи этапа, — иначе они будут отброшены.\n"
        )
    return (
        f"Сценарий задачи: {scenario.label}.\n"
        f"Этапы сценария:\n{stages_overview}\n\n"
        f"Текущее состояние задачи (JSON):\n{json.dumps(state, ensure_ascii=False)}\n\n"
        f"Текущий этап: {stage.name}. Допустимые следующие этапы: "
        f"{', '.join(stage.next_stages) or 'нет (задача завершена)'}.\n"
        f"Обязательные для перехода ключи текущего этапа: "
        f"{', '.join(stage.required_keys) or 'нет'}.\n"
        f"{gate_note}"
        f"Ключи, которые можно заполнять на этом этапе (любые другие будут "
        f"отброшены):\n{allowed_keys or '- (нет)'}\n\n"
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
    paused_note = ""
    if task.get("paused"):
        paused_note = " · ⏸ на паузе" + (
            f" с {task['paused_at']}" if task.get("paused_at") else ""
        )
    lines = [
        f"📋 Задача: {scenario.label}",
        f"Цель: {task.get('goal') or '—'}",
        f"Этап: {stage.label} ({position} из {len(scenario.stages)}){paused_note}",
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

    history = render_transitions(task)
    if history:
        lines.append("История переходов:\n" + history)

    return "\n".join(lines)


def render_transitions(task: dict) -> str:
    """История переходов задачи для /smart_agent_task_show и сводки после паузы —
    включая ОТКЛОНЁННЫЕ попытки, чтобы «автомат не пустил» можно было увидеть, а не
    только предположить."""
    scenario = scenario_of(task)
    rendered = []
    for entry in transitions(task):
        by = _BY_LABELS.get(entry.get("by", ""), entry.get("by", ""))
        note = f" — {entry['note']}" if entry.get("note") else ""
        if entry.get("by") == BY_REJECTED or entry.get("from") == entry.get("to"):
            where = stage_label(scenario, entry.get("to", ""))
            rendered.append(f"- {where} · {by}{note}")
        else:
            rendered.append(
                f"- {stage_label(scenario, entry.get('from', ''))} → "
                f"{stage_label(scenario, entry.get('to', ''))} · {by}{note}"
            )
    return "\n".join(rendered)


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
