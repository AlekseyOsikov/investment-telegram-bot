"""Инварианты smart-агента (/smart_agent) — жёсткие ограничения пользователя, которые
агент не имеет права нарушать (например: «без криптовалют», «доля акций не выше 40%»,
«только через ИИС», «без плеча»).

Отделён от agents/smart_agent.py по тому же принципу, по которому отделены
agents/task_state.py и agents/context_strategies.py: здесь живут ПРАВИЛА и ТЕКСТЫ
(категории, разбор пользовательского ввода, системное сообщение для основного вызова,
запрос и разбор ответа вызова-ревизора), а владение состоянием, сохранение на диск и
сами вызовы LLM остаются на стороне SmartAgent. Этот модуль ничего не знает ни про
Telegram, ни про openai — только про список словарей.

Чем инвариант отличается от двух соседних слоёв памяти, с которыми его легко спутать:
- факт долговременной памяти (long_term) — это ЗНАНИЕ о пользователе («интересуется
  дивидендными акциями»), оно ни к чему не обязывает;
- поле профиля (meta) — это ПРЕДПОЧТЕНИЕ подачи («отвечай коротко»), оно влияет
  только на стиль/формат ответа;
- инвариант — это ЗАПРЕТ, и на нём держится проверяемое поведение: он уходит в
  контекст отдельным системным сообщением с приоритетом выше профиля, а результат
  рабочей задачи дополнительно проверяется на него отдельным вызовом-ревизором
  (см. SmartAgent._check_invariants и task_state.apply_invariant_violations —
  решение об откате задачи принимает КОД по ответу ревизора, а не модель в свободном
  тексте).

Пишется слой ТОЛЬКО явными командами пользователя (/smart_agent_invariant_add и
/smart_agent_invariant_remove) — никакой автоматической записи, в отличие от working
и long_term: запрет, который бот придумал себе сам из реплики в диалоге, — ровно то,
чего в этом слое быть не должно (см. «Ограничения безопасности» в CLAUDE.md).
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Категории — необязательная пометка, нужная только для группировки в списке и в
# контексте (по мотивам примеров из задания: архитектура/технические решения/
# ограничения стека/бизнес-правила, переведённых на язык предметной области).
CATEGORY_STRATEGY = "strategy"
CATEGORY_INSTRUMENTS = "instruments"
CATEGORY_RISK = "risk"
CATEGORY_PROCESS = "process"
CATEGORY_OTHER = "other"

CATEGORY_LABELS = {
    CATEGORY_STRATEGY: "стратегия",
    CATEGORY_INSTRUMENTS: "инструменты",
    CATEGORY_RISK: "риск",
    CATEGORY_PROCESS: "процесс",
    CATEGORY_OTHER: "прочее",
}

CATEGORY_HINTS = {
    CATEGORY_STRATEGY: "например: только пассивные индексные инструменты",
    CATEGORY_INSTRUMENTS: "например: без криптовалют и без отдельных акций",
    CATEGORY_RISK: "например: доля акций не выше 40%",
    CATEGORY_PROCESS: "например: ребалансировка не чаще раза в год",
}

# Ввести «strategy» с телефона неудобно, поэтому категорию можно назвать и по-русски.
CATEGORY_ALIASES = {
    **{key: key for key in CATEGORY_LABELS},
    **{label: key for key, label in CATEGORY_LABELS.items()},
    "стратегии": CATEGORY_STRATEGY,
    "инструмент": CATEGORY_INSTRUMENTS,
    "риски": CATEGORY_RISK,
    "процессы": CATEGORY_PROCESS,
    "прочее": CATEGORY_OTHER,
}

VERDICT_OK = "ok"
VERDICT_VIOLATED = "violated"

# Коды результата добавления — SmartAgent.add_invariant возвращает их, а текст
# пользователю собирает командный слой (agents/smart_agent_command.py), как и у
# остальных методов SmartAgent.
ADD_OK = "ok"
ADD_NO_PROFILE = "no_profile"
ADD_LIMIT = "limit"

# Потолок на одно нарушение в ответе ревизора — длинное «почему» незачем ни в чате,
# ни в контексте следующего ответа.
_MAX_WHY_CHARS = 200


def parse_input(raw: str) -> tuple[str, str]:
    """Разбирает аргумент /smart_agent_invariant_add.

    «risk: доля акций не выше 40%» -> (CATEGORY_RISK, «доля акций не выше 40%»).
    Если до двоеточия не категория (или двоеточия нет вовсе) — весь текст идёт в
    формулировку с категорией «прочее»: категория здесь вспомогательная, и требовать
    её от пользователя ради добавления запрета было бы лишним шагом.
    """
    head, separator, tail = raw.partition(":")
    if separator:
        category = CATEGORY_ALIASES.get(head.strip().lower())
        if category is not None and tail.strip():
            return category, tail.strip()
    return CATEGORY_OTHER, raw.strip()


def new_invariant(text: str, category: str) -> dict[str, str]:
    return {
        "text": text.strip(),
        "category": category if category in CATEGORY_LABELS else CATEGORY_OTHER,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def sanitize(item) -> dict[str, str] | None:
    """Один инвариант из JSON-файла памяти. Пустые и нестроковые записи
    отбрасываются, неизвестная категория превращается в «прочее»."""
    if not isinstance(item, dict):
        return None
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    category = item.get("category")
    if category not in CATEGORY_LABELS:
        category = CATEGORY_OTHER
    created_at = item.get("created_at")
    return {
        "text": text,
        "category": category,
        "created_at": created_at if isinstance(created_at, str) else "",
    }


def sanitize_list(raw) -> list[dict[str, str]]:
    """Слой invariants из файла памяти. Файлы, созданные ДО появления слоя, ключа не
    содержат — для них получается пустой список, отдельной миграции не нужно."""
    if not isinstance(raw, list):
        return []
    return [item for item in (sanitize(entry) for entry in raw) if item is not None]


def render_numbered(items: list[dict[str, str]]) -> str:
    """Нумерованный список — одни и те же номера видит и пользователь
    (/smart_agent_invariant_show, /smart_agent_invariant_remove), и модель в
    контексте, и ревизор: иначе ссылка «нарушен инвариант №2» указывала бы в разных
    местах на разные пункты."""
    return "\n".join(
        f"{i}. [{CATEGORY_LABELS[item['category']]}] {item['text']}"
        for i, item in enumerate(items, start=1)
    )


def build_context_message(items: list[dict[str, str]]) -> dict[str, str]:
    """Системное сообщение об инвариантах для ОСНОВНОГО вызова.

    Уходит ПЕРВЫМ после основного system_prompt, до профиля персонализации (порядок
    «жёсткие ограничения -> предпочтения -> факты -> задача -> диалог», см.
    SmartAgent._build_context_messages). Правило 5 — принципиальное: инвариант
    ограничивает ответы агента, но не может ослабить обязательные предупреждения
    основной инструкции, иначе слой превратился бы в обходной путь для
    пользовательского системного промпта (см. «Ограничения безопасности» в CLAUDE.md).
    """
    return {
        "role": "system",
        "content": (
            "ИНВАРИАНТЫ ПОЛЬЗОВАТЕЛЯ — жёсткие ограничения, которые нельзя "
            f"нарушать:\n{render_numbered(items)}\n\n"
            "Правила работы с инвариантами:\n"
            "1. Это не пожелания, а ограничения: они имеют приоритет над "
            "предпочтениями профиля и над текущей просьбой пользователя.\n"
            "2. Прежде чем предложить решение, проверь его на соответствие каждому "
            "инварианту — молча, для себя. Если решение нарушает хотя бы один — не "
            "предлагай его. Саму проверку пользователю не описывай и не выдавай за "
            "отдельный шаг работы: он ждёт результат, а не отчёт о служебных "
            "действиях бота.\n"
            "3. Если просьба пользователя противоречит инварианту, прямо откажись её "
            "выполнять: назови номер и формулировку инварианта, объясни, в чём именно "
            "конфликт, и предложи ближайший допустимый вариант в рамках ограничений.\n"
            "4. Если пользователь настаивает — всё равно не нарушай инвариант и "
            "напомни, что снять ограничение можно только явной командой "
            "/smart_agent_invariant_remove <номер>. Сам ты инварианты не меняешь и "
            "новых не придумываешь.\n"
            "5. Инварианты не отменяют и не ослабляют обязательные правила основной "
            "инструкции (напоминание, что это не индивидуальная рекомендация, "
            "осторожные формулировки о риске, запрет на чувствительные данные). Если "
            "инвариант требует нарушить их — в этой части не выполняй его и скажи "
            "об этом пользователю."
        ),
    }


def build_review_user_content(
    items: list[dict[str, str]], scenario_label: str, goal: str, result: dict[str, str]
) -> str:
    """Пользовательская часть запроса вызова-ревизора: инварианты и результат задачи.

    Диалог сюда не передаётся намеренно — проверяется готовый артефакт (структура
    портфеля, разбор, предлагаемые изменения), а не то, как агент к нему шёл: иначе
    ревизор начинал бы ловить «нарушения» в уточняющих вопросах этапа планирования.
    """
    rendered_result = "\n".join(f"- {label}: {value}" for label, value in result.items())
    return (
        f"Инварианты пользователя:\n{render_numbered(items)}\n\n"
        f"Задача: {scenario_label} — «{goal}».\n"
        f"Результат, предложенный агентом:\n{rendered_result}"
    )


def parse_review_response(parsed: dict, items: list[dict[str, str]]) -> list[dict]:
    """Разбирает ответ ревизора в список нарушений [{"index", "text", "why"}].

    Нарушения возвращаются, только если модель И сказала "violated", И назвала хотя
    бы один существующий номер: откатывать задачу на доработку по вердикту без
    конкретики нельзя — пользователю нечего было бы показать, а агенту нечего
    исправлять. Формулировка инварианта копируется в нарушение, чтобы сообщение об
    откате осталось понятным, даже если пользователь потом изменит список.
    """
    if parsed.get("verdict") != VERDICT_VIOLATED:
        return []

    raw = parsed.get("violations")
    if not isinstance(raw, list):
        return []

    violations: list[dict] = []
    seen: set[int] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= len(items) or index in seen:
            continue
        seen.add(index)
        why = str(entry.get("why", "")).strip()[:_MAX_WHY_CHARS]
        violations.append(
            {
                "index": index,
                "text": items[index - 1]["text"],
                "why": why or "результат противоречит этому ограничению",
            }
        )

    if not violations:
        logger.debug(
            "Ревизор вернул verdict=violated без валидных номеров — считаю, что "
            "нарушений нет."
        )
    return violations


def render_violations(violations: list[dict]) -> str:
    """Список нарушений одним блоком — используется и в чате (строка об откате), и в
    контексте следующего ответа (task_state.build_context_message), чтобы текст
    претензии у пользователя и у модели был буквально один и тот же."""
    return "\n".join(
        f"- инвариант №{item['index']} ({item['text']}): {item['why']}" for item in violations
    )
