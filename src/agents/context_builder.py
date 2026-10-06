"""Сборка контекста LLM для `/smart_agent` из включённых слоёв активного профиля — чистые
функции без состояния, файловой системы и сети (tests/test_context_builder.py). Отделена от
agents/smart_agent.py: порядок и тексты сообщений проверяются тестами без `SmartAgent`.

Порядок слоёв — от самого жёсткого и стабильного к самому свежему: инварианты -> профиль ->
инструменты -> справочные материалы -> долговременная память -> рабочая задача ->
краткосрочный диалог. Каждое служебное сообщение уходит ОТДЕЛЬНЫМ системным сообщением и не
подменяет `SYSTEM_PROMPT`; оговорки о том, что слой не ослабляет обязательные предупреждения и
осторожные формулировки, сохраняй при правке текстов (иначе слой превратится в
пользовательский системный промпт)."""

from __future__ import annotations

from . import invariants, rag_context, task_state
from .memory_state import (
    LAYER_INVARIANTS,
    LAYER_LONG_TERM,
    LAYER_PROFILE,
    LAYER_SHORT_TERM,
    LAYER_WORKING,
    PROFILE_FIELD_LABELS,
    PROFILE_FIELDS,
)


def profile_meta_message(meta: dict[str, str]) -> dict[str, str] | None:
    """Системное сообщение с профилем персонализации активного чата — влияет
    ТОЛЬКО на стиль/формат/объём ответа, о чём сообщение явно предупреждает
    модель, а не отменяет обязательные предупреждения основного system_prompt
    (см. «Правила предметной области» в CLAUDE.md). Возвращает
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


def long_term_message(facts: list[dict[str, str]]) -> dict[str, str]:
    facts_text = "\n".join(f"- {fact['text']}" for fact in facts)
    return {
        "role": "system",
        "content": (
            "Долговременная память о пользователе (часть сохранена им "
            "явно командой /smart_agent_remember, часть извлечена из "
            f"диалога):\n{facts_text}"
        ),
    }


def rag_messages(materials: rag_context.Materials) -> list[dict[str, str]] | None:
    """Правила обращения с материалами — когда фрагменты в запросе есть; при состоявшемся
    поиске без фрагментов (модель вызвана при доступных инструментах или активной задаче,
    иначе ответ — «не знаю» без вызова) — указание не выдавать ответ за подтверждённый
    материалами."""
    if materials.chunks:
        return [rag_context.build_rules_message()]
    if materials.searched:
        return [rag_context.build_no_materials_message()]
    return None


def build_context_messages(
    system_prompt: str,
    profile: dict | None,
    enabled_layers: dict[str, bool],
    short_term_pairs: int,
    tools_messages: list[dict[str, str]] | None = None,
    rag_messages: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Собирает контекст LLM из явно ВКЛЮЧЁННЫХ слоёв (`enabled_layers`) активного профиля —
    в отличие от Agent, здесь нет автоматического выбора одной стратегии: пользователь сам
    решает и что сохранять, и какие слои участвуют в запросе. `profile` — None, если активного
    профиля нет (тогда только системный промпт).

    tools_messages — готовые сообщения слоя tools (доступные, недоступные, выключенные
    источники — см. agents/market_tools.py); может быть несколько сразу. Их передаёт ask()
    только когда слой включён и хотя бы один источник настроен, поэтому проверка слоя здесь не
    повторяется. Инварианты остаются первыми и прямо получают приоритет над ними.

    rag_messages — правила обращения со справочными материалами (rag_messages()); ask()
    передаёт их только когда фрагменты в запрос действительно попали. Идут сразу после
    сообщений tools, до долговременной памяти; сами фрагменты — в последнем user-сообщении."""
    messages = [{"role": "system", "content": system_prompt}]
    if profile is None:
        return messages

    # Инварианты идут ПЕРВЫМИ, до профиля персонализации: это ограничения, а
    # профиль — предпочтения подачи, и приоритет между ними проговорён прямо в
    # тексте сообщения (см. invariants.build_context_message).
    if enabled_layers[LAYER_INVARIANTS] and profile["invariants"]:
        messages.append(invariants.build_context_message(profile["invariants"]))

    if enabled_layers[LAYER_PROFILE]:
        meta_message = profile_meta_message(profile["meta"])
        if meta_message:
            messages.append(meta_message)

    if tools_messages:
        messages.extend(tools_messages)

    if rag_messages:
        messages.extend(rag_messages)

    if enabled_layers[LAYER_LONG_TERM] and profile["long_term"]:
        messages.append(long_term_message(profile["long_term"]))

    if enabled_layers[LAYER_WORKING] and profile["working"] is not None:
        # Весь текст про этап/шаг/ожидание/недостающие пункты собирает сам
        # автомат (agents/task_state.py) — там же, где определены условия
        # перехода, чтобы модель и код не расходились в том, чего не хватает.
        messages.append(
            {"role": "system", "content": task_state.build_context_message(profile["working"])}
        )

    if enabled_layers[LAYER_SHORT_TERM]:
        window_size = 2 * short_term_pairs
        short_term = profile["short_term"]
        messages.extend(short_term[-window_size:] if window_size > 0 else [])

    return messages
