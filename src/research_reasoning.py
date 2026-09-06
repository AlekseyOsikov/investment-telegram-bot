"""Режим исследования (/research_reasoning): сравнение способов рассуждения DeepSeek API.

Отдельный от основного прокси-потока модуль, по аналогии с research_response_format.py:
это техническое исследование того, как разные приёмы построения промпта влияют на
качество решения произвольной задачи (не обязательно инвестиционной), а не часть
обычного сценария использования бота. Поэтому системный промпт здесь намеренно не
позиционирует модель как инвестиционного ассистента (в отличие от config.SYSTEM_PROMPT) —
задачи в этом режиме могут быть любыми, а не только про инвестиции.

Сценарий 4 интерактивен: после выбора сценария бот показывает отдельную клавиатуру
выбора роли эксперта (аналитик/инженер/критик) и позволяет запросить решение от
нескольких экспертов подряд для одной и той же задачи, не возвращаясь в главное меню
сценариев между запросами.

Сценарий 5 запускает сценарии 1-4 (включая всех трёх экспертов по отдельности, итого
6 обращений к DeepSeek) параллельно в пуле потоков, затем одним отдельным запросом
просит модель сравнить полученные ответы между собой — отличаются ли они и какой из
них точнее — и показывает и все исходные ответы, и итоговое сравнение.

main.py подключает фичу через build_reasoning_conversation_handler() — единственную
точку интеграции с остальным приложением.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import (
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    TELEGRAM_MESSAGE_LIMIT,
)
from main_client import main_client

logger = logging.getLogger(__name__)

WAITING_TASK, CHOOSING_SCENARIO, CHOOSING_EXPERT = range(3)

# Этот режим сравнивает приёмы рассуждения на уровне промпта (пошагово, самопромпт,
# группа экспертов), а не собственное скрытое рассуждение модели — оно не только не
# нужно для сравнения, но и мешает: на моделях с рассуждениями весь max_tokens может
# уйти на reasoning_content, оставляя видимый ответ пустым (см. _content_or_reasoning_
# fallback). "thinking" не входит в типизированную сигнатуру chat.completions.create
# в openai SDK, поэтому передаётся через extra_body.
DISABLE_THINKING = {"thinking": {"type": "disabled"}}

# Нарочно нейтральный системный промпт: этот режим исследует приёмы рассуждения на
# произвольных задачах, а не отвечает на инвестиционные вопросы, поэтому здесь не
# используется config.SYSTEM_PROMPT с его инвестиционной ролью и дисклеймерами.
REASONING_SYSTEM_PROMPT = (
    "Ты — ассистент, который решает поставленную пользователем задачу. Отвечай по "
    "существу. Определи язык, на котором сформулирована задача, и веди видимый ответ "
    "(поле content) строго на этом же языке от первого до последнего слова — даже если "
    "внутренние рассуждения (reasoning) велись на другом языке, финальный видимый ответ "
    "переведи на язык задачи."
)

EXPERT_ROLES = {
    "analyst": (
        "Аналитик",
        "Ты — аналитик. Прежде чем дать решение, разбери исходные данные и факты задачи, "
        "оцени возможные варианты и их последствия.",
    ),
    "engineer": (
        "Инженер",
        "Ты — инженер. Подойди к задаче практически: раздели её на конкретные шаги "
        "реализации и предложи работающее пошаговое решение.",
    ),
    "critic": (
        "Критик",
        "Ты — критик. Сначала укажи на возможные ошибки, слабые места и неверные "
        "допущения в постановке задачи или в очевидном решении, затем дай выверенный ответ.",
    ),
}

REASONING_INTRO_TEXT = (
    f"🔬 Режим исследования способов рассуждения {MAIN_CLIENT_LABEL} API.\n\n"
    "Это техническое исследование, а не инвестиционный сервис — задача может быть любой, "
    "не обязательно про инвестиции.\n\n"
    "👉 Введите задачу, которую нужно решить."
)


# --------------------------------------------------------------------------- #
# 5 способов рассуждения
# --------------------------------------------------------------------------- #


def _extract_usage(response) -> dict[str, int] | None:
    """Достаёт минимальную статистику по токенам из ответа DeepSeek, если она есть."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _content_or_reasoning_fallback(message) -> str | None:
    """Возвращает видимый content, а если он пуст — обрезанный reasoning_content.

    На моделях с рассуждениями (deepseek-reasoner и т.п.) весь лимит max_tokens
    может целиком уйти на скрытые размышления, оставляя видимый content пустым при
    finish_reason == "length" (тот же случай уже описан в
    research_response_format.call_deepseek_max_tokens) — здесь это актуально для
    любого сценария, а не только с явным ограничением токенов пользователем.
    """
    content = message.content
    if not content:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            # reasoning_content — сырой ход мыслей модели, а не финальный ответ: язык
            # инструкции про "отвечай на языке задачи" на него не распространяется,
            # поэтому он может оказаться на другом языке (часто на английском) даже
            # если задача была на русском — это ожидаемо, а не баг перевода.
            return (
                "[Модель ещё не начала видимый ответ, вот её рассуждения "
                "(могут быть на другом языке, чем задача)]\n\n" + reasoning
            )
    return content


def _sum_usage(
    first: dict[str, int] | None, second: dict[str, int] | None
) -> dict[str, int] | None:
    """Складывает статистику по токенам двух вызовов API (для сценария 3, два запроса)."""
    if first is None and second is None:
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a, b = (first or {}).get(key), (second or {}).get(key)
        if a is None and b is None:
            continue
        result[key] = (a or 0) + (b or 0)
    return result or None


def call_reasoning_direct(
    task: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Способ 1: прямой ответ без дополнительных инструкций (контроль)."""
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    choice = response.choices[0]
    return _content_or_reasoning_fallback(choice.message), choice.finish_reason, _extract_usage(response)


def call_reasoning_step_by_step(
    task: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Способ 2: в промпт добавляется инструкция «решай пошагово»."""
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": f"{task}\n\nРешай пошагово."},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    choice = response.choices[0]
    return _content_or_reasoning_fallback(choice.message), choice.finish_reason, _extract_usage(response)


def call_reasoning_self_prompt(
    task: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Способ 3: модель сначала составляет промпт для решения задачи, затем решает по нему.

    Два последовательных вызова API: первый просит только текст промпта (без решения
    самой задачи), второй использует этот промпт как пользовательское сообщение для
    получения решения. Сгенерированный промпt выводится перед итоговым ответом —
    это и есть предмет исследования данного сценария.
    """
    prompt_response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Составь подробный промпт (инструкцию для себя же), который поможет "
                    "качественно решить следующую задачу. Не решай саму задачу, выведи "
                    f"только текст промпта.\n\nЗадача: {task}"
                ),
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    generated_prompt = _content_or_reasoning_fallback(prompt_response.choices[0].message)
    if not generated_prompt:
        generated_prompt = f"(промпт не сгенерирован: {MAIN_CLIENT_LABEL} вернул пустой ответ на первом шаге)"

    solve_response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{generated_prompt}\n\n"
                    "Используя этот промпт как инструкцию, реши следующую задачу и выведи "
                    f"окончательный ответ (не переписывай сам промпт).\n\nЗадача: {task}"
                ),
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    choice = solve_response.choices[0]
    final_answer = _content_or_reasoning_fallback(choice.message)
    if not final_answer:
        final_answer = f"⚠️ {MAIN_CLIENT_LABEL} вернул пустой ответ на сгенерированный промпт."
    combined_answer = (
        f"🧭 Сгенерированный промпт:\n{generated_prompt}\n\n"
        f"📝 Ответ по этому промпту:\n{final_answer}"
    )
    usage = _sum_usage(_extract_usage(prompt_response), _extract_usage(solve_response))
    return combined_answer, choice.finish_reason, usage


def call_reasoning_expert(
    task: str, expert_id: str
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Способ 4: решение задачи от лица одного эксперта из группы (аналитик/инженер/критик)."""
    _, expert_instruction = EXPERT_ROLES[expert_id]
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": f"{REASONING_SYSTEM_PROMPT} {expert_instruction}"},
            {"role": "user", "content": task},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    choice = response.choices[0]
    return _content_or_reasoning_fallback(choice.message), choice.finish_reason, _extract_usage(response)


# Подзадачи сценария 5 — те же вызовы, что стоят за сценариями 1-4 (эксперты — по
# отдельности на каждую роль), запускаются параллельно в пуле потоков.
_COMPARE_ALL_SUB_SCENARIOS: list[tuple[str, object, object]] = [
    ("Прямой ответ", call_reasoning_direct, None),
    ("Пошаговое рассуждение", call_reasoning_step_by_step, None),
    ("Промпт от модели для себя", call_reasoning_self_prompt, None),
    ("Эксперт: Аналитик", call_reasoning_expert, "analyst"),
    ("Эксперт: Инженер", call_reasoning_expert, "engineer"),
    ("Эксперт: Критик", call_reasoning_expert, "critic"),
]

COMPARE_ALL_INSTRUCTION = (
    "Ниже приведены ответы на одну и ту же задачу, полученные разными способами "
    "рассуждения. Сравни их и ответь на два вопроса:\n"
    "1. Отличаются ли ответы друг от друга (и чем именно)?\n"
    "2. Какой способ дал наиболее точный результат и почему?"
)


def call_reasoning_compare_all(
    task: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Способ 5: параллельно запускает способы 1-4 и просит модель сравнить их ответы.

    Каждая подзадача выполняется в своём потоке (обращения к API — блокирующий
    HTTP-вызов openai SDK, поэтому распараллеливание через ThreadPoolExecutor, а не
    asyncio, соответствует тому, как остальные сценарии этого модуля уже вызывают
    DeepSeek синхронно). Отдельная ошибка одной подзадачи не прерывает сравнение —
    вместо ответа этой подзадачи в сравнение уходит пометка об ошибке. Если не удалась
    ни одна подзадача, до финального сравнительного запроса дело всё равно доходит:
    модель увидит только пометки об ошибках и напишет об этом в сравнении.
    """
    sub_results: dict[str, tuple[str | None, str | None]] = {}
    sub_usages: list[dict[str, int]] = []

    with ThreadPoolExecutor(max_workers=len(_COMPARE_ALL_SUB_SCENARIOS)) as executor:
        future_to_label = {
            executor.submit(fn, task, extra): label
            for label, fn, extra in _COMPARE_ALL_SUB_SCENARIOS
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                answer, finish_reason, usage = future.result()
            except Exception:  # noqa: BLE001 — одна упавшая подзадача не должна рушить сравнение
                logger.exception("Подзадача «%s» сценария 5 завершилась ошибкой.", label)
                answer, finish_reason, usage = None, None, None
            sub_results[label] = (answer, finish_reason)
            if usage:
                sub_usages.append(usage)

    ordered_labels = [label for label, _, _ in _COMPARE_ALL_SUB_SCENARIOS]

    comparison_input = "\n\n".join(
        f"### {label}\n{sub_results[label][0] or '[нет ответа: ошибка при обращении к API]'}"
        for label in ordered_labels
    )
    comparison_response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Исходная задача: {task}\n\n{comparison_input}\n\n{COMPARE_ALL_INSTRUCTION}"
                ),
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    choice = comparison_response.choices[0]
    comparison_answer = _content_or_reasoning_fallback(choice.message)

    result_parts = []
    for label in ordered_labels:
        answer, finish_reason = sub_results[label]
        text = answer or "⚠️ Не удалось получить ответ (ошибка API)."
        if finish_reason == "length":
            text += "\n⚠️ Обрезано из-за ограничения max_tokens."
        result_parts.append(f"🔹 {label}:\n{text}")
    result_parts.append(
        "🧩 Сравнение способов:\n" + (comparison_answer or f"{MAIN_CLIENT_LABEL} вернул пустой ответ.")
    )
    combined_answer = "\n\n".join(result_parts)

    usage = _extract_usage(comparison_response)
    for sub_usage in sub_usages:
        usage = _sum_usage(usage, sub_usage)
    return combined_answer, choice.finish_reason, usage


SCENARIO_HANDLERS = {
    "1": ("Прямой ответ", call_reasoning_direct),
    "2": ("Пошаговое рассуждение", call_reasoning_step_by_step),
    "3": ("Промпт от модели для себя", call_reasoning_self_prompt),
    "4": ("Группа экспертов", None),  # запускается через выбор роли, см. CHOOSING_EXPERT
    "5": ("Сравнение всех способов", call_reasoning_compare_all),
}


# --------------------------------------------------------------------------- #
# Вспомогательные функции: клавиатуры, форматирование, запуск сценария
# --------------------------------------------------------------------------- #


def _build_reasoning_scenario_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1️⃣ Прямой ответ", callback_data="reasoning:scenario:1"),
                InlineKeyboardButton("2️⃣ Пошагово", callback_data="reasoning:scenario:2"),
            ],
            [
                InlineKeyboardButton("3️⃣ Промпт от модели", callback_data="reasoning:scenario:3"),
                InlineKeyboardButton("4️⃣ Группа экспертов", callback_data="reasoning:scenario:4"),
            ],
            [InlineKeyboardButton("5️⃣ Сравнить все способы", callback_data="reasoning:scenario:5")],
            [InlineKeyboardButton("❓ Новая задача", callback_data="reasoning:new_task")],
            [InlineKeyboardButton("🚪 Выйти из исследования", callback_data="reasoning:exit")],
        ]
    )


def _build_expert_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"🧑‍💼 {label}", callback_data=f"reasoning:expert:{expert_id}")]
            for expert_id, (label, _) in EXPERT_ROLES.items()
        ]
        + [[InlineKeyboardButton("↩️ К списку сценариев", callback_data="reasoning:expert_back")]]
    )


def _format_reasoning_answer(answer: str | None, finish_reason: str | None) -> str:
    """Готовит текст ответа сценария к отправке в Telegram."""
    text = answer or f"{MAIN_CLIENT_LABEL} вернул пустой ответ."
    if finish_reason == "length":
        text += "\n\n⚠️ ВНИМАНИЕ: Ответ был ОБРЕЗАН из-за ограничения max_tokens!"
    return text


def _format_scenario_stats(finish_reason: str | None, usage: dict[str, int] | None) -> str:
    """Минимальная статистика по итогам сценария: причина остановки и токены."""
    parts = [f"finish_reason={finish_reason or 'н/д'}"]
    if usage:
        parts.append(f"prompt_tokens={usage.get('prompt_tokens', 'н/д')}")
        parts.append(f"completion_tokens={usage.get('completion_tokens', 'н/д')}")
        parts.append(f"total_tokens={usage.get('total_tokens', 'н/д')}")
    return "📈 " + ", ".join(parts)


def _run_reasoning_scenario(
    handler_fn, task: str, extra: object = None
) -> tuple[str | None, str | None, str | None]:
    """Выполняет сценарий, возвращает (текст_ответа, текст_ошибки, статистика).

    При ошибке текст_ответа и статистика отсутствуют (None) — обращения к API не
    случилось или оно не завершилось валидным ответом.
    """
    try:
        answer, finish_reason, usage = handler_fn(task, extra)
    except AuthenticationError:
        logger.error(
            "Ошибка аутентификации %s API — проверьте %s.", MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR
        )
        return None, (
            f"❌ Ошибка авторизации на сервере {MAIN_CLIENT_LABEL}. "
            "Администратору бота нужно проверить API-ключ."
        ), None
    except RateLimitError:
        logger.warning("Превышен лимит запросов к %s API.", MAIN_CLIENT_LABEL)
        return None, (
            f"⏳ Сервис {MAIN_CLIENT_LABEL} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        ), None
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к %s API.", MAIN_CLIENT_LABEL)
        return None, f"⏳ {MAIN_CLIENT_LABEL} не ответил вовремя. Попробуй отправить запрос ещё раз.", None
    except APIConnectionError:
        logger.error("Не удалось подключиться к %s API.", MAIN_CLIENT_LABEL)
        return None, (
            f"🌐 Не получилось подключиться к серверу {MAIN_CLIENT_LABEL}. "
            "Проверь соединение и попробуй позже."
        ), None
    except APIStatusError as exc:
        logger.error("%s API вернул ошибку: %s", MAIN_CLIENT_LABEL, exc)
        return None, f"⚠️ Сервер {MAIN_CLIENT_LABEL} вернул ошибку. Попробуй позже.", None
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к %s API (исследование).", MAIN_CLIENT_LABEL)
        return None, "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже.", None

    formatted = _format_reasoning_answer(answer, finish_reason)
    stats = _format_scenario_stats(finish_reason, usage)
    return formatted, None, stats


async def _send_scenario_result(
    message,
    label: str,
    handler_fn,
    task: str,
    extra: object,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """Запускает сценарий и отправляет результат + статистику + клавиатуру дальше."""
    await message.chat.send_action(action=ChatAction.TYPING)
    result_text, error_text, stats_text = _run_reasoning_scenario(handler_fn, task, extra)

    await message.reply_text(f"📊 Способ: {label}")
    if error_text:
        await message.reply_text(error_text)
    else:
        for i in range(0, len(result_text), TELEGRAM_MESSAGE_LIMIT):
            await message.reply_text(result_text[i : i + TELEGRAM_MESSAGE_LIMIT])
        if stats_text:
            await message.reply_text(stats_text)

    await message.reply_text("👉 Что дальше?", reply_markup=reply_markup)


# --------------------------------------------------------------------------- #
# Обработчики диалога
# --------------------------------------------------------------------------- #


async def reasoning_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим исследования (/research_reasoning)."""
    await update.message.reply_text(REASONING_INTRO_TEXT)
    return WAITING_TASK


async def reasoning_receive_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сохраняет задачу для исследования и показывает выбор способа рассуждения."""
    task = update.message.text

    if not task or not task.strip():
        await update.message.reply_text("👉 Пожалуйста, отправь текст задачи.")
        return WAITING_TASK

    if len(task) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Задача слишком длинная ({len(task)} символов). Максимум — {MAX_INPUT_CHARS}."
        )
        return WAITING_TASK

    context.user_data["reasoning_task"] = task
    await update.message.reply_text(
        "Задача сохранена.\n\n👉 Выберите способ рассуждения:",
        reply_markup=_build_reasoning_scenario_keyboard(),
    )
    return CHOOSING_SCENARIO


async def reasoning_scenario_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки способа рассуждения / новая задача / выход."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    data = query.data

    if data == "reasoning:exit":
        context.user_data.pop("reasoning_task", None)
        await query.message.reply_text(
            "Исследование завершено. Можешь просто написать вопрос — отвечу в обычном режиме."
        )
        return ConversationHandler.END

    if data == "reasoning:new_task":
        await query.message.reply_text("👉 Введите новую задачу для исследования.")
        return WAITING_TASK

    task = context.user_data.get("reasoning_task")
    if not task:
        await query.message.reply_text(
            "Не найдена сохранённая задача. Начните заново командой /research_reasoning."
        )
        return ConversationHandler.END

    scenario_id = data.rsplit(":", 1)[-1]

    if scenario_id == "4":
        await query.message.reply_text(
            "👉 Выберите роль эксперта, от лица которого нужно решить задачу:",
            reply_markup=_build_expert_keyboard(),
        )
        return CHOOSING_EXPERT

    label, handler_fn = SCENARIO_HANDLERS[scenario_id]
    await _send_scenario_result(
        query.message, label, handler_fn, task, None, _build_reasoning_scenario_keyboard()
    )
    return CHOOSING_SCENARIO


async def reasoning_expert_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает выбор роли эксперта (сценарий 4) или возврат к списку сценариев."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    data = query.data

    if data == "reasoning:expert_back":
        await query.message.reply_text(
            "👉 Выберите способ рассуждения:", reply_markup=_build_reasoning_scenario_keyboard()
        )
        return CHOOSING_SCENARIO

    task = context.user_data.get("reasoning_task")
    if not task:
        await query.message.reply_text(
            "Не найдена сохранённая задача. Начните заново командой /research_reasoning."
        )
        return ConversationHandler.END

    expert_id = data.rsplit(":", 1)[-1]
    label = f"Группа экспертов — {EXPERT_ROLES[expert_id][0]}"
    await _send_scenario_result(
        query.message, label, call_reasoning_expert, task, expert_id, _build_expert_keyboard()
    )
    return CHOOSING_EXPERT


async def reasoning_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — принудительный выход из режима исследования."""
    context.user_data.pop("reasoning_task", None)
    await update.message.reply_text("Исследование прервано. Возвращаюсь в обычный режим.")
    return ConversationHandler.END


def build_reasoning_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler режима исследования рассуждений для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("research_reasoning", reasoning_command)],
        states={
            WAITING_TASK: [MessageHandler(text_filter, reasoning_receive_task)],
            CHOOSING_SCENARIO: [
                CallbackQueryHandler(reasoning_scenario_callback, pattern=r"^reasoning:"),
            ],
            CHOOSING_EXPERT: [
                CallbackQueryHandler(reasoning_expert_callback, pattern=r"^reasoning:"),
            ],
        },
        fallbacks=[CommandHandler("cancel", reasoning_cancel)],
    )
