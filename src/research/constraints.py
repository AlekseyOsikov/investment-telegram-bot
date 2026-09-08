"""Режим исследования (/research_constraints): сравнение сценариев ограничений API на ответ.

Отдельный от основного прокси-потока модуль, по аналогии с reasoning.py: это
техническое исследование влияния разных ограничений API — без ограничений (контроль),
формат ответа (`response_format`, JSON), `max_tokens`, стоп-слова (`stop`) — на форму
ответа, а не часть обычного сценария использования бота. В отличие от reasoning.py/
temperature.py/models.py, здесь используется инвестиционный `config.SYSTEM_PROMPT`, а не
нейтральный промпт: предмет исследования — сами ограничения API поверх штатного
инвестиционного ответа бота, а не поведение модели на произвольных задачах.

Сценарии 3 и 4 интерактивны: после выбора сценария бот дополнительно спрашивает
параметр (max_tokens / список стоп-слов) прямо у пользователя, а не берёт его из
фиксированной константы — это и есть предмет исследования в этих двух сценариях.
Оба ограничения выполняются на стороне API (через параметры запроса), а не
постобработкой ответа ботом — это экономит токены и время ответа.

Часть докстрингов/комментариев ниже описывает то, что было эмпирически найдено конкретно
на DeepSeek (например, что его API не поддерживает
`response_format={"type": "json_schema"}`) — сами сценарии всё равно идут через
выбранного `MAIN_CLIENT`, а не всегда через DeepSeek напрямую, и на другом провайдере
(Kimi) это конкретное ограничение не проверялось.

Перехват ошибок API и статистика по токенам вынесены в research/_shared.py (общие для
нескольких research-режимов, см. его докстринг) — здесь остаётся то, что специфично
именно для этого режима: сами 4 сценария вызова API и JSON-специфичное форматирование
ответа сценария 2 (проверка формата ответа).

main.py подключает фичу через build_constraints_conversation_handler() — единственную
точку интеграции с остальным приложением.
"""

from __future__ import annotations

import json
import logging

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
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    TELEGRAM_MESSAGE_LIMIT,
)
from providers.main_client import main_client

from ._shared import build_cancel_handler, content_or_reasoning_fallback, extract_usage, run_scenario

logger = logging.getLogger(__name__)

WAITING_QUESTION, CHOOSING_SCENARIO, WAITING_MAX_TOKENS, WAITING_STOP_WORDS = range(4)

# Лимит на количество стоп-последовательностей за один запрос — так было эмпирически
# найдено на DeepSeek (совпадает с лимитом OpenAI API); на Kimi отдельно не проверялось.
CONSTRAINTS_MAX_STOP_WORDS = 4

# Эмпирически найдено на DeepSeek: его API не поддерживает OpenAI-style
# response_format={"type": "json_schema"} (structured outputs, падает с 400 "This
# response_format type is unavailable now") — доступен только базовый JSON-режим
# {"type": "json_object"}, поэтому нужная структура ответа задаётся текстом прямо в
# пользовательском сообщении, а не схемой на стороне API; корректность и состав полей
# проверяются уже на стороне бота в _format_constraints_answer. На Kimi отдельно не
# проверялось — сценарий всё равно идёт через выбранного MAIN_CLIENT.
CONSTRAINTS_JSON_RESPONSE_FORMAT = {"type": "json_object"}

CONSTRAINTS_JSON_FIELDS = ["ticker", "company_name", "sector", "summary"]
CONSTRAINTS_JSON_LIST_KEY = "companies"

# Top-level должен оставаться JSON-объектом (этого требует response_format
# {"type": "json_object"}), поэтому список компаний оборачивается в один ключ,
# а не возвращается как «голый» массив.
CONSTRAINTS_JSON_INSTRUCTION = (
    "Ответь строго в формате JSON-объекта (только JSON, без markdown-разметки и "
    f'пояснений вне него) с единственным полем "{CONSTRAINTS_JSON_LIST_KEY}" — списком '
    "объектов, по одному объекту на каждую компанию/тикер из вопроса (если компания "
    "одна — список из одного элемента). Каждый объект должен содержать следующие и "
    "только следующие поля:\n"
    '- "ticker" (string) — биржевой тикер (как есть, латиницей)\n'
    '- "company_name" (string) — полное название компании\n'
    '- "sector" (string) — сектор/отрасль компании\n'
    '- "summary" (string) — краткая обосновывающая информация, максимум 100 слов\n'
    "Значения полей company_name, sector и summary должны быть на том же языке, "
    "на котором задан сам вопрос (ticker — всегда как есть, латиницей)."
)

CONSTRAINTS_INTRO_TEXT = (
    f"🔬 Режим исследования влияния ограничений {MAIN_CLIENT_LABEL} API (включая формат "
    "ответа) на ответ.\n\n"
    "⚠️ Напоминание: бот не является лицензированным финансовым советником, а ответы в "
    "этом режиме — часть технического исследования, а не инвестиционная рекомендация.\n\n"
    "👉 Введите вопрос для исследования влияния ограничений на ответ."
)


# --------------------------------------------------------------------------- #
# 4 сценария вызова API
# --------------------------------------------------------------------------- #


def call_constraint_no_restrictions(
    question: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 1: обычный запрос без дополнительных ограничений (контроль)."""
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason, extract_usage(response)


def call_constraint_json_schema(
    question: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 2: проверка формата ответа — JSON-режим со структурой, заданной в промпте."""
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\n{CONSTRAINTS_JSON_INSTRUCTION}"},
        ],
        response_format=CONSTRAINTS_JSON_RESPONSE_FORMAT,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason, extract_usage(response)


def call_constraint_max_tokens(
    question: str, max_tokens: int
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 3: жёсткое ограничение длины ответа через API-параметр max_tokens.

    Ограничение выполняется на стороне API (модель прекращает генерацию по достижении
    лимита), а не постобработкой полного ответа ботом — это экономит токены и время по
    сравнению с обрезкой уже сгенерированного текста. У моделей с рассуждениями весь
    лимит может уйти на скрытые размышления, оставляя видимый content пустым —
    content_or_reasoning_fallback подставляет в этом случае обрезанный reasoning_content
    вместо пустоты (см. её докстринг в research/_shared.py).
    """
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    content = content_or_reasoning_fallback(choice.message)
    return content, choice.finish_reason, extract_usage(response)


def call_constraint_stop_sequence(
    question: str, stop_words: list[str]
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 4: остановка генерации на первом совпадении со стоп-словом пользователя."""
    response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stop=stop_words,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason, extract_usage(response)


SCENARIO_HANDLERS = {
    "1": ("Без ограничений", call_constraint_no_restrictions),
    "2": ("Формат ответа: JSON-режим (структура задана в промпте)", call_constraint_json_schema),
    "3": ("Ограничение max_tokens", call_constraint_max_tokens),
    "4": ("Свои стоп-слова", call_constraint_stop_sequence),
}


# --------------------------------------------------------------------------- #
# Вспомогательные функции: клавиатура, форматирование, запуск сценария
# --------------------------------------------------------------------------- #


def _build_constraints_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1️⃣ Без ограничений", callback_data="constraints:scenario:1"),
                InlineKeyboardButton("2️⃣ Формат ответа (JSON)", callback_data="constraints:scenario:2"),
            ],
            [
                InlineKeyboardButton("3️⃣ Max tokens", callback_data="constraints:scenario:3"),
                InlineKeyboardButton("4️⃣ Свои стоп-слова", callback_data="constraints:scenario:4"),
            ],
            [InlineKeyboardButton("❓ Новый вопрос", callback_data="constraints:new_question")],
            [InlineKeyboardButton("🚪 Выйти из исследования", callback_data="constraints:exit")],
        ]
    )


def _strip_code_fence(text: str) -> str:
    """Снимает ```json ... ``` обёртку, если модель прислала её вопреки инструкции."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text.removeprefix("```")
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _format_constraints_answer(scenario_id: str, answer: str | None, finish_reason: str | None) -> str:
    """Готовит текст ответа сценария к отправке в Telegram."""
    if scenario_id == "2":
        if not answer:
            return f"⚠️ Введённый запрос не может быть обработан: {MAIN_CLIENT_LABEL} вернул пустой ответ."
        try:
            parsed = json.loads(_strip_code_fence(answer))
        except (json.JSONDecodeError, TypeError):
            parsed = None

        invalid_structure = (
            f"⚠️ Введённый запрос не может быть обработан: {MAIN_CLIENT_LABEL} вернул невалидный "
            "JSON для заданной структуры. Попробуйте переформулировать вопрос."
        )
        if not isinstance(parsed, dict):
            return invalid_structure
        companies = parsed.get(CONSTRAINTS_JSON_LIST_KEY)
        if not isinstance(companies, list) or not companies:
            return invalid_structure

        for company in companies:
            if not isinstance(company, dict):
                return invalid_structure
            missing_fields = [f for f in CONSTRAINTS_JSON_FIELDS if f not in company]
            if missing_fields:
                return (
                    f"⚠️ Введённый запрос не может быть обработан: {MAIN_CLIENT_LABEL} вернул JSON без "
                    f"обязательных полей ({', '.join(missing_fields)}). "
                    "Попробуйте переформулировать вопрос."
                )

        return "```json\n" + json.dumps(parsed, ensure_ascii=False, indent=2) + "\n```"

    text = answer or f"{MAIN_CLIENT_LABEL} вернул пустой ответ."
    if finish_reason == "length":
        text += "\n\n⚠️ ВНИМАНИЕ: Ответ был ОБРЕЗАН из-за ограничения max_tokens!"
    return text


def _run_constraints_scenario(
    scenario_id: str, question: str, extra: object = None
) -> tuple[str | None, str | None, str | None]:
    """Выполняет сценарий, возвращает (текст_ответа, текст_ошибки, статистика)."""
    _, handler_fn = SCENARIO_HANDLERS[scenario_id]
    return run_scenario(
        handler_fn,
        question,
        extra,
        format_answer=lambda answer, finish_reason: _format_constraints_answer(
            scenario_id, answer, finish_reason
        ),
    )


async def _send_scenario_result(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    scenario_id: str,
    question: str,
    extra: object,
    label_suffix: str = "",
) -> None:
    """Запускает сценарий и отправляет результат + статистику + клавиатуру дальше."""
    label = SCENARIO_HANDLERS[scenario_id][0]
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    result_text, error_text, stats_text = _run_constraints_scenario(scenario_id, question, extra)

    await message.reply_text(f"📊 Сценарий: {label}{label_suffix}")
    if error_text:
        await message.reply_text(error_text)
    else:
        for i in range(0, len(result_text), TELEGRAM_MESSAGE_LIMIT):
            await message.reply_text(result_text[i : i + TELEGRAM_MESSAGE_LIMIT])
        if stats_text:
            await message.reply_text(stats_text)

    await message.reply_text(
        "👉 Выберите следующий сценарий или завершите исследование:",
        reply_markup=_build_constraints_keyboard(),
    )


# --------------------------------------------------------------------------- #
# Обработчики диалога
# --------------------------------------------------------------------------- #


async def constraints_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим исследования (/research_constraints)."""
    await update.message.reply_text(CONSTRAINTS_INTRO_TEXT)
    return WAITING_QUESTION


async def constraints_receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сохраняет вопрос для исследования и показывает выбор сценария."""
    question = update.message.text

    if not question or not question.strip():
        await update.message.reply_text("👉 Пожалуйста, отправь текстовый вопрос.")
        return WAITING_QUESTION

    if len(question) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            "⚠️ Вопрос слишком длинный "
            f"({len(question)} символов). Максимум — {MAX_INPUT_CHARS}."
        )
        return WAITING_QUESTION

    context.user_data["constraints_question"] = question
    await update.message.reply_text(
        f"Вопрос сохранён.\n\n👉 Выберите сценарий запроса к {MAIN_CLIENT_LABEL}:",
        reply_markup=_build_constraints_keyboard(),
    )
    return CHOOSING_SCENARIO


async def constraints_scenario_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки сценария / новый вопрос / выход."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    data = query.data

    if data == "constraints:exit":
        context.user_data.pop("constraints_question", None)
        await query.message.reply_text(
            "Исследование завершено. Можешь просто написать вопрос — отвечу в обычном режиме."
        )
        return ConversationHandler.END

    if data == "constraints:new_question":
        await query.message.reply_text(
            "👉 Введите новый вопрос для исследования влияния ограничений на ответ."
        )
        return WAITING_QUESTION

    question = context.user_data.get("constraints_question")
    if not question:
        await query.message.reply_text(
            "Не найден сохранённый вопрос для исследования. Начните заново командой /research_constraints."
        )
        return ConversationHandler.END

    scenario_id = data.rsplit(":", 1)[-1]

    if scenario_id == "3":
        await query.message.reply_text(
            "👉 Введите максимальное количество токенов ответа (max_tokens, целое число):"
        )
        return WAITING_MAX_TOKENS

    if scenario_id == "4":
        await query.message.reply_text(
            "👉 Введите стоп-слова через запятую "
            f"(не более {CONSTRAINTS_MAX_STOP_WORDS}), например: точка, конец"
        )
        return WAITING_STOP_WORDS

    await _send_scenario_result(query.message, context, scenario_id, question, extra=None)
    return CHOOSING_SCENARIO


async def constraints_max_tokens_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает значение max_tokens для сценария 3 и запускает его."""
    text = (update.message.text or "").strip()
    if not text.isdigit() or int(text) <= 0:
        await update.message.reply_text(
            "👉 Пожалуйста, введите положительное целое число (например, 50)."
        )
        return WAITING_MAX_TOKENS

    question = context.user_data.get("constraints_question")
    if not question:
        await update.message.reply_text(
            "Не найден сохранённый вопрос для исследования. Начните заново командой /research_constraints."
        )
        return ConversationHandler.END

    max_tokens = int(text)
    await _send_scenario_result(
        update.message,
        context,
        "3",
        question,
        extra=max_tokens,
        label_suffix=f" (max_tokens={max_tokens})",
    )
    return CHOOSING_SCENARIO


async def constraints_stop_words_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает список стоп-слов для сценария 4 и запускает его."""
    raw_text = update.message.text or ""
    stop_words = [w.strip() for w in raw_text.split(",") if w.strip()]

    if not stop_words:
        await update.message.reply_text(
            "👉 Пожалуйста, укажите хотя бы одно стоп-слово (через запятую, если их несколько)."
        )
        return WAITING_STOP_WORDS

    if len(stop_words) > CONSTRAINTS_MAX_STOP_WORDS:
        await update.message.reply_text(
            f"⚠️ {MAIN_CLIENT_LABEL} API поддерживает не более {CONSTRAINTS_MAX_STOP_WORDS} стоп-последовательностей.\n"
            f"👉 Укажите не более {CONSTRAINTS_MAX_STOP_WORDS} через запятую."
        )
        return WAITING_STOP_WORDS

    question = context.user_data.get("constraints_question")
    if not question:
        await update.message.reply_text(
            "Не найден сохранённый вопрос для исследования. Начните заново командой /research_constraints."
        )
        return ConversationHandler.END

    await _send_scenario_result(
        update.message,
        context,
        "4",
        question,
        extra=stop_words,
        label_suffix=f" ({', '.join(stop_words)})",
    )
    return CHOOSING_SCENARIO


constraints_cancel = build_cancel_handler("constraints_question")


def build_constraints_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler режима исследования для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("research_constraints", constraints_command)],
        states={
            WAITING_QUESTION: [MessageHandler(text_filter, constraints_receive_question)],
            CHOOSING_SCENARIO: [
                CallbackQueryHandler(constraints_scenario_callback, pattern=r"^constraints:"),
            ],
            WAITING_MAX_TOKENS: [MessageHandler(text_filter, constraints_max_tokens_input)],
            WAITING_STOP_WORDS: [MessageHandler(text_filter, constraints_stop_words_input)],
        },
        fallbacks=[CommandHandler("cancel", constraints_cancel)],
    )
