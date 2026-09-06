"""Режим исследования (/research_response_format): сравнение сценариев ограничений DeepSeek API.

Отдельный от основного прокси-потока модуль: это техническое исследование влияния
параметров DeepSeek API (response_format, max_tokens, stop) на форму ответа, а не
часть обычного сценария использования бота.

Сценарии 3 и 4 интерактивны: после выбора сценария бот дополнительно спрашивает
параметр (max_tokens / список стоп-слов) прямо у пользователя, а не берёт его из
фиксированной константы — это и есть предмет исследования в этих двух сценариях.
Оба ограничения выполняются на стороне DeepSeek (через параметры запроса), а не
постобработкой ответа ботом — это экономит токены и время ответа.

main.py подключает фичу через build_conversation_handler() — единственную точку
интеграции с остальным приложением.
"""

from __future__ import annotations

import json
import logging

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
    DEEPSEEK_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    TELEGRAM_MESSAGE_LIMIT,
    deepseek_client,
)

logger = logging.getLogger(__name__)

WAITING_QUESTION, CHOOSING_SCENARIO, WAITING_MAX_TOKENS, WAITING_STOP_WORDS = range(4)

# DeepSeek (как и OpenAI) принимает не более 4 стоп-последовательностей за запрос.
RESEARCH_MAX_STOP_WORDS = 4

# DeepSeek API не поддерживает response_format={"type": "json_schema"} (structured
# outputs из OpenAI) — запрос с ним падает с ошибкой 400 "This response_format type
# is unavailable now". Доступен только базовый JSON-режим {"type": "json_object"},
# поэтому нужная структура ответа задаётся текстом прямо в пользовательском
# сообщении, а не схемой на стороне API; корректность и состав полей проверяются
# уже на стороне бота в _format_research_answer.
RESEARCH_JSON_RESPONSE_FORMAT = {"type": "json_object"}

RESEARCH_JSON_FIELDS = ["ticker", "company_name", "sector", "summary"]
RESEARCH_JSON_LIST_KEY = "companies"

# Top-level должен оставаться JSON-объектом (этого требует response_format
# {"type": "json_object"}), поэтому список компаний оборачивается в один ключ,
# а не возвращается как «голый» массив.
RESEARCH_JSON_INSTRUCTION = (
    "Ответь строго в формате JSON-объекта (только JSON, без markdown-разметки и "
    f'пояснений вне него) с единственным полем "{RESEARCH_JSON_LIST_KEY}" — списком '
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

RESEARCH_INTRO_TEXT = (
    "🔬 Режим исследования влияния ограничений DeepSeek API на ответ.\n\n"
    "⚠️ Напоминание: бот не является лицензированным финансовым советником, а ответы в "
    "этом режиме — часть технического исследования, а не инвестиционная рекомендация.\n\n"
    "👉 Введите вопрос для исследования влияния ограничений на ответ."
)


# --------------------------------------------------------------------------- #
# 4 сценария вызова DeepSeek API
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


def call_deepseek_no_restrictions(
    question: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 1: обычный запрос без дополнительных ограничений (контроль)."""
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason, _extract_usage(response)


def call_deepseek_json_schema(
    question: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 2: JSON-режим DeepSeek со структурой, заданной в промпте."""
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\n{RESEARCH_JSON_INSTRUCTION}"},
        ],
        response_format=RESEARCH_JSON_RESPONSE_FORMAT,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason, _extract_usage(response)


def call_deepseek_max_tokens(
    question: str, max_tokens: int
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 3: жёсткое ограничение длины ответа через API-параметр max_tokens.

    Ограничение выполняется на стороне DeepSeek (модель прекращает генерацию по
    достижении лимита), а не постобработкой полного ответа ботом — это экономит
    токены и время по сравнению с обрезкой уже сгенерированного текста.
    """
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        # У моделей с рассуждениями (например, deepseek-reasoner) весь лимит
        # max_tokens может уйти на скрытые рассуждения (reasoning_content), не
        # дойдя до видимого ответа: content пуст, а finish_reason всё равно
        # "length". Показываем обрезанные рассуждения вместо пустоты — обрезка
        # всё так же видна пользователю, просто на уровне "мыслей" модели.
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            content = "[Модель ещё не начала видимый ответ, вот её рассуждения]\n\n" + reasoning
    return content, choice.finish_reason, _extract_usage(response)


def call_deepseek_stop_sequence(
    question: str, stop_words: list[str]
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 4: остановка генерации на первом совпадении со стоп-словом пользователя."""
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stop=stop_words,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason, _extract_usage(response)


SCENARIO_HANDLERS = {
    "1": ("Без ограничений", call_deepseek_no_restrictions),
    "2": ("JSON-режим (структура задана в промпте)", call_deepseek_json_schema),
    "3": ("Ограничение max_tokens", call_deepseek_max_tokens),
    "4": ("Свои стоп-слова", call_deepseek_stop_sequence),
}


# --------------------------------------------------------------------------- #
# Вспомогательные функции: клавиатура, форматирование, запуск сценария
# --------------------------------------------------------------------------- #


def _build_research_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1️⃣ Без ограничений", callback_data="research:scenario:1"),
                InlineKeyboardButton("2️⃣ JSON-режим", callback_data="research:scenario:2"),
            ],
            [
                InlineKeyboardButton("3️⃣ Max tokens", callback_data="research:scenario:3"),
                InlineKeyboardButton("4️⃣ Свои стоп-слова", callback_data="research:scenario:4"),
            ],
            [InlineKeyboardButton("❓ Новый вопрос", callback_data="research:new_question")],
            [InlineKeyboardButton("🚪 Выйти из исследования", callback_data="research:exit")],
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


def _format_research_answer(scenario_id: str, answer: str | None, finish_reason: str | None) -> str:
    """Готовит текст ответа сценария к отправке в Telegram."""
    if scenario_id == "2":
        if not answer:
            return "⚠️ Введённый запрос не может быть обработан: DeepSeek вернул пустой ответ."
        try:
            parsed = json.loads(_strip_code_fence(answer))
        except (json.JSONDecodeError, TypeError):
            parsed = None

        invalid_structure = (
            "⚠️ Введённый запрос не может быть обработан: DeepSeek вернул невалидный "
            "JSON для заданной структуры. Попробуйте переформулировать вопрос."
        )
        if not isinstance(parsed, dict):
            return invalid_structure
        companies = parsed.get(RESEARCH_JSON_LIST_KEY)
        if not isinstance(companies, list) or not companies:
            return invalid_structure

        for company in companies:
            if not isinstance(company, dict):
                return invalid_structure
            missing_fields = [f for f in RESEARCH_JSON_FIELDS if f not in company]
            if missing_fields:
                return (
                    "⚠️ Введённый запрос не может быть обработан: DeepSeek вернул JSON без "
                    f"обязательных полей ({', '.join(missing_fields)}). "
                    "Попробуйте переформулировать вопрос."
                )

        return "```json\n" + json.dumps(parsed, ensure_ascii=False, indent=2) + "\n```"

    text = answer or "DeepSeek вернул пустой ответ."
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


def _run_research_scenario(
    scenario_id: str, question: str, extra: object = None
) -> tuple[str | None, str | None, str | None]:
    """Выполняет сценарий, возвращает (текст_ответа, текст_ошибки, статистика).

    При ошибке текст_ответа и статистика отсутствуют (None) — обращения к API не
    случилось или оно не завершилось валидным ответом.
    """
    _, handler_fn = SCENARIO_HANDLERS[scenario_id]
    try:
        answer, finish_reason, usage = handler_fn(question, extra)
    except AuthenticationError:
        logger.error("Ошибка аутентификации DeepSeek API — проверьте DEEPSEEK_API_KEY.")
        return None, (
            "❌ Ошибка авторизации на сервере DeepSeek. "
            "Администратору бота нужно проверить API-ключ."
        ), None
    except RateLimitError:
        logger.warning("Превышен лимит запросов к DeepSeek API.")
        return None, (
            "⏳ Сервис DeepSeek временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        ), None
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к DeepSeek API.")
        return None, "⏳ DeepSeek не ответил вовремя. Попробуй отправить запрос ещё раз.", None
    except APIConnectionError:
        logger.error("Не удалось подключиться к DeepSeek API.")
        return None, (
            "🌐 Не получилось подключиться к серверу DeepSeek. "
            "Проверь соединение и попробуй позже."
        ), None
    except APIStatusError as exc:
        logger.error("DeepSeek API вернул ошибку: %s", exc)
        return None, "⚠️ Сервер DeepSeek вернул ошибку. Попробуй позже.", None
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к DeepSeek API (исследование).")
        return None, "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже.", None

    formatted = _format_research_answer(scenario_id, answer, finish_reason)
    stats = _format_scenario_stats(finish_reason, usage)
    return formatted, None, stats


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
    result_text, error_text, stats_text = _run_research_scenario(scenario_id, question, extra)

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
        reply_markup=_build_research_keyboard(),
    )


# --------------------------------------------------------------------------- #
# Обработчики диалога
# --------------------------------------------------------------------------- #


async def research_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим исследования (/research_response_format)."""
    await update.message.reply_text(RESEARCH_INTRO_TEXT)
    return WAITING_QUESTION


async def research_receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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

    context.user_data["research_question"] = question
    await update.message.reply_text(
        "Вопрос сохранён.\n\n👉 Выберите сценарий запроса к DeepSeek:",
        reply_markup=_build_research_keyboard(),
    )
    return CHOOSING_SCENARIO


async def research_scenario_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки сценария / новый вопрос / выход."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    data = query.data

    if data == "research:exit":
        context.user_data.pop("research_question", None)
        await query.message.reply_text(
            "Исследование завершено. Можешь просто написать вопрос — отвечу в обычном режиме."
        )
        return ConversationHandler.END

    if data == "research:new_question":
        await query.message.reply_text(
            "👉 Введите новый вопрос для исследования влияния ограничений на ответ."
        )
        return WAITING_QUESTION

    question = context.user_data.get("research_question")
    if not question:
        await query.message.reply_text(
            "Не найден сохранённый вопрос для исследования. Начните заново командой /research_response_format."
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
            f"(не более {RESEARCH_MAX_STOP_WORDS}), например: точка, конец"
        )
        return WAITING_STOP_WORDS

    await _send_scenario_result(query.message, context, scenario_id, question, extra=None)
    return CHOOSING_SCENARIO


async def research_max_tokens_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает значение max_tokens для сценария 3 и запускает его."""
    text = (update.message.text or "").strip()
    if not text.isdigit() or int(text) <= 0:
        await update.message.reply_text(
            "👉 Пожалуйста, введите положительное целое число (например, 50)."
        )
        return WAITING_MAX_TOKENS

    question = context.user_data.get("research_question")
    if not question:
        await update.message.reply_text(
            "Не найден сохранённый вопрос для исследования. Начните заново командой /research_response_format."
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


async def research_stop_words_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает список стоп-слов для сценария 4 и запускает его."""
    raw_text = update.message.text or ""
    stop_words = [w.strip() for w in raw_text.split(",") if w.strip()]

    if not stop_words:
        await update.message.reply_text(
            "👉 Пожалуйста, укажите хотя бы одно стоп-слово (через запятую, если их несколько)."
        )
        return WAITING_STOP_WORDS

    if len(stop_words) > RESEARCH_MAX_STOP_WORDS:
        await update.message.reply_text(
            f"⚠️ DeepSeek API поддерживает не более {RESEARCH_MAX_STOP_WORDS} стоп-последовательностей.\n"
            f"👉 Укажите не более {RESEARCH_MAX_STOP_WORDS} через запятую."
        )
        return WAITING_STOP_WORDS

    question = context.user_data.get("research_question")
    if not question:
        await update.message.reply_text(
            "Не найден сохранённый вопрос для исследования. Начните заново командой /research_response_format."
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


async def research_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — принудительный выход из режима исследования."""
    context.user_data.pop("research_question", None)
    await update.message.reply_text("Исследование прервано. Возвращаюсь в обычный режим.")
    return ConversationHandler.END


def build_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler режима исследования для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("research_response_format", research_command)],
        states={
            WAITING_QUESTION: [MessageHandler(text_filter, research_receive_question)],
            CHOOSING_SCENARIO: [
                CallbackQueryHandler(research_scenario_callback, pattern=r"^research:"),
            ],
            WAITING_MAX_TOKENS: [MessageHandler(text_filter, research_max_tokens_input)],
            WAITING_STOP_WORDS: [MessageHandler(text_filter, research_stop_words_input)],
        },
        fallbacks=[CommandHandler("cancel", research_cancel)],
    )
