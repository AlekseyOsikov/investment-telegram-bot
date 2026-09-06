"""Режим исследования (/research_temperature): сравнение температур DeepSeek API.

Отдельный от основного прокси-потока модуль, по аналогии с research_reasoning.py: это
техническое исследование того, как параметр temperature влияет на форму и содержание
ответа на произвольную задачу (не обязательно инвестиционную), а не часть обычного
сценария использования бота. Поэтому системный промпт здесь, как и в
research_reasoning.py, намеренно нейтральный, а не investment-специфичный
config.SYSTEM_PROMPT — задачи в этом режиме могут быть любыми.

Сценарии 1-4 отправляют одну и ту же задачу с одним и тем же промптом при разных
фиксированных значениях temperature — 0, 0.7, 1.2 и 2 (весь допустимый для DeepSeek/
OpenAI API диапазон [0, 2]). Значения не запрашиваются у пользователя (в отличие от
max_tokens/стоп-слов в research_response_format.py), это предмет исследования, а не
настраиваемый параметр.

Сценарий 5 запускает сценарии 1-4 параллельно в пуле потоков, затем одним отдельным
запросом просит модель сравнить полученные ответы между собой по точности,
креативности и разнообразию и сформулировать рекомендации, для каких задач лучше
подходит каждое значение temperature — показывает и все исходные ответы, и итоговое
сравнение.

main.py подключает фичу через build_temperature_conversation_handler() — единственную
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
    DEEPSEEK_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    TELEGRAM_MESSAGE_LIMIT,
    deepseek_client,
)

logger = logging.getLogger(__name__)

WAITING_TASK, CHOOSING_SCENARIO = range(2)

# Этот режим сравнивает влияние temperature на уже сгенерированный видимый ответ, а не
# собственное скрытое рассуждение модели — оно не только не нужно для сравнения, но и
# мешает: на моделях с рассуждениями весь max_tokens может уйти на reasoning_content,
# оставляя видимый ответ пустым (см. _content_or_reasoning_fallback). "thinking" не
# входит в типизированную сигнатуру chat.completions.create в openai SDK, поэтому
# передаётся через extra_body — как и в research_reasoning.py.
DISABLE_THINKING = {"thinking": {"type": "disabled"}}

# Нарочно нейтральный системный промпт: этот режим исследует влияние temperature на
# произвольных задачах, а не отвечает на инвестиционные вопросы, поэтому здесь не
# используется config.SYSTEM_PROMPT с его инвестиционной ролью и дисклеймерами.
TEMPERATURE_SYSTEM_PROMPT = (
    "Ты — ассистент, который решает поставленную пользователем задачу. Отвечай по "
    "существу. Определи язык, на котором сформулирована задача, и веди видимый ответ "
    "(поле content) строго на этом же языке от первого до последнего слова."
)

TEMPERATURE_INTRO_TEXT = (
    "🔬 Режим исследования влияния temperature на ответ DeepSeek API.\n\n"
    "Это техническое исследование, а не инвестиционный сервис — задача может быть любой, "
    "не обязательно про инвестиции.\n\n"
    "👉 Введите задачу, которую нужно решить."
)


# --------------------------------------------------------------------------- #
# 5 сценариев вызова DeepSeek API
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

    На моделях с рассуждениями (deepseek-reasoner и т.п.) весь лимит max_tokens может
    целиком уйти на скрытые размышления, оставляя видимый content пустым при
    finish_reason == "length" — тот же случай, что и в research_reasoning.py.
    """
    content = message.content
    if not content:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return (
                "[Модель ещё не начала видимый ответ, вот её рассуждения]\n\n" + reasoning
            )
    return content


def _sum_usage(
    first: dict[str, int] | None, second: dict[str, int] | None
) -> dict[str, int] | None:
    """Складывает статистику по токенам двух вызовов API (для сценария 5)."""
    if first is None and second is None:
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a, b = (first or {}).get(key), (second or {}).get(key)
        if a is None and b is None:
            continue
        result[key] = (a or 0) + (b or 0)
    return result or None


def call_temperature(
    task: str, temperature: float
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Запрашивает решение задачи у DeepSeek с заданным значением temperature."""
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": TEMPERATURE_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=DISABLE_THINKING,
    )
    choice = response.choices[0]
    return _content_or_reasoning_fallback(choice.message), choice.finish_reason, _extract_usage(response)


def call_temperature_0(task: str, _extra: object = None) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 1: temperature = 0 — детерминированный, наиболее предсказуемый ответ."""
    return call_temperature(task, 0)


def call_temperature_0_7(task: str, _extra: object = None) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 2: temperature = 0.7 — сбалансированное значение по умолчанию."""
    return call_temperature(task, 0.7)


def call_temperature_1_2(task: str, _extra: object = None) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 3: temperature = 1.2 — повышенная случайность/креативность."""
    return call_temperature(task, 1.2)


def call_temperature_2(task: str, _extra: object = None) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Сценарий 4: temperature = 2 — максимум допустимого диапазона API."""
    return call_temperature(task, 2)


# Подзадачи сценария 5 — те же вызовы, что стоят за сценариями 1-4, запускаются
# параллельно в пуле потоков.
_COMPARE_ALL_SUB_SCENARIOS: list[tuple[str, object, object]] = [
    ("temperature = 0", call_temperature_0, None),
    ("temperature = 0.7", call_temperature_0_7, None),
    ("temperature = 1.2", call_temperature_1_2, None),
    ("temperature = 2", call_temperature_2, None),
]

COMPARE_ALL_INSTRUCTION = (
    "Ниже приведены ответы на одну и ту же задачу, полученные при разных значениях "
    "параметра temperature (0, 0.7, 1.2 и 2) при прочих равных условиях. Сравни их по "
    "трём критериям:\n"
    "1. Точность — где ответ фактически точнее, последовательнее и меньше похож на "
    "галлюцинацию.\n"
    "2. Креативность — где ответ более оригинальный и неожиданный по содержанию или "
    "формулировкам.\n"
    "3. Разнообразие — насколько ответы вообще отличаются друг от друга по содержанию, "
    "структуре и формулировкам.\n"
    "Затем сформулируй рекомендации: для каких типов задач лучше подходит каждое из "
    "значений temperature (0 / 0.7 / 1.2 / 2)."
)


def call_temperature_compare_all(
    task: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Способ 5: параллельно запускает температуры 1-4 и просит модель сравнить ответы.

    Каждая подзадача выполняется в своём потоке (обращения к API — блокирующий HTTP-
    вызов openai SDK, поэтому распараллеливание через ThreadPoolExecutor, а не asyncio,
    соответствует тому, как остальные сценарии этого модуля уже вызывают DeepSeek
    синхронно). Отдельная ошибка одной подзадачи не прерывает сравнение — вместо ответа
    этой подзадачи в сравнение уходит пометка об ошибке. Если не удалась ни одна
    подзадача, до финального сравнительного запроса дело всё равно доходит: модель
    увидит только пометки об ошибках и напишет об этом в сравнении.
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
    comparison_response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": TEMPERATURE_SYSTEM_PROMPT},
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
        "🧩 Сравнение и рекомендации:\n" + (comparison_answer or "DeepSeek вернул пустой ответ.")
    )
    combined_answer = "\n\n".join(result_parts)

    usage = _extract_usage(comparison_response)
    for sub_usage in sub_usages:
        usage = _sum_usage(usage, sub_usage)
    return combined_answer, choice.finish_reason, usage


SCENARIO_HANDLERS = {
    "1": ("temperature = 0", call_temperature_0),
    "2": ("temperature = 0.7", call_temperature_0_7),
    "3": ("temperature = 1.2", call_temperature_1_2),
    "4": ("temperature = 2", call_temperature_2),
    "5": ("Сравнение всех температур", call_temperature_compare_all),
}


# --------------------------------------------------------------------------- #
# Вспомогательные функции: клавиатура, форматирование, запуск сценария
# --------------------------------------------------------------------------- #


def _build_temperature_scenario_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1️⃣ temperature = 0", callback_data="temperature:scenario:1"),
                InlineKeyboardButton("2️⃣ temperature = 0.7", callback_data="temperature:scenario:2"),
            ],
            [
                InlineKeyboardButton("3️⃣ temperature = 1.2", callback_data="temperature:scenario:3"),
                InlineKeyboardButton("4️⃣ temperature = 2", callback_data="temperature:scenario:4"),
            ],
            [InlineKeyboardButton("5️⃣ Сравнить все температуры", callback_data="temperature:scenario:5")],
            [InlineKeyboardButton("❓ Новая задача", callback_data="temperature:new_task")],
            [InlineKeyboardButton("🚪 Выйти из исследования", callback_data="temperature:exit")],
        ]
    )


def _format_temperature_answer(answer: str | None, finish_reason: str | None) -> str:
    """Готовит текст ответа сценария к отправке в Telegram."""
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


def _run_temperature_scenario(
    handler_fn, task: str, extra: object = None
) -> tuple[str | None, str | None, str | None]:
    """Выполняет сценарий, возвращает (текст_ответа, текст_ошибки, статистика).

    При ошибке текст_ответа и статистика отсутствуют (None) — обращения к API не
    случилось или оно не завершилось валидным ответом.
    """
    try:
        answer, finish_reason, usage = handler_fn(task, extra)
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

    formatted = _format_temperature_answer(answer, finish_reason)
    stats = _format_scenario_stats(finish_reason, usage)
    return formatted, None, stats


async def _send_scenario_result(
    message,
    label: str,
    handler_fn,
    task: str,
    extra: object,
) -> None:
    """Запускает сценарий и отправляет результат + статистику + клавиатуру дальше."""
    await message.chat.send_action(action=ChatAction.TYPING)
    result_text, error_text, stats_text = _run_temperature_scenario(handler_fn, task, extra)

    await message.reply_text(f"📊 Сценарий: {label}")
    if error_text:
        await message.reply_text(error_text)
    else:
        for i in range(0, len(result_text), TELEGRAM_MESSAGE_LIMIT):
            await message.reply_text(result_text[i : i + TELEGRAM_MESSAGE_LIMIT])
        if stats_text:
            await message.reply_text(stats_text)

    await message.reply_text(
        "👉 Что дальше?", reply_markup=_build_temperature_scenario_keyboard()
    )


# --------------------------------------------------------------------------- #
# Обработчики диалога
# --------------------------------------------------------------------------- #


async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим исследования (/research_temperature)."""
    await update.message.reply_text(TEMPERATURE_INTRO_TEXT)
    return WAITING_TASK


async def temperature_receive_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сохраняет задачу для исследования и показывает выбор значения temperature."""
    task = update.message.text

    if not task or not task.strip():
        await update.message.reply_text("👉 Пожалуйста, отправь текст задачи.")
        return WAITING_TASK

    if len(task) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Задача слишком длинная ({len(task)} символов). Максимум — {MAX_INPUT_CHARS}."
        )
        return WAITING_TASK

    context.user_data["temperature_task"] = task
    await update.message.reply_text(
        "Задача сохранена.\n\n👉 Выберите значение temperature:",
        reply_markup=_build_temperature_scenario_keyboard(),
    )
    return CHOOSING_SCENARIO


async def temperature_scenario_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки сценария / новая задача / выход."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    data = query.data

    if data == "temperature:exit":
        context.user_data.pop("temperature_task", None)
        await query.message.reply_text(
            "Исследование завершено. Можешь просто написать вопрос — отвечу в обычном режиме."
        )
        return ConversationHandler.END

    if data == "temperature:new_task":
        await query.message.reply_text("👉 Введите новую задачу для исследования.")
        return WAITING_TASK

    task = context.user_data.get("temperature_task")
    if not task:
        await query.message.reply_text(
            "Не найдена сохранённая задача. Начните заново командой /research_temperature."
        )
        return ConversationHandler.END

    scenario_id = data.rsplit(":", 1)[-1]
    label, handler_fn = SCENARIO_HANDLERS[scenario_id]
    await _send_scenario_result(query.message, label, handler_fn, task, None)
    return CHOOSING_SCENARIO


async def temperature_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — принудительный выход из режима исследования."""
    context.user_data.pop("temperature_task", None)
    await update.message.reply_text("Исследование прервано. Возвращаюсь в обычный режим.")
    return ConversationHandler.END


def build_temperature_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler режима исследования temperature для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("research_temperature", temperature_command)],
        states={
            WAITING_TASK: [MessageHandler(text_filter, temperature_receive_task)],
            CHOOSING_SCENARIO: [
                CallbackQueryHandler(temperature_scenario_callback, pattern=r"^temperature:"),
            ],
        },
        fallbacks=[CommandHandler("cancel", temperature_cancel)],
    )
