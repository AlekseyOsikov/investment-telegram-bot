"""Режим исследования (/research_models): сравнение моделей DeepSeek и Kimi.

Отдельный от основного прокси-потока модуль, по аналогии с research_reasoning.py и
research_temperature.py: техническое исследование того, как разные модели отвечают на
одну и ту же произвольную задачу (не обязательно инвестиционную), а не часть обычного
сценария использования бота. Системный промпт, как и в этих двух модулях, намеренно
нейтральный, а не investment-специфичный config.SYSTEM_PROMPT.

Сценарии 1-4 отправляют одну и ту же задачу одной из четырёх фиксированных моделей —
две модели Kimi (kimi_client.py) и две модели DeepSeek (deepseek_client.py), порядок
задан от самой сильной к самой слабой, см. MODEL_CATALOG. В отличие от temperature в
research_temperature.py, конкретные идентификаторы моделей и клиент, через который
каждая из них вызывается, не хардкодятся, а берутся из deepseek_client.py/kimi_client.py
(в конечном счёте — из переменных окружения DEEPSEEK_MODEL_PRO, DEEPSEEK_MODEL_FLASH,
KIMI_MODEL_K3, KIMI_MODEL_K2_6, см. .env.example): у разных провайдеров идентификаторы
моделей меняются быстрее кода, поэтому здесь они — настройка, а не предмет
исследования. Если конкретный идентификатор недоступен на используемом API-эндпоинте,
вызов завершится ошибкой DeepSeek/OpenAI SDK, которая перехватывается общим блоком
обработки ошибок (см. _run_models_scenario), как и любая другая ошибка API в проекте.

Стоимость (USD) в MODEL_CATALOG — иллюстративные оценочные величины для целей сравнения
между сценариями этого модуля, а не проверенные официальные тарифы конкретных
провайдеров. При появлении официальных тарифов значения нужно заменить.

Сценарий 5 запускает сценарии 1-4 не полностью параллельно: разные провайдеры
опрашиваются параллельно (свой поток на провайдера), а модели одного провайдера — по
очереди внутри этого потока, т.к. Kimi отвечает RateLimitError (429, "max organization
concurrency: 1") на два одновременных запроса от одного аккаунта — см. докстринг
call_model_compare_all и _group_sub_scenarios_by_client(). Затем одним отдельным
запросом к MAIN_CLIENT/MAIN_MODEL (main_client.py, config.py) — тому же провайдеру и той
же модели, что и в остальных research-режимах без собственного выбора модели
(research_reasoning.py, research_temperature.py, research_response_format.py), а не к
одной из четырёх сравниваемых моделей — просит сравнить полученные ответы вместе со
статистикой (токены, время ответа, стоимость) по качеству, скорости и ресурсоёмкости и
дать краткий вывод — показывает и все исходные ответы со статистикой, и итоговое
сравнение.

main.py подключает фичу через build_models_conversation_handler() — единственную точку
интеграции с остальным приложением.
"""

from __future__ import annotations

import logging
import time
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
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    TELEGRAM_MESSAGE_LIMIT,
)
from deepseek_client import DEEPSEEK_MODEL_FLASH, DEEPSEEK_MODEL_PRO, deepseek_client
from kimi_client import KIMI_MODEL_K2_6, KIMI_MODEL_K3, kimi_client
from main_client import main_client

logger = logging.getLogger(__name__)

WAITING_TASK, CHOOSING_SCENARIO = range(2)

# Нарочно нейтральный системный промпт: этот режим исследует ответы разных моделей на
# произвольных задачах, а не отвечает на инвестиционные вопросы, поэтому здесь не
# используется config.SYSTEM_PROMPT с его инвестиционной ролью и дисклеймерами.
MODELS_SYSTEM_PROMPT = (
    "Ты — ассистент, который решает поставленную пользователем задачу. Отвечай по "
    "существу. Определи язык, на котором сформулирована задача, и веди видимый ответ "
    "(поле content) строго на этом же языке от первого до последнего слова."
)

# Порядок от самой сильной модели к самой слабой — задаёт порядок кнопок 1-4 (сценарий 5
# отправляет итоговый запрос на сравнение через MAIN_CLIENT/MAIN_MODEL, а не через одну
# из этих четырёх моделей, см. докстринг модуля). "client" привязывает каждую модель к её
# провайдеру: две модели Kimi идут через kimi_client, две DeepSeek — через
# deepseek_client. Цены — иллюстративные оценки для сравнения между сценариями, см.
# докстринг модуля.
MODEL_CATALOG: list[dict[str, object]] = [
    {
        "id": KIMI_MODEL_K3,
        "label": "Kimi K3",
        "client": kimi_client,
        "price_input_per_million": 3.0,
        "price_output_per_million": 15.0,
    },
    {
        "id": DEEPSEEK_MODEL_PRO,
        "label": "DeepSeek V4 Pro",
        "client": deepseek_client,
        "price_input_per_million": 0.66,
        "price_output_per_million": 1.98,
    },
    {
        "id": KIMI_MODEL_K2_6,
        "label": "Kimi K2.6",
        "client": kimi_client,
        "price_input_per_million": 0.95,
        "price_output_per_million": 4.0,
    },
    {
        "id": DEEPSEEK_MODEL_FLASH,
        "label": "DeepSeek V4 Flash",
        "client": deepseek_client,
        "price_input_per_million": 0.22,
        "price_output_per_million": 0.66,
    },
]

MODELS_INTRO_TEXT = (
    "🔬 Режим исследования моделей DeepSeek и Kimi.\n\n"
    "Это техническое исследование, а не инвестиционный сервис — задача может быть любой, "
    "не обязательно про инвестиции.\n\n"
    "👉 Введите задачу, которую нужно решить."
)


# --------------------------------------------------------------------------- #
# 5 сценариев вызова DeepSeek/Kimi API
# --------------------------------------------------------------------------- #


def _extract_usage(response) -> dict[str, int] | None:
    """Достаёт минимальную статистику по токенам из ответа API, если она есть."""
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

    На моделях с рассуждениями весь лимит max_tokens может целиком уйти на скрытые
    размышления, оставляя видимый content пустым при finish_reason == "length" — тот же
    случай, что и в research_reasoning.py и research_temperature.py.
    """
    content = message.content
    if not content:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return "[Модель ещё не начала видимый ответ, вот её рассуждения]\n\n" + reasoning
    return content


def _calculate_cost_usd(model_id: str, usage: dict[str, int] | None) -> float | None:
    """Считает иллюстративную стоимость вызова по тарифам из MODEL_CATALOG."""
    if not usage:
        return None
    catalog_entry = next((m for m in MODEL_CATALOG if m["id"] == model_id), None)
    if catalog_entry is None:
        return None
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return (
        prompt_tokens * catalog_entry["price_input_per_million"]
        + completion_tokens * catalog_entry["price_output_per_million"]
    ) / 1_000_000


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


def _sum_cost(first: float | None, second: float | None) -> float | None:
    """Складывает стоимость двух вызовов API (для сценария 5)."""
    if first is None and second is None:
        return None
    return (first or 0.0) + (second or 0.0)


def call_model(
    task: str, client, model_id: str
) -> tuple[str | None, str | None, dict[str, int] | None, float, float | None]:
    """Запрашивает решение задачи у указанной модели через указанный клиент.

    Возвращает (ответ, finish_reason, статистика_токенов, время_ответа_сек, стоимость_usd).
    """
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": MODELS_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    elapsed_seconds = time.monotonic() - start
    choice = response.choices[0]
    usage = _extract_usage(response)
    cost_usd = _calculate_cost_usd(model_id, usage)
    return _content_or_reasoning_fallback(choice.message), choice.finish_reason, usage, elapsed_seconds, cost_usd


def call_model_kimi_k3(task: str, _extra: object = None):
    """Сценарий 1: kimi-k3 (kimi_client) — самая сильная модель из четырёх."""
    return call_model(task, kimi_client, KIMI_MODEL_K3)


def call_model_deepseek_v4_pro(task: str, _extra: object = None):
    """Сценарий 2: deepseek-v4-pro (deepseek_client)."""
    return call_model(task, deepseek_client, DEEPSEEK_MODEL_PRO)


def call_model_kimi_k2_6(task: str, _extra: object = None):
    """Сценарий 3: kimi-k2.6 (kimi_client)."""
    return call_model(task, kimi_client, KIMI_MODEL_K2_6)


def call_model_deepseek_v4_flash(task: str, _extra: object = None):
    """Сценарий 4: deepseek-v4-flash (deepseek_client) — самая слабая модель из четырёх."""
    return call_model(task, deepseek_client, DEEPSEEK_MODEL_FLASH)


# Подзадачи сценария 5 — те же вызовы, что стоят за сценариями 1-4. Клиент указан явно
# (а не только через fn), чтобы _group_sub_scenarios_by_client() могло сгруппировать
# модели одного провайдера, не вызывая их и не заглядывая внутрь call_model().
_COMPARE_ALL_SUB_SCENARIOS: list[tuple[str, object, object]] = [
    ("Kimi K3", kimi_client, call_model_kimi_k3),
    ("DeepSeek V4 Pro", deepseek_client, call_model_deepseek_v4_pro),
    ("Kimi K2.6", kimi_client, call_model_kimi_k2_6),
    ("DeepSeek V4 Flash", deepseek_client, call_model_deepseek_v4_flash),
]


def _group_sub_scenarios_by_client() -> list[list[tuple[str, object]]]:
    """Группирует подзадачи сценария 5 по клиенту-провайдеру, сохраняя порядок первого
    появления провайдера в _COMPARE_ALL_SUB_SCENARIOS.
    """
    groups: dict[int, list[tuple[str, object]]] = {}
    order: list[int] = []
    for label, client, fn in _COMPARE_ALL_SUB_SCENARIOS:
        client_key = id(client)
        if client_key not in groups:
            groups[client_key] = []
            order.append(client_key)
        groups[client_key].append((label, fn))
    return [groups[key] for key in order]

COMPARE_ALL_INSTRUCTION = (
    "Ниже приведены ответы разных моделей на одну и ту же задачу вместе со статистикой "
    "каждого вызова (токены, время ответа, стоимость в долларах). Сравни их по трём "
    "критериям:\n"
    "1. Качество ответов — какая модель точнее и полнее решила задачу.\n"
    "2. Скорость — какая модель ответила быстрее.\n"
    "3. Ресурсоёмкость — какая модель израсходовала меньше токенов и обошлась дешевле.\n"
    "Сделай краткий вывод: какая модель предпочтительна и в каких случаях имеет смысл "
    "выбрать более слабую, но быструю и дешёвую модель вместо самой сильной."
)


def _format_stats_line(
    finish_reason: str | None,
    usage: dict[str, int] | None,
    elapsed_seconds: float | None,
    cost_usd: float | None,
) -> str:
    parts = [f"finish_reason={finish_reason or 'н/д'}"]
    if usage:
        parts.append(f"prompt_tokens={usage.get('prompt_tokens', 'н/д')}")
        parts.append(f"completion_tokens={usage.get('completion_tokens', 'н/д')}")
        parts.append(f"total_tokens={usage.get('total_tokens', 'н/д')}")
    if elapsed_seconds is not None:
        parts.append(f"время={elapsed_seconds:.2f}с")
    if cost_usd is not None:
        parts.append(f"стоимость=${cost_usd:.6f}")
    return ", ".join(parts)


def call_model_compare_all(
    task: str, _extra: object = None
) -> tuple[str | None, str | None, dict[str, int] | None, float, float | None]:
    """Способ 5: параллельно запускает все 4 модели и просит MAIN_CLIENT/MAIN_MODEL их сравнить.

    Сравнивающий запрос идёт через main_client/MAIN_MODEL (config.py, main_client.py) —
    тот же провайдер и модель, что и в остальных research-режимах без собственного
    выбора модели (research_reasoning.py, research_temperature.py,
    research_response_format.py), а не через одну из четырёх сравниваемых моделей.

    Разные провайдеры опрашиваются параллельно (свой поток на провайдера), а модели
    одного провайдера — последовательно внутри этого потока: Kimi возвращает
    RateLimitError (429, "max organization concurrency: 1") на второй одновременный
    запрос от одного аккаунта, поэтому его две модели (kimi-k3, kimi-k2.6) нельзя
    вызывать по-настоящему параллельно. DeepSeek такого ограничения не показывал, но
    последовательный вызов внутри провайдера — общее правило для всех, а не только для
    Kimi, чтобы не зависеть от того, какой конкретно провайдер сейчас чувствителен к
    конкурентности. Отдельная ошибка одной подзадачи не прерывает ни её группу, ни
    сравнение в целом — вместо ответа этой подзадачи в сравнение уходит пометка об
    ошибке. Время ответа каждой модели меряется внутри её собственного вызова
    (call_model), а не относительно фазы в целом — это и есть предмет сравнения
    "скорость" (и оно всё ещё осмысленно при последовательном вызове внутри провайдера:
    время каждой конкретной модели не включает время ожидания других моделей).
    """
    sub_results: dict[str, tuple[str | None, str | None, dict[str, int] | None, float | None, float | None]] = {}

    def _run_group_sequentially(
        entries: list[tuple[str, object]]
    ) -> dict[str, tuple[str | None, str | None, dict[str, int] | None, float | None, float | None]]:
        group_results = {}
        for label, fn in entries:
            try:
                group_results[label] = fn(task)
            except Exception:  # noqa: BLE001 — одна упавшая подзадача не должна рушить сравнение
                logger.exception("Подзадача «%s» сценария 5 завершилась ошибкой.", label)
                group_results[label] = (None, None, None, None, None)
        return group_results

    provider_groups = _group_sub_scenarios_by_client()
    parallel_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(provider_groups)) as executor:
        futures = [executor.submit(_run_group_sequentially, group) for group in provider_groups]
        for future in as_completed(futures):
            sub_results.update(future.result())
    parallel_elapsed = time.monotonic() - parallel_start

    ordered_labels = [label for label, _client, _fn in _COMPARE_ALL_SUB_SCENARIOS]

    comparison_input = "\n\n".join(
        f"### {label}\nОтвет: {sub_results[label][0] or '[нет ответа: ошибка при обращении к API]'}\n"
        f"Статистика: {_format_stats_line(*sub_results[label][1:])}"
        for label in ordered_labels
    )
    comparison_start = time.monotonic()
    comparison_response = main_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": MODELS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Исходная задача: {task}\n\n{comparison_input}\n\n{COMPARE_ALL_INSTRUCTION}"
                ),
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    comparison_elapsed = time.monotonic() - comparison_start
    choice = comparison_response.choices[0]
    comparison_answer = _content_or_reasoning_fallback(choice.message)
    comparison_usage = _extract_usage(comparison_response)
    # MAIN_MODEL обычно не входит в MODEL_CATALOG (это отдельная, "судейская" модель),
    # поэтому для неё чаще всего нет тарифа — _calculate_cost_usd в этом случае вернёт
    # None, и стоимость сравнивающего запроса просто не попадёт в итоговую статистику.
    comparison_cost = _calculate_cost_usd(MAIN_MODEL, comparison_usage)

    result_parts = []
    for label in ordered_labels:
        answer, finish_reason, usage, elapsed, cost = sub_results[label]
        text = answer or "⚠️ Не удалось получить ответ (ошибка API)."
        if finish_reason == "length":
            text += "\n⚠️ Обрезано из-за ограничения max_tokens."
        stats_line = _format_stats_line(finish_reason, usage, elapsed, cost)
        result_parts.append(f"🔹 {label}:\n{text}\n📈 {stats_line}")
    result_parts.append(
        f"🧩 Сравнение и вывод (по мнению {MAIN_CLIENT_LABEL} / {MAIN_MODEL}):\n"
        + (comparison_answer or "Модель вернула пустой ответ.")
    )
    combined_answer = "\n\n".join(result_parts)

    total_usage = comparison_usage
    total_cost = comparison_cost
    for _answer, _finish_reason, usage, _elapsed, cost in sub_results.values():
        total_usage = _sum_usage(total_usage, usage)
        total_cost = _sum_cost(total_cost, cost)
    total_elapsed = parallel_elapsed + comparison_elapsed

    return combined_answer, choice.finish_reason, total_usage, total_elapsed, total_cost


SCENARIO_HANDLERS = {
    "1": ("Kimi K3 (сильнейшая)", call_model_kimi_k3),
    "2": ("DeepSeek V4 Pro", call_model_deepseek_v4_pro),
    "3": ("Kimi K2.6", call_model_kimi_k2_6),
    "4": ("DeepSeek V4 Flash (слабейшая)", call_model_deepseek_v4_flash),
    "5": ("Сравнение всех моделей", call_model_compare_all),
}


# --------------------------------------------------------------------------- #
# Вспомогательные функции: клавиатура, форматирование, запуск сценария
# --------------------------------------------------------------------------- #


def _build_models_scenario_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("1️⃣ Kimi K3", callback_data="models:scenario:1"),
                InlineKeyboardButton("2️⃣ DeepSeek V4 Pro", callback_data="models:scenario:2"),
            ],
            [
                InlineKeyboardButton("3️⃣ Kimi K2.6", callback_data="models:scenario:3"),
                InlineKeyboardButton("4️⃣ DeepSeek V4 Flash", callback_data="models:scenario:4"),
            ],
            [InlineKeyboardButton("5️⃣ Сравнить все модели", callback_data="models:scenario:5")],
            [InlineKeyboardButton("❓ Новая задача", callback_data="models:new_task")],
            [InlineKeyboardButton("🚪 Выйти из исследования", callback_data="models:exit")],
        ]
    )


def _format_model_answer(answer: str | None, finish_reason: str | None) -> str:
    """Готовит текст ответа сценария к отправке в Telegram."""
    text = answer or "Модель вернула пустой ответ."
    if finish_reason == "length":
        text += "\n\n⚠️ ВНИМАНИЕ: Ответ был ОБРЕЗАН из-за ограничения max_tokens!"
    return text


def _format_scenario_stats(
    finish_reason: str | None,
    usage: dict[str, int] | None,
    elapsed_seconds: float | None,
    cost_usd: float | None,
) -> str:
    """Статистика по итогам сценария: причина остановки, токены, время, стоимость."""
    return "📈 " + _format_stats_line(finish_reason, usage, elapsed_seconds, cost_usd)


def _run_models_scenario(
    handler_fn, task: str, extra: object = None
) -> tuple[str | None, str | None, str | None]:
    """Выполняет сценарий, возвращает (текст_ответа, текст_ошибки, статистика).

    При ошибке текст_ответа и статистика отсутствуют (None) — обращения к API не
    случилось или оно не завершилось валидным ответом. В отличие от остальных
    research-модулей, здесь вызов может идти как к DeepSeek, так и к Kimi (см.
    MODEL_CATALOG), поэтому сообщения об ошибках ниже сформулированы нейтрально, без
    привязки к конкретному провайдеру. Ошибка "модель не найдена" (некорректный или
    недоступный на этом API model_id) приходит от OpenAI SDK как APIStatusError и
    перехватывается тем же блоком, что и остальные ошибки API.
    """
    try:
        answer, finish_reason, usage, elapsed, cost = handler_fn(task, extra)
    except AuthenticationError:
        logger.error("Ошибка аутентификации API — проверьте DEEPSEEK_API_KEY/KIMI_API_KEY.")
        return None, (
            "❌ Ошибка авторизации у провайдера этой модели. "
            "Администратору бота нужно проверить соответствующий API-ключ."
        ), None
    except RateLimitError:
        logger.warning("Превышен лимит запросов к API.")
        return None, (
            "⏳ Сервис временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        ), None
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к API.")
        return None, "⏳ Модель не ответила вовремя. Попробуй отправить запрос ещё раз.", None
    except APIConnectionError:
        logger.error("Не удалось подключиться к API.")
        return None, (
            "🌐 Не получилось подключиться к серверу провайдера. "
            "Проверь соединение и попробуй позже."
        ), None
    except APIStatusError as exc:
        logger.error("API вернул ошибку: %s", exc)
        return None, (
            "⚠️ Сервер вернул ошибку (возможно, эта модель недоступна на используемом "
            "API-эндпоинте). Попробуй другой сценарий."
        ), None
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к API (исследование моделей).")
        return None, "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже.", None

    formatted = _format_model_answer(answer, finish_reason)
    stats = _format_scenario_stats(finish_reason, usage, elapsed, cost)
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
    result_text, error_text, stats_text = _run_models_scenario(handler_fn, task, extra)

    await message.reply_text(f"📊 Сценарий: {label}")
    if error_text:
        await message.reply_text(error_text)
    else:
        for i in range(0, len(result_text), TELEGRAM_MESSAGE_LIMIT):
            await message.reply_text(result_text[i : i + TELEGRAM_MESSAGE_LIMIT])
        if stats_text:
            await message.reply_text(stats_text)

    await message.reply_text("👉 Что дальше?", reply_markup=_build_models_scenario_keyboard())


# --------------------------------------------------------------------------- #
# Обработчики диалога
# --------------------------------------------------------------------------- #


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим исследования (/research_models)."""
    await update.message.reply_text(MODELS_INTRO_TEXT)
    return WAITING_TASK


async def models_receive_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сохраняет задачу для исследования и показывает выбор модели."""
    task = update.message.text

    if not task or not task.strip():
        await update.message.reply_text("👉 Пожалуйста, отправь текст задачи.")
        return WAITING_TASK

    if len(task) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Задача слишком длинная ({len(task)} символов). Максимум — {MAX_INPUT_CHARS}."
        )
        return WAITING_TASK

    context.user_data["models_task"] = task
    await update.message.reply_text(
        "Задача сохранена.\n\n👉 Выберите модель:",
        reply_markup=_build_models_scenario_keyboard(),
    )
    return CHOOSING_SCENARIO


async def models_scenario_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки модели / новая задача / выход."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    data = query.data

    if data == "models:exit":
        context.user_data.pop("models_task", None)
        await query.message.reply_text(
            "Исследование завершено. Можешь просто написать вопрос — отвечу в обычном режиме."
        )
        return ConversationHandler.END

    if data == "models:new_task":
        await query.message.reply_text("👉 Введите новую задачу для исследования.")
        return WAITING_TASK

    task = context.user_data.get("models_task")
    if not task:
        await query.message.reply_text(
            "Не найдена сохранённая задача. Начните заново командой /research_models."
        )
        return ConversationHandler.END

    scenario_id = data.rsplit(":", 1)[-1]
    label, handler_fn = SCENARIO_HANDLERS[scenario_id]
    await _send_scenario_result(query.message, label, handler_fn, task, None)
    return CHOOSING_SCENARIO


async def models_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — принудительный выход из режима исследования."""
    context.user_data.pop("models_task", None)
    await update.message.reply_text("Исследование прервано. Возвращаюсь в обычный режим.")
    return ConversationHandler.END


def build_models_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler режима исследования моделей для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("research_models", models_command)],
        states={
            WAITING_TASK: [MessageHandler(text_filter, models_receive_task)],
            CHOOSING_SCENARIO: [
                CallbackQueryHandler(models_scenario_callback, pattern=r"^models:"),
            ],
        },
        fallbacks=[CommandHandler("cancel", models_cancel)],
    )
