"""Команда /agent_compare — параллельное сравнение трёх стратегий управления
контекстом (Sliding Window / Sticky Facts / Branching; "summary" сюда не входит) на
одном и том же диалоге, для ручной оценки качества/устойчивости/расхода токенов —
инструмент тестирования, а не альтернативный способ отвечать пользователю на
инвестиционные вопросы (в этом смысле он ближе к research-режимам, чем к /agent).

Устроено как fan-out поверх трёх независимых Agent (agents/agent.py), у каждого своя
стратегия зафиксирована на весь сеанс и свой файл истории —
AGENT_HISTORY_DIR/<chat_id>_compare_<стратегия>.json, отдельно от обычного файла
<chat_id>.json команды /agent того же чата (см. докстринг Agent.__init__ про
составной идентификатор). Экземпляры кэшируются в _compare_agents по chat_id, как и
_agents в agent_command.py — по одному набору из трёх на чат за время жизни процесса.

На каждый текстовый вопрос пользователя все три Agent.ask() вызываются параллельно
через ThreadPoolExecutor (_ask_all_strategies) — тот же приём, что уже используется в
research/temperature.py и research/models.py (сценарий 5, "сравнить все"): обращение к
LLM через openai SDK — блокирующий HTTP-вызов, поэтому распараллеливание через потоки,
а не asyncio. В отличие от research-сценариев, здесь важен ФИКСИРОВАННЫЙ порядок
вывода (Sliding Window → Sticky Facts → Branching), одинаковый на каждом ходу, а не
порядок завершения — поэтому результаты забираются из future.result() в порядке
COMPARE_STRATEGIES, а не через as_completed(). Ответ каждой стратегии уходит в чат
отдельными сообщениями с префиксом-меткой и своей строкой статистики токенов
(_format_token_stats, переиспользована из agent_command.py) сразу после ответа —
сравнение "на лету" глазами пользователя, а не только в финальном отчёте. Сбой одной
стратегии не должен мешать показать ответы двух других — ошибка каждой обрабатывается
по отдельности (api_error_to_message из research/_shared.py, тот же перевод исключений
OpenAI SDK на русский, что и в остальном боте) и просто занимает место этой стратегии
в том же порядке вывода.

/agent_compare_report — отдельная команда сравнения (не /summary, чтобы не путать с
одноимённой стратегией управления контекстом, которая в это сравнение не входит).
Работает, только если чат сейчас в режиме сравнения (см. active_mode.py) — берёт
последний ответ ассистента из истории каждого из трёх Agent и одним отдельным вызовом
через main_client/MAIN_MODEL (не через одну из трёх сравниваемых стратегий — тот же
принцип, что и в сравнивающем запросе research/temperature.py и research/models.py)
просит модель сравнить их по качеству/устойчивости/токенам (AGENT_COMPARE_REPORT_SYSTEM_PROMPT
в config.py). Не "открывает" ничего нового пользователю — все три ответа уже видны в
чате выше, отчёт лишь их сопоставляет.

/agent_compare_reset — очищает историю всех трёх теневых агентов сразу (аналог
/agent_reset), работает независимо от активного режима, как и /agent_reset.

Взаимоисключение с /agent — см. active_mode.py: вход в /agent_compare отклоняется,
если чат сейчас в обычном режиме /agent, и наоборот.

main.py подключает через build_agent_compare_conversation_handler(),
build_agent_compare_report_handler() и build_agent_compare_reset_handler().
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from telegram import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ChatAction
from telegram.ext import (
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import (
    AGENT_COMPARE_REPORT_SYSTEM_PROMPT,
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    TELEGRAM_MESSAGE_LIMIT,
)
from providers.main_client import main_client
from research._shared import api_error_to_message

from .active_mode import (
    AGENT_MODE,
    COMPARE_MODE,
    clear_active_mode,
    get_active_mode,
    set_active_mode,
)
from .agent import Agent, AgentAnswer
from .agent_command import _format_token_stats

logger = logging.getLogger(__name__)

WAITING_QUESTION = 0

# Фиксированный порядок вывода — одинаковый на каждом ходу (см. докстринг модуля про
# то, почему это не порядок завершения параллельных вызовов). "summary" не участвует —
# сравниваем только эти три, как и попросили.
COMPARE_STRATEGIES = ["sliding_window", "sticky_facts", "branching"]
COMPARE_STRATEGY_LABELS = {
    "sliding_window": "Sliding Window",
    "sticky_facts": "Sticky Facts",
    "branching": "Branching",
}

EXIT_BUTTON_TEXT = "🚪 Выйти из режима сравнения"
COMPARE_KEYBOARD = ReplyKeyboardMarkup([[KeyboardButton(EXIT_BUTTON_TEXT)]], resize_keyboard=True)

COMPARE_INTRO_TEXT = (
    "🧪 Режим сравнения стратегий.\n\n"
    "Каждый твой вопрос параллельно уйдёт трём стратегиям управления контекстом "
    f"({', '.join(COMPARE_STRATEGY_LABELS.values())}) — ответ каждой придёт отдельным "
    "сообщением по порядку, со своей статистикой токенов.\n\n"
    "👉 Введи вопрос. Команда /agent_compare_report в любой момент сравнит последние "
    "ответы всех трёх. /agent_compare_reset очищает историю всех трёх сразу. Чтобы "
    "выйти, нажми кнопку внизу (или отправь /cancel)."
)

# {chat_id: {стратегия: Agent}} — по одному набору из трёх теневых агентов на чат за
# время жизни процесса (аналог _agents в agent_command.py). Каждый Agent хранит файл
# AGENT_HISTORY_DIR/<chat_id>_compare_<стратегия>.json — отдельно и от обычного
# /agent того же чата, и от теневых агентов других чатов.
_compare_agents: dict[int, dict[str, Agent]] = {}


def _get_compare_agents(chat_id: int) -> dict[str, Agent]:
    agents = _compare_agents.get(chat_id)
    if agents is None:
        agents = {}
        for strategy in COMPARE_STRATEGIES:
            agent = Agent(f"{chat_id}_compare_{strategy}")
            # Стратегия каждого теневого агента зафиксирована на весь сеанс сравнения
            # — явно проставляем её при создании (идемпотентно и для уже существующего
            # на диске файла из прошлого сеанса), а не полагаемся на
            # AGENT_CONTEXT_STRATEGY по умолчанию, которая может быть другой.
            agent.set_strategy(strategy)
            agents[strategy] = agent
        _compare_agents[chat_id] = agents
    return agents


def _ask_all_strategies(
    agents: dict[str, Agent], user_text: str
) -> dict[str, AgentAnswer | Exception]:
    """Параллельно вызывает ask() у всех трёх стратегий (см. докстринг модуля про
    ThreadPoolExecutor). Возвращает по каждой стратегии либо её AgentAnswer, либо
    выброшенное исключение — сбой одной не мешает получить и показать остальные
    (см. docstring модуля).
    """
    results: dict[str, AgentAnswer | Exception] = {}
    with ThreadPoolExecutor(max_workers=len(COMPARE_STRATEGIES)) as executor:
        futures = {
            strategy: executor.submit(agents[strategy].ask, user_text)
            for strategy in COMPARE_STRATEGIES
        }
        # Порядок ниже — COMPARE_STRATEGIES (фиксированный порядок вывода), а не
        # порядок завершения (как в as_completed()) — все три уже запущены parallel
        # вызовом submit() выше, .result() здесь просто ждёт конкретную из них.
        for strategy in COMPARE_STRATEGIES:
            try:
                results[strategy] = futures[strategy].result()
            except Exception as exc:  # noqa: BLE001 — сбой одной стратегии не должен рушить сравнение
                results[strategy] = exc
    return results


async def _exit_compare_mode(update: Update) -> int:
    """Общий выход из режима сравнения — по кнопке и по /cancel."""
    clear_active_mode(update.effective_chat.id)
    await update.message.reply_text(
        "Режим сравнения стратегий завершён. Возвращаюсь в обычный режим.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def agent_compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим сравнения (/agent_compare).

    Взаимоисключается с /agent (agents/agent_command.py, см. active_mode.py) — если
    чат уже в обычном режиме агента, вход отклоняется с подсказкой сначала выйти
    оттуда.
    """
    chat_id = update.effective_chat.id
    if get_active_mode(chat_id) == AGENT_MODE:
        await update.message.reply_text(
            "⚠️ Сейчас активен обычный режим агента (/agent). Сначала выйди из него "
            "(кнопка выхода или /cancel), потом заходи в /agent_compare."
        )
        return ConversationHandler.END

    set_active_mode(chat_id, COMPARE_MODE)
    await update.message.reply_text(COMPARE_INTRO_TEXT, reply_markup=COMPARE_KEYBOARD)
    return WAITING_QUESTION


async def agent_compare_receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает вопрос пользователя, рассылает его всем трём стратегиям параллельно
    и отправляет три ответа по очереди (см. докстринг модуля)."""
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if user_text == EXIT_BUTTON_TEXT:
        return await _exit_compare_mode(update)

    if not user_text or not user_text.strip():
        await update.message.reply_text(
            "👉 Пожалуйста, отправь текстовый вопрос.", reply_markup=COMPARE_KEYBOARD
        )
        return WAITING_QUESTION

    if len(user_text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Вопрос слишком длинный ({len(user_text)} символов). Максимум — {MAX_INPUT_CHARS}.",
            reply_markup=COMPARE_KEYBOARD,
        )
        return WAITING_QUESTION

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    agents = _get_compare_agents(chat_id)
    results = _ask_all_strategies(agents, user_text)

    for strategy in COMPARE_STRATEGIES:
        label = COMPARE_STRATEGY_LABELS[strategy]
        result = results[strategy]

        if isinstance(result, Exception):
            message = api_error_to_message(result, MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR)
            if message is None:
                logger.exception(
                    "Непредвиденная ошибка при обращении к %s API (/agent_compare, %s).",
                    MAIN_CLIENT_LABEL,
                    strategy,
                    exc_info=result,
                )
                message = "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."
            await update.message.reply_text(
                f"🔹 {label}:\n{message}", reply_markup=COMPARE_KEYBOARD
            )
            continue

        answer = result.text
        chunks = [
            answer[i : i + TELEGRAM_MESSAGE_LIMIT]
            for i in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT)
        ]
        for i, chunk in enumerate(chunks):
            text = f"🔹 {label}:\n{chunk}" if i == 0 else chunk
            await update.message.reply_text(text, reply_markup=COMPARE_KEYBOARD)
        await update.message.reply_text(_format_token_stats(result), reply_markup=COMPARE_KEYBOARD)

    return WAITING_QUESTION


async def agent_compare_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — тот же выход из режима сравнения, что и по кнопке."""
    return await _exit_compare_mode(update)


async def agent_compare_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_compare_report — сравнивает последние ответы всех трёх стратегий
    одним вызовом LLM (см. докстринг модуля). Доступна, только если чат сейчас в
    режиме сравнения — иначе сравнивать нечего осмысленно (теневые агенты могут не
    существовать или быть пустыми).
    """
    chat_id = update.effective_chat.id
    if get_active_mode(chat_id) != COMPARE_MODE:
        await update.message.reply_text(
            "⚠️ Команда доступна только в режиме сравнения стратегий — сначала зайди "
            "в /agent_compare и задай хотя бы один вопрос."
        )
        return

    agents = _get_compare_agents(chat_id)
    last_answers: dict[str, str] = {}
    for strategy, agent in agents.items():
        history = agent.get_history()
        if history and history[-1]["role"] == "assistant":
            last_answers[strategy] = history[-1]["content"]

    if not last_answers:
        await update.message.reply_text(
            "📭 Пока нет ни одного ответа для сравнения — сначала задай вопрос в режиме сравнения."
        )
        return

    comparison_input = "\n\n".join(
        f"### {COMPARE_STRATEGY_LABELS[strategy]}\n"
        f"{last_answers.get(strategy, '[нет ответа — эта стратегия не ответила]')}"
        for strategy in COMPARE_STRATEGIES
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        response = main_client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": AGENT_COMPARE_REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": comparison_input},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except AuthenticationError:
        logger.error(
            "Ошибка аутентификации %s API — проверьте %s.", MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR
        )
        await update.message.reply_text(
            f"❌ Ошибка авторизации на сервере {MAIN_CLIENT_LABEL}. "
            "Администратору бота нужно проверить API-ключ."
        )
        return
    except RateLimitError:
        logger.warning("Превышен лимит запросов к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ Сервис {MAIN_CLIENT_LABEL} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        )
        return
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ {MAIN_CLIENT_LABEL} не ответил вовремя. Попробуй /agent_compare_report ещё раз."
        )
        return
    except APIConnectionError:
        logger.error("Не удалось подключиться к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"🌐 Не получилось подключиться к серверу {MAIN_CLIENT_LABEL}. "
            "Проверь соединение и попробуй позже."
        )
        return
    except APIStatusError as exc:
        logger.error("%s API вернул ошибку: %s", MAIN_CLIENT_LABEL, exc)
        await update.message.reply_text(
            f"⚠️ Сервер {MAIN_CLIENT_LABEL} вернул ошибку. Попробуй позже."
        )
        return
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception(
            "Непредвиденная ошибка при обращении к %s API (/agent_compare_report).",
            MAIN_CLIENT_LABEL,
        )
        await update.message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."
        )
        return

    answer = response.choices[0].message.content or (
        f"{MAIN_CLIENT_LABEL} вернул пустой ответ. Попробуй /agent_compare_report ещё раз."
    )
    text = "🧩 Сравнение стратегий:\n\n" + answer
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MESSAGE_LIMIT])


async def agent_compare_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_compare_reset — очищает историю всех трёх теневых агентов сразу
    (аналог /agent_reset). Работает независимо от активного режима, как и /agent_reset."""
    chat_id = update.effective_chat.id
    agents = _get_compare_agents(chat_id)
    for agent in agents.values():
        agent.reset()
    await update.message.reply_text(
        "🗑 История всех трёх стратегий сравнения очищена. "
        "Следующий вопрос каждая увидит как первый."
    )


def build_agent_compare_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler команды /agent_compare для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("agent_compare", agent_compare_command)],
        states={WAITING_QUESTION: [MessageHandler(text_filter, agent_compare_receive_question)]},
        fallbacks=[CommandHandler("cancel", agent_compare_cancel)],
    )


def build_agent_compare_report_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_compare_report для регистрации в main.py."""
    return CommandHandler("agent_compare_report", agent_compare_report_command)


def build_agent_compare_reset_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_compare_reset для регистрации в main.py."""
    return CommandHandler("agent_compare_reset", agent_compare_reset_command)
