"""Команда /smart_agent — LLM-агент с явно разделённой моделью памяти (см. докстринг
agents/smart_agent.py и раздел «Управление памятью smart-агента» в CLAUDE.md).

Устроена по тому же общему принципу, что и /agent (agent_command.py): весь цикл
«вопрос -> вызов LLM -> ответ» инкапсулирован в SmartAgent (agents/smart_agent.py),
здесь — только Telegram-часть (получить текст, показать индикатор набора, отправить
ответ, перевести ошибку API в сообщение на русском, та же параметризация через
MAIN_CLIENT_LABEL/MAIN_API_KEY_ENV_VAR). Это НЕЗАВИСИМЫЙ от /agent режим (свой класс,
свой файл памяти, свои команды) — единственная связь с /agent/agent_compare —
взаимоисключение через active_mode.py, чтобы один чат не оказался "внутри" нескольких
ConversationHandler-ов сразу (см. докстринг active_mode.py).

Команды:
- /smart_agent — вход в диалог «вопрос за вопросом» (как /agent), до /cancel или
  кнопки выхода.
- /smart_agent_remember <текст> — сохраняет факт в долговременную память ДОСЛОВНО.
- /smart_agent_forget <номер> — удаляет факт по номеру (см. /smart_agent_long_show).
- /smart_agent_long_show — показывает все факты долговременной памяти с номерами.
- /smart_agent_task_start <цель> — начинает рабочую задачу (заменяет предыдущую).
- /smart_agent_task_set <ключ> <значение> — кладёт данные в текущую рабочую задачу.
- /smart_agent_task_show — показывает текущую рабочую задачу.
- /smart_agent_task_done — завершает и очищает текущую рабочую задачу.
- /smart_agent_show — показывает все три слоя памяти как есть, их статус
  включено/выключено и системные сообщения, реально ушедшие в LLM на последний вопрос
  (см. SmartAgent.get_last_context_messages) — способ проверить, что попадает в каждый
  слой и как это влияет на ответ.
- /smart_agent_toggle <short|working|long> — включает/выключает слой в СБОРКЕ
  контекста без удаления данных — так можно сравнить ответ на один и тот же вопрос с
  разными слоями включёнными/выключенными.
- /smart_agent_reset — очищает все три слоя памяти разом (не трогает
  enabled_layers — настройка режима, а не данные, как и /agent_reset не трогает
  стратегию).

main.py подключает через build_smart_agent_conversation_handler() и по одному
CommandHandler на каждую из остальных команд (build_smart_agent_*_handler()).
"""

from __future__ import annotations

import logging

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

from config import MAIN_API_KEY_ENV_VAR, MAIN_CLIENT_LABEL, MAX_INPUT_CHARS, TELEGRAM_MESSAGE_LIMIT

from .active_mode import (
    AGENT_MODE,
    COMPARE_MODE,
    SMART_AGENT_MODE,
    clear_active_mode,
    get_active_mode,
    set_active_mode,
)
from .smart_agent import SmartAgent, SmartAgentAnswer

logger = logging.getLogger(__name__)

WAITING_QUESTION = 0

LAYER_LABELS = {
    "short_term": "Краткосрочная (текущий диалог)",
    "working": "Рабочая (данные текущей задачи)",
    "long_term": "Долговременная (факты)",
}
# Короткие алиасы для /smart_agent_toggle — вводить "long_term" в Telegram неудобно.
LAYER_ALIASES = {"short": "short_term", "working": "working", "long": "long_term"}

EXIT_BUTTON_TEXT = "🚪 Выйти из режима smart-агента"
SMART_AGENT_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(EXIT_BUTTON_TEXT)]], resize_keyboard=True
)

SMART_AGENT_INTRO_TEXT = (
    "🧠 Режим smart-агента.\n\n"
    "В отличие от /agent, здесь память явно разделена на три слоя: краткосрочная "
    "(этот диалог), рабочая (данные текущей задачи, /smart_agent_task_start) и "
    "долговременная (факты, которые ты явно сохраняешь через /smart_agent_remember). "
    "Что попадает в каждый слой — решаешь только ты, никакой автоматики. "
    "/smart_agent_show покажет содержимое всех слоёв и что из них ушло в последний "
    "ответ.\n\n"
    "👉 Введи вопрос. Чтобы выйти, нажми кнопку внизу (или отправь /cancel)."
)

# Экземпляр SmartAgent хранит память конкретного чата (см. докстринг SmartAgent) —
# кэшируем по chat_id, как _agents в agent_command.py.
_smart_agents: dict[int, SmartAgent] = {}


def _get_smart_agent(chat_id: int) -> SmartAgent:
    agent = _smart_agents.get(chat_id)
    if agent is None:
        agent = SmartAgent(chat_id)
        _smart_agents[chat_id] = agent
    return agent


def _format_token_stats(result: SmartAgentAnswer) -> str:
    context_tokens = result.context_tokens if result.context_tokens is not None else "н/д"
    response = result.response_tokens if result.response_tokens is not None else "н/д"
    return (
        f"📈 Токены: запрос ≈{result.request_tokens_approx} (по символам), "
        f"контекст={context_tokens}, ответ={response}"
    )


async def _exit_smart_agent_mode(update: Update) -> int:
    clear_active_mode(update.effective_chat.id)
    await update.message.reply_text(
        "Режим smart-агента завершён. Возвращаюсь в обычный режим.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def smart_agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим smart-агента (/smart_agent).

    Взаимоисключается с /agent и /agent_compare (см. active_mode.py) — если чат уже в
    одном из этих режимов, вход отклоняется с подсказкой сначала выйти оттуда.
    """
    chat_id = update.effective_chat.id
    active_mode = get_active_mode(chat_id)
    if active_mode == AGENT_MODE:
        await update.message.reply_text(
            "⚠️ Сейчас активен обычный режим агента (/agent). Сначала выйди из него "
            "(кнопка выхода или /cancel), потом заходи в /smart_agent."
        )
        return ConversationHandler.END
    if active_mode == COMPARE_MODE:
        await update.message.reply_text(
            "⚠️ Сейчас активен режим сравнения стратегий (/agent_compare). Сначала "
            "выйди из него (кнопка выхода или /cancel), потом заходи в /smart_agent."
        )
        return ConversationHandler.END

    set_active_mode(chat_id, SMART_AGENT_MODE)
    await update.message.reply_text(SMART_AGENT_INTRO_TEXT, reply_markup=SMART_AGENT_KEYBOARD)
    return WAITING_QUESTION


async def smart_agent_receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает вопрос пользователя, вызывает SmartAgent и отправляет ответ обратно —
    обработка ошибок API дословно повторяет agent_receive_question (agent_command.py):
    независимый агент, но тот же принцип перевода ошибок OpenAI SDK на русский."""
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if user_text == EXIT_BUTTON_TEXT:
        return await _exit_smart_agent_mode(update)

    if not user_text or not user_text.strip():
        await update.message.reply_text(
            "👉 Пожалуйста, отправь текстовый вопрос.", reply_markup=SMART_AGENT_KEYBOARD
        )
        return WAITING_QUESTION

    if len(user_text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Вопрос слишком длинный ({len(user_text)} символов). Максимум — {MAX_INPUT_CHARS}.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        result = _get_smart_agent(chat_id).ask(user_text)
    except AuthenticationError:
        logger.error(
            "Ошибка аутентификации %s API — проверьте %s.", MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR
        )
        await update.message.reply_text(
            f"❌ Ошибка авторизации на сервере {MAIN_CLIENT_LABEL}. "
            "Администратору бота нужно проверить API-ключ.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except RateLimitError:
        logger.warning("Превышен лимит запросов к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ Сервис {MAIN_CLIENT_LABEL} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ {MAIN_CLIENT_LABEL} не ответил вовремя. Попробуй отправить вопрос ещё раз.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except APIConnectionError:
        logger.error("Не удалось подключиться к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"🌐 Не получилось подключиться к серверу {MAIN_CLIENT_LABEL}. "
            "Проверь соединение и попробуй позже.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except APIStatusError as exc:
        logger.error("%s API вернул ошибку: %s", MAIN_CLIENT_LABEL, exc)
        await update.message.reply_text(
            f"⚠️ Сервер {MAIN_CLIENT_LABEL} вернул ошибку. Попробуй позже.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception(
            "Непредвиденная ошибка при обращении к %s API (smart-агент).", MAIN_CLIENT_LABEL
        )
        await update.message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже.",
            reply_markup=SMART_AGENT_KEYBOARD,
        )
        return WAITING_QUESTION

    answer = result.text
    chunks = [answer[i : i + TELEGRAM_MESSAGE_LIMIT] for i in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT)]
    for chunk in chunks:
        await update.message.reply_text(chunk, reply_markup=SMART_AGENT_KEYBOARD)

    await update.message.reply_text(_format_token_stats(result), reply_markup=SMART_AGENT_KEYBOARD)

    return WAITING_QUESTION


async def smart_agent_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _exit_smart_agent_mode(update)


# --------------------------------------------------------------------------- #
# Долговременная память
# --------------------------------------------------------------------------- #


async def smart_agent_remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_remember <текст> — сохраняет факт в долговременную память
    дословно (см. SmartAgent.remember). Работает независимо от того, находится ли
    пользователь сейчас в режиме /smart_agent."""
    if not context.args:
        await update.message.reply_text(
            "👉 Формат: /smart_agent_remember <текст факта>\n"
            "⚠️ Не сохраняй сюда номера счетов/карт, паспортные данные и другие "
            "чувствительные данные — эта команда ничего не фильтрует."
        )
        return

    fact = " ".join(context.args)
    agent = _get_smart_agent(update.effective_chat.id)
    agent.remember(fact)
    await update.message.reply_text(f"🧠 Сохранено в долговременную память: «{fact}».")


async def smart_agent_forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_forget <номер> — удаляет факт по номеру из
    /smart_agent_long_show."""
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("👉 Формат: /smart_agent_forget <номер>")
        return

    index = int(context.args[0])
    agent = _get_smart_agent(update.effective_chat.id)
    if not agent.forget(index):
        await update.message.reply_text(
            f"⚠️ Факта с номером {index} нет — посмотри актуальные номера в /smart_agent_long_show."
        )
        return
    await update.message.reply_text(f"🗑 Факт №{index} удалён из долговременной памяти.")


async def smart_agent_long_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_long_show — печатает все факты долговременной памяти с
    номерами (номера использует /smart_agent_forget)."""
    facts = _get_smart_agent(update.effective_chat.id).get_long_term_facts()
    if not facts:
        await update.message.reply_text(
            "📭 Долговременная память пуста. Сохранить факт — /smart_agent_remember <текст>."
        )
        return

    lines = ["🧠 Долговременная память:"]
    lines.extend(f"{i}. {fact}" for i, fact in enumerate(facts, start=1))
    await update.message.reply_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Рабочая память (текущая задача)
# --------------------------------------------------------------------------- #


async def smart_agent_task_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_start <цель> — начинает рабочую задачу, заменяя
    предыдущую, если она была (см. SmartAgent.start_task)."""
    if not context.args:
        await update.message.reply_text("👉 Формат: /smart_agent_task_start <цель задачи>")
        return

    goal = " ".join(context.args)
    agent = _get_smart_agent(update.effective_chat.id)
    had_previous_task = agent.get_working() is not None
    agent.start_task(goal)
    note = " Предыдущая рабочая задача заменена." if had_previous_task else ""
    await update.message.reply_text(f"📋 Рабочая задача начата: «{goal}».{note}")


async def smart_agent_task_set_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_set <ключ> <значение> — кладёт данные в текущую
    рабочую задачу. Требует уже начатую задачу (/smart_agent_task_start)."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "👉 Формат: /smart_agent_task_set <ключ> <значение>\n"
            "⚠️ Не сохраняй сюда номера счетов/карт, паспортные данные и другие "
            "чувствительные данные — эта команда ничего не фильтрует."
        )
        return

    key, value = context.args[0], " ".join(context.args[1:])
    agent = _get_smart_agent(update.effective_chat.id)
    if not agent.set_task_data(key, value):
        await update.message.reply_text(
            "⚠️ Сейчас нет активной рабочей задачи. Начни её — /smart_agent_task_start <цель>."
        )
        return
    await update.message.reply_text(f"📋 В рабочую задачу сохранено: {key} = {value}.")


async def smart_agent_task_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_show — показывает текущую рабочую задачу как есть."""
    working = _get_smart_agent(update.effective_chat.id).get_working()
    if working is None:
        await update.message.reply_text(
            "📭 Сейчас нет активной рабочей задачи. Начать — /smart_agent_task_start <цель>."
        )
        return

    lines = [
        f"📋 Рабочая задача: {working['goal']}",
        f"Статус: {working['status']}. Начата: {working['created_at']}.",
    ]
    if working["data"]:
        lines.append("Данные:")
        lines.extend(f"- {key}: {value}" for key, value in working["data"].items())
    else:
        lines.append("Данных пока нет — добавить: /smart_agent_task_set <ключ> <значение>.")
    await update.message.reply_text("\n".join(lines))


async def smart_agent_task_done_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_done — завершает и очищает текущую рабочую задачу."""
    agent = _get_smart_agent(update.effective_chat.id)
    if not agent.finish_task():
        await update.message.reply_text("📭 Сейчас нет активной рабочей задачи.")
        return
    await update.message.reply_text("✅ Рабочая задача завершена и очищена.")


# --------------------------------------------------------------------------- #
# Наблюдаемость и переключение слоёв
# --------------------------------------------------------------------------- #


async def smart_agent_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_show — показывает все три слоя памяти раздельно как есть,
    их статус включено/выключено (/smart_agent_toggle) и системные сообщения, реально
    ушедшие в LLM на последний вопрос (SmartAgent.get_last_context_messages) — так
    видно, что именно попало в каждый слой и что из этого реально повлияло на ответ.
    """
    agent = _get_smart_agent(update.effective_chat.id)
    enabled_layers = agent.get_enabled_layers()

    def status(layer: str) -> str:
        return "включена" if enabled_layers[layer] else "выключена"

    lines = ["🧠 Слои памяти smart-агента:\n"]

    short_term = agent.get_short_term()
    lines.append(
        f"1️⃣ Краткосрочная ({status('short_term')}), пар вопрос-ответ: {len(short_term) // 2}."
    )

    working = agent.get_working()
    if working is None:
        lines.append(f"2️⃣ Рабочая ({status('working')}): активной задачи нет.")
    else:
        lines.append(
            f"2️⃣ Рабочая ({status('working')}): «{working['goal']}», "
            f"данных: {len(working['data'])}."
        )

    facts = agent.get_long_term_facts()
    lines.append(f"3️⃣ Долговременная ({status('long_term')}), фактов: {len(facts)}.")

    last_context = agent.get_last_context_messages()
    if last_context:
        rendered = "\n".join(f"— {m['content']}" for m in last_context if m["role"] == "system")
        lines.append(
            "\n📨 Системный контекст последнего вопроса (то, что реально ушло в LLM):\n" + rendered
        )
    else:
        lines.append("\n📨 Ещё не было ни одного вопроса — контекст последнего вызова пуст.")

    lines.append(
        "\nПодробности: /smart_agent_long_show, /smart_agent_task_show. "
        "Переключить слой — /smart_agent_toggle <short|working|long>."
    )

    text = "\n".join(lines)
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MESSAGE_LIMIT])


async def smart_agent_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_toggle <short|working|long> — включает/выключает слой в
    СБОРКЕ контекста без удаления данных (см. SmartAgent.set_layer_enabled) — так
    можно сравнить ответ на один и тот же вопрос с разным набором включённых слоёв."""
    if len(context.args) != 1 or context.args[0] not in LAYER_ALIASES:
        await update.message.reply_text(
            "👉 Формат: /smart_agent_toggle <short|working|long>"
        )
        return

    layer = LAYER_ALIASES[context.args[0]]
    agent = _get_smart_agent(update.effective_chat.id)
    new_value = not agent.get_enabled_layers()[layer]
    agent.set_layer_enabled(layer, new_value)
    state = "включена" if new_value else "выключена"
    await update.message.reply_text(f"✅ Слой «{LAYER_LABELS[layer]}» теперь {state} в контексте.")


async def smart_agent_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_reset — очищает все три слоя памяти разом. Не трогает
    enabled_layers (настройка режима, а не данные, см. SmartAgent.reset_all)."""
    _get_smart_agent(update.effective_chat.id).reset_all()
    await update.message.reply_text("🗑 Вся память smart-агента (все три слоя) очищена.")


# --------------------------------------------------------------------------- #
# Сборка обработчиков для main.py
# --------------------------------------------------------------------------- #


def build_smart_agent_conversation_handler() -> ConversationHandler:
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("smart_agent", smart_agent_command)],
        states={WAITING_QUESTION: [MessageHandler(text_filter, smart_agent_receive_question)]},
        fallbacks=[CommandHandler("cancel", smart_agent_cancel)],
    )


def build_smart_agent_remember_handler() -> CommandHandler:
    return CommandHandler("smart_agent_remember", smart_agent_remember_command)


def build_smart_agent_forget_handler() -> CommandHandler:
    return CommandHandler("smart_agent_forget", smart_agent_forget_command)


def build_smart_agent_long_show_handler() -> CommandHandler:
    return CommandHandler("smart_agent_long_show", smart_agent_long_show_command)


def build_smart_agent_task_start_handler() -> CommandHandler:
    return CommandHandler("smart_agent_task_start", smart_agent_task_start_command)


def build_smart_agent_task_set_handler() -> CommandHandler:
    return CommandHandler("smart_agent_task_set", smart_agent_task_set_command)


def build_smart_agent_task_show_handler() -> CommandHandler:
    return CommandHandler("smart_agent_task_show", smart_agent_task_show_command)


def build_smart_agent_task_done_handler() -> CommandHandler:
    return CommandHandler("smart_agent_task_done", smart_agent_task_done_command)


def build_smart_agent_show_handler() -> CommandHandler:
    return CommandHandler("smart_agent_show", smart_agent_show_command)


def build_smart_agent_toggle_handler() -> CommandHandler:
    return CommandHandler("smart_agent_toggle", smart_agent_toggle_command)


def build_smart_agent_reset_handler() -> CommandHandler:
    return CommandHandler("smart_agent_reset", smart_agent_reset_command)
