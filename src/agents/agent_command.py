"""Команда /agent — LLM-агент с памятью диалога, оформленный как отдельная сущность.

В отличие от handle_message (main.py), где вызов LLM выполняется инлайн внутри
Telegram-обработчика, здесь весь цикл «принять вопрос -> вызвать API -> разобрать
ответ» инкапсулирован в классе Agent (agents/agent.py). Обработчики этого модуля
отвечают только за Telegram-часть: получить текст вопроса, показать индикатор
набора, отправить готовый ответ агента и превратить возможную ошибку API в
сообщение на русском — по тому же принципу, что и handle_message, с той же
параметризацией через MAIN_CLIENT_LABEL/MAIN_API_KEY_ENV_VAR (см. «Архитектура» в
CLAUDE.md).

Команда работает как диалог (ConversationHandler): после /agent пользователь может
задавать вопросы один за другим, пока не выйдет из режима. В отличие от основного
потока и от прежней stateless-версии этого агента, каждый вызов видит всю
предыдущую переписку этого чата с агентом — история хранится в JSON-файле на диске
(см. докстринг Agent) и переживает и повторный вход в /agent, и перезапуск всего
бота. Экземпляры Agent кэшируются по chat_id в _agents (см. _get_agent) — один на
чат за всё время жизни процесса, а не общий на всех пользователей, как раньше.
Команда /agent_reset (agent_reset_command) отдельно от диалога полностью очищает
историю чата, а /agent_history (agent_history_command) печатает её как есть —
обе работают и внутри режима агента, и вне его, не входят в ConversationHandler и
не меняют его состояние. Выход из режима самого
диалога (не путать с очисткой истории) — кнопка на постоянной клавиатуре под полем
ввода (ReplyKeyboardMarkup, а не inline-кнопка под конкретным сообщением, как в
research-режимах) или, как и раньше, команда /cancel — оба пути ведут в один и тот
же _exit_agent_mode; история диалога при этом не очищается, чтобы следующий /agent
мог его продолжить. Клавиатура с кнопкой выхода прикрепляется к каждому ответу
бота, пока пользователь остаётся в состоянии WAITING_QUESTION (а не только один раз
к приветствию) — так кнопка гарантированно не пропадает из чата ни после ошибок
валидации/API, ни после ответа модели, а исчезает только через ReplyKeyboardRemove в
_exit_agent_mode. После каждого ответа агента (но не после ошибок API) отдельным
сообщением отправляется статистика по токенам (_format_token_stats) — см. докстринги
AgentAnswer и Agent.ask в agents/agent.py про то, откуда берётся каждое из чисел.

Переполнение контекста (BadRequestError с кодом/текстом про context length) —
единственная ошибка API агента, в которой пользователю показывается сырая причина от
сервера и явная подсказка /agent_reset, а не общий текст "попробуй позже" — см.
комментарий внутри except APIStatusError в agent_receive_question.

Управление контекстом (см. докстринг Agent в agents/agent.py про сами 4 стратегии —
sliding_window/sticky_facts/branching/summary) переключается командой /agent_context
(инлайн-кнопки, работает вне зависимости от того, находится ли пользователь в режиме
/agent — как /agent_reset/agent_history). Команды /agent_checkpoint, /agent_branch и
/agent_switch_branch управляют чекпоинтами и ветками диалога и имеют смысл только при
активной стратегии Branching — вне неё отклоняются с подсказкой переключиться (см.
_require_branching_strategy); ветки при этом хранятся всегда (общий слой хранения для
всех стратегий, см. докстринг Agent), просто эти три команды ничего не показывают о
них, пока выбрана другая стратегия.

main.py подключает команду через build_agent_conversation_handler(),
build_agent_reset_handler(), build_agent_history_handler(),
build_agent_context_handlers(), build_agent_checkpoint_handler(),
build_agent_branch_handler() и build_agent_switch_branch_handlers() — единственные
точки интеграции с остальным приложением.
"""

from __future__ import annotations

import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
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
    AGENT_ENABLED_STRATEGIES,
    AGENT_STRATEGY_LABELS,
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAX_INPUT_CHARS,
    TELEGRAM_MESSAGE_LIMIT,
)

from .active_mode import (
    AGENT_MODE,
    COMPARE_MODE,
    clear_active_mode,
    get_active_mode,
    set_active_mode,
)
from .agent import Agent, AgentAnswer

# Стратегия, при которой команды /agent_checkpoint, /agent_branch, /agent_switch_branch
# осмысленны (см. докстринг Agent про то, почему ветки — общий слой хранения, но
# управлять ими имеет смысл только пока активна именно эта стратегия).
BRANCHING_STRATEGY = "branching"

logger = logging.getLogger(__name__)

WAITING_QUESTION = 0

# Текст кнопки одновременно и подпись на клавиатуре, и «команда» — MessageHandler
# сравнивает с ним обычный текст сообщения, который Telegram отправляет при нажатии
# кнопки ReplyKeyboardMarkup (в отличие от inline-кнопок, у них нет callback_data).
EXIT_BUTTON_TEXT = "🚪 Выйти из режима агента"

AGENT_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(EXIT_BUTTON_TEXT)]], resize_keyboard=True
)

def _agent_intro_text(strategy: str) -> str:
    return (
        "🤖 Режим агента.\n\n"
        f"Задавай вопросы — агент помнит историю этого диалога и передаёт её в "
        f"{MAIN_CLIENT_LABEL} при каждом следующем вопросе, даже после перезапуска бота.\n\n"
        f"Текущая стратегия управления контекстом: {AGENT_STRATEGY_LABELS[strategy]} "
        "(сменить — /agent_context).\n\n"
        "👉 Введи вопрос. Чтобы выйти, нажми кнопку внизу (или отправь /cancel). "
        "Команда /agent_reset в любой момент очищает историю."
    )

# Экземпляр Agent хранит историю диалога конкретного чата (см. докстринг Agent),
# поэтому, в отличие от прежней stateless-версии, не может быть один на все чаты —
# кэшируем по chat_id и переиспользуем в пределах жизни процесса; при первом
# обращении после перезапуска процесса Agent сам подхватит историю с диска.
_agents: dict[int, Agent] = {}


def _get_agent(chat_id: int) -> Agent:
    agent = _agents.get(chat_id)
    if agent is None:
        agent = Agent(chat_id)
        _agents[chat_id] = agent
    return agent


def _format_token_stats(result: AgentAnswer) -> str:
    """Строка статистики по токенам, отправляемая отдельным сообщением после ответа
    агента (см. докстринг AgentAnswer/Agent.ask в agents/agent.py про то, что значит
    каждое из чисел).
    """
    context_tokens = result.context_tokens if result.context_tokens is not None else "н/д"
    response = result.response_tokens if result.response_tokens is not None else "н/д"
    return (
        f"📈 Токены: запрос ≈{result.request_tokens_approx} (по символам), "
        f"контекст={context_tokens}, ответ={response}"
    )


async def _exit_agent_mode(update: Update) -> int:
    """Общий выход из режима агента — по кнопке и по /cancel (см. build_..._handler)."""
    clear_active_mode(update.effective_chat.id)
    await update.message.reply_text(
        "Режим агента завершён. Возвращаюсь в обычный режим.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа в режим агента (/agent).

    Взаимоисключается с /agent_compare (agents/compare_command.py, см. active_mode.py
    про то, почему) — если чат уже в режиме сравнения, вход отклоняется с подсказкой
    сначала выйти оттуда.
    """
    chat_id = update.effective_chat.id
    if get_active_mode(chat_id) == COMPARE_MODE:
        await update.message.reply_text(
            "⚠️ Сейчас активен режим сравнения стратегий (/agent_compare). "
            "Сначала выйди из него (кнопка выхода или /cancel), потом заходи в /agent."
        )
        return ConversationHandler.END

    set_active_mode(chat_id, AGENT_MODE)
    strategy = _get_agent(chat_id).get_strategy()
    await update.message.reply_text(_agent_intro_text(strategy), reply_markup=AGENT_KEYBOARD)
    return WAITING_QUESTION


async def agent_receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимает вопрос пользователя, вызывает агента и отправляет ответ обратно."""
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if user_text == EXIT_BUTTON_TEXT:
        return await _exit_agent_mode(update)

    if not user_text or not user_text.strip():
        await update.message.reply_text(
            "👉 Пожалуйста, отправь текстовый вопрос.", reply_markup=AGENT_KEYBOARD
        )
        return WAITING_QUESTION

    if len(user_text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Вопрос слишком длинный ({len(user_text)} символов). Максимум — {MAX_INPUT_CHARS}.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        result = _get_agent(chat_id).ask(user_text)
    except AuthenticationError:
        logger.error(
            "Ошибка аутентификации %s API — проверьте %s.", MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR
        )
        await update.message.reply_text(
            f"❌ Ошибка авторизации на сервере {MAIN_CLIENT_LABEL}. "
            "Администратору бота нужно проверить API-ключ.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except RateLimitError:
        logger.warning("Превышен лимит запросов к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ Сервис {MAIN_CLIENT_LABEL} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except (APITimeoutError, TimeoutError):
        logger.warning("Тайм-аут запроса к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"⏳ {MAIN_CLIENT_LABEL} не ответил вовремя. Попробуй отправить вопрос ещё раз.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except APIConnectionError:
        logger.error("Не удалось подключиться к %s API.", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            f"🌐 Не получилось подключиться к серверу {MAIN_CLIENT_LABEL}. "
            "Проверь соединение и попробуй позже.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except APIStatusError as exc:
        # BadRequestError (400) — подкласс APIStatusError, отдельного except для него не
        # заводим (перехватился бы этой же веткой), а разбираем частный случай прямо
        # здесь: переполнение контекста диалога — единственная ошибка API агента, где
        # пользователю имеет смысл показать сырую причину от сервера и явно подсказать
        # /agent_reset, а не только общий текст "попробуй позже", т.к. история у /agent
        # ничем не ограничена (см. CLAUDE.md) и повтор того же вопроса ту же ошибку
        # повторит снова.
        if isinstance(exc, BadRequestError):
            reason = exc.message or str(exc)
            is_context_length_error = (
                exc.code == "context_length_exceeded"
                or "context length" in reason.lower()
                or "maximum context" in reason.lower()
            )
            if is_context_length_error:
                logger.warning(
                    "Превышена максимальная длина контекста %s API (агент): %s",
                    MAIN_CLIENT_LABEL,
                    exc,
                )
                await update.message.reply_text(
                    f"⚠️ История диалога с агентом стала слишком большой для {MAIN_CLIENT_LABEL} "
                    "(превышена максимальная длина контекста).\n"
                    f"Причина от сервера: {reason}\n\n"
                    "Используй /agent_reset, чтобы очистить историю и продолжить.",
                    reply_markup=AGENT_KEYBOARD,
                )
                return WAITING_QUESTION

        logger.error("%s API вернул ошибку: %s", MAIN_CLIENT_LABEL, exc)
        await update.message.reply_text(
            f"⚠️ Сервер {MAIN_CLIENT_LABEL} вернул ошибку. Попробуй позже.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION
    except Exception:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        logger.exception("Непредвиденная ошибка при обращении к %s API (агент).", MAIN_CLIENT_LABEL)
        await update.message.reply_text(
            "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже.",
            reply_markup=AGENT_KEYBOARD,
        )
        return WAITING_QUESTION

    # Клавиатуру достаточно прикрепить к последнему куску ответа — Telegram и так
    # держит её показанной до следующего reply_markup, но делаем это явно на каждом
    # сообщении бота в этом состоянии (см. докстринг модуля), а не только на первом.
    answer = result.text
    chunks = [answer[i : i + TELEGRAM_MESSAGE_LIMIT] for i in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT)]
    for chunk in chunks:
        await update.message.reply_text(chunk, reply_markup=AGENT_KEYBOARD)

    await update.message.reply_text(_format_token_stats(result), reply_markup=AGENT_KEYBOARD)

    return WAITING_QUESTION


async def agent_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Fallback /cancel — тот же выход из режима агента, что и по кнопке (см. _exit_agent_mode)."""
    return await _exit_agent_mode(update)


async def agent_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_reset — очищает историю диалога агента для этого чата.

    Работает независимо от того, находится ли пользователь сейчас в режиме /agent
    (см. докстринг модуля) — не входит в ConversationHandler и не меняет его
    состояние.
    """
    chat_id = update.effective_chat.id
    _get_agent(chat_id).reset()
    await update.message.reply_text(
        "🗑 История диалога с агентом очищена. Следующий вопрос агент увидит как первый."
    )


async def agent_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_history — печатает сохранённую историю диалога агента для этого чата.

    Как и /agent_reset, работает независимо от того, находится ли пользователь
    сейчас в режиме /agent — не входит в ConversationHandler и не меняет его
    состояние. Печатает историю АКТИВНОЙ ветки как есть (не сокращённый контекст,
    см. докстринг Agent) — при нескольких ветках (стратегия Branching) отдельной
    строкой указывает, какая ветка активна и какие ещё существуют. Не ограничена по
    длине, поэтому результат режется на части по TELEGRAM_MESSAGE_LIMIT так же, как в
    handle_message (main.py).
    """
    chat_id = update.effective_chat.id
    agent = _get_agent(chat_id)
    history = agent.get_history()
    branches = agent.list_branches()

    if not history:
        await update.message.reply_text(
            "📭 История диалога с агентом пуста. Используй /agent, чтобы задать вопрос."
        )
        return

    lines = [f"📜 История диалога с агентом ({len(history) // 2} вопрос(ов)):"]
    if len(branches) > 1:
        lines.append(
            f"🌿 Активная ветка: «{agent.get_active_branch()}». "
            f"Все ветки: {', '.join(branches)}."
        )
    for i in range(0, len(history), 2):
        pair_number = i // 2 + 1
        lines.append(f"\n{pair_number}. 🙋 {history[i]['content']}")
        if i + 1 < len(history):
            lines.append(f"🤖 {history[i + 1]['content']}")

    text = "\n".join(lines)
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MESSAGE_LIMIT])


async def agent_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_mode — показывает текущий режим чата: активную стратегию
    управления контекстом и, только при стратегии Branching, активную ветку диалога
    (вне Branching понятие "активная ветка" не видно пользователю нигде ещё —
    /agent_checkpoint/agent_branch/agent_switch_branch тоже работают только при ней,
    см. _require_branching_strategy). Как и /agent_reset/agent_history/agent_context,
    работает независимо от того, находится ли пользователь сейчас в режиме /agent —
    не входит в ConversationHandler и не меняет его состояние.
    """
    agent = _get_agent(update.effective_chat.id)
    strategy = agent.get_strategy()
    strategy_label = AGENT_STRATEGY_LABELS[strategy]

    lines = [f"⚙️ Стратегия управления контекстом: {strategy_label}."]
    if strategy == BRANCHING_STRATEGY:
        branches = agent.list_branches()
        if len(branches) > 1:
            lines.append(
                f"🌿 Активная ветка: «{agent.get_active_branch()}». "
                f"Все ветки: {', '.join(branches)}."
            )
        else:
            lines.append(f"🌿 Активная ветка: «{agent.get_active_branch()}».")

    await update.message.reply_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# /agent_context — переключение стратегии управления контекстом
# --------------------------------------------------------------------------- #

_AGENT_CONTEXT_CALLBACK_PREFIX = "agent_ctx:"


def _strategy_keyboard(current: str) -> InlineKeyboardMarkup:
    """Кнопки только для стратегий из AGENT_ENABLED_STRATEGIES (config.py) — оператор
    бота может скрыть часть стратегий из выбора; если текущая стратегия чата в список
    не входит (например, его сузили уже после того, как чат её выбрал), кнопки для
    неё просто не будет — сама стратегия при этом продолжает работать как обычно."""
    buttons = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if name == current else ''}{label}",
                callback_data=f"{_AGENT_CONTEXT_CALLBACK_PREFIX}{name}",
            )
        ]
        for name, label in AGENT_STRATEGY_LABELS.items()
        if name in AGENT_ENABLED_STRATEGIES
    ]
    return InlineKeyboardMarkup(buttons)


async def agent_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_context — показывает текущую стратегию и кнопки для переключения
    на одну из остальных. Работает независимо от того, находится ли пользователь в
    режиме /agent, как /agent_reset и /agent_history."""
    agent = _get_agent(update.effective_chat.id)
    current_label = AGENT_STRATEGY_LABELS[agent.get_strategy()]
    await update.message.reply_text(
        f"👉 Текущая стратегия управления контекстом: {current_label}.\nВыбери другую:",
        reply_markup=_strategy_keyboard(agent.get_strategy()),
    )


async def agent_context_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатие кнопки выбора стратегии из agent_context_command."""
    query = update.callback_query
    await query.answer()
    strategy = query.data[len(_AGENT_CONTEXT_CALLBACK_PREFIX) :]
    if strategy not in AGENT_STRATEGY_LABELS or strategy not in AGENT_ENABLED_STRATEGIES:
        return
    agent = _get_agent(update.effective_chat.id)
    agent.set_strategy(strategy)
    await query.edit_message_text(
        f"✅ Стратегия управления контекстом переключена на: {AGENT_STRATEGY_LABELS[strategy]}."
    )


# --------------------------------------------------------------------------- #
# /agent_checkpoint, /agent_branch, /agent_switch_branch — стратегия Branching
# --------------------------------------------------------------------------- #

_AGENT_SWITCH_BRANCH_CALLBACK_PREFIX = "agent_switch_branch:"


def _require_branching_strategy(agent: Agent) -> str | None:
    """Возвращает текст ошибки, если активная стратегия чата — не Branching, иначе
    None. Чекпоинты и ветки осмысленны только при этой стратегии (см. BRANCHING_STRATEGY
    выше и докстринг Agent про то, почему ветки — общий слой хранения, но управлять
    ими имеет смысл только при ней) — вне неё команды отклоняются с подсказкой
    переключиться через /agent_context, а не молча работают поверх другой стратегии.
    """
    if agent.get_strategy() != BRANCHING_STRATEGY:
        return (
            "⚠️ Чекпоинты и ветки доступны только при стратегии "
            f"{AGENT_STRATEGY_LABELS[BRANCHING_STRATEGY]}.\n"
            f"Сейчас выбрана: {AGENT_STRATEGY_LABELS[agent.get_strategy()]}.\n"
            "Переключись командой /agent_context."
        )
    return None


async def agent_checkpoint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_checkpoint <имя> — помечает текущую точку активной ветки."""
    agent = _get_agent(update.effective_chat.id)
    error = _require_branching_strategy(agent)
    if error:
        await update.message.reply_text(error)
        return

    if not context.args:
        await update.message.reply_text("👉 Формат: /agent_checkpoint <имя>")
        return

    name = context.args[0]
    agent.create_checkpoint(name)
    await update.message.reply_text(
        f"📍 Чекпоинт «{name}» сохранён в ветке «{agent.get_active_branch()}»."
    )


async def agent_branch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_branch <чекпоинт> <ветка> — создаёт одну новую ветку от
    указанного чекпоинта (см. Agent.create_branch). Чтобы получить несколько веток
    от одной точки (например, классические две), вызови команду повторно с тем же
    чекпоинтом и другим именем ветки."""
    agent = _get_agent(update.effective_chat.id)
    error = _require_branching_strategy(agent)
    if error:
        await update.message.reply_text(error)
        return

    if len(context.args) != 2:
        await update.message.reply_text("👉 Формат: /agent_branch <чекпоинт> <ветка>")
        return

    checkpoint_name, branch_name = context.args
    if not agent.checkpoint_exists(checkpoint_name):
        await update.message.reply_text(f"⚠️ Чекпоинт «{checkpoint_name}» не найден.")
        return
    if agent.branch_exists(branch_name):
        await update.message.reply_text(
            "⚠️ Ветка с таким именем уже существует — выбери другое имя."
        )
        return

    agent.create_branch(checkpoint_name, branch_name)
    await update.message.reply_text(
        f"🌿 Создана ветка «{branch_name}» от чекпоинта «{checkpoint_name}».\n"
        "Переключиться на неё — /agent_switch_branch."
    )


def _branch_keyboard(agent: Agent) -> InlineKeyboardMarkup:
    current = agent.get_active_branch()
    buttons = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if name == current else ''}{name}",
                callback_data=f"{_AGENT_SWITCH_BRANCH_CALLBACK_PREFIX}{name}",
            )
        ]
        for name in agent.list_branches()
    ]
    return InlineKeyboardMarkup(buttons)


async def agent_switch_branch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /agent_switch_branch — показывает кнопки со всеми ветками чата и
    переключает активную по выбору (см. Agent.switch_branch)."""
    agent = _get_agent(update.effective_chat.id)
    error = _require_branching_strategy(agent)
    if error:
        await update.message.reply_text(error)
        return

    await update.message.reply_text(
        f"👉 Текущая ветка: «{agent.get_active_branch()}». Выбери, на какую переключиться:",
        reply_markup=_branch_keyboard(agent),
    )


async def agent_switch_branch_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатие кнопки выбора ветки из agent_switch_branch_command."""
    query = update.callback_query
    await query.answer()
    agent = _get_agent(update.effective_chat.id)
    error = _require_branching_strategy(agent)
    if error:
        await query.edit_message_text(error)
        return

    branch_name = query.data[len(_AGENT_SWITCH_BRANCH_CALLBACK_PREFIX) :]
    if not agent.branch_exists(branch_name):
        await query.edit_message_text("⚠️ Такой ветки уже нет.")
        return
    agent.switch_branch(branch_name)
    await query.edit_message_text(f"✅ Активная ветка переключена на «{branch_name}».")


def build_agent_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler команды /agent для регистрации в main.py."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("agent", agent_command)],
        states={WAITING_QUESTION: [MessageHandler(text_filter, agent_receive_question)]},
        fallbacks=[CommandHandler("cancel", agent_cancel)],
    )


def build_agent_reset_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_reset для регистрации в main.py."""
    return CommandHandler("agent_reset", agent_reset_command)


def build_agent_history_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_history для регистрации в main.py."""
    return CommandHandler("agent_history", agent_history_command)


def build_agent_mode_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_mode для регистрации в main.py."""
    return CommandHandler("agent_mode", agent_mode_command)


def build_agent_context_handlers() -> list:
    """Собирает обработчики команды /agent_context (переключатель стратегий) для
    регистрации в main.py — команду и обработчик нажатий на её inline-кнопки."""
    return [
        CommandHandler("agent_context", agent_context_command),
        CallbackQueryHandler(agent_context_callback, pattern=f"^{_AGENT_CONTEXT_CALLBACK_PREFIX}"),
    ]


def build_agent_checkpoint_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_checkpoint для регистрации в main.py."""
    return CommandHandler("agent_checkpoint", agent_checkpoint_command)


def build_agent_branch_handler() -> CommandHandler:
    """Собирает CommandHandler команды /agent_branch для регистрации в main.py."""
    return CommandHandler("agent_branch", agent_branch_command)


def build_agent_switch_branch_handlers() -> list:
    """Собирает обработчики команды /agent_switch_branch для регистрации в main.py —
    команду и обработчик нажатий на её inline-кнопки."""
    return [
        CommandHandler("agent_switch_branch", agent_switch_branch_command),
        CallbackQueryHandler(
            agent_switch_branch_callback, pattern=f"^{_AGENT_SWITCH_BRANCH_CALLBACK_PREFIX}"
        ),
    ]
