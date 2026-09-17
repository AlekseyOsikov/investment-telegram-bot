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

ПЕРСОНАЛИЗАЦИЯ (профили): на чат может быть заведено несколько именованных профилей
(например, «Консервативный»/«Агрессивный») — каждый со своими предпочтениями (стиль,
уровень опыта, формат ответа, отношение к риску, горизонт, интересы, что не
затрагивать) И СВОЕЙ НЕЗАВИСИМОЙ копией всех трёх слоёв памяти (см. докстринг
SmartAgent). Если у чата нет активного профиля, /smart_agent сам показывает выбор
профиля (кнопки: существующие профили + «➕ Новый профиль») ПЕРЕД тем, как перейти к
циклу вопросов — так же и после удаления активного профиля
(/smart_agent_profile_delete) выбор показывается сразу, а не оставляет чат в
подвешенном состоянии. Создание нового профиля — короткая анкета (по одному вопросу
на поле, с возможностью пропустить). Всё это реализовано ДВАЖДЫ по форме, но
переиспользует общую логику анкеты (_handle_new_profile_name/_handle_profile_field_answer):
- как часть ConversationHandler-а самой команды /smart_agent (выбор/анкета ведёт
  прямо в цикл вопросов, состояния PROFILE_PICK/WAITING_PROFILE_NAME/
  WAITING_PROFILE_FIELD);
- как ОТДЕЛЬНЫЙ ConversationHandler команды /smart_agent_profile (управление
  профилем в любой момент, даже посреди диалога — по тому же принципу, что
  /agent_switch_branch работает независимо от того, находится ли пользователь в
  режиме /agent; выбор/анкета здесь заканчивается подтверждением, а не циклом
  вопросов). Регистрируется в main.py ПЕРЕД ConversationHandler-ом /smart_agent —
  порядок важен: пока эта анкета активна, именно она должна первой перехватывать
  обычный текст (ответы на её вопросы), а не WAITING_QUESTION команды /smart_agent
  (см. комментарий в main.py у регистрации).
Инлайн-кнопки выбора/создания профиля используют РАЗНЫЕ префиксы callback_data для
этих двух путей (_START_PROFILE_CALLBACK_PREFIX/_MANAGE_PROFILE_CALLBACK_PREFIX) —
иначе нажатие кнопки одного пикера могло бы быть перехвачено обработчиком другого
ConversationHandler-а. Обработчик кнопок "manage" зарегистрирован ЕЩЁ и как entry
point своего ConversationHandler-а — это позволяет ему подхватывать нажатия и на
пикер, отправленный ВНЕ какого-либо диалога (после /smart_agent_profile_delete).

Все данные, кроме meta профиля (создание — анкета/entry-callback, правка — только
/smart_agent_profile_set), продолжают писаться теми же принципами, что и раньше —
просто теперь в разрезе активного профиля, а не общими на весь чат:
- /smart_agent — вход в диалог «вопрос за вопросом» (как /agent), до /cancel или
  кнопки выхода; если нет активного профиля — сначала выбор/создание профиля.
- /smart_agent_profile — выбрать другой профиль или создать новый (работает в любой
  момент, не только при входе в /smart_agent).
- /smart_agent_profile_set <ключ> <значение> — точечно поправить одно поле профиля
  без анкеты заново.
- /smart_agent_profile_show — показать все профили и поля активного.
- /smart_agent_profile_delete <имя> — удалить профиль целиком (вместе со всей его
  памятью); если удалён активный — сразу показывается выбор нового.
- /smart_agent_remember <текст> — сохраняет факт в долговременную память активного
  профиля ДОСЛОВНО.
- /smart_agent_forget <номер> — удаляет факт по номеру (см. /smart_agent_long_show).
- /smart_agent_long_show — показывает все факты долговременной памяти активного
  профиля с номерами.
- /smart_agent_task_start <цель> — начинает рабочую задачу активного профиля
  (заменяет предыдущую).
- /smart_agent_task_set <ключ> <значение> — кладёт данные в текущую рабочую задачу.
- /smart_agent_task_show — показывает текущую рабочую задачу.
- /smart_agent_task_done — завершает и очищает текущую рабочую задачу.
- /smart_agent_show — показывает профиль и все три слоя памяти активного профиля как
  есть, их статус включено/выключено и системные сообщения, реально ушедшие в LLM на
  последний вопрос (см. SmartAgent.get_last_context_messages) — способ проверить, что
  попадает в каждый слой и как это влияет на ответ.
- /smart_agent_toggle <profile|short|working|long> — включает/выключает слой в
  СБОРКЕ контекста без удаления данных (общая настройка на чат, не per-profile) — так
  можно сравнить ответ на один и тот же вопрос с разными слоями включёнными/
  выключенными.
- /smart_agent_reset — очищает три слоя памяти АКТИВНОГО ПРОФИЛЯ (не трогает сам
  профиль, его meta, другие профили и enabled_layers).

Команды без активного профиля (кроме /smart_agent и /smart_agent_profile*) отклоняются
с подсказкой выбрать/создать профиль — см. _require_active_profile().

main.py подключает через build_smart_agent_conversation_handler(),
build_smart_agent_profile_conversation_handler() и по одному CommandHandler на
каждую из остальных команд (build_smart_agent_*_handler()).
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

from config import MAIN_API_KEY_ENV_VAR, MAIN_CLIENT_LABEL, MAX_INPUT_CHARS, TELEGRAM_MESSAGE_LIMIT

from .active_mode import (
    AGENT_MODE,
    COMPARE_MODE,
    SMART_AGENT_MODE,
    clear_active_mode,
    get_active_mode,
    set_active_mode,
)
from .smart_agent import PROFILE_FIELD_LABELS, PROFILE_FIELDS, SmartAgent, SmartAgentAnswer

logger = logging.getLogger(__name__)

# --- Состояния ConversationHandler-а команды /smart_agent --- #
WAITING_QUESTION = 0
PROFILE_PICK = 1
WAITING_PROFILE_NAME = 2
WAITING_PROFILE_FIELD = 3

# --- Состояния ConversationHandler-а команды /smart_agent_profile (управление
# профилем в любой момент, отдельно от цикла вопросов выше) --- #
MANAGE_PROFILE_PICK = 0
MANAGE_WAITING_NAME = 1
MANAGE_WAITING_FIELD = 2

LAYER_LABELS = {
    "profile": "Профиль (предпочтения персонализации)",
    "short_term": "Краткосрочная (текущий диалог)",
    "working": "Рабочая (данные текущей задачи)",
    "long_term": "Долговременная (факты)",
}
# Короткие алиасы для /smart_agent_toggle — вводить "long_term" в Telegram неудобно.
LAYER_ALIASES = {
    "profile": "profile",
    "short": "short_term",
    "working": "working",
    "long": "long_term",
}

EXIT_BUTTON_TEXT = "🚪 Выйти из режима smart-агента"
SMART_AGENT_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(EXIT_BUTTON_TEXT)]], resize_keyboard=True
)

SMART_AGENT_INTRO_TEXT = (
    "🧠 Режим smart-агента.\n\n"
    "Память разделена на три слоя: краткосрочная (этот диалог), рабочая (данные "
    "текущей задачи, /smart_agent_task_start) и долговременная (факты, "
    "/smart_agent_remember) — и хранится в разрезе твоего текущего профиля "
    "(/smart_agent_profile покажет и позволит переключить). Что попадает в каждый "
    "слой — решаешь только ты, никакой автоматики. /smart_agent_show покажет "
    "содержимое всех слоёв и что из них ушло в последний ответ.\n\n"
    "👉 Введи вопрос. Чтобы выйти, нажми кнопку внизу (или отправь /cancel)."
)

# Примеры-подсказки к вопросам анкеты создания профиля (см. PROFILE_FIELDS в
# agents/smart_agent.py) — единственное место с этими формулировками.
PROFILE_FIELD_EXAMPLES = {
    "style": "например: просто и по-дружески / нейтрально-деловой / с терминологией",
    "experience_level": "например: новичок / есть опыт / разбираюсь профессионально",
    "format": "например: коротко / развёрнуто с пояснениями / списком по пунктам",
    "risk_tolerance": "например: консервативный / умеренный / агрессивный",
    "horizon": "например: краткосрочный / среднесрочный / долгосрочный",
    "interests": "например: акции, ETF, недвижимость",
    "excluded_topics": "например: не предлагать криптовалюту",
}

MAX_PROFILE_NAME_LENGTH = 20
SKIP_WORD = "пропустить"

# Транзитное состояние анкеты (имя нового профиля + собранные поля + текущий индекс
# поля) хранится в context.user_data — общий для user_data механизм PTB, доступный
# из обработчиков ОБОИХ ConversationHandler-ов ниже (см. докстринг модуля).
_UD_NEW_NAME = "sa_new_profile_name"
_UD_NEW_VALUES = "sa_new_profile_values"
_UD_NEW_FIELD_INDEX = "sa_new_profile_field_index"

_START_PROFILE_CALLBACK_PREFIX = "sa_start:"
_MANAGE_PROFILE_CALLBACK_PREFIX = "sa_manage:"

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


def _require_active_profile(agent: SmartAgent) -> str | None:
    """Возвращает текст ошибки, если у чата нет активного профиля, иначе None — вся
    память (кроме meta профилей и самого их списка) хранится в разрезе активного
    профиля (см. докстринг SmartAgent), поэтому большинству команд он нужен."""
    if agent.get_active_profile_name() is None:
        return (
            "⚠️ Сейчас нет активного профиля — вся память smart-агента (краткосрочная, "
            "рабочая, долговременная) хранится в разрезе профиля.\n"
            "Выбери или создай профиль — /smart_agent_profile."
        )
    return None


def _profile_keyboard(agent: SmartAgent, callback_prefix: str) -> InlineKeyboardMarkup:
    """Кнопки: по одной на каждый существующий профиль (✅ у активного) + «➕ Новый
    профиль». callback_prefix различает, какой из двух путей анкеты обрабатывает
    нажатие (см. докстринг модуля про _START_PROFILE_CALLBACK_PREFIX/
    _MANAGE_PROFILE_CALLBACK_PREFIX)."""
    current = agent.get_active_profile_name()
    buttons = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if name == current else ''}{name}",
                callback_data=f"{callback_prefix}switch:{name}",
            )
        ]
        for name in agent.list_profiles()
    ]
    buttons.append(
        [InlineKeyboardButton("➕ Новый профиль", callback_data=f"{callback_prefix}new")]
    )
    return InlineKeyboardMarkup(buttons)


def _validate_profile_name(name: str) -> str | None:
    """Возвращает текст ошибки, если имя профиля непригодно, иначе None. ':' и
    переносы строк запрещены, т.к. используются как разделитель в callback_data
    кнопок пикера (см. _profile_keyboard)."""
    if not name:
        return "⚠️ Имя профиля не может быть пустым."
    if ":" in name or "\n" in name:
        return "⚠️ Имя профиля не должно содержать «:» или переносы строк."
    if len(name) > MAX_PROFILE_NAME_LENGTH:
        return f"⚠️ Имя профиля слишком длинное (максимум {MAX_PROFILE_NAME_LENGTH} символов)."
    return None


def _profile_field_prompt(index: int) -> str:
    field = PROFILE_FIELDS[index]
    label = PROFILE_FIELD_LABELS[field]
    example = PROFILE_FIELD_EXAMPLES[field]
    return (
        f"👉 {label} ({example}).\n"
        f"Отправь текст или «{SKIP_WORD}», чтобы пропустить этот пункт "
        f"({index + 1}/{len(PROFILE_FIELDS)})."
    )


# --------------------------------------------------------------------------- #
# Общая логика анкеты создания профиля — переиспользуется и циклом /smart_agent
# (ведёт в WAITING_QUESTION), и отдельной командой /smart_agent_profile (ведёт в
# ConversationHandler.END) через параметры состояний/колбэк finish.
# --------------------------------------------------------------------------- #


async def _handle_new_profile_name(
    update: Update, context: ContextTypes.DEFAULT_TYPE, name_state: int, field_state: int
) -> int:
    text = (update.message.text or "").strip()
    error = _validate_profile_name(text)
    if error:
        await update.message.reply_text(error)
        return name_state

    agent = _get_smart_agent(update.effective_chat.id)
    if agent.profile_exists(text):
        await update.message.reply_text(
            "⚠️ Профиль с таким именем уже существует — выбери другое имя."
        )
        return name_state

    context.user_data[_UD_NEW_NAME] = text
    context.user_data[_UD_NEW_VALUES] = {}
    context.user_data[_UD_NEW_FIELD_INDEX] = 0
    await update.message.reply_text(_profile_field_prompt(0))
    return field_state


async def _handle_profile_field_answer(
    update: Update, context: ContextTypes.DEFAULT_TYPE, field_state: int, finish
) -> int:
    text = (update.message.text or "").strip()
    index = context.user_data.get(_UD_NEW_FIELD_INDEX, 0)
    field = PROFILE_FIELDS[index]
    if text.lower() != SKIP_WORD:
        context.user_data[_UD_NEW_VALUES][field] = text

    index += 1
    if index < len(PROFILE_FIELDS):
        context.user_data[_UD_NEW_FIELD_INDEX] = index
        await update.message.reply_text(_profile_field_prompt(index))
        return field_state

    name = context.user_data.pop(_UD_NEW_NAME)
    values = context.user_data.pop(_UD_NEW_VALUES)
    context.user_data.pop(_UD_NEW_FIELD_INDEX, None)
    agent = _get_smart_agent(update.effective_chat.id)
    agent.create_profile(name, values)
    return await finish(update, context, name)


# --------------------------------------------------------------------------- #
# /smart_agent — вход в диалог (с выбором/созданием профиля, если его ещё нет)
# --------------------------------------------------------------------------- #


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
    одном из этих режимов, вход отклоняется с подсказкой сначала выйти оттуда. Если у
    чата нет активного профиля, вместо цикла вопросов сначала показывается выбор/
    создание профиля (см. докстринг модуля).
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
    agent = _get_smart_agent(chat_id)
    if agent.get_active_profile_name() is not None:
        await update.message.reply_text(SMART_AGENT_INTRO_TEXT, reply_markup=SMART_AGENT_KEYBOARD)
        return WAITING_QUESTION

    await update.message.reply_text(
        "👉 Прежде чем начать, выбери профиль (влияет на стиль и формат ответов) "
        "или создай новый:",
        reply_markup=_profile_keyboard(agent, _START_PROFILE_CALLBACK_PREFIX),
    )
    return PROFILE_PICK


async def smart_agent_start_pick_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки пикера, показанного smart_agent_command при входе
    без активного профиля — в отличие от manage-пикера (/smart_agent_profile), ведёт
    прямо в цикл вопросов WAITING_QUESTION."""
    query = update.callback_query
    await query.answer()
    data = query.data[len(_START_PROFILE_CALLBACK_PREFIX) :]
    agent = _get_smart_agent(update.effective_chat.id)

    if data == "new":
        await query.edit_message_text(
            "👉 Введи название нового профиля (например: «Консервативный»)."
        )
        return WAITING_PROFILE_NAME

    name = data[len("switch:") :]
    if not agent.profile_exists(name):
        await query.edit_message_text("⚠️ Такого профиля уже нет. Отправь /smart_agent ещё раз.")
        return ConversationHandler.END

    agent.switch_profile(name)
    await query.edit_message_text(f"✅ Активный профиль: «{name}».")
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=SMART_AGENT_INTRO_TEXT, reply_markup=SMART_AGENT_KEYBOARD
    )
    return WAITING_QUESTION


async def smart_agent_receive_profile_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _handle_new_profile_name(update, context, WAITING_PROFILE_NAME, WAITING_PROFILE_FIELD)


async def _finish_new_profile_start_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str) -> int:
    await update.message.reply_text(f"✅ Профиль «{name}» создан и активирован.")
    await update.message.reply_text(SMART_AGENT_INTRO_TEXT, reply_markup=SMART_AGENT_KEYBOARD)
    return WAITING_QUESTION


async def smart_agent_receive_profile_field(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _handle_profile_field_answer(
        update, context, WAITING_PROFILE_FIELD, _finish_new_profile_start_chat
    )


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

    agent = _get_smart_agent(chat_id)
    if agent.get_active_profile_name() is None:
        # Редкий случай: активный профиль удалили командой /smart_agent_profile_delete
        # прямо посреди этого диалога (WAITING_QUESTION) — предлагаем выбрать новый,
        # а не падаем на agent.ask() (см. его защитный RuntimeError в smart_agent.py).
        await update.message.reply_text(
            "⚠️ Активный профиль был удалён. Выбери другой или создай новый:",
            reply_markup=_profile_keyboard(agent, _START_PROFILE_CALLBACK_PREFIX),
        )
        return PROFILE_PICK

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        result = agent.ask(user_text)
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
    context.user_data.pop(_UD_NEW_NAME, None)
    context.user_data.pop(_UD_NEW_VALUES, None)
    context.user_data.pop(_UD_NEW_FIELD_INDEX, None)
    return await _exit_smart_agent_mode(update)


# --------------------------------------------------------------------------- #
# /smart_agent_profile — управление профилем в любой момент (не только при входе)
# --------------------------------------------------------------------------- #


async def smart_agent_profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Команда /smart_agent_profile — показывает текущий профиль и кнопки выбора
    другого/создания нового. Работает независимо от того, находится ли пользователь
    сейчас в режиме /smart_agent (см. докстринг модуля про порядок регистрации в
    main.py, обеспечивающий это)."""
    agent = _get_smart_agent(update.effective_chat.id)
    current = agent.get_active_profile_name()
    intro = f"👉 Текущий профиль: «{current}»." if current else "👉 Сейчас нет активного профиля."
    await update.message.reply_text(
        intro + " Выбери другой или создай новый:",
        reply_markup=_profile_keyboard(agent, _MANAGE_PROFILE_CALLBACK_PREFIX),
    )
    return MANAGE_PROFILE_PICK


async def smart_agent_manage_pick_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает нажатие кнопки manage-пикера — и как обработчик состояния
    MANAGE_PROFILE_PICK, и как ЕЩЁ ОДИН entry point того же ConversationHandler-а
    (см. build_smart_agent_profile_conversation_handler): второе нужно, чтобы кнопки
    пикера, отправленного /smart_agent_profile_delete_command ВНЕ какого-либо
    диалога, тоже обрабатывались, а не повисали без ответа."""
    query = update.callback_query
    await query.answer()
    data = query.data[len(_MANAGE_PROFILE_CALLBACK_PREFIX) :]
    agent = _get_smart_agent(update.effective_chat.id)

    if data == "new":
        await query.edit_message_text(
            "👉 Введи название нового профиля (например: «Консервативный»)."
        )
        return MANAGE_WAITING_NAME

    name = data[len("switch:") :]
    if not agent.profile_exists(name):
        await query.edit_message_text("⚠️ Такого профиля уже нет.")
        return ConversationHandler.END

    agent.switch_profile(name)
    await query.edit_message_text(f"✅ Активный профиль переключён на «{name}».")
    return ConversationHandler.END


async def manage_profile_receive_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _handle_new_profile_name(update, context, MANAGE_WAITING_NAME, MANAGE_WAITING_FIELD)


async def _finish_new_profile_manage(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str) -> int:
    await update.message.reply_text(f"✅ Профиль «{name}» создан и активирован.")
    return ConversationHandler.END


async def manage_profile_receive_field(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _handle_profile_field_answer(
        update, context, MANAGE_WAITING_FIELD, _finish_new_profile_manage
    )


async def manage_profile_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop(_UD_NEW_NAME, None)
    context.user_data.pop(_UD_NEW_VALUES, None)
    context.user_data.pop(_UD_NEW_FIELD_INDEX, None)
    await update.message.reply_text("Отменено.")
    return ConversationHandler.END


async def smart_agent_profile_set_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_profile_set <ключ> <значение> — точечно правит одно поле
    профиля без анкеты заново (см. SmartAgent.update_profile_field)."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    if len(context.args) < 2 or context.args[0] not in PROFILE_FIELDS:
        fields = ", ".join(PROFILE_FIELDS)
        await update.message.reply_text(
            "👉 Формат: /smart_agent_profile_set <ключ> <значение>\n"
            f"Доступные ключи: {fields}\n"
            "⚠️ Не сохраняй сюда номера счетов/карт, паспортные данные и другие "
            "чувствительные данные — эта команда ничего не фильтрует."
        )
        return

    key, value = context.args[0], " ".join(context.args[1:])
    agent.update_profile_field(key, value)
    await update.message.reply_text(
        f"✅ В профиле «{agent.get_active_profile_name()}» обновлено: "
        f"{PROFILE_FIELD_LABELS[key]} = {value}."
    )


async def smart_agent_profile_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_profile_show — список всех профилей чата и поля активного."""
    agent = _get_smart_agent(update.effective_chat.id)
    profiles = agent.list_profiles()
    current = agent.get_active_profile_name()

    if not profiles:
        await update.message.reply_text(
            "📭 Профилей ещё нет. Создать — /smart_agent_profile."
        )
        return

    lines = [
        f"👤 Профили ({len(profiles)}): "
        + ", ".join(f"«{name}»{' (активный)' if name == current else ''}" for name in profiles)
    ]
    if current:
        meta = agent.get_profile_meta(current) or {}
        lines.append(f"\nАктивный профиль «{current}»:")
        for field in PROFILE_FIELDS:
            value = meta.get(field) or "—"
            lines.append(f"- {PROFILE_FIELD_LABELS[field]}: {value}")
    else:
        lines.append("\nАктивного профиля нет — выбери через /smart_agent_profile.")

    await update.message.reply_text("\n".join(lines))


async def smart_agent_profile_delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_profile_delete <имя> — удаляет профиль целиком вместе со
    всей его памятью. Если удалён АКТИВНЫЙ профиль, сразу показывает пикер выбора
    нового — чат не остаётся без профиля до следующей явной команды (см. докстринг
    модуля и SmartAgent.delete_profile)."""
    if not context.args:
        await update.message.reply_text("👉 Формат: /smart_agent_profile_delete <имя>")
        return

    name = " ".join(context.args)
    agent = _get_smart_agent(update.effective_chat.id)
    was_active = agent.get_active_profile_name() == name
    if not agent.delete_profile(name):
        await update.message.reply_text(f"⚠️ Профиль «{name}» не найден.")
        return

    await update.message.reply_text(f"🗑 Профиль «{name}» удалён (вместе со всей его памятью).")
    if was_active:
        await update.message.reply_text(
            "👉 Активный профиль удалён. Выбери другой или создай новый:",
            reply_markup=_profile_keyboard(agent, _MANAGE_PROFILE_CALLBACK_PREFIX),
        )


# --------------------------------------------------------------------------- #
# Долговременная память (активного профиля)
# --------------------------------------------------------------------------- #


async def smart_agent_remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_remember <текст> — сохраняет факт в долговременную память
    активного профиля дословно (см. SmartAgent.remember)."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    if not context.args:
        await update.message.reply_text(
            "👉 Формат: /smart_agent_remember <текст факта>\n"
            "⚠️ Не сохраняй сюда номера счетов/карт, паспортные данные и другие "
            "чувствительные данные — эта команда ничего не фильтрует."
        )
        return

    fact = " ".join(context.args)
    agent.remember(fact)
    await update.message.reply_text(f"🧠 Сохранено в долговременную память: «{fact}».")


async def smart_agent_forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_forget <номер> — удаляет факт по номеру из
    /smart_agent_long_show."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("👉 Формат: /smart_agent_forget <номер>")
        return

    index = int(context.args[0])
    if not agent.forget(index):
        await update.message.reply_text(
            f"⚠️ Факта с номером {index} нет — посмотри актуальные номера в /smart_agent_long_show."
        )
        return
    await update.message.reply_text(f"🗑 Факт №{index} удалён из долговременной памяти.")


async def smart_agent_long_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_long_show — печатает все факты долговременной памяти
    активного профиля с номерами (номера использует /smart_agent_forget)."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    facts = agent.get_long_term_facts()
    if not facts:
        await update.message.reply_text(
            "📭 Долговременная память пуста. Сохранить факт — /smart_agent_remember <текст>."
        )
        return

    lines = ["🧠 Долговременная память:"]
    lines.extend(f"{i}. {fact}" for i, fact in enumerate(facts, start=1))
    await update.message.reply_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Рабочая память (текущая задача активного профиля)
# --------------------------------------------------------------------------- #


async def smart_agent_task_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_start <цель> — начинает рабочую задачу активного
    профиля, заменяя предыдущую, если она была (см. SmartAgent.start_task)."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    if not context.args:
        await update.message.reply_text("👉 Формат: /smart_agent_task_start <цель задачи>")
        return

    goal = " ".join(context.args)
    had_previous_task = agent.get_working() is not None
    agent.start_task(goal)
    note = " Предыдущая рабочая задача заменена." if had_previous_task else ""
    await update.message.reply_text(f"📋 Рабочая задача начата: «{goal}».{note}")


async def smart_agent_task_set_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_set <ключ> <значение> — кладёт данные в текущую
    рабочую задачу активного профиля. Требует уже начатую задачу
    (/smart_agent_task_start)."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "👉 Формат: /smart_agent_task_set <ключ> <значение>\n"
            "⚠️ Не сохраняй сюда номера счетов/карт, паспортные данные и другие "
            "чувствительные данные — эта команда ничего не фильтрует."
        )
        return

    key, value = context.args[0], " ".join(context.args[1:])
    if not agent.set_task_data(key, value):
        await update.message.reply_text(
            "⚠️ Сейчас нет активной рабочей задачи. Начни её — /smart_agent_task_start <цель>."
        )
        return
    await update.message.reply_text(f"📋 В рабочую задачу сохранено: {key} = {value}.")


async def smart_agent_task_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_task_show — показывает текущую рабочую задачу активного
    профиля как есть."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    working = agent.get_working()
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
    """Команда /smart_agent_task_done — завершает и очищает текущую рабочую задачу
    активного профиля."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    if not agent.finish_task():
        await update.message.reply_text("📭 Сейчас нет активной рабочей задачи.")
        return
    await update.message.reply_text("✅ Рабочая задача завершена и очищена.")


# --------------------------------------------------------------------------- #
# Наблюдаемость и переключение слоёв
# --------------------------------------------------------------------------- #


async def smart_agent_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_show — показывает профиль и три слоя памяти АКТИВНОГО
    ПРОФИЛЯ раздельно как есть, их статус включено/выключено (/smart_agent_toggle) и
    системные сообщения, реально ушедшие в LLM на последний вопрос
    (SmartAgent.get_last_context_messages) — так видно, что именно попало в каждый
    слой и что из этого реально повлияло на ответ.
    """
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    enabled_layers = agent.get_enabled_layers()

    def status(layer: str) -> str:
        return "включена" if enabled_layers[layer] else "выключена"

    current = agent.get_active_profile_name()
    lines = [f"🧠 Профиль «{current}» — слои памяти:\n"]

    meta = agent.get_profile_meta(current) or {}
    filled_meta = [f"- {PROFILE_FIELD_LABELS[f]}: {meta[f]}" for f in PROFILE_FIELDS if meta.get(f)]
    lines.append(
        f"0️⃣ Профиль ({status('profile')}): "
        + ("\n" + "\n".join(filled_meta) if filled_meta else "поля не заполнены")
    )

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
        "\nПодробности: /smart_agent_profile_show, /smart_agent_long_show, "
        "/smart_agent_task_show. Переключить слой — "
        "/smart_agent_toggle <profile|short|working|long>."
    )

    text = "\n".join(lines)
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MESSAGE_LIMIT])


async def smart_agent_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_toggle <profile|short|working|long> — включает/выключает
    слой в СБОРКЕ контекста без удаления данных (см. SmartAgent.set_layer_enabled) —
    так можно сравнить ответ на один и тот же вопрос с разным набором включённых
    слоёв. Общая настройка на весь чат, не per-profile — не требует активного
    профиля."""
    if len(context.args) != 1 or context.args[0] not in LAYER_ALIASES:
        await update.message.reply_text(
            "👉 Формат: /smart_agent_toggle <profile|short|working|long>"
        )
        return

    layer = LAYER_ALIASES[context.args[0]]
    agent = _get_smart_agent(update.effective_chat.id)
    new_value = not agent.get_enabled_layers()[layer]
    agent.set_layer_enabled(layer, new_value)
    state = "включена" if new_value else "выключена"
    await update.message.reply_text(f"✅ Слой «{LAYER_LABELS[layer]}» теперь {state} в контексте.")


async def smart_agent_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /smart_agent_reset — очищает три слоя памяти АКТИВНОГО ПРОФИЛЯ разом.
    Не трогает сам профиль, его meta, другие профили и enabled_layers (настройка
    режима, а не данные, см. SmartAgent.reset_all) — для удаления профиля целиком
    есть отдельная команда /smart_agent_profile_delete."""
    agent = _get_smart_agent(update.effective_chat.id)
    error = _require_active_profile(agent)
    if error:
        await update.message.reply_text(error)
        return

    agent.reset_all()
    await update.message.reply_text(
        f"🗑 Память профиля «{agent.get_active_profile_name()}» (все три слоя) очищена."
    )


# --------------------------------------------------------------------------- #
# Сборка обработчиков для main.py
# --------------------------------------------------------------------------- #


def build_smart_agent_conversation_handler() -> ConversationHandler:
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    return ConversationHandler(
        entry_points=[CommandHandler("smart_agent", smart_agent_command)],
        states={
            PROFILE_PICK: [
                CallbackQueryHandler(
                    smart_agent_start_pick_callback, pattern=f"^{_START_PROFILE_CALLBACK_PREFIX}"
                )
            ],
            WAITING_PROFILE_NAME: [MessageHandler(text_filter, smart_agent_receive_profile_name)],
            WAITING_PROFILE_FIELD: [MessageHandler(text_filter, smart_agent_receive_profile_field)],
            WAITING_QUESTION: [MessageHandler(text_filter, smart_agent_receive_question)],
        },
        fallbacks=[CommandHandler("cancel", smart_agent_cancel)],
    )


def build_smart_agent_profile_conversation_handler() -> ConversationHandler:
    """Собирает ConversationHandler команды /smart_agent_profile — управление
    профилем в любой момент, отдельно от цикла вопросов /smart_agent (см. докстринг
    модуля). Регистрировать в main.py НУЖНО ДО build_smart_agent_conversation_handler()
    — см. комментарий там."""
    text_filter = filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE
    manage_pick_pattern = f"^{_MANAGE_PROFILE_CALLBACK_PREFIX}"
    return ConversationHandler(
        entry_points=[
            CommandHandler("smart_agent_profile", smart_agent_profile_command),
            # Позволяет пикеру, отправленному /smart_agent_profile_delete ВНЕ
            # какого-либо диалога, тоже обрабатывать свои кнопки (см. докстринг
            # smart_agent_manage_pick_callback).
            CallbackQueryHandler(smart_agent_manage_pick_callback, pattern=manage_pick_pattern),
        ],
        states={
            MANAGE_PROFILE_PICK: [
                CallbackQueryHandler(smart_agent_manage_pick_callback, pattern=manage_pick_pattern)
            ],
            MANAGE_WAITING_NAME: [MessageHandler(text_filter, manage_profile_receive_name)],
            MANAGE_WAITING_FIELD: [MessageHandler(text_filter, manage_profile_receive_field)],
        },
        fallbacks=[CommandHandler("cancel", manage_profile_cancel)],
    )


def build_smart_agent_profile_set_handler() -> CommandHandler:
    return CommandHandler("smart_agent_profile_set", smart_agent_profile_set_command)


def build_smart_agent_profile_show_handler() -> CommandHandler:
    return CommandHandler("smart_agent_profile_show", smart_agent_profile_show_command)


def build_smart_agent_profile_delete_handler() -> CommandHandler:
    return CommandHandler("smart_agent_profile_delete", smart_agent_profile_delete_command)


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
