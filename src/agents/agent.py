"""Agent — LLM-агент как отдельная сущность, а не голый вызов API.

Инкапсулирует полный цикл обращения к LLM-провайдеру: сборку сообщений (системный
промпт + контекст диалога + вопрос пользователя), сам вызов OpenAI-совместимого API и
разбор ответа (включая fallback на случай пустого content). По умолчанию использует
main_client/MAIN_MODEL — того же провайдера и модель, что и основной поток бота
(main.py), поэтому не требует собственного выбора провайдера и не нарушает
ограничение «никакого runtime-переключения модели» (см. «Ограничения безопасности» в
CLAUDE.md): провайдер и модель агента по-прежнему настраиваются только через
MAIN_CLIENT/MAIN_MODEL в .env, а не параметром, доступным пользователю чата.

В отличие от основного потока бота (main.py), Agent хранит историю диалога — по
явному запросу пользователя (см. CLAUDE.md, раздел «Ограничения безопасности», про
то, что это осознанное исключение из правила «никакой памяти без явного запроса», а
не тихое нарушение). История хранится в JSON-файле на чат (AGENT_HISTORY_DIR/
<chat_id>.json, см. config.py) и переживает и перезапуск процесса бота, и повторный
вход в /agent. Один экземпляр Agent обслуживает один чат (см. agents/agent_command.py,
где экземпляры кэшируются по chat_id).

Хранение — ветки диалога, и ВСЁ производное от истории состояние хранится ТОЖЕ по
веткам, не только сырые сообщения. self._branches — словарь именованных веток
{имя ветки: {"messages": [...], "facts": {...}, "facts_processed_pairs": N,
"summary": ..., "summarized_pairs": M}} плюс указатель активной ветки
(self._active_branch) — это общий слой хранения для ВСЕХ стратегий управления
контекстом, а не только для Branching: до первого использования веток у чата ровно
одна ветка "main", и всё работает как раньше. Каждая ветка хранит свою
последовательность сообщений целиком без ограничений (см. get_history() и
/agent_history в agent_command.py) — сокращённый контекст, уходящий в LLM, строится
поверх сообщений активной ветки, а не заменяет их. Такое устройство выбрано вместо
альтернативы "один общий список + пометка ветки у каждого сообщения" (см. обсуждение
в задаче) как более простое: общий префикс до чекпоинта дублируется в каждой новой
ветке, но для истории чата с ботом это несущественные объёмы, а взамен все стратегии
продолжают работать с обычным плоским списком сообщений одной ветки, как и раньше, без
дополнительной фильтрации.

facts/facts_processed_pairs/summary/summarized_pairs — тоже per-branch, а не общие на
весь чат: у каждой ветки своя независимая сводка/словарь фактов, производные от её
собственной последовательности сообщений, и переключение между ветками (switch_branch)
ничего не сбрасывает и не смешивает — просто дальнейшие self._facts/self._summary/
self._facts_processed_pairs/self._summarized_pairs (см. properties ниже) начинают
относиться к новой активной ветке. Это осознанное отличие от более раннего поведения
(когда переключение ветки обнуляло summary/facts): раз у каждой ветки своя история
сообщений, у неё естественно должны быть и свои производные от этой истории summary/
facts, накапливающиеся независимо и не теряющиеся при переключении туда-обратно.
Новая ветка (create_branch) при этом стартует с ПУСТЫМИ facts/summary, а не наследует
их от ветки чекпоинта — так счётчики facts_processed_pairs/summarized_pairs гарантированно
не превышают число сообщений в новой (обычно более короткой на момент создания) ветке;
если нужно "досчитать" summary/facts для новой ветки, это происходит естественным
образом на её собственных последующих вопросах, теми же механизмами, что и для любой
другой ветки.

Реализовано как четыре property (_facts/_facts_processed_pairs/_summary/
_summarized_pairs), читающих и пишущих в self._branches[self._active_branch][...] —
весь остальной код Agent и стратегии в context_strategies.py (которые читают/пишут
agent._facts/agent._summary/agent._summarized_pairs напрямую) продолжают работать с
этими именами как раньше, не задумываясь о том, что за ними теперь стоит конкретная
ветка, а не общее для чата поле.

Управление контекстом — 4 переключаемые стратегии, вынесенные в отдельные классы в
agents/context_strategies.py (ContextStrategy и 4 реализации, реестр STRATEGIES по
имени) — см. докстринг этого модуля про сам интерфейс (build_messages/after_turn) и
про то, почему стратегии читают/пишут "protected" состояние Agent напрямую, а не
дублируют его у себя. В LLM с каждым вызовом ask() уходит не вся история активной
ветки целиком, а результат работы текущей стратегии (self._strategy, переключается
командой /agent_context — agents/agent_command.py, хранится per-chat в JSON-файле, а
не в .env, см. докстринг AGENT_CONTEXT_STRATEGY в config.py). Сама сборка LLM-вызовов
для сворачивания summary/facts (_summarize_chunk/_update_facts) и общие для всех
стратегий помощники (_active_messages/_recent_window) остаются методами Agent —
стратегии в context_strategies.py только решают, когда их вызывать и что подставить
в контекст, а не как обратиться к LLM-клиенту:

- "sliding_window" — последние AGENT_CONTEXT_RECENT_PAIRS пар вопрос-ответ активной
  ветки, всё остальное отбрасывается из контекста (не из хранения).
- "sticky_facts" — отдельный словарь ключ-значение активной ветки (self._facts: цель,
  ограничения, предпочтения, решения, договорённости), обновляемый вызовом LLM ДВАЖДЫ
  вокруг каждого вопроса пользователя — ПЕРЕД ним (before_turn, догоняет бэклог ещё не
  учтённых пар с прошлых ходов ЭТОЙ ветки, если он есть — self._facts_processed_pairs,
  _update_facts, см. ниже) и сразу ПОСЛЕ получения ответа (after_turn, учитывает
  саму эту пару) — так facts остаются актуальными и для контекста текущего
  вопроса, и сразу после него, а не только к следующему вопросу. В контекст уходит
  facts (одним системным сообщением) + те же последние AGENT_CONTEXT_RECENT_PAIRS
  пар.
- "branching" — вся история активной ветки целиком, без обрезки и без summary/facts:
  суть стратегии в независимом продолжении диалога в каждой ветке, а не в экономии
  контекста (см. ниже про сами ветки).
- "summary" (стратегия по умолчанию, в т.ч. для чатов, у которых JSON-файл истории
  ещё в старом формате без understanding о ветках/стратегиях/facts, см.
  _load_state) — прежний механизм: более старая часть периодически сворачивается в
  текстовую сводку активной ветки (self._summary), и каждый запрос содержит summary
  (если он уже есть) плюс "свежий хвост" — все пары вопрос-ответ активной ветки, ещё
  не попавшие в summary. AGENT_CONTEXT_RECENT_PAIRS — минимальный размер этого
  хвоста: столько последних пар никогда не сворачиваются. Как только сверх него
  накапливается ещё AGENT_SUMMARY_CHUNK_PAIRS пар, они сворачиваются в summary
  отдельным вызовом того же main_client/MAIN_MODEL (с AGENT_SUMMARY_SYSTEM_PROMPT
  вместо основного SYSTEM_PROMPT) — старая сводка передаётся вместе с новым блоком,
  и модель возвращает объединённую сводку (инкрементальный rollup, а не пересборка с
  нуля при каждом срабатывании). "Свежий хвост" — это буквально «всё, что ещё не
  попало в summary» (self._active_messages()[2 * self._summarized_pairs :]), а не
  отдельное окно фиксированного размера.

Переключение стратегии (Agent.set_strategy) не трогает summary/facts — они продолжают
накапливаться в фоне своей стратегии (summary — только пока активна "summary", facts —
только пока активна "sticky_facts", см. _before_turn/_after_turn), и снова становятся
актуальными, если пользователь вернётся к той же стратегии позже. Это не то же самое,
что переключение ВЕТКИ (см. ниже) — здесь речь про одну и ту же ветку, только с другой
стратегией поверх неё.

Ветки диалога (стратегия "branching"). checkpoint (create_checkpoint) — это просто
запомненная длина активной ветки на момент вызова; create_branch создаёт одну новую
именованную ветку как независимую копию сообщений до этой длины (с пустыми facts/
summary — см. выше) — дальше она растёт независимо; вызов можно повторить с тем же
чекпоинтом и другим именем ветки сколько угодно раз, чтобы получить несколько веток
от одной точки (в т.ч. классические две). switch_branch переключает, какая ветка
активна для последующих ask() — и, соответственно, чьи facts/summary видны через
self._facts/self._summary и т.п. Команды-обёртки (/agent_checkpoint, /agent_branch,
/agent_switch_branch, см. agents/agent_command.py) осмысленны и разрешены только при
активной стратегии "branching" — иначе у них нет чёткой семантики (что значит "окно"
или "summary" сразу в нескольких ветках), поэтому Agent сам не проверяет
self._strategy в create_checkpoint/create_branch/switch_branch — эту проверку делает
вызывающий код команд.

Полная сырая история при этом продолжает храниться на диске без ограничений — общий
инвариант, не привязанный к конкретной стратегии. /agent_history печатает историю
активной ветки как есть, а не сокращённый контекст. /agent_reset очищает все ветки
(вместе с их facts/summary), чекпоинты, но НЕ трогает выбранную стратегию — это
настройка режима работы с агентом, а не часть очищаемого диалога.

Если вызов на сворачивание summary или на обновление facts сам завершится ошибкой
API (или невалидным JSON — для facts), это не должно ронять уже полученный ответ
пользователю на его текущий вопрос: _maybe_update_summary/_update_facts — единственные
места в Agent, где исключения OpenAI SDK перехватываются внутри самого класса (в
остальном ask() их сознательно не ловит, см. ниже), — при ошибке summary/facts
остаются прежними, а попытка обновить их повторяется при одном из следующих вопросов.

Ошибки OpenAI SDK (аутентификация, лимиты, таймауты и т.д.) в основном вызове ask()
намеренно не перехватываются здесь — агент отвечает только за построение запроса и
разбор ответа, а перевод ошибки API в сообщение пользователю на русском — забота
вызывающего Telegram-обработчика (agents/agent_command.py), по тому же принципу, что
и handle_message в main.py.

ask() дополнительно считает и возвращает (в AgentAnswer) статистику по токенам —
DeepSeek/Kimi API не даёт токены текущего вопроса и ответа по отдельности, только
usage.prompt_tokens (весь промпт, отправленный в этом вызове: системный промпт +
контекст, построенный текущей стратегией, + вопрос) и usage.completion_tokens (ответ),
поэтому:
- "токены ответа" — это usage.completion_tokens как есть;
- "токены контекста" — это usage.prompt_tokens как есть (весь промпт этого вызова, а
  не размер сырой истории на диске — см. "Управление контекстом" выше про то, что
  именно строит текущая стратегия);
- "токены нового вопроса" — только приблизительно, по длине текста вопроса (len // 2,
  без обращения к API — делитель подобран эмпирически под русский текст: кириллица в
  BPE-токенайзерах вроде cl100k обычно кодируется куда менее эффективно, чем латиница,
  ближе к ~2 символам на токен, а не к ~4, как для английского).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import (
    AGENT_CONTEXT_RECENT_PAIRS,
    AGENT_CONTEXT_STRATEGY,
    AGENT_FACTS_MAX_TOKENS,
    AGENT_FACTS_SYSTEM_PROMPT,
    AGENT_HISTORY_DIR,
    AGENT_STRATEGY_LABELS,
    AGENT_SUMMARY_CHUNK_PAIRS,
    AGENT_SUMMARY_MAX_TOKENS,
    AGENT_SUMMARY_SYSTEM_PROMPT,
    MAIN_MODEL,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from providers.main_client import main_client

from .context_strategies import STRATEGIES

logger = logging.getLogger(__name__)

# Имя единственной ветки, с которой начинается любой чат (и в которую превращается
# плоский список сообщений при чтении файла истории в старом, "доветочном" формате).
DEFAULT_BRANCH = "main"


@dataclass
class AgentAnswer:
    """Результат ask() — текст ответа плюс статистика по токенам (см. докстринг модуля)."""

    text: str
    request_tokens_approx: int
    context_tokens: int | None
    response_tokens: int | None


class Agent:
    """Агент с историей диалога, сохраняемой в JSON-файле на чат (см. докстринг модуля).

    Сырая история хранится по веткам без ограничения по длине (по явному решению —
    см. CLAUDE.md), но в LLM с каждым вызовом ask() уходит не она вся, а результат
    работы текущей стратегии управления контекстом (self._strategy) — см. "Управление
    контекстом" в докстринге модуля. facts/summary и их счётчики тоже хранятся по
    веткам (см. докстринг модуля) — доступ к ним для активной ветки дают properties
    _facts/_facts_processed_pairs/_summary/_summarized_pairs ниже.
    """

    def __init__(
        self,
        chat_id: int | str,
        client=main_client,
        model: str = MAIN_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        history_dir: str = AGENT_HISTORY_DIR,
        recent_pairs: int = AGENT_CONTEXT_RECENT_PAIRS,
        summary_chunk_pairs: int = AGENT_SUMMARY_CHUNK_PAIRS,
        summary_max_tokens: int = AGENT_SUMMARY_MAX_TOKENS,
        summary_system_prompt: str = AGENT_SUMMARY_SYSTEM_PROMPT,
        facts_max_tokens: int = AGENT_FACTS_MAX_TOKENS,
        facts_system_prompt: str = AGENT_FACTS_SYSTEM_PROMPT,
        default_strategy: str = AGENT_CONTEXT_STRATEGY,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        # chat_id используется только для имени файла — обычно это реальный chat_id
        # (int), но agents/compare_command.py передаёт составной строковый
        # идентификатор (f"{chat_id}_compare_<стратегия>"), чтобы у каждой из трёх
        # "теневых" стратегий сравнения был свой файл истории, не пересекающийся ни
        # с обычным /agent того же чата, ни друг с другом.
        self._history_path = Path(history_dir) / f"{chat_id}.json"
        self._recent_pairs = recent_pairs
        self._summary_chunk_pairs = summary_chunk_pairs
        self._summary_max_tokens = summary_max_tokens
        self._summary_system_prompt = summary_system_prompt
        self._facts_max_tokens = facts_max_tokens
        self._facts_system_prompt = facts_system_prompt
        self._default_strategy = default_strategy
        (
            self._branches,
            self._active_branch,
            self._checkpoints,
            self._strategy,
        ) = self._load_state()

    @staticmethod
    def _new_branch_state(messages: list[dict[str, str]] | None = None) -> dict:
        """Пустое (или с готовыми messages) состояние одной ветки — сырые сообщения
        плюс facts/facts_processed_pairs/summary/summarized_pairs именно ЭТОЙ ветки
        (см. докстринг модуля про то, почему они per-branch, а не общие на чат)."""
        return {
            "messages": messages if messages is not None else [],
            "facts": {},
            "facts_processed_pairs": 0,
            "summary": None,
            "summarized_pairs": 0,
        }

    def _empty_state(
        self, messages: list[dict[str, str]] | None = None
    ) -> tuple[dict[str, dict], str, dict[str, dict], str]:
        return (
            {DEFAULT_BRANCH: self._new_branch_state(messages)},
            DEFAULT_BRANCH,
            {},
            self._default_strategy,
        )

    def _load_state(self) -> tuple[dict[str, dict], str, dict[str, dict], str]:
        """Читает состояние чата из JSON-файла — ветки (с их facts/summary),
        активную ветку, чекпоинты и стратегию.

        Поддерживает несколько форматов файла, от старого к новому (более старые
        форматы также содержали поле "last_context_tokens" — статистику "токены по
        разнице", которой больше нет; оно просто игнорируется при чтении):
        1. Голый список сообщений (до добавления статистики по токенам) — становится
           единственной веткой DEFAULT_BRANCH с пустыми facts/summary.
        2. {"messages": [...], "summary": ..., "summarized_pairs": ...} (до добавления
           веток/стратегий/facts) — messages и summary/summarized_pairs переносятся в
           единственную ветку DEFAULT_BRANCH, стратегия — default_strategy
           (AGENT_CONTEXT_STRATEGY из config.py, по умолчанию "summary" — чтобы
           поведение уже существующих чатов не изменилось молча).
        3. {"branches": {имя: [сообщения]}, "facts": ..., "facts_processed_pairs": ...,
           "summary": ..., "summarized_pairs": ...} (ветки уже есть, но facts/summary
           ещё общие на весь чат, а не per-branch) — тогда, т.к. переключение ветки
           раньше как раз обнуляло facts/summary, сохранённые значения относятся
           именно к активной на момент сохранения ветке: переносим их только туда,
           остальные ветки получают пустые facts/summary (см. _new_branch_state).
        4. Текущий формат — {"branches": {имя: {"messages": [...], "facts": ...,
           "facts_processed_pairs": ..., "summary": ..., "summarized_pairs": ...}}}:
           у каждой ветки уже своё собственное facts/summary-состояние, читаем как
           есть, с валидацией отдельных полей.
        Отсутствие/повреждение файла — пустое состояние с одной веткой DEFAULT_BRANCH.
        """
        try:
            raw = self._history_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._empty_state()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Повреждённый файл истории агента %s — начинаю с пустой истории.",
                self._history_path,
            )
            return self._empty_state()

        if isinstance(data, list):
            return self._empty_state(data)

        if not isinstance(data, dict):
            return self._empty_state()

        branches_raw = data.get("branches")
        if isinstance(branches_raw, dict) and branches_raw:
            active_branch = data.get("active_branch")
            if not isinstance(active_branch, str) or active_branch not in branches_raw:
                active_branch = next(iter(branches_raw))
            checkpoints = data.get("checkpoints")
            checkpoints = checkpoints if isinstance(checkpoints, dict) else {}
            strategy = data.get("strategy")
            strategy = strategy if strategy in AGENT_STRATEGY_LABELS else self._default_strategy

            first_entry = next(iter(branches_raw.values()))
            if isinstance(first_entry, dict) and "messages" in first_entry:
                # Формат 4 — у каждой ветки уже своё facts/summary-состояние.
                branches = {}
                for name, entry in branches_raw.items():
                    if not isinstance(entry, dict):
                        branches[name] = self._new_branch_state()
                        continue
                    messages = entry.get("messages")
                    messages = messages if isinstance(messages, list) else []
                    facts = entry.get("facts")
                    facts = facts if isinstance(facts, dict) else {}
                    facts_processed_pairs = entry.get("facts_processed_pairs")
                    facts_processed_pairs = (
                        facts_processed_pairs if isinstance(facts_processed_pairs, int) else 0
                    )
                    summary = entry.get("summary")
                    summary = summary if isinstance(summary, str) else None
                    summarized_pairs = entry.get("summarized_pairs")
                    summarized_pairs = (
                        summarized_pairs if isinstance(summarized_pairs, int) else 0
                    )
                    branches[name] = {
                        "messages": messages,
                        "facts": facts,
                        "facts_processed_pairs": facts_processed_pairs,
                        "summary": summary,
                        "summarized_pairs": summarized_pairs,
                    }
            else:
                # Формат 3 — ветки уже есть, но facts/summary ещё общие на весь чат
                # (переносим их только в активную ветку, см. докстринг метода).
                global_facts = data.get("facts")
                global_facts = global_facts if isinstance(global_facts, dict) else {}
                global_facts_processed_pairs = data.get("facts_processed_pairs")
                global_facts_processed_pairs = (
                    global_facts_processed_pairs
                    if isinstance(global_facts_processed_pairs, int)
                    else 0
                )
                global_summary = data.get("summary")
                global_summary = global_summary if isinstance(global_summary, str) else None
                global_summarized_pairs = data.get("summarized_pairs")
                global_summarized_pairs = (
                    global_summarized_pairs if isinstance(global_summarized_pairs, int) else 0
                )

                branches = {}
                for name, messages in branches_raw.items():
                    messages = messages if isinstance(messages, list) else []
                    if name == active_branch:
                        branches[name] = {
                            "messages": messages,
                            "facts": global_facts,
                            "facts_processed_pairs": global_facts_processed_pairs,
                            "summary": global_summary,
                            "summarized_pairs": global_summarized_pairs,
                        }
                    else:
                        branches[name] = self._new_branch_state(messages)

            return branches, active_branch, checkpoints, strategy

        # Формат 2 — плоский список сообщений, до добавления веток/стратегий/facts.
        messages = data.get("messages")
        messages = messages if isinstance(messages, list) else []
        summary = data.get("summary")
        summary = summary if isinstance(summary, str) else None
        summarized_pairs = data.get("summarized_pairs")
        summarized_pairs = summarized_pairs if isinstance(summarized_pairs, int) else 0
        branch_state = self._new_branch_state(messages)
        branch_state["summary"] = summary
        branch_state["summarized_pairs"] = summarized_pairs
        return {DEFAULT_BRANCH: branch_state}, DEFAULT_BRANCH, {}, self._default_strategy

    def _save_history(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "branches": self._branches,
            "active_branch": self._active_branch,
            "checkpoints": self._checkpoints,
            "strategy": self._strategy,
        }
        self._history_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        """Полностью очищает историю диалога этого чата: все ветки (вместе с их
        facts/summary) и чекпоинты (см. /agent_reset в agent_command.py). Выбранная
        стратегия (self._strategy) НЕ сбрасывается — это настройка режима работы с
        агентом, а не часть очищаемого диалога.
        """
        self._branches = {DEFAULT_BRANCH: self._new_branch_state()}
        self._active_branch = DEFAULT_BRANCH
        self._checkpoints = {}
        self._history_path.unlink(missing_ok=True)

    def _active_messages(self) -> list[dict[str, str]]:
        """Сырые сообщения активной ветки (НЕ копия — внутреннее использование в
        Agent и в стратегиях context_strategies.py, которые читают/пишут его как
        часть общего с Agent состояния, см. докстринг context_strategies.py).
        Внешнему коду (agent_command.py) отдаём только копии — см. get_history()."""
        return self._branches[self._active_branch]["messages"]

    def get_history(self) -> list[dict[str, str]]:
        """Возвращает копию сохранённой истории активной ветки (см. /agent_history в
        agent_command.py).

        Копия, а не сама внутренняя история — чтобы вызывающий код не мог случайно
        исказить состояние агента через возвращённый список.
        """
        return list(self._active_messages())

    # --- facts/summary активной ветки (см. докстринг модуля про то, почему per-branch) --- #

    @property
    def _facts(self) -> dict[str, str]:
        return self._branches[self._active_branch]["facts"]

    @_facts.setter
    def _facts(self, value: dict[str, str]) -> None:
        self._branches[self._active_branch]["facts"] = value

    @property
    def _facts_processed_pairs(self) -> int:
        return self._branches[self._active_branch]["facts_processed_pairs"]

    @_facts_processed_pairs.setter
    def _facts_processed_pairs(self, value: int) -> None:
        self._branches[self._active_branch]["facts_processed_pairs"] = value

    @property
    def _summary(self) -> str | None:
        return self._branches[self._active_branch]["summary"]

    @_summary.setter
    def _summary(self, value: str | None) -> None:
        self._branches[self._active_branch]["summary"] = value

    @property
    def _summarized_pairs(self) -> int:
        return self._branches[self._active_branch]["summarized_pairs"]

    @_summarized_pairs.setter
    def _summarized_pairs(self, value: int) -> None:
        self._branches[self._active_branch]["summarized_pairs"] = value

    # --- Стратегия управления контекстом (см. /agent_context в agent_command.py) --- #

    def get_strategy(self) -> str:
        return self._strategy

    def set_strategy(self, strategy: str) -> None:
        """Переключает стратегию управления контекстом для этого чата. Не проверяет
        strategy на допустимость — эту проверку делает вызывающий код команды (см.
        AGENT_STRATEGY_LABELS в config.py).

        summary/facts НЕ сбрасываются — они не теряются и снова станут актуальны,
        если пользователь вернётся к той же стратегии позже (см. докстринг модуля).
        """
        self._strategy = strategy
        self._save_history()

    # --- Ветки диалога (см. /agent_checkpoint, /agent_branch, /agent_switch_branch) --- #

    def get_active_branch(self) -> str:
        return self._active_branch

    def list_branches(self) -> list[str]:
        return list(self._branches.keys())

    def branch_exists(self, name: str) -> bool:
        return name in self._branches

    def checkpoint_exists(self, name: str) -> bool:
        return name in self._checkpoints

    def create_checkpoint(self, name: str) -> None:
        """Помечает текущую длину активной ветки как чекпоинт с именем name.
        Перезаписывает существующий чекпоинт с тем же именем, если он был."""
        self._checkpoints[name] = {
            "branch": self._active_branch,
            "length": len(self._active_messages()),
        }
        self._save_history()

    def create_branch(self, checkpoint_name: str, branch_name: str) -> None:
        """Создаёт новую ветку branch_name как независимую копию сообщений ветки
        чекпоинта checkpoint_name вплоть до его длины на момент создания — см.
        "Хранение — ветки диалога" в докстринге модуля про то, почему это дублирование
        общего префикса, а не общий список с пометками. Facts/summary новой ветки
        начинаются с чистого листа (не наследуются от ветки чекпоинта — см. докстринг
        модуля про то, почему). Вызов можно повторить с тем же checkpoint_name и
        другим branch_name сколько угодно раз, чтобы получить несколько веток от
        одного чекпоинта (в т.ч. классический сценарий "две ветки от одной точки") —
        метод намеренно создаёт только одну ветку за раз, а не фиксированные две,
        чтобы не навязывать это число. Не переключает активную ветку — для этого
        switch_branch/agent_switch_branch. Вызывающий код обязан заранее проверить
        checkpoint_exists(checkpoint_name) и что branch_name ещё не занято (см.
        branch_exists) — сам метод этого не проверяет.
        """
        checkpoint = self._checkpoints[checkpoint_name]
        prefix = self._branches[checkpoint["branch"]]["messages"][: checkpoint["length"]]
        self._branches[branch_name] = self._new_branch_state(list(prefix))
        self._save_history()

    def switch_branch(self, name: str) -> None:
        """Переключает активную ветку. НЕ сбрасывает summary/summarized_pairs/facts/
        facts_processed_pairs — они хранятся per-branch (см. докстринг модуля), и
        после переключения self._facts/self._summary и т.п. просто начинают
        относиться к новой активной ветке — её собственное facts/summary (если уже
        накапливалось) остаётся на месте и доступно сразу. Вызывающий код обязан
        заранее проверить branch_exists(name).
        """
        self._active_branch = name
        self._save_history()

    def get_facts(self) -> dict[str, str]:
        return dict(self._facts)

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        """Достаёт prompt_tokens/completion_tokens из ответа API — источник для
        context_tokens/response_tokens в AgentAnswer (см. докстринг модуля)."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return None
        logger.debug("Usage от API (агент): %r", usage)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _recent_window(self, active: list[dict[str, str]]) -> list[dict[str, str]]:
        """Последние 2 * recent_pairs сообщений активной ветки (используется в
        sliding_window и sticky_facts) — общий помощник, а не inline-срез в двух
        местах, т.к. "срез отрицательным нулём" (active[-0:]) вернул бы весь список,
        а не пустой, если recent_pairs == 0."""
        window_size = 2 * self._recent_pairs
        return active[-window_size:] if window_size > 0 else []

    def _build_context_messages(self) -> list[dict[str, str]]:
        """Собирает то, что уходит в LLM вместо полной истории активной ветки: системный
        промпт (общий для всех стратегий) + то, что вернёт build_messages() текущей
        стратегии (agents/context_strategies.py, реестр STRATEGIES по self._strategy)
        — см. "Управление контекстом" в докстринге модуля.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(STRATEGIES[self._strategy].build_messages(self))
        return messages

    def _summarize_chunk(self, chunk: list[dict[str, str]]) -> str:
        """Сворачивает старый summary (если есть) + новый блок сообщений chunk в один
        обновлённый summary — один вызов main_client/MAIN_MODEL с
        AGENT_SUMMARY_SYSTEM_PROMPT (см. докстринг модуля). Может выбросить
        исключение OpenAI SDK — перехват на совести вызывающего.
        """
        rendered_chunk = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Агент'}: {m['content']}" for m in chunk
        )
        user_content = (
            f"Текущая сводка:\n{self._summary or '(пока пустая — это первое сворачивание)'}"
            f"\n\nНовый фрагмент диалога, который нужно включить в сводку:\n{rendered_chunk}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._summary_system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self._summary_max_tokens,
            timeout=self._timeout,
        )
        content = response.choices[0].message.content
        return content or (self._summary or "")

    def _maybe_update_summary(self) -> None:
        """Сворачивает в summary очередные блоки активной ветки по summary_chunk_pairs
        пар, пока за пределами recent_pairs остаётся хотя бы summary_chunk_pairs
        несвёрнутых пар (см. "Управление контекстом" в докстринге модуля). Ошибка
        вызова на сворачивание не прерывает ask() и не портит уже полученный ответ
        пользователю — просто логируется, summary/summarized_pairs остаются прежними,
        и хвост продолжает расти до следующей попытки.
        """
        active = self._active_messages()
        while True:
            total_pairs = len(active) // 2
            unsummarized_pairs = total_pairs - self._summarized_pairs
            if unsummarized_pairs - self._recent_pairs < self._summary_chunk_pairs:
                return

            chunk_start = 2 * self._summarized_pairs
            chunk_end = chunk_start + 2 * self._summary_chunk_pairs
            chunk = active[chunk_start:chunk_end]

            try:
                new_summary = self._summarize_chunk(chunk)
            except Exception:  # noqa: BLE001 — сворачивание не должно ронять ask()
                logger.warning(
                    "Не удалось обновить summary агента (chat history %s) — "
                    "попробую снова при следующем вопросе.",
                    self._history_path,
                    exc_info=True,
                )
                return

            self._summary = new_summary
            self._summarized_pairs += self._summary_chunk_pairs

    def _call_facts_api(self, user_content: str, max_tokens: int) -> dict[str, str] | None:
        """Один вызов LLM на обновление facts (response_format={"type": "json_object"})
        — общий для обеих попыток в _update_facts (обычной и повторной с удвоенным
        max_tokens). Возвращает распарсенный словарь фактов, None если API вернул
        пустой content (не ошибка — трактуем как "нет обновления" и не меняем facts,
        см. _update_facts) или содержимое не оказалось JSON-объектом. Может выбросить
        json.JSONDecodeError (в т.ч. из-за обрезки ответа по max_tokens — см.
        AGENT_FACTS_MAX_TOKENS в config.py) либо исключение OpenAI SDK — оба
        перехватывает вызывающий _update_facts().
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._facts_system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            timeout=self._timeout,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        return {str(key): str(value) for key, value in parsed.items()}

    def _update_facts(self) -> None:
        """Обновляет self._facts (активной ветки) вызовом LLM с
        AGENT_FACTS_SYSTEM_PROMPT (см. докстринг модуля про стратегию
        "sticky_facts") — вызывается из StickyFactsStrategy.before_turn() (перед
        отправкой вопроса, чтобы догнать бэклог с прошлых ходов) И из
        StickyFactsStrategy.after_turn() (сразу после получения ответа на текущий,
        чтобы учесть саму эту пару без задержки до следующего вопроса) — см.
        докстринг StickyFactsStrategy про то, зачем именно два вызова. В отличие от
        summary, который сворачивается раз в chunk, оба вызова здесь работают с
        одним и тем же понятием "бэклог" (см. ниже) и не отличаются по логике —
        метод сам не знает, из какого из двух хуков его вызвали.

        В LLM уходит не только последняя пара вопрос-ответ, а весь "бэклог" —
        сообщения активной ветки, ещё не учтённые в facts ЭТОЙ ветки
        (self._active_messages()[2 * self._facts_processed_pairs :]), по тому же
        принципу, что несвёрнутый хвост в summary (self._summarized_pairs). При
        вызове из before_turn (до отправки вопроса) в бэклог могут попасть только
        пары ПРЕДЫДУЩИХ ходов — текущий вопрос ещё не задан; при вызове из
        after_turn (после ответа) в бэклоге обычно ровно одна свежая пара — та, что
        только что добавилась (если только предыдущий before_turn не подвёл, тогда
        бэклог шире). Если бэклог уже пуст (второй из двух вызовов не находит новых
        пар — обычный случай), функция просто ничего не делает (см. ранний return
        ниже) — лишнего вызова к LLM не происходит. Бэклог поверх этого нужен для
        устойчивости к ошибкам: если сбой случился на прошлом ходу, пара
        вопрос-ответ из него не теряется насовсем — счётчик facts_processed_pairs
        продвигается ТОЛЬКО при успешном обновлении, поэтому пропущенные пары
        копятся в бэклоге до следующей успешной попытки (а не пропадают, как было
        бы, если бы каждый вызов видел только одну конкретную пару).

        Ответ модели — не текст, а JSON-объект (response_format=
        {"type": "json_object"}, см. _call_facts_api), который целиком заменяет
        старый словарь фактов (модель сама решает, что оставить/обновить/сжать — см.
        правила в самом промпте, включая лимит на число ключей, не позволяющий
        словарю расти неограниченно). Как и json_object в research/constraints.py
        (сценарий 2), формат проверен на DeepSeek; для Kimi отдельно не проверялся.

        Если json.loads падает (типичная причина — ответ обрезан по max_tokens,
        растущий словарь фактов не влезает в лимит), делается ОДИН повтор с
        удвоенным max_tokens прямо сейчас, не дожидаясь следующего вопроса
        пользователя — это чинит подавляющее большинство таких обрезок. Любая другая
        ошибка (или повторный сбой) не должна ронять уже полученный ответ
        пользователю — facts и facts_processed_pairs остаются прежними, попытка
        обновить их повторяется на следующем вопросе с тем же (уже подросшим)
        бэклогом (тот же принцип, что и в _maybe_update_summary).
        """
        unprocessed = self._active_messages()[2 * self._facts_processed_pairs :]
        if not unprocessed:
            return

        rendered = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Агент'}: {m['content']}"
            for m in unprocessed
        )
        user_content = (
            f"Текущий словарь фактов (JSON):\n{json.dumps(self._facts, ensure_ascii=False)}"
            f"\n\nНовые пары вопрос-ответ, ещё не учтённые в словаре:\n{rendered}"
        )

        try:
            facts = self._call_facts_api(user_content, self._facts_max_tokens)
        except json.JSONDecodeError:
            logger.info(
                "Ответ на обновление facts агента (chat history %s) не распарсился "
                "как JSON (похоже на обрезку по max_tokens=%d) — повторяю запрос с "
                "удвоенным лимитом.",
                self._history_path,
                self._facts_max_tokens,
            )
            try:
                facts = self._call_facts_api(user_content, self._facts_max_tokens * 2)
            except Exception:  # noqa: BLE001 — обновление facts не должно ронять ask()
                logger.warning(
                    "Не удалось обновить facts агента даже после повтора (chat "
                    "history %s) — попробую снова при следующем вопросе.",
                    self._history_path,
                    exc_info=True,
                )
                return
        except Exception:  # noqa: BLE001 — обновление facts не должно ронять ask()
            logger.warning(
                "Не удалось обновить facts агента (chat history %s) — "
                "попробую снова при следующем вопросе.",
                self._history_path,
                exc_info=True,
            )
            return

        if facts is not None:
            self._facts = facts
        self._facts_processed_pairs += len(unprocessed) // 2

    def _before_turn(self) -> None:
        """Побочное действие текущей стратегии ДО сборки контекста и отправки
        вопроса в LLM — делегирует в before_turn() текущей стратегии (см.
        agents/context_strategies.py): для "sticky_facts" догоняет бэклог ещё не
        учтённых в facts пар с прошлых ходов (agent._update_facts()), чтобы контекст
        ТЕКУЩЕГО вопроса строился на уже актуальных facts, а не устаревших; для
        остальных стратегий — ничего.
        """
        STRATEGIES[self._strategy].before_turn(self)

    def _after_turn(self, user_text: str, answer_text: str) -> None:
        """Побочное действие текущей стратегии после успешно сохранённой пары
        вопрос-ответ — делегирует в after_turn() текущей стратегии (см.
        agents/context_strategies.py): обновление summary для "summary" и facts для
        "sticky_facts" (учитывает саму эту, только что дописанную пару — см. докстринг
        StickyFactsStrategy про то, почему facts обновляются и здесь, и в
        _before_turn()); для "sliding_window"/"branching" — ничего.
        """
        STRATEGIES[self._strategy].after_turn(self, user_text, answer_text)

    def ask(self, user_text: str) -> AgentAnswer:
        """Отправляет вопрос пользователя в LLM вместе с контекстом чата, построенным
        текущей стратегией (см. "Управление контекстом" в докстринге модуля, не вся
        сохранённая история активной ветки), и возвращает ответ.

        Перед сборкой контекста вызывает _before_turn() — для "sticky_facts" это
        означает, что бэклог ещё не учтённых facts с прошлых ходов (в т.ч. из-за
        прошлых сбоев) догоняется ДО того, как контекст текущего вопроса будет
        собран, а не только после получения ответа на него. После получения ответа
        _after_turn() для "sticky_facts" дополнительно учитывает саму эту, только
        что дописанную пару — так facts остаются актуальными и для текущего вопроса
        (через _before_turn), и сразу после него, не дожидаясь следующего вопроса
        (см. докстринги _before_turn()/_after_turn()/StickyFactsStrategy).

        Может выбросить исключение OpenAI SDK (сетевые ошибки, ошибки API и т.д.) —
        перехват и перевод в сообщение пользователю на русском выполняет вызывающий
        код, см. докстринг модуля. История дописывается только после успешного
        ответа API — неудачный вызов не искажает сохранённый диалог. Статистика по
        токенам в возвращённом AgentAnswer — см. докстринг модуля про то, откуда
        берётся каждое из значений.
        """
        self._before_turn()

        # Делитель 2, а не общепринятые для английского языка 4 символа/токен — см.
        # докстринг модуля про то, почему кириллица в BPE-токенайзерах менее эффективна.
        request_tokens_approx = len(user_text) // 2

        messages = self._build_context_messages()
        messages.append({"role": "user", "content": user_text})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_output_tokens,
            timeout=self._timeout,
        )
        content = response.choices[0].message.content
        answer = content or "Модель вернула пустой ответ. Попробуй переформулировать вопрос."

        usage = self._extract_usage(response)
        context_tokens = usage["prompt_tokens"] if usage else None
        response_tokens = usage["completion_tokens"] if usage else None

        self._active_messages().append({"role": "user", "content": user_text})
        self._active_messages().append({"role": "assistant", "content": answer})
        self._after_turn(user_text, answer)
        self._save_history()

        return AgentAnswer(
            text=answer,
            request_tokens_approx=request_tokens_approx,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
        )
