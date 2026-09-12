"""Agent — LLM-агент как отдельная сущность, а не голый вызов API.

Инкапсулирует полный цикл обращения к LLM-провайдеру: сборку сообщений (системный
промпт + история диалога + вопрос пользователя), сам вызов OpenAI-совместимого API и
разбор ответа (включая fallback на случай пустого content). По умолчанию использует
main_client/MAIN_MODEL — того же провайдера и модель, что и основной поток бота
(main.py), поэтому не требует собственного выбора провайдера и не нарушает
ограничение «никакого runtime-переключения модели» (см. «Ограничения безопасности» в
CLAUDE.md): провайдер и модель агента по-прежнему настраиваются только через
MAIN_CLIENT/MAIN_MODEL в .env, а не параметром, доступным пользователю чата.

В отличие от основного потока бота (main.py) и от прежней stateless-версии этого
агента, Agent хранит историю диалога — по явному запросу пользователя (см.
CLAUDE.md, раздел «Ограничения безопасности», про то, что это осознанное исключение
из правила «никакой памяти без явного запроса», а не тихое нарушение). История
хранится в JSON-файле на чат (AGENT_HISTORY_DIR/<chat_id>.json, см. config.py) и
переживает и перезапуск процесса бота, и повторный вход в /agent — конструктор
загружает файл, если он есть, а ask() дописывает в него очередную пару
"вопрос-ответ" после каждого успешного обращения к API. Один экземпляр Agent
обслуживает один чат (см. agents/agent_command.py, где экземпляры кэшируются по
chat_id) — это не тот же объект, что раньше переиспользовался на все чаты сразу,
т.к. теперь у каждого чата собственное состояние (история).

Управление контекстом. Чтобы в LLM не уходила вся история целиком по мере её
роста, более старая часть периодически сворачивается в текстовую сводку
(summary), и в каждый запрос идёт не вся история, а summary (если он уже есть)
плюс "свежий хвост" — все пары вопрос-ответ, ещё не попавшие в summary.
AGENT_CONTEXT_RECENT_PAIRS (config.py) задаёт минимальный размер этого хвоста —
столько последних пар никогда не сворачиваются; когда сверх них накапливается
ещё AGENT_SUMMARY_CHUNK_PAIRS пар, они сворачиваются в summary отдельным вызовом
того же main_client/MAIN_MODEL (с AGENT_SUMMARY_SYSTEM_PROMPT вместо основного
SYSTEM_PROMPT) — старый summary передаётся вместе с новым блоком, и модель
возвращает объединённую сводку (инкрементальный rollup, а не пересборка с нуля).
Полная сырая история при этом по-прежнему хранится на диске без ограничений (см.
get_history() и /agent_history в agent_command.py) — summary влияет только на то,
что уходит в LLM, а не на то, что хранится. "Свежий хвост" — это буквально "всё,
что ещё не попало в summary" (self._history[2 * self._summarized_pairs :]), а не
отдельное жёстко фиксированное окно, поэтому "дыры" между уже свёрнутой частью и
несвёрнутым хвостом не бывает — см. _maybe_update_summary(). Если вызов на
сворачивание сам завершится ошибкой API, summary остаётся прежним, а хвост
просто растёт до следующей попытки при одном из следующих вопросов — ответ
пользователю, ради которого была вызвана эта попытка, к этому моменту уже
получен и не теряется (см. try/except внутри _maybe_update_summary()).

Ошибки OpenAI SDK (аутентификация, лимиты, таймауты и т.д.) намеренно не
перехватываются здесь — агент отвечает только за построение запроса и разбор ответа,
а перевод ошибки API в сообщение пользователю на русском — забота вызывающего
Telegram-обработчика (agents/agent_command.py), по тому же принципу, что и
handle_message в main.py.

ask() дополнительно считает и возвращает (в AgentAnswer) статистику по токенам —
DeepSeek/Kimi API не даёт токены текущего вопроса и ответа по отдельности, только
usage.prompt_tokens (весь промпт: системный промпт + история + вопрос целиком) и
usage.completion_tokens (ответ), поэтому:
- "токены ответа" — это usage.completion_tokens как есть;
- "токены всей истории" — это usage.prompt_tokens как есть (весь промпт, отправленный
  в этом вызове);
- "токены нового вопроса" оцениваются двумя способами: приблизительно, по длине
  текста вопроса (len // 2, без обращения к API — делитель подобран эмпирически под
  русский текст: кириллица в BPE-токенайзерах вроде cl100k обычно кодируется куда
  менее эффективно, чем латиница, ближе к ~2 символам на токен, а не к ~4, как для
  английского), и через разницу — из текущего
  usage.prompt_tokens вычитается last_context_tokens ПРЕДЫДУЩЕГО вызова этого чата
  (сохраняется в JSON-файле истории). last_context_tokens — это prompt_tokens +
  completion_tokens предыдущего вызова МИНУС его reasoning_tokens (если API их
  возвращает, см. _extract_usage) — вычитать нужно именно то, что реально попало в
  content и, значит, в сохранённую историю, а не весь completion_tokens: у моделей
  с рассуждениями заметная часть completion_tokens уходит на скрытые размышления,
  которые в историю не попадают, и без поправки на них "разница" на следующем шаге
  уходит в минус. Точность дополнительно ограничена служебными токенами на разметку сообщений чата
  (несколько токенов на сообщение) — они не в счёт. Для первого вопроса в чате (или
  сразу после reset()) базы для сравнения ещё нет — разница возвращается как None.

С добавлением управления контекстом (см. выше) usage.prompt_tokens отражает не
токены ВСЕЙ сохранённой истории, а токены того, что реально ушло в этом
запросе — summary (если есть) плюс несвёрнутый хвост плюс системный промпт и
вопрос. Сразу после того, как сработала свёртка очередного блока в summary,
last_context_tokens сбрасывается в None по тому же принципу, что и для первого
вопроса/сразу после reset() — см. докстринг _maybe_update_summary() про то, почему
попытка скорректировать его оценкой размера свёртки ненадёжна: запрос на
сворачивание тянет за собой свой собственный system-промпт и текстовую обвязку,
которых нет в основном контексте, и без токенайзера провайдера отделить их вклад
от вклада самого сворачиваемого блока в его usage.prompt_tokens нельзя. Поэтому
request_tokens_diff на первом вопросе после свёртки — тоже None, а со следующего
вопроса снова считается как обычно, уже из настоящего usage.prompt_tokens.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import (
    AGENT_CONTEXT_RECENT_PAIRS,
    AGENT_HISTORY_DIR,
    AGENT_SUMMARY_CHUNK_PAIRS,
    AGENT_SUMMARY_MAX_TOKENS,
    AGENT_SUMMARY_SYSTEM_PROMPT,
    MAIN_MODEL,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from providers.main_client import main_client

logger = logging.getLogger(__name__)


@dataclass
class AgentAnswer:
    """Результат ask() — текст ответа плюс статистика по токенам (см. докстринг модуля)."""

    text: str
    request_tokens_diff: int | None
    request_tokens_approx: int
    history_tokens: int | None
    response_tokens: int | None


class Agent:
    """Агент с историей диалога, сохраняемой в JSON-файле на чат (см. докстринг модуля).

    Сырая история не ограничена по длине (по явному решению — см. CLAUDE.md) и
    хранится на диске целиком, но в LLM с каждым вызовом ask() уходит не она вся,
    а summary + несвёрнутый хвост — см. "Управление контекстом" в докстринге модуля.
    """

    def __init__(
        self,
        chat_id: int,
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
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._history_path = Path(history_dir) / f"{chat_id}.json"
        self._recent_pairs = recent_pairs
        self._summary_chunk_pairs = summary_chunk_pairs
        self._summary_max_tokens = summary_max_tokens
        self._summary_system_prompt = summary_system_prompt
        (
            self._history,
            self._last_context_tokens,
            self._summary,
            self._summarized_pairs,
        ) = self._load_state()

    def _load_state(self) -> tuple[list[dict[str, str]], int | None, str | None, int]:
        """Читает историю чата, last_context_tokens и состояние summary из JSON-файла.

        Поддерживает три формата файла: самый старый — голый список сообщений (до
        добавления статистики по токенам, last_context_tokens в этом случае
        неизвестен), промежуточный — {"messages": [...], "last_context_tokens": ...}
        (до добавления управления контекстом) и текущий — с дополнительными
        "summary"/"summarized_pairs". В файлах без summary/summarized_pairs (оба
        старых формата) считаем, что сворачивания ещё не было — summary=None,
        summarized_pairs=0, следующий же вызов ask() досчитает их с этой точки.
        Отсутствие/повреждение файла — пустая история.
        """
        try:
            raw = self._history_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], None, None, 0

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Повреждённый файл истории агента %s — начинаю с пустой истории.",
                self._history_path,
            )
            return [], None, None, 0

        if isinstance(data, list):
            return data, None, None, 0

        if isinstance(data, dict):
            messages = data.get("messages")
            last_context_tokens = data.get("last_context_tokens")
            summary = data.get("summary")
            summarized_pairs = data.get("summarized_pairs")
            return (
                messages if isinstance(messages, list) else [],
                last_context_tokens if isinstance(last_context_tokens, int) else None,
                summary if isinstance(summary, str) else None,
                summarized_pairs if isinstance(summarized_pairs, int) else 0,
            )

        return [], None, None, 0

    def _save_history(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "messages": self._history,
            "last_context_tokens": self._last_context_tokens,
            "summary": self._summary,
            "summarized_pairs": self._summarized_pairs,
        }
        self._history_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        """Полностью очищает историю диалога этого чата (см. /agent_reset в agent_command.py).

        Вместе с историей сбрасывается и last_context_tokens, и summary/
        summarized_pairs — следующий вопрос снова станет "первым" и для расчёта
        разницы токенов (см. докстринг модуля), и для управления контекстом.
        """
        self._history = []
        self._last_context_tokens = None
        self._summary = None
        self._summarized_pairs = 0
        self._history_path.unlink(missing_ok=True)

    def get_history(self) -> list[dict[str, str]]:
        """Возвращает копию сохранённой истории (см. /agent_history в agent_command.py).

        Копия, а не сама внутренняя история — чтобы вызывающий код не мог случайно
        исказить состояние агента через возвращённый список.
        """
        return list(self._history)

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        """Достаёт prompt_tokens/completion_tokens (и, если есть, reasoning_tokens) из
        ответа API. reasoning_tokens (usage.completion_tokens_details.reasoning_tokens,
        как в OpenAI-совместимой схеме reasoning-моделей) — часть completion_tokens,
        которая ушла на скрытые размышления и НЕ попадает в content, а значит и в
        сохранённую историю; без вычитания этой части last_context_tokens в ask()
        завышается и уводит следующий расчёт "по разнице" в минус. Если поле
        отсутствует у конкретного провайдера/модели — считаем reasoning_tokens = 0
        (поведение как раньше).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return None
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) or 0
        logger.debug("Usage от API (агент): %r", usage)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    def _build_context_messages(self) -> list[dict[str, str]]:
        """Собирает то, что уходит в LLM вместо полной истории: summary (если есть)
        плюс "свежий хвост" — все пары, ещё не свёрнутые в summary (см. "Управление
        контекстом" в докстринге модуля). Хвост — это self._history[2 *
        self._summarized_pairs :], а не отдельное окно фиксированного размера, поэтому
        между "уже в summary" и "ещё как есть" не остаётся пар, которые не попадают
        никуда.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Краткое содержание более ранней части этого диалога:\n{self._summary}"
                    ),
                }
            )
        messages.extend(self._history[2 * self._summarized_pairs :])
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
        """Сворачивает в summary очередные блоки по summary_chunk_pairs пар, пока за
        пределами recent_pairs остаётся хотя бы summary_chunk_pairs несвёрнутых пар
        (см. "Управление контекстом" в докстринге модуля). Ошибка вызова на
        сворачивание не прерывает ask() и не портит уже полученный ответ пользователю
        — просто логируется, summary/summarized_pairs остаются прежними, и хвост
        продолжает расти до следующей попытки.

        После каждого успешного сворачивания сбрасывает last_context_tokens в None,
        а не пытается скорректировать его оценкой. Запрос на сворачивание отправляет
        свой собственный system-промпт (AGENT_SUMMARY_SYSTEM_PROMPT) и текстовую
        обвязку ("Текущая сводка:...", "Новый фрагмент диалога..."), которых нет и не
        будет в основном контексте — его usage.prompt_tokens поэтому НЕ измеряет
        размер того, что реально покидает контекст, и не позволяет надёжно оценить
        поправку (сколько именно токенов приходится на посторонний промпт/обвязку, а
        сколько — на сам сворачиваемый блок, без доступа к токенайзеру провайдера не
        узнать). Вместо неточной коррекции используется тот же принцип, что уже
        применяется к первому вопросу в чате и к вопросу сразу после reset() — там
        тоже последней базы для сравнения ещё нет, и request_tokens_diff возвращается
        как None (см. докстринг модуля): свёртка меняет контекст ничуть не менее
        структурно, чем reset, поэтому честнее показать "нет базы для сравнения" на
        один этот вопрос, чем недостоверную оценку. Со следующего же вопроса
        last_context_tokens снова считается из настоящего usage.prompt_tokens
        основного вызова (см. ask()), уже корректно отражающего сокращённый контекст.
        """
        while True:
            total_pairs = len(self._history) // 2
            unsummarized_pairs = total_pairs - self._summarized_pairs
            if unsummarized_pairs - self._recent_pairs < self._summary_chunk_pairs:
                return

            chunk_start = 2 * self._summarized_pairs
            chunk_end = chunk_start + 2 * self._summary_chunk_pairs
            chunk = self._history[chunk_start:chunk_end]

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
            self._last_context_tokens = None

    def ask(self, user_text: str) -> AgentAnswer:
        """Отправляет вопрос пользователя в LLM вместе с контекстом чата (summary +
        несвёрнутый хвост истории — см. "Управление контекстом" в докстринге модуля,
        не вся сохранённая история) и возвращает ответ.

        Может выбросить исключение OpenAI SDK (сетевые ошибки, ошибки API и т.д.) —
        перехват и перевод в сообщение пользователю на русском выполняет вызывающий
        код, см. докстринг модуля. История дописывается только после успешного
        ответа API — неудачный вызов не искажает сохранённый диалог. Статистика по
        токенам в возвращённом AgentAnswer — см. докстринг модуля про то, откуда
        берётся каждое из четырёх значений.
        """
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
        history_tokens = usage["prompt_tokens"] if usage else None
        response_tokens = usage["completion_tokens"] if usage else None
        request_tokens_diff = (
            usage["prompt_tokens"] - self._last_context_tokens
            if usage and self._last_context_tokens is not None
            else None
        )
        # В историю попадает только видимый content, а не completion_tokens целиком —
        # вычитаем reasoning_tokens (см. докстринг _extract_usage), иначе база для
        # следующего diff завышается и результат уходит в минус.
        self._last_context_tokens = (
            usage["prompt_tokens"] + usage["completion_tokens"] - usage["reasoning_tokens"]
            if usage
            else None
        )

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": answer})
        self._maybe_update_summary()
        self._save_history()

        return AgentAnswer(
            text=answer,
            request_tokens_diff=request_tokens_diff,
            request_tokens_approx=request_tokens_approx,
            history_tokens=history_tokens,
            response_tokens=response_tokens,
        )
