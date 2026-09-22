# CLAUDE.md

Этот файл содержит инструкции для Claude Code (claude.ai/code) при работе с кодом в этом репозитории.

## Что это за проект

Telegram-бот для инвестиционных рекомендаций. Пользователь задаёт вопрос в личном чате, бот
пересылает его в DeepSeek API (OpenAI-совместимый) и возвращает ответ как есть. Сейчас это
stateless-прокси без истории диалога и без админ-команд; проект будет дорабатываться в сторону
полноценного инвестиционного ассистента, поэтому архитектурные и доменные решения ниже стоит
пересматривать по мере роста функциональности, а не считать зафиксированными навсегда.

Логика разнесена по нескольким файлам и двум пакетам в `src/`:
- `config.py` — переменные окружения и константы, не привязанные к конкретному
  LLM-провайдеру: Telegram-токен, `MAIN_CLIENT`/`MAIN_MODEL` (см. "Конфигурация"),
  `SYSTEM_PROMPT`, лимиты, логирование. Единственное место, вызывающее `load_dotenv()`.
- `providers/` — подключения к конкретным LLM-провайдерам и выбор основного клиента:
  - `providers/deepseek_client.py` — подключение к DeepSeek: `DEEPSEEK_API_KEY`/
    `DEEPSEEK_BASE_URL`, клиент `deepseek_client`, модели `DEEPSEEK_MODEL_PRO`/
    `DEEPSEEK_MODEL_FLASH` (только для `/research_models`).
  - `providers/kimi_client.py` — подключение к Kimi/Moonshot AI: `KIMI_API_KEY`/
    `KIMI_BASE_URL`, клиент `kimi_client`, модели `KIMI_MODEL_K3`/`KIMI_MODEL_K2_6`
    (только для `/research_models`).
  - `providers/main_client.py` — выбирает `deepseek_client` или `kimi_client` по
    `MAIN_CLIENT` и экспортирует результат как `main_client`; основной поток и три
    research-режима без собственного выбора модели импортируют клиента отсюда, а не
    из `providers/deepseek_client.py` напрямую (см. "Конфигурация" и "Архитектура").
- `research/` — технические режимы исследования API, каждый — не используется в
  обычном потоке сообщений:
  - `research/_shared.py` — общий каркас нескольких режимов: перехват ошибок OpenAI
    SDK и перевод их в сообщение на русском (`run_scenario`/`api_error_to_message`),
    статистика по токенам (`extract_usage`/`sum_usage`/`format_scenario_stats`),
    fallback на `reasoning_content` (`content_or_reasoning_fallback`), фабрика
    обработчика `/cancel` (`build_cancel_handler`) — см. "Архитектура" про то, какой
    режим что из него использует и почему `research/models.py` не использует всё.
  - `research/constraints.py` — режим `/research_constraints`, исследование ограничений API на ответ, включая формат ответа (см. ниже).
  - `research/reasoning.py` — режим `/research_reasoning`, исследование способов рассуждения DeepSeek API.
  - `research/temperature.py` — режим `/research_temperature`, исследование влияния параметра `temperature` на ответ DeepSeek API.
  - `research/models.py` — режим `/research_models`, сравнение четырёх моделей DeepSeek и Kimi по качеству, скорости и стоимости ответа.
- `agents/` — агенты: сущности, инкапсулирующие цикл «вопрос пользователя -> вызов
  LLM -> разбор ответа» отдельно от Telegram-обработчиков:
  - `agents/agent.py` — класс `Agent`, использует `main_client`/`MAIN_MODEL` (как и
    `main.py`). В отличие от остального бота, хранит историю диалога — в JSON-файле
    на chat_id в `AGENT_HISTORY_DIR` (`config.py`) — это осознанное, явно
    запрошенное пользователем исключение из общего правила «никакой памяти», см.
    «Ограничения безопасности» ниже. Сырые сообщения хранятся по именованным
    веткам без ограничений (одна ветка `main` по умолчанию), но в LLM с каждым
    вопросом уходит не активная ветка целиком, а результат одной из 4
    переключаемых стратегий управления контекстом (Sliding Window / Sticky Facts /
    Branching / Summary+хвост) — см. «Управление контекстом агента» в разделе
    «Архитектура». Сами стратегии — в `agents/context_strategies.py`.
  - `agents/context_strategies.py` — 4 класса стратегий управления контекстом
    (`ContextStrategy` и его реализации `SlidingWindowStrategy`/
    `StickyFactsStrategy`/`BranchingStrategy`/`SummaryStrategy`) и реестр по имени
    `STRATEGIES`, используемый `Agent._before_turn()`/`Agent._build_context_messages()`/
    `Agent._after_turn()`. Стратегии читают/пишут "protected" состояние `Agent`
    напрямую (активную ветку, `facts`, `summary`) — персистентность и вызовы
    LLM-клиента остаются на стороне `Agent`, стратегии решают только, что уходит в
    контекст и когда обновлять `summary`/`facts` (см. докстринг модуля). `facts`
    обновляется ДВАЖДЫ вокруг каждого хода — в `before_turn()` (до отправки
    вопроса, догоняет бэклог с прошлых ходов) и в `after_turn()` (сразу после
    ответа, учитывает саму эту пару); `summary` — только в `after_turn()`
    (см. «Управление контекстом агента» в разделе «Архитектура» про то, почему
    именно так).
  - `agents/agent_command.py` — команда `/agent` (`ConversationHandler`, диалог
    «вопрос за вопросом» до `/cancel` или кнопки выхода), команда `/agent_reset`
    (полностью очищает историю чата), команда `/agent_history` (печатает
    сохранённую историю активной ветки как есть, без ограничения длины — режется
    на части по `TELEGRAM_MESSAGE_LIMIT`, как и обычные ответы), команда
    `/agent_mode` (печатает текущий режим чата: активную стратегию управления
    контекстом и, только если она — Branching, активную ветку — вне Branching
    понятие ветки нигде пользователю не показывается, как и в `/agent_checkpoint`/
    `/agent_branch`/`/agent_switch_branch` ниже), команда `/agent_context` (кнопки выбора одной из
    стратегий управления контекстом, доступных согласно `AGENT_ENABLED_STRATEGIES`)
    и команды `/agent_checkpoint`/`/agent_branch`/`/agent_switch_branch` (чекпоинты
    и ветки диалога — работают только при стратегии Branching) — все восемь
    строятся через `build_agent_conversation_handler()`/`build_agent_reset_handler()`/
    `build_agent_history_handler()`/`build_agent_mode_handler()`/
    `build_agent_context_handlers()`/`build_agent_checkpoint_handler()`/
    `build_agent_branch_handler()`/`build_agent_switch_branch_handlers()`.
  - `agents/active_mode.py` — общий в памяти процесса трекер `{chat_id: "agent" |
    "compare" | "smart_agent"}`, используемый `agent_command.py`, `compare_command.py`
    и `smart_agent_command.py`, чтобы `/agent`, `/agent_compare` и `/smart_agent` были
    попарно взаимоисключающими для одного чата (у каждого свой независимый
    `ConversationHandler`, и без этой отметки ничто не мешало бы быть "внутри"
    нескольких сразу) — отдельный модуль, а не общее состояние в одном из трёх, чтобы
    не создавать цикл импорта.
  - `agents/compare_command.py` — команда `/agent_compare`: инструмент тестирования
    (ближе по духу к `research/*`, чем к `/agent`), параллельно сравнивающий три
    ФИКСИРОВАННЫЕ стратегии управления контекстом (Sliding Window, Sticky Facts,
    Branching — "summary" не участвует) на одном диалоге. Каждая стратегия — свой
    `Agent` (`agents/agent.py`) со своим файлом истории
    (`AGENT_HISTORY_DIR/<chat_id>_compare_<стратегия>.json`, отдельно от обычного
    `<chat_id>.json` того же чата), кэшируются в `_compare_agents` по chat_id, как
    `_agents` в `agent_command.py`. На каждый вопрос все три `Agent.ask()`
    вызываются параллельно через `ThreadPoolExecutor` (тот же приём, что в
    `research/temperature.py`/`research/models.py`, сценарий 5), но результаты
    выводятся в чат в ФИКСИРОВАННОМ порядке (Sliding Window → Sticky Facts →
    Branching, `COMPARE_STRATEGIES`) — не в порядке завершения — каждый отдельными
    сообщениями с меткой-префиксом и своей строкой токенов
    (`_format_token_stats`, переиспользована из `agent_command.py`). Сбой одной
    стратегии не мешает показать другие — ошибка каждой обрабатывается отдельно
    через `api_error_to_message` из `research/_shared.py`. Команда
    `/agent_compare_report` (не `/summary` — так называется другая, не участвующая
    в сравнении стратегия) берёт последний ответ каждой из трёх и одним вызовом
    через `main_client`/`MAIN_MODEL` (`AGENT_COMPARE_REPORT_SYSTEM_PROMPT` в
    `config.py`) просит сравнить их по качеству/устойчивости/токенам — работает,
    только если чат сейчас в режиме сравнения. `/agent_compare_reset` очищает
    историю всех трёх сразу, работает независимо от активного режима (как
    `/agent_reset`). Три сборщика — `build_agent_compare_conversation_handler()`/
    `build_agent_compare_report_handler()`/`build_agent_compare_reset_handler()`.
  - `agents/smart_agent.py` — класс `SmartAgent`: НЕЗАВИСИМЫЙ от `Agent` LLM-агент с
    явно разделённой моделью памяти (три слоя вместо 4 переключаемых стратегий одной
    истории) плюс слой ИНВАРИАНТОВ — жёстких ограничений пользователя, которые агент
    не имеет права нарушать (`agents/invariants.py`, см. «Инварианты smart-агента» в
    разделе «Архитектура») — см. «Управление памятью smart-агента» в разделе
    «Архитектура».
    Рабочий слой — конечный автомат задачи, правила которого живут в
    `agents/task_state.py`, а владение состоянием, сохранение на диск и сами вызовы
    LLM (включая технический вызов `update_task_state`) — здесь. Поверх
    трёх слоёв — именованные ПРОФИЛИ ПЕРСОНАЛИЗАЦИИ: чат может завести несколько
    профилей, каждый со своими предпочтениями (стиль, формат, риск и т.п.,
    `PROFILE_FIELDS`) и СВОЕЙ независимой копией всех трёх слоёв памяти — переключение
    профиля переключает диалог/задачу/факты целиком, как переключение ветки Branching
    у `Agent`. Тоже использует `main_client`/`MAIN_MODEL`, тоже хранит состояние в
    JSON-файле на chat_id, но в отдельном каталоге (`AGENT_MEMORY_DIR`, не
    `AGENT_HISTORY_DIR`) и в своём формате — файлы `/agent` и `/smart_agent` одного
    чата не пересекаются.
  - `agents/invariants.py` — правила слоя инвариантов `/smart_agent`: реестр
    категорий, разбор пользовательского ввода (`parse_input`), системное сообщение
    для основного вызова (`build_context_message`), запрос и разбор ответа
    вызова-ревизора (`build_review_user_content`/`parse_review_response`), общий
    рендер списка и нарушений. Отделён от `agents/smart_agent.py` по тому же
    принципу, что `agents/task_state.py`: здесь правила и тексты, там — владение
    состоянием, сохранение на диск и вызовы LLM. Про механизм целиком см.
    «Инварианты smart-агента» в разделе «Архитектура».
  - `agents/task_state.py` — правила конечного автомата рабочей задачи
    `/smart_agent`: реестр сценариев (`SCENARIOS`: `portfolio`/`asset`/`review`),
    их этапы (`planning` → `execution` → `validation` → `done`, с откатом
    `validation` → `execution`), условия перехода — включая ГЕЙТЫ
    (`TaskStage.gate`: подтверждение вводных и приёмка результата), разбор и
    валидация JSON-ответа технического вызова (фильтр по ключам ТЕКУЩЕГО этапа),
    короткая история переходов вместе с отклонёнными попытками
    (`record_transition`), а также все тексты про состояние (контекст для LLM,
    строка состояния в чат, сводка для возобновления). Здесь же — откат задачи на
    доработку, когда её результат нарушил инварианты (`apply_invariant_violations`,
    `INVARIANT_VIOLATIONS_KEY`); сам список инвариантов автомату не принадлежит.
    Отделён от
    `agents/smart_agent.py` по тому же принципу, что `agents/context_strategies.py`
    от `agents/agent.py`: здесь правила автомата, там — владение состоянием,
    сохранение на диск и вызовы LLM-клиента. Про сам механизм см. «Конечный
    автомат рабочей задачи» в разделе «Архитектура».
  - `agents/smart_agent_command.py` — команда `/smart_agent` (`ConversationHandler`,
    диалог «вопрос за вопросом» до `/cancel` или кнопки выхода, по образцу `/agent`;
    если у чата нет активного профиля, сначала показывает его выбор/создание — см.
    «Управление памятью smart-агента»), команда `/smart_agent_profile` (отдельный
    `ConversationHandler`, выбор/создание профиля в любой момент, даже посреди
    диалога), `/smart_agent_profile_set`/`/smart_agent_profile_show`/
    `/smart_agent_profile_delete` (точечная правка, просмотр и удаление профиля) и
    команды управления тремя слоями памяти АКТИВНОГО ПРОФИЛЯ:
    `/smart_agent_remember`/`/smart_agent_forget`/`/smart_agent_long_show`
    (долговременная), `/smart_agent_task_start`/`/smart_agent_task_set`/
    `/smart_agent_task_show`/`/smart_agent_task_pause`/`/smart_agent_task_resume`/
    `/smart_agent_task_stage`/`/smart_agent_task_done` (рабочая — задача как
    конечный автомат, см. «Конечный автомат рабочей задачи»), а также команды слоя
    ИНВАРИАНТОВ `/smart_agent_invariant_add`/`/smart_agent_invariant_show`/
    `/smart_agent_invariant_remove` (см. «Инварианты smart-агента»),
    `/smart_agent_show` (профиль + инварианты + все три слоя как есть + последний
    собранный контекст LLM) и `/smart_agent_toggle` (включить/выключить слой, включая
    профиль и инварианты, в контексте без удаления данных) и `/smart_agent_reset`
    (очистить три слоя активного профиля — инварианты и `meta` при этом остаются).
    Строятся через `build_smart_agent_conversation_handler()`,
    `build_smart_agent_profile_conversation_handler()` и по одному
    `build_smart_agent_*_handler()` на каждую из остальных девятнадцати команд.
- `main.py` — обычный прокси-поток (`/start`, `/help`, `handle_message`) и точка входа
  приложения; подключает четыре режима исследования через `build_constraints_conversation_handler()`,
  `build_reasoning_conversation_handler()`, `build_temperature_conversation_handler()`,
  `build_models_conversation_handler()`, агента через все восемь `build_agent_*` из
  `agents/agent_command.py`, сравнение стратегий через три `build_agent_compare_*`
  из `agents/compare_command.py`, а также smart-агента через все двадцать один
  `build_smart_agent_*` из `agents/smart_agent_command.py` —
  `build_smart_agent_profile_conversation_handler()` регистрируется ПЕРЕД
  `build_smart_agent_conversation_handler()` (см. «Управление памятью smart-агента»
  про то, почему порядок важен).

## Команды

```bash
make install   # pip install -r requirements.txt
make run       # python src/main.py
make test      # pytest tests/ — юнит-тесты конечного автомата задачи
make clean     # удаляет __pycache__, .pytest_cache, .ruff_cache, артефакты build/dist
```

Линтинг/форматирование (ruff настроен в `pyproject.toml`, но не входит в requirements.txt):

```bash
pip install ruff
ruff check src/
ruff format src/
```

Внимание: на момент появления тестов `ruff check src/` на существующем коде не
проходит начисто (в основном `E501` в давно написанных строках), и `ruff format`
переформатировал бы половину проекта. При доработке сверяй, что твои изменения не
ДОБАВЛЯЮТ замечаний, а не гонись за нулём на всём репозитории.

Тесты (`pytest` тоже не входит в requirements.txt, по тому же принципу, что ruff):

```bash
pip install pytest
pytest tests/
```

Проверка синтаксиса без запуска:

```bash
python3 -m py_compile src/config.py src/providers/deepseek_client.py src/providers/kimi_client.py src/providers/main_client.py src/research/_shared.py src/research/constraints.py src/research/reasoning.py src/research/temperature.py src/research/models.py src/agents/agent.py src/agents/agent_command.py src/agents/context_strategies.py src/agents/active_mode.py src/agents/compare_command.py src/agents/smart_agent.py src/agents/smart_agent_command.py src/agents/task_state.py src/agents/invariants.py src/main.py
```

`tests/` — юнит-тесты правил конечного автомата рабочей задачи
(`tests/test_task_state.py`, ~20 тестов) плюс `tests/conftest.py`, который
добавляет `src/` в `sys.path` (пакета/установки у проекта нет, запуск —
`python src/main.py`). Тесты покрывают именно `agents/task_state.py`: это чистые
функции без сети, Telegram и LLM, поэтому ни моков клиента, ни фикстур с файлами
памяти там нет — при доработке автомата дописывай тесты сюда, а не заводи моки
на `SmartAgent`.

## Процесс работы: OpenSpec

Проект подключён к [OpenSpec](https://github.com/Fission-AI/OpenSpec) — spec-driven
процессу, в котором заметное изменение сначала описывается артефактами планирования
(предложение, дельта-спека, дизайн, задачи), и только потом пишется код. Это касается
ПРОЦЕССА доработки, а не рантайма бота: `src/` от OpenSpec никак не зависит, каталог
`openspec/` приложение не читает, а `requirements.txt`/`Makefile` про него не знают.

Что лежит в репозитории:
- `openspec/config.yaml` — конфигурация (`schema: spec-driven`). Все опциональные
  секции (`context`, `rules`, `operations`) пока закомментированы, то есть действуют
  значения по умолчанию; проектный контекст для генерации артефактов при
  необходимости дописывается именно сюда, а не в тексты скиллов.
- `openspec/specs/` — ГЛАВНЫЕ спецификации («что система должна делать»), по каталогу
  на capability, внутри `spec.md`. Сейчас пуст.
- `openspec/changes/<change-id>/` — изменение в работе: `proposal.md` (что и зачем),
  `design.md` (как), `specs/<capability-path>/spec.md` (ДЕЛЬТА к главной спеке, а не
  её копия), `tasks.md` (шаги реализации). Сейчас активных изменений нет.
- `openspec/changes/archive/YYYY-MM-DD-<change-id>/` — завершённые изменения.
- `.claude/skills/openspec-*` и `.claude/commands/opsx/*` — скиллы и слэш-команды,
  сгенерированные `openspec init`. Это генерируемые файлы: обновляются CLI, вручную
  их править не нужно.

Каталог `openspec/` (как и `.claude/`) коммитится в git — в отличие от `data/`,
планирование и спеки являются частью истории проекта.

**CLI.** Скиллы работают через CLI `openspec` (Node-пакет `@fission-ai/openspec`,
`npm install -g @fission-ai/openspec@latest`, требуется Node.js 20.19+). Проверка —
`openspec --version`; на момент подключения OpenSpec к проекту CLI в окружении
разработчика установлен не был, и без него скиллы `openspec-*`/`/opsx:*` до конца не
отработают (команды вроде `openspec list --json`, `openspec status`, `openspec
instructions`, `openspec validate`, `openspec archive` — их рабочий инструмент). Если
CLI недоступен, не имитируй его: не создавай файлы в `openspec/` руками и не пытайся
воспроизвести формат артефактов по памяти — сообщи, что нужен CLI, и либо поставь его
по просьбе пользователя, либо выполни задачу обычным порядком, без OpenSpec.

Порядок работы (слэш-команды Claude Code):
1. `/opsx:explore` — продумать идею или проблему до создания изменения.
2. `/opsx:propose` — создать изменение и сгенерировать все артефакты разом.
3. `/opsx:update` — переработать артефакты уже созданного изменения, сохранив их
   согласованность между собой.
4. `/opsx:apply` — реализация по `tasks.md`; ТОЛЬКО на этом шаге меняется код в `src/`.
5. `/opsx:sync` — перенести дельту в `openspec/specs/` без архивации изменения.
6. `/opsx:archive` — завершить изменение: дельта уезжает в главные спеки, само
   изменение — в `openspec/changes/archive/`.

Что нужно соблюдать при работе в этом процессе:
- **Граница планирования.** `explore`/`propose`/`update` не трогают код, даже если
  исходная просьба звучит как «сделай/почини». После готовых артефактов нужно
  остановиться и дождаться отдельной явной просьбы на реализацию (`/opsx:apply`) — не
  начинать её в том же ответе.
- **Не всё идёт через OpenSpec.** Опечатки, правка документации, мелкий рефакторинг,
  ответ на вопрос по коду делаются обычным порядком. OpenSpec — для изменений,
  меняющих наблюдаемое поведение бота: новая команда, новый слой памяти, новый
  сценарий рабочей задачи, новая стратегия контекста и т.п.
- **Формат дельта-спеки:** разделы `## ADDED Requirements` / `## MODIFIED Requirements`
  / `## REMOVED Requirements` / `## RENAMED Requirements`, внутри — `### Requirement:
  <название>` и минимум один `#### Scenario: <название>` на требование. В MODIFIED
  требование переносится ЦЕЛИКОМ (тело плюс все сохраняющиеся сценарии) — урезанный
  блок `openspec validate`/`openspec archive` отклонят. В главных спеках
  (`openspec/specs/`) заголовков операций быть не должно: там один раздел
  `## Requirements`.
- **Спеки и этот файл не дублируют друг друга.** `openspec/specs/` отвечает на вопрос
  «что система должна делать» (требования и сценарии), CLAUDE.md — «как устроен код и
  почему решение именно такое». Поэтому после `/opsx:archive` (или сразу вместе с
  реализацией) обновляй CLAUDE.md и README.md, если изменение затронуло архитектуру,
  конфигурацию или набор команд бота — спека этого не заменяет.
- **Существующая функциональность в спеки не переносилась**: `openspec/specs/` пуст
  намеренно, источник истины по уже написанному коду — этот файл и сам код. Не
  переписывай весь текущий бот в спеки задним числом как побочный эффект другой
  задачи; ретроспективное покрытие спеками — отдельное решение владельца проекта.
- **Доменные и security-ограничения действуют и внутри OpenSpec.** Правила из
  «Правил предметной области» и «Ограничений безопасности» ниже не отменяются тем,
  что что-то записано в `proposal.md`: предложение, вводящее сбор денежных сумм,
  пользовательский системный промпт или рантайм-переключение модели/провайдера,
  по-прежнему требует явного обсуждения с владельцем проекта, а не молчаливой
  реализации «по утверждённой спеке».

## Конфигурация

Все секреты и настраиваемые параметры загружаются из переменных окружения через
`python-dotenv` (`.env`, в gitignore). `.env.example` документирует каждую переменную.
`load_dotenv()` вызывается только в `config.py`; `providers/deepseek_client.py` и
`providers/kimi_client.py` делают `import config` ради побочного эффекта (dotenv и
`logging.basicConfig()` должны отработать раньше, чем эти модули прочитают переменные
окружения) и сами `load_dotenv()` не вызывают.

`MAIN_CLIENT` (`config.py`, значения `"deepseek"` или `"kimi"`, по умолчанию
`"deepseek"`) выбирает провайдера основного потока бота (`main.py`) и трёх
research-режимов, которые сами не выбирают конкретную модель (`research/reasoning.py`,
`research/temperature.py`, `research/constraints.py`) — `research/models.py` от него
не зависит, см. его отдельное описание ниже. `MAIN_MODEL` (`config.py`) — модель,
которая используется с этим провайдером; значение по умолчанию зависит от `MAIN_CLIENT`
(`deepseek-v4-flash` для DeepSeek, `kimi-k3` для Kimi). Сам клиент для основного потока
собирается в `providers/main_client.py` (`main_client = deepseek_client if MAIN_CLIENT
== "deepseek" else kimi_client`) — этот модуль, а не `config.py`, импортирует оба
клиентских модуля, чтобы не создавать цикл импорта (см. докстринг `config.py`).

**Обязательность DEEPSEEK_API_KEY/KIMI_API_KEY зависит от MAIN_CLIENT** — это ключевое
поведение, которое нужно сохранять при доработке: `providers/deepseek_client.py`
завершает процесс при старте, только если `MAIN_CLIENT == "deepseek"` и
`DEEPSEEK_API_KEY` не задан; симметрично `providers/kimi_client.py` завершает процесс,
только если `MAIN_CLIENT == "kimi"` и `KIMI_API_KEY` не задан. Ключ провайдера, который
не выбран как `MAIN_CLIENT`, остаётся опциональным — он нужен только сценариям
`/research_models` на моделях этого провайдера, и его отсутствие лишь логирует
предупреждение (клиент создаётся с плейсхолдером вместо пустого ключа — см. докстрings
обоих `providers/*_client.py` про то, зачем: конструктор `OpenAI()` иначе поднял бы
ошибку сразу при пустом `api_key`, ещё до первого реального запроса).
`TELEGRAM_BOT_TOKEN` и допустимость самого значения `MAIN_CLIENT` (только
`"deepseek"`/`"kimi"`) проверяются в `config._validate_config()`. При добавлении новой
обязательной или опциональной переменной окружения ориентируйся на это же разделение:
провайдер-специфичные переменные — в соответствующий `providers/*_client.py`, общие для
бота — в `config.py`.

`RESEARCH` (`config.py`, булево значение `true`/`false`, также принимает `1`/`0`,
`yes`/`no`, `on`/`off`, по умолчанию `true`; допустимость значения проверяется в
`_validate_config()` как и `MAIN_CLIENT`) — включает или выключает исследовательские
и служебные команды: `/research_constraints`, `/research_reasoning`,
`/research_temperature`, `/research_models`, `/agent_compare`,
`/agent_compare_report`, `/agent_compare_reset`, `/agent_mode`, `/agent_context`. Это
решение оператора бота (например, скрыть технические эксперименты на проде), а не
то, что пользователь чата переключает сам. При `RESEARCH=false` эти команды не
регистрируются как обработчики в `main.py` (`RESEARCH_ENABLED`) и не упоминаются в
тексте `/help` — попытка отправить такую команду боту при выключенном флаге ничем не
отличается от отправки любой другой незарегистрированной команды (бот молча её
игнорирует, отдельного сообщения об отказе нет). Флаг не затрагивает `/agent`,
`/agent_reset`, `/agent_history` и `/agent_checkpoint`/`/agent_branch`/
`/agent_switch_branch` — они не являются исследовательскими и остаются доступны
всегда, независимо от `RESEARCH`. При добавлении новой исследовательской команды
добавляй её проверку в этот же список (регистрация в `main.py` и текст `/help`), а
не создавай для неё отдельный флаг.

У опциональных переменных (`DEEPSEEK_BASE_URL`, `KIMI_BASE_URL`, `REQUEST_TIMEOUT_SECONDS`,
`MAX_OUTPUT_TOKENS`, `MAX_INPUT_CHARS` и т.д.) значения по умолчанию заданы прямо в коде —
`config.py`/`providers/deepseek_client.py`/`providers/kimi_client.py` являются источником
истины по составу настроек, а не данный документ.

Параметры сценариев `/research_constraints` (структура JSON-ответа, лимит токенов,
stop-последовательность) и `/research_temperature` (значения `temperature`) заданы
константами в соответствующих модулях и намеренно не вынесены в `.env` — они предмет
самого исследования, а не настройка поведения бота. Идентификаторы моделей в
`/research_models` (`DEEPSEEK_MODEL_PRO`, `DEEPSEEK_MODEL_FLASH`, `KIMI_MODEL_K3`,
`KIMI_MODEL_K2_6`) — исключение из этого принципа: они вынесены в `.env`, потому что
конкретные идентификаторы моделей у провайдеров меняются быстрее кода бота, и здесь это
настройка (что именно доступно на счёте пользователя), а не предмет исследования.

`AGENT_HISTORY_DIR` (`config.py`, по умолчанию `data/agent_history`) — каталог, где
`agents/agent.py` хранит по одному JSON-файлу истории диалога на chat_id; каталог не
коммитится (см. `.gitignore`). Это единственное место в проекте, где диалог
сохраняется на диск — см. «Ограничения безопасности» ниже.

`AGENT_CONTEXT_RECENT_PAIRS`/`AGENT_SUMMARY_CHUNK_PAIRS`/`AGENT_SUMMARY_MAX_TOKENS`
(`config.py`, по умолчанию 10/10/500) настраивают управление контекстом `/agent` —
см. «Управление контекстом агента» в разделе «Архитектура» про сам механизм.
`AGENT_SUMMARY_SYSTEM_PROMPT` (`config.py`) — отдельный от основного `SYSTEM_PROMPT`
системный промпт для технической задачи сворачивания истории в сводку, не вынесен в
`.env` по тому же принципу, что и `SYSTEM_PROMPT`.

`AGENT_CONTEXT_STRATEGY` (`config.py`, допустимые значения `"summary"`
(по умолчанию)/`"sliding_window"`/`"sticky_facts"`/`"branching"`, проверяется в
`_validate_config()` как и `MAIN_CLIENT`) — стратегия управления контекстом `/agent`
по умолчанию для новых чатов и для миграции файлов истории в старом (доветочном)
формате. Это настройка оператора бота только в этом смысле — сам пользователь чата
переключает стратегию для своего диалога в любой момент командой `/agent_context`,
и этот per-chat выбор хранится в JSON-файле истории, а не в `.env` (см. «Управление
контекстом агента» ниже про то, почему это не нарушает запрет на runtime-переключение
модели/провайдера/системного промпта из «Ограничений безопасности»).
`AGENT_STRATEGY_LABELS` (`config.py`) — человекочитаемые подписи стратегий для кнопок
`/agent_context` и текста `/agent`, единственное место с этими формулировками.

`AGENT_ENABLED_STRATEGIES` (`config.py`, по умолчанию все 4 стратегии через запятую,
проверяется в `_validate_config()`: список не может быть пустым, каждое значение —
одно из `_SUPPORTED_AGENT_STRATEGIES`, и `AGENT_CONTEXT_STRATEGY` обязана в него
входить) — какие из стратегий вообще предлагаются кнопками `/agent_context`
(`agents/agent_command.py`: `_strategy_keyboard()` показывает только их,
`agent_context_callback()` отклоняет попытку переключиться на отключённую
стратегию). Это решение оператора бота (например, временно скрыть ещё не обкатанную
стратегию), а не то, что пользователь чата может менять сам — не путать с
`AGENT_CONTEXT_STRATEGY` выше, который выбирает стратегию по умолчанию, а не то,
какие стратегии видны. Список не ограничивает поведение самого `Agent`: если у чата
уже сохранена стратегия, впоследствии исключённая из списка, `Agent` продолжает
работать по ней как обычно — список влияет только на набор кнопок.

`AGENT_FACTS_SYSTEM_PROMPT`/`AGENT_FACTS_MAX_TOKENS` (`config.py`, по умолчанию
лимит 500) — по тому же принципу, что `AGENT_SUMMARY_SYSTEM_PROMPT`/
`AGENT_SUMMARY_MAX_TOKENS`, но для стратегии Sticky Facts: промпт просит модель
вернуть строго JSON-объект (словарь фактов), а не текст сводки.

`AGENT_COMPARE_REPORT_SYSTEM_PROMPT` (`config.py`) — системный промпт для
`/agent_compare_report` (`agents/compare_command.py`): просит сравнить три ответа
по качеству/устойчивости/токенам, теми же тремя критериями, что обсуждались как
метрики сравнения стратегий. Не инвестиционный совет, дисклеймеры из
`SYSTEM_PROMPT` не нужны — по тому же принципу, что и у `AGENT_SUMMARY_SYSTEM_PROMPT`/
`AGENT_FACTS_SYSTEM_PROMPT`.

`AGENT_MEMORY_DIR` (`config.py`, по умолчанию `data/smart_agent_memory`) — каталог,
где `agents/smart_agent.py` хранит по одному JSON-файлу памяти на chat_id; каталог не
коммитится (см. `.gitignore`), тем же принципом, что и `AGENT_HISTORY_DIR`, но
отдельно от него — файлы `/agent` и `/smart_agent` одного чата не должны путаться.
`AGENT_MEMORY_SHORT_TERM_PAIRS` (по умолчанию 10) — сколько последних пар
вопрос-ответ краткосрочного слоя уходит в контекст LLM (аналог
`AGENT_CONTEXT_RECENT_PAIRS` для `/agent`, но не общий с ним — у каждого агента своя
переменная, т.к. они независимы). `AGENT_MEMORY_LONG_TERM_MAX_FACTS` (по умолчанию
20) — максимум фактов в долговременной памяти smart-агента; при превышении
вытесняется самый старый факт, но СНАЧАЛА автоматически извлечённые (см.
«Управление памятью smart-агента» ниже про то, почему это простой лимит по
количеству, а не по токенам, как у `/agent`, и почему порядок вытеснения не
произвольный).

`AGENT_TASK_STATE_SYSTEM_PROMPT`/`AGENT_TASK_STATE_MAX_TOKENS` (`config.py`, по
умолчанию лимит 800) и `AGENT_TASK_START_SYSTEM_PROMPT`/
`AGENT_TASK_START_MAX_TOKENS` (по умолчанию 200) — два системных промпта
технического вызова, двигающего конечный автомат рабочей задачи `/smart_agent`
(см. «Конечный автомат рабочей задачи» в разделе «Архитектура»). Не вынесены в
`.env` по тому же принципу, что `AGENT_SUMMARY_SYSTEM_PROMPT`/
`AGENT_FACTS_SYSTEM_PROMPT`. Их ДВА, а не один, потому что вызов происходит после
каждого ответа агента, в том числе когда активной задачи нет: в этом случае нужно
решить только «начинается ли задача и какая», и гонять ради этого полный разбор
состояния — лишние токены на каждом разовом вопросе. Оба промпта явно запрещают
извлекать номера счетов/карт, паспортные данные и денежные суммы — при
доработке текстов сохраняй этот запрет, он и есть замена прежней защите «пишет
только пользователь явной командой». `AGENT_TASK_STATE_SYSTEM_PROMPT` вдобавок
описывает ключи-гейты (`verdict`, `brief`/`brief_verdict`) и правило «заполняй
только ключи текущего этапа»: сами ограничения держит код
(`task_state.normalize_data_updates`), но модели незачем тратить ходы на
обновления, которые всё равно будут отброшены.

`AGENT_MEMORY_MAX_INVARIANTS`/`AGENT_INVARIANT_MAX_CHARS` (`config.py`, по умолчанию
10 и 300) и `AGENT_INVARIANTS_SYSTEM_PROMPT`/`AGENT_INVARIANTS_MAX_TOKENS` (по
умолчанию лимит 400) — слой инвариантов `/smart_agent` (см. «Инварианты
smart-агента» в разделе «Архитектура»). Промпт — вызова-ревизора, проверяющего
готовый результат рабочей задачи на соответствие инвариантам; не вынесен в `.env` по
тому же принципу, что `AGENT_TASK_STATE_SYSTEM_PROMPT`. Лимит на количество здесь
работает НЕ как `AGENT_MEMORY_LONG_TERM_MAX_FACTS`: при переполнении ничего не
вытесняется, а добавление отклоняется — незаметно выбросить заданный пользователем
запрет нельзя. Лимит длины держит инвариант проверяемым правилом в одну фразу.

## Архитектура

Приложение на `python-telegram-bot` (v20+, async). Основной поток обработки сообщений
(`main.py`) — один на всех, `handle_message`:

1. `filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE` ограничивает обработчик
   сообщений только обычным текстом в личных чатах — групповые чаты и не-текстовый контент
   молча отфильтровываются самим фильтром, а не обрабатываются с последующим отказом.
2. `handle_message` проверяет длину сообщения (`MAX_INPUT_CHARS`), отправляет индикатор
   `ChatAction.TYPING`, затем вызывает провайдера через клиент SDK `openai`
   (`main_client` из `providers/main_client.py` — фактически `deepseek_client` или
   `kimi_client` в зависимости от `MAIN_CLIENT`, см. "Конфигурация"), с моделью `MAIN_MODEL`.
3. Каждый вызов провайдера отправляет ровно два сообщения: `SYSTEM_PROMPT` и текущий текст
   пользователя — история намеренно не подмешивается (см. докстринг модуля). Каждый запрос
   полностью независим от предыдущих.
4. Исключения OpenAI SDK (`AuthenticationError`, `RateLimitError`, `APITimeoutError`,
   `APIConnectionError`, `APIStatusError`, плюс общий fallback) перехватываются по
   отдельности в `handle_message` и превращаются в конкретное сообщение об ошибке на
   русском для пользователя, с упоминанием `MAIN_CLIENT_LABEL` ("DeepSeek"/"Kimi" в
   зависимости от `MAIN_CLIENT`) вместо захардкоженного "DeepSeek" — при добавлении
   новой обработки ошибок следуй этому же паттерну «перехватить, назвать реального
   провайдера через MAIN_CLIENT_LABEL и перевести на язык пользователя», а не одному
   общему `except` и не захардкоженному названию провайдера.
5. Длинные ответы режутся на части по `TELEGRAM_MESSAGE_LIMIT` (4000 символов, с запасом от
   лимита Telegram в 4096) и отправляются несколькими сообщениями.
6. `error_handler` зарегистрирован как общий обработчик ошибок приложения (PTB) для всего,
   что не было перехвачено в try/except внутри `handle_message` (например, ошибки самого
   Telegram API).

Все строки, которые видит пользователь бота (`/start`, `/help`, сообщения об ошибках) — на
русском языке.

`research/constraints.py`, `research/reasoning.py` и `research/temperature.py` все
импортируют `main_client`/`MAIN_MODEL` так же, как `main.py` (см. "Конфигурация"). Их
пользовательский текст и обработка ошибок параметризованы через `MAIN_CLIENT_LABEL`/
`MAIN_API_KEY_ENV_VAR` по тому же принципу, что и `handle_message` в `main.py` —
сохраняй этот принцип при доработке текстов и не возвращай туда захардкоженное
"DeepSeek". `research/constraints.py` — исключение: часть его докстрингов/комментариев
(про сценарий 2 и лимит стоп-слов, см. "Особенности сценариев" ниже) описывает то, что
было эмпирически найдено конкретно на DeepSeek, а не гарантированно верно для любого
`MAIN_CLIENT` — сами сценарии всё равно идут через выбранного `MAIN_CLIENT`, а не всегда
через DeepSeek напрямую; сохраняй эту оговорку при доработке, а не выдавай найденное на
DeepSeek за универсальное свойство API. Сам перехват ошибок и сборка статистики для этих трёх режимов вынесены в
общий `research/_shared.py` (`run_scenario`, `api_error_to_message`,
`format_scenario_stats`, `extract_usage`, `content_or_reasoning_fallback`, `sum_usage`,
`build_cancel_handler`) — три research-режима буквально дублировали этот код до
рефакторинга, поэтому изменение принципа обработки ошибок API или статистики нужно
вносить в `research/_shared.py`, а не в трёх местах по отдельности; специфику
конкретного режима (сами сценарии, форматирование ответа, тексты кнопок) оставляй в
его собственном модуле. `research/models.py` не использует `run_scenario`/
`api_error_to_message` из `_shared.py` — его сообщения об ошибках намеренно
провайдер-нейтральны, а статистика шире (см. его отдельное описание ниже) — но
`extract_usage`/`content_or_reasoning_fallback`/`sum_usage`/`build_cancel_handler`
использует наравне с остальными.

Отдельно, в `research/constraints.py`, живёт технический режим `/research_constraints` — `ConversationHandler` с
4 состояниями (ввод вопроса → выбор одного из 4 сценариев вызова API через
inline-кнопки → для сценариев 3 и 4 дополнительно запрашивается параметр текстом),
не пересекающийся с основным потоком `handle_message`. Это не альтернативный способ
отвечать пользователю на инвестиционные вопросы, а инструмент исследования поведения
самого API (включая формат ответа, сценарий 2) и постобработки ответа — интеграция с
`main.py` ограничена одной функцией `build_constraints_conversation_handler()`.

Особенности сценариев, важные при доработке:
- Сценарий 2 (`call_constraint_json_schema`, тестирует формат ответа): DeepSeek не
  поддерживает OpenAI-style `response_format={"type": "json_schema"}` (structured
  outputs, падает с 400 "This response_format type is unavailable now") —
  используется `{"type": "json_object"}`, а нужные поля задаются текстом в
  пользовательском сообщении, а не схемой на стороне API; не возвращай `json_schema`
  обратно. Это найдено конкретно на DeepSeek — на Kimi (если он выбран как
  `MAIN_CLIENT`) отдельно не проверялось.
- Сценарий 3 (`call_constraint_max_tokens`): значение `max_tokens` запрашивается у
  пользователя (состояние `WAITING_MAX_TOKENS`) и передаётся в API как есть —
  ограничение выполняется на стороне API (экономит токены и время), а не
  постобработкой уже сгенерированного полного ответа ботом. На моделях с
  рассуждениями (например, deepseek-reasoner) весь лимит может целиком уйти на скрытые
  размышления, оставляя видимый `content` пустым при `finish_reason == "length"` —
  в этом случае `content_or_reasoning_fallback` (`research/_shared.py`) подставляет
  обрезанный `reasoning_content` вместо пустоты, это поведение нужно сохранить при
  доработке.
- Сценарий 4 (`call_constraint_stop_sequence`): стоп-слова тоже запрашиваются у
  пользователя (состояние `WAITING_STOP_WORDS`), через запятую, не более
  `CONSTRAINTS_MAX_STOP_WORDS` (4 — лимит, эмпирически найденный на DeepSeek/OpenAI
  API на `stop`; на Kimi отдельно не проверялся).

Условные обозначения сообщений `/research_constraints` (сохраняй при доработке текстов):
- 👉 — перед любым сообщением, ожидающим выбора (кнопки) или текстового ввода
  от пользователя — визуально отличает такие сообщения от результата.
- 📊 — перед результатом сценария ("Сценарий: …").
- 📈 — перед минимальной статистикой ответа (`finish_reason` и токены из
  `response.usage`, см. `format_scenario_stats` в `research/_shared.py`), отправляется
  отдельным сообщением сразу после результата сценария (кроме случаев ошибки API, когда
  `response.usage` недоступен).

`research/temperature.py` (режим `/research_temperature`) устроен по тому же принципу,
что и `research/reasoning.py`: `ConversationHandler` с 2 состояниями (ввод задачи →
выбор сценария из 5 через inline-кнопки), нейтральный (не инвестиционный) системный
промпт и отключённое "thinking" (`extra_body={"thinking": {"type": "disabled"}}`), т.к.
предмет исследования — видимый ответ при разных `temperature`, а не скрытые рассуждения
модели. Сценарии 1-4 — фиксированные значения `temperature` (0 / 0.7 / 1.2 / 2, весь
допустимый диапазон API), не запрашиваемые у пользователя. Сценарий 5 запускает
сценарии 1-4 параллельно через `ThreadPoolExecutor`, затем отдельным запросом просит
модель (`main_client`/`MAIN_MODEL`) сравнить ответы по точности, креативности и
разнообразию и дать рекомендации, для каких задач подходит каждое значение — при
доработке сохраняй эти три критерия сравнения.

`research/models.py` (режим `/research_models`) устроен по тому же принципу, что и
`research/temperature.py`: `ConversationHandler` с 2 состояниями, нейтральный системный
промпт. Сценарии 1-4 — четыре модели из `MODEL_CATALOG`, отсортированные от самой
сильной (слот "Kimi K3") к самой слабой ("DeepSeek V4 Flash"). В отличие от остальных
research-режимов, идентификаторы моделей здесь не хардкодятся в модуле, а приходят из
`providers/deepseek_client.py`/`providers/kimi_client.py` (в конечном счёте — из `.env`,
см. раздел "Конфигурация"): `MODEL_CATALOG` — список словарей `{"id", "label", "client",
...цены}`, каждый со своим клиентом (`kimi_client` для двух моделей Kimi,
`deepseek_client` для двух моделей DeepSeek) — `call_model()` принимает и `client`, и
`model_id` явно, а не всегда использует один и тот же клиент. При добавлении/замене
модели меняй элемент `MODEL_CATALOG` (и переменную окружения по умолчанию в
`providers/deepseek_client.py`/`providers/kimi_client.py`), а не переменную основного
потока `MAIN_MODEL`. В отличие от `research/reasoning.py`/`research/temperature.py`,
здесь "thinking" не отключается —
предмет сравнения включает и штатное поведение модели в целом. Статистика каждого
вызова (`_format_scenario_stats`) расширена по сравнению с другими research-режимами:
помимо `finish_reason` и токенов включает время ответа (`time.monotonic()` вокруг
вызова) и стоимость в USD, посчитанную по иллюстративным тарифам из `MODEL_CATALOG`
(`price_input_per_million`/`price_output_per_million` — это не проверенные официальные
цены, см. докстринг модуля). Сценарий 5 запускает сценарии 1-4 параллельно через
`ThreadPoolExecutor`, затем одним отдельным запросом через `main_client`/`MAIN_MODEL` —
тот же провайдер и модель, что и в остальных research-режимах без собственного выбора
модели, а не через одну из четырёх сравниваемых моделей — передаёт ей все ответы вместе
с их статистикой (токены, время, стоимость) и просит сравнить качество, скорость и
ресурсоёмкость — при доработке сохраняй эти три критерия сравнения, то, что сравнивающий
запрос идёт через `MAIN_CLIENT`, а не через одну из моделей `MODEL_CATALOG`, и то, что
статистика передаётся модели в самом запросе на сравнение, а не только показывается
пользователю. Обработка ошибок API в этом модуле
(`_run_models_scenario`) сформулирована провайдер-нейтрально (не "DeepSeek", а
"провайдер"/"API"), т.к. вызов может уйти как к DeepSeek, так и к Kimi.

### Управление контекстом агента

`agents/agent.py` (класс `Agent`) не отправляет в LLM всю сохранённую историю
диалога целиком — вместо этого в LLM с каждым вопросом уходит результат работы
ОДНОЙ из 4 переключаемых стратегий (`Agent._strategy`, per-chat состояние в
JSON-файле истории, переключается командой `/agent_context` —
`agents/agent_command.py`, кнопки со списком `AGENT_STRATEGY_LABELS`). Это не
нарушает запрет на runtime-переключение модели/провайдера/системного промпта из
«Ограничений безопасности»: стратегия влияет только на то, как собирается
контекст для уже выбранного `MAIN_CLIENT`/`MAIN_MODEL`/`SYSTEM_PROMPT`, а не на
сам выбор провайдера/модели/промпта.

Хранение — ветки диалога, общий слой для всех 4 стратегий. Сырые сообщения чата
хранятся не одним списком, а словарём именованных веток (`Agent._branches:
{имя: {"messages": [...], "facts": ..., "facts_processed_pairs": ..., "summary":
..., "summarized_pairs": ...}}`) плюс указателем активной ветки
(`Agent._active_branch`) — до первого использования Branching у чата ровно одна
ветка `"main"`, и всё работает как раньше. Каждая ветка хранит свою
последовательность сообщений целиком без ограничений — сокращённый контекст,
уходящий в LLM, строится поверх сообщений активной ветки, а не заменяет их.
Выбрано вместо альтернативы «один общий список + пометка ветки у сообщения»
как более простое: при создании ветки от чекпоинта общий префикс дублируется
в неё, но для истории чата с ботом это несущественный объём, а взамен все
стратегии продолжают работать с обычным плоским списком сообщений одной
ветки, без фильтрации по пометкам.

**`facts`/`facts_processed_pairs`/`summary`/`summarized_pairs` — ТОЖЕ per-branch**,
а не общие на весь чат: у каждой ветки своя независимая сводка/словарь фактов,
производные от её собственной последовательности сообщений. Доступ к ним для
активной ветки дают четыре `@property` на `Agent` (`_facts`/`_facts_processed_pairs`/
`_summary`/`_summarized_pairs`), читающие/пишущие
`self._branches[self._active_branch][...]` — весь остальной код `Agent` и
стратегии в `agents/context_strategies.py` (которые обращаются к
`agent._facts`/`agent._summary`/`agent._summarized_pairs` напрямую) продолжают
работать с этими именами как раньше, не зная, что за ними теперь стоит
конкретная ветка. Новая ветка (`Agent.create_branch()`) стартует с ПУСТЫМИ
facts/summary, а не наследует их от ветки чекпоинта — так счётчики
`facts_processed_pairs`/`summarized_pairs` гарантированно не превышают число
сообщений в новой (обычно более короткой на момент создания) ветке; сохраняй
это при доработке, не подставляй туда состояние ветки-источника.

Сами 4 стратегии (`Agent._build_context_messages()`):
- **`sliding_window`** — последние `AGENT_CONTEXT_RECENT_PAIRS` пар вопрос-ответ
  активной ветки, всё остальное отбрасывается из контекста (не из хранения на
  диске).
- **`sticky_facts`** — отдельный словарь ключ-значение (`Agent._facts`: цель,
  ограничения, предпочтения, решения, договорённости), обновляемый вызовом LLM
  (`Agent._update_facts()`, `AGENT_FACTS_SYSTEM_PROMPT`, `response_format=
  {"type": "json_object"}`) ДВАЖДЫ вокруг каждого вопроса пользователя (см.
  `StickyFactsStrategy` в `agents/context_strategies.py`):
  - `before_turn()` — ДО сборки контекста и отправки вопроса в LLM: догоняет
    бэклог ещё не учтённых пар с ПРЕДЫДУЩИХ ходов (если он есть), чтобы контекст
    ТЕКУЩЕГО вопроса строился уже на актуальных facts, а не устаревших.
  - `after_turn()` — сразу ПОСЛЕ получения ответа на текущий вопрос (пара уже
    дописана в активную ветку): учитывает саму эту пару без задержки до
    следующего вопроса.
  `_update_facts()` — общий код для обоих вызовов, ничего не знает о том, из
  какого хука его вызвали, и просто ничего не делает, если бэклог уже пуст (см.
  ниже) — поэтому вызывать его дважды безопасно и не создаёт лишней нагрузки,
  когда обновлять нечего. При доработке сохраняй оба вызова — это не дублирование
  по ошибке, а намеренный компромисс между "контекст всегда свежий" и "facts не
  отстают от последнего ответа". В контекст уходит `facts` одним системным
  сообщением + те же последние `AGENT_CONTEXT_RECENT_PAIRS` пар. Как и
  `json_object` в `research/constraints.py` (сценарий 2), формат проверен на
  DeepSeek — для Kimi отдельно не проверялся.
  В LLM на каждое обновление уходит не единственная пара, а весь бэклог
  сообщений активной ветки, ещё не учтённых в `facts` (`Agent._facts_processed_pairs`,
  тот же принцип, что несвёрнутый хвост `summary`/`summarized_pairs` ниже) — на
  момент `before_turn()` туда попадают пары ПРЕДЫДУЩИХ ходов (текущий вопрос
  попасть не может — ответа на него ещё нет), на момент `after_turn()` обычно
  ровно одна свежая пара (если только `before_turn()` до этого не подвёл — тогда
  бэклог шире). Счётчик продвигается ТОЛЬКО при успешном обновлении, поэтому
  пара из неудачной попытки не пропадает молча, а остаётся в бэклоге до
  следующей успешной попытки (раньше терялась — этот бэклог добавлен именно для
  этого, не убирай его при доработке). Растущий без ограничения словарь фактов
  рано или поздно не помещается в `AGENT_FACTS_MAX_TOKENS`, и ответ модели
  обрезается посередине JSON
  (`JSONDecodeError: Unterminated string`) — от этого есть два уровня защиты:
  правило в `AGENT_FACTS_SYSTEM_PROMPT` держать словарь компактным (не более
  ~20 ключей, объединять/заменять вместо накопления), и один автоматический
  повтор в `_update_facts()` с удвоенным `max_tokens` сразу при
  `JSONDecodeError`, не дожидаясь следующего вопроса пользователя. Если и
  повтор не помог (или ошибка другого рода), `facts`/`facts_processed_pairs`
  остаются прежними, попытка повторяется перед следующим вопросом (см.
  try/except в `_update_facts()`), а не роняют `ask()`.
- **`branching`** — вся история активной ветки целиком, без обрезки и без
  summary/facts: суть стратегии в независимом продолжении диалога в каждой
  ветке, а не в экономии контекста. `Agent.create_checkpoint(name)` запоминает
  текущую длину активной ветки; `Agent.create_branch(checkpoint, name)` создаёт
  ОДНУ новую ветку как независимую копию сообщений до этой длины (с пустыми
  facts/summary — новая ветка не наследует их от ветки чекпоинта, см. «Хранение —
  ветки диалога» выше) — дальше она растёт независимо; чтобы получить несколько
  веток от одного чекпоинта (в т.ч. классические две), команду/метод вызывают
  повторно с тем же чекпоинтом и другим именем ветки — метод намеренно не
  создаёт фиксированное число веток за один вызов. `Agent.switch_branch(name)`
  переключает, какая ветка активна для последующих `ask()` (и, соответственно,
  чьи facts/summary видны). Команды-обёртки
  (`/agent_checkpoint`, `/agent_branch`, `/agent_switch_branch`) осмысленны и
  разрешены только при активной стратегии `"branching"` —
  `_require_branching_strategy()` в `agent_command.py` отклоняет их с подсказкой
  переключиться, иначе поведение (что значит "окно" или "summary" сразу в
  нескольких ветках) не определено; сам `Agent` эту проверку не делает — ветки
  как слой хранения работают независимо от текущей стратегии, проверка на
  совести вызывающего кода команд.
- **`summary`** (стратегия по умолчанию — `AGENT_CONTEXT_STRATEGY` в `config.py`
  — в т.ч. для чатов с JSON-файлом истории ещё в старом, доветочном формате, см.
  `Agent._load_state()`) — прежний механизм: более старая часть периодически
  сворачивается в текстовую сводку (`summary`), и каждый запрос содержит
  `summary` (если он уже есть) плюс "свежий хвост" — все пары вопрос-ответ
  активной ветки, ещё не попавшие в сводку. `AGENT_CONTEXT_RECENT_PAIRS` —
  минимальный размер этого хвоста: столько последних пар никогда не
  сворачиваются. Как только сверх него накапливается ещё
  `AGENT_SUMMARY_CHUNK_PAIRS` пар, они сворачиваются в `summary` отдельным
  вызовом того же `main_client`/`MAIN_MODEL` (с `AGENT_SUMMARY_SYSTEM_PROMPT`
  вместо основного `SYSTEM_PROMPT`) — старая сводка передаётся вместе с новым
  блоком, и модель возвращает объединённую сводку (инкрементальный rollup, а не
  пересборка с нуля при каждом срабатывании — сохраняй этот принцип при
  доработке). "Свежий хвост" — это буквально «всё, что ещё не попало в
  summary», а не отдельное окно фиксированного размера — см.
  `Agent._maybe_update_summary()`.

Переключение стратегии (`Agent.set_strategy()`) НЕ трогает `summary`/`facts`
(и их счётчики `summarized_pairs`/`facts_processed_pairs`) — они продолжают
накапливаться только пока активна их стратегия (см. `Agent._after_turn()`) и снова
становятся актуальны при возврате к ней. `Agent.switch_branch()` — раз facts/summary
per-branch (см. выше) — ТОЖЕ их не сбрасывает: после переключения `self._facts`/
`self._summary` и т.п. просто начинают указывать на состояние НОВОЙ активной ветки,
а состояние ветки, с которой ушли, остаётся на месте и доступно сразу при
возврате на неё (см. докстринг `Agent.switch_branch()`) — это осознанное отличие
от более раннего поведения (когда переключение ветки обнуляло summary/facts),
сохраняй его при доработке.

Полная сырая история при этом продолжает храниться на диске без ограничений —
общий инвариант, не привязанный к конкретной стратегии. `/agent_history`
(`agent_command.py`) печатает историю АКТИВНОЙ ВЕТКИ как есть, а не сокращённый
контекст — при нескольких ветках отдельной строкой указывает, какая ветка
активна и какие ещё существуют; сохраняй это при доработке, а не переключай на
показ `summary`/`facts`. `/agent_reset` очищает все ветки (вместе с их
`facts`/`facts_processed_pairs`/`summary`/`summarized_pairs`) и чекпоинты, но
НЕ трогает выбранную стратегию — это настройка режима работы с агентом, а не
часть очищаемого диалога.

Если вызов на сворачивание `summary` или на обновление `facts` сам завершится
ошибкой API (или невалидным JSON — для `facts`), это не должно ронять уже
полученный ответ пользователю на его текущий вопрос:
`Agent._maybe_update_summary()`/`Agent._update_facts()` — единственные места в
`Agent`, где исключения OpenAI SDK перехватываются внутри самого класса (в
остальном `ask()` их сознательно не ловит, см. выше), — при ошибке
`summary`/`facts` остаются прежними, а попытка обновить их повторяется при
одном из следующих вопросов.

### Управление памятью smart-агента

`agents/smart_agent.py` (класс `SmartAgent`) — НЕЗАВИСИМЫЙ от `Agent` агент: вместо
одной истории диалога с переключаемой стратегией управления контекстом (см.
«Управление контекстом агента» выше) хранит память в трёх явно разделённых слоях,
каждый со своим смыслом и своим способом записи, плюс отдельный слой ОГРАНИЧЕНИЙ —
инварианты (см. «Инварианты smart-агента» ниже). Все слои хранятся В РАЗРЕЗЕ
ПРОФИЛЯ (см. «Профили персонализации» ниже) внутри одного JSON-файла
(`AGENT_MEMORY_DIR/<chat_id>.json`):

- **`short_term`** (краткосрочная, текущий диалог активного профиля) — список сырых
  сообщений (`{"role", "content"}`), пишется АВТОМАТИЧЕСКИ на каждый вызов
  `SmartAgent.ask()` — единственный слой без явной команды на запись, т.к. это и есть
  сам диалог. Хранится на диске без ограничения (как история `/agent`), но в контекст
  LLM уходит не вся история, а только последние `AGENT_MEMORY_SHORT_TERM_PAIRS` пар.
- **`working`** (рабочая, текущая задача активного профиля) — одна активная задача,
  оформленная как КОНЕЧНЫЙ АВТОМАТ (`{"task_type", "goal", "stage", "step",
  "expected_action", "paused", "paused_at", "awaiting_correction", "data",
  "processed_pairs", "created_at"}` или `None`, см. «Конечный автомат рабочей
  задачи» ниже). Пишется
  и явно (`/smart_agent_task_start <сценарий> <цель>` — `SmartAgent.start_task`,
  заменяет предыдущую задачу, а не копит несколько параллельно;
  `/smart_agent_task_set <ключ> <значение>` — `SmartAgent.set_task_data`;
  `/smart_agent_task_pause`/`/smart_agent_task_resume`/`/smart_agent_task_stage` —
  `pause_task`/`resume_task`/`set_task_stage`; `/smart_agent_task_done` —
  `finish_task`, очищает), и АВТОМАТИЧЕСКИ — отдельным техническим вызовом LLM
  после каждого ответа агента (`SmartAgent.update_task_state`).
- **`long_term`** (долговременная, факты активного профиля) — список фактов
  `{"text", "source", "created_at"}`. Пишется и явно командой
  `/smart_agent_remember <текст>` (`SmartAgent.remember`) ДОСЛОВНО (`source:
  "user"`), и автоматически тем же техническим вызовом (`source: "auto"` — сюда
  попадают и извлечённые из диалога факты, и итог завершённой задачи). Удаляется по
  номеру `/smart_agent_forget <номер>` (`SmartAgent.forget`, номера — как в
  `/smart_agent_long_show`, который показывает и происхождение факта). При
  превышении `AGENT_MEMORY_LONG_TERM_MAX_FACTS` вытесняется самый старый факт, но
  СНАЧАЛА автоматический и никогда — только что добавленный (`SmartAgent._append_fact`):
  иначе автоизвлечение постепенно вымыло бы из памяти всё, что пользователь
  сохранил руками, а при полной памяти новый факт удалялся бы сразу же после
  добавления. Сохраняй этот порядок при доработке. Лимит по-прежнему простой, по
  количеству, а не по токенам, как `AGENT_FACTS_MAX_TOKENS` у стратегии Sticky
  Facts `/agent`: там лимит защищает от обрезки JSON-ответа LLM при сворачивании,
  здесь сами факты в LLM на запись не отправляются.

Все геттеры/мутаторы этих трёх слоёв (`get_short_term`/`get_working`/
`get_long_term_facts`/`remember`/`forget`/`start_task`/`set_task_data`/`finish_task`/
`pause_task`/`resume_task`/`set_task_stage`/`update_task_state`/`reset_all`)
требуют активного профиля — при его отсутствии геттеры возвращают
пустое значение, а мутаторы возвращают `False` без изменения состояния;
`agents/smart_agent_command.py` проверяет это ДО вызова через
`_require_active_profile()` и отвечает подсказкой выбрать/создать профиль
(`/smart_agent_profile`), а не даёт командам молча падать или работать «в никуда».

Ни один слой не фильтрует содержимое на вход, поэтому защита от чувствительных
данных держится на двух вещах сразу, и обе нужно сохранять при доработке: (1)
предупреждения в текстах команд, где пишет пользователь (`/smart_agent_remember`,
`/smart_agent_task_set`, `/smart_agent_task_start`, а также анкета и
`/smart_agent_profile_set` — см. ниже); (2) прямой запрет извлекать такие данные в
системных промптах технического вызова (`AGENT_TASK_STATE_SYSTEM_PROMPT`/
`AGENT_TASK_START_SYSTEM_PROMPT`, правила про номера счетов/карт и денежные суммы)
— с появлением автоматической записи одних только предупреждений пользователю уже
недостаточно. То же ограничение из «Правил предметной области» ниже, применённое ко
всем слоям, а не только к `long_term`.

#### Инварианты smart-агента

Рядом с тремя слоями памяти у профиля есть слой ОГРАНИЧЕНИЙ — инварианты
(`agents/invariants.py`): жёсткие правила пользователя («без криптовалют», «доля
акций не выше 40%», «только через ИИС»), которые агент не имеет права нарушать. Это
НЕ четвёртый слой памяти, и путать их не стоит: `long_term` хранит ЗНАНИЕ о
пользователе, `meta` профиля — ПРЕДПОЧТЕНИЕ подачи, а инвариант — ЗАПРЕТ, и именно
поэтому он подкреплён проверкой кодом, а не только формулировкой в промпте.

Хранение — список `{"text", "category", "created_at"}` внутри профиля
(`SmartAgent._profiles[<имя>]["invariants"]`), по тем же правилам, что и три слоя
памяти: у каждого профиля свой набор, новый профиль стартует с ПУСТЫМ (ничего не
наследует, как новая ветка Branching у `Agent`), файлы старого формата без этого
ключа получают пустой список (`invariants.sanitize_list`, отдельной миграции не
нужно). Категория (`strategy`/`instruments`/`risk`/`process`/`other`) —
необязательная пометка для группировки, распознаётся префиксом до двоеточия в
команде (`invariants.parse_input`: `«риск: доля акций не выше 40%»`).

**Пишется слой ТОЛЬКО явными командами пользователя** — `/smart_agent_invariant_add`
и `/smart_agent_invariant_remove`. Автоматической записи, в отличие от
`working`/`long_term`, здесь нет: запрет, который бот вывел сам из реплики в
диалоге, — ровно то, чего в этом слое быть не должно. При доработке не распространяй
на инварианты автоматическое извлечение, тем же правилом, что и на `meta` профиля.

**Лимит работает БЕЗ вытеснения**: при достижении `AGENT_MEMORY_MAX_INVARIANTS`
добавление отклоняется (`SmartAgent.add_invariant` возвращает `invariants.ADD_LIMIT`),
а не вытесняет самый старый, как `_append_fact` у `long_term` — иначе ограничение
переставало бы действовать незаметно для пользователя.

Инвариант действует на ДВУХ уровнях сразу, и оба нужно сохранять при доработке:

1. **Контекст любого ответа.** `invariants.build_context_message()` уходит ПЕРВЫМ
   после `SYSTEM_PROMPT`, до профиля персонализации (порядок сборки: инварианты →
   профиль → долговременная память → рабочая задача → краткосрочный диалог):
   нумерованный список плюс правила — приоритет выше профиля и выше текущей просьбы;
   проверять решение на инварианты модель должна МОЛЧА (правило 2): пользователь ждёт
   результат, а не отчёт о служебных действиях бота, и анонс «дальше проверю на
   соответствие инвариантам» он читает как обещание отдельного шага работы;
   при конфликте отказаться, назвать НОМЕР и формулировку, объяснить конфликт и
   предложить допустимую альтернативу; при настойчивости не уступать, а напомнить,
   что снять ограничение можно только командой `/smart_agent_invariant_remove`.
   Отдельным правилом прописано, что инвариант НЕ отменяет и не ослабляет
   обязательные предупреждения `SYSTEM_PROMPT` — иначе слой стал бы обходным путём
   для пользовательского системного промпта (см. «Ограничения безопасности»); это та
   же оговорка, что и в `_profile_meta_message`, и убирать её нельзя.
2. **Проверка результата рабочей задачи КОДОМ.** Как только на этапе `validation`
   появляется ещё НЕ ПРОВЕРЕННЫЙ результат (`task_state.needs_invariant_check`
   сравнивает `result_payload` со снимком уже проверенного —
   `INVARIANTS_CHECKED_KEY`), `SmartAgent._check_invariants()` делает
   ОТДЕЛЬНЫЙ вызов-ревизор (`AGENT_INVARIANTS_SYSTEM_PROMPT`, `response_format=
   {"type": "json_object"}`, один повтор с удвоенным лимитом — по образцу
   `update_task_state`). Ревизору уходит только готовый артефакт
   (`task_state.result_payload`), а НЕ диалог: иначе он ловил бы «нарушения» в
   уточняющих вопросах этапа планирования. При `verdict: "violated"` с хотя бы одним
   существующим номером задачу откатывает код
   (`task_state.apply_invariant_violations`) — тем же `_roll_back`, что и просьба
   пользователя о правках: результат СОХРАНЯЕТСЯ (агент правит его, а не сочиняет
   заново), `verdict` снимается, ставится `awaiting_correction`, и вперёд автомат не
   пойдёт, пока не появится НОВЫЙ результат. Претензии живут в задаче
   (`INVARIANT_VIOLATIONS_KEY`, вне `data` — `data` фильтруется словарём сценария) и
   снимаются автоматически, как только переписан обязательный ключ текущего этапа.

Привязка к самому РЕЗУЛЬТАТУ, а не к факту перехода на `validation`, — не
косметика: пока проверка висела на переходе, задачу, переведённую на проверку
вручную (`/smart_agent_task_stage validation`), ревизор не видел вовсе (перехода не
было), и её результат уходил в `done` непроверенным. Снимок нужен, чтобы тот же
результат не проверялся заново на каждом ходу диалога о приёмке; ошибка вызова
означает «нарушений нет» и повторной проверки того же результата не будет —
служебная проверка не имеет права стопорить задачу (см. ниже).

Ревизор проверяет ГОТОВЫЙ РЕЗУЛЬТАТ сразу, как только он появился, — то есть ДО
приёмки пользователем, а не после: этап `validation` — это согласие человека, и
согласовывать вариант, нарушающий его же ограничения, бессмысленно. Поэтому строка
об откате может прийти в том же ходу, в котором агент показал новую структуру
(в т.ч. после просьбы «поправь доли»): порядок всегда «ответ агента → служебные
строки», и откат относится к только что показанному варианту, а не к прежнему.

Что ревизор НЕ считает нарушением (правила 2-3 `AGENT_INVARIANTS_SYSTEM_PROMPT`):
упоминание запрещённого инструмента в пояснении («криптовалюты исключены», «вместо
биткоина — золото») и доля 0% — это соблюдение инварианта. Нарушение — когда
запрещённое реально входит в решение (ненулевая доля, рекомендация купить, совет
обойти ограничение). Формулировка добавлена именно потому, что пояснение к
исправленному результату почти всегда называет то, что из него убрали, и без
оговорки ревизор мог откатить корректный вариант.

Почему проверка смещена в сторону пропусков, а не ложных срабатываний: `verdict:
"violated"` без валидных номеров нарушением НЕ считается
(`invariants.parse_review_response`) — откатывать задачу, не имея что показать
пользователю и что исправлять агенту, бессмысленно; ошибка вызова или неразобранный
JSON тоже означают «нарушений нет» (`_call_json_with_retry`) — служебная проверка не
имеет права ни уронить уже отправленный ответ, ни застопорить задачу, а инварианты
при этом всё равно лежат в контексте основного ответа.

Видимость: строка об откате в чат («⛔ Результат нарушает инварианты…») печатается
ВМЕСТО обычной «✅ Этап: …», т.к. переход в этом случае — движение НАЗАД; та же
формулировка уходит в контекст следующего ответа
(`task_state.build_context_message`), отмечается в строке состояния и в
`/smart_agent_task_show`; полный список — `/smart_agent_invariant_show` и отдельным
блоком в `/smart_agent_show`.

Слой переключается `/smart_agent_toggle invariants` наравне с остальными, и это
единственный слой, выключение которого снимает ЕЩЁ И проверку кодом (ревизор не
вызывается): иначе бот возвращал бы задачу на доработку, ссылаясь на ограничения,
которых в его контексте в этот момент нет. Чтобы выключение не осталось незамеченным,
после каждого ответа печатается предупреждение, пока инварианты заданы, но слой
выключен (`_task_service_lines`). `/smart_agent_reset` инварианты НЕ очищает (как и
`meta` профиля) — это заданный пользователем режим работы, а не накопленные данные
диалога; снять ограничение можно только точечно по номеру.

#### Конечный автомат рабочей задачи

Рабочий слой `/smart_agent` — не просто «цель + словарь данных», а конечный автомат
с этапом, текущим шагом и ожидаемым действием. Правила автомата вынесены в
`agents/task_state.py` (см. его докстринг), состояние живёт в `working` активного
профиля.

Имена этапов инженерные, а означают шаги ИНВЕСТИЦИОННОГО процесса
(профилирование → предложение → приёмка → зафиксировано), и путать их с
одноимёнными понятиями других доменов нельзя — при доработке текстов и докстрингов
сохраняй эту оговорку:
- `planning` — профилирование: сбор вводных пользователя и подтверждение сводки;
- `execution` — ФОРМИРОВАНИЕ ПРЕДЛОЖЕНИЯ, а НЕ исполнение сделок (в финансах
  execution — это как раз исполнение у брокера, чего бот не делает и не должен, см.
  «Правила предметной области»). Пользователю этап и подписан по смыслу:
  «формирование портфеля»/«разбор»/«предложение изменений» — инженерных имён он не
  видит вообще;
- `validation` — ПРИЁМКА результата пользователем («устраивает или нужны правки»), а
  не машинная проверка. Машинная проверка в проекте есть, но это отдельный слой —
  ревизор инвариантов (см. «Инварианты smart-агента»);
- `done` — работа зафиксирована, итог перенесён в долговременную память.

Сценарии (`SCENARIOS`) — фиксированный реестр, все на ОДНОМ скелете этапов
`planning` → `execution` → `validation` → `done` (с откатом `validation` →
`execution`, если пользователь просит правки). Различаются только словарём ключей
`data`, наличием гейта на `planning` и текстами инструкций:

| Сценарий | `planning` собирает | Гейт вводных | `execution` заполняет |
|---|---|---|---|
| `portfolio` — составление портфеля | `goal_type`, `horizon`, `risk`, `constraints`, затем `brief`/`brief_verdict` | да | `allocation` (доли в %) |
| `asset` — разбор актива/идеи | `asset`, `horizon`, `role_in_portfolio` | нет | `analysis` |
| `review` — ревизия портфеля | `current_allocation` (только доли), `concern`, `constraints`, затем `brief`/`brief_verdict` | да | `proposed_changes` |

Код автомата про конкретные сценарии ничего не знает — новый сценарий добавляется
одной записью в `SCENARIOS`, без изменений в логике переходов; гейт — тоже свойство
записи реестра (`TaskStage.gate`), а не ветка в коде.

**Контекст основного ответа называет и СЛЕДУЮЩИЙ этап**
(`task_state.next_forward_stage` в `build_context_message`), плюс прямо запрещает
выдумывать собственные названия этапов и анонсировать пользователю служебные
проверки бота как шаг работы. Без этой строки модель знала только текущий этап и
досочиняла пользователю несуществующие («дальше — наполнение блоков, потом проверка
на соответствие инвариантам»), то есть обещала то, чего автомат не делает. Сохраняй
это при доработке текстов: всё, что модель говорит про порядок работы, должно
совпадать с реестром этапов.

**Условие перехода — это список обязательных ключей `data` текущего этапа
(`TaskStage.required_keys`) плюс, если у этапа есть гейт, конкретное значение
ключа-гейта, и проверяет это КОД (`task_state.missing_keys`/`TaskStage.gate_satisfied`/
`_forward_target`), а не модель.** Тот же список недостающих пунктов уходит и в
контекст основного ответа (`task_state.build_context_message`), поэтому вежливый
отказ на преждевременное «переходи уже к портфелю» всегда совпадает с тем, что
реально произойдёт с состоянием.

**ГЕЙТ (`TaskStage.gate`) — пара «ключ, требуемое значение»: решение ЧЕЛОВЕКА,
которое из данных не выводится.** Их два:
- `planning` → `execution` у `portfolio`/`review`: `brief_verdict == "confirmed"`.
  Агент сначала собирает параметры, потом проговаривает СВОДКУ ВВОДНЫХ (`brief` —
  как он понял цель/горизонт/риск/ограничения) и только после прямого подтверждения
  начинает работу. Это и есть реализация «нельзя делать реализацию до утверждённого
  плана», сформулированная по-доменному: в инвестиционном консультировании
  подтверждают не «план работ», а вводные, на которых строится рекомендация. Сводка
  хранится ключом, а не остаётся вопросом в чате, чтобы после паузы её можно было
  показать заново из состояния, без обращения к LLM.
  Отказ (`brief_verdict == "corrected"`) откатывать некуда — это первый этап,
  поэтому гейт сбрасывается НА МЕСТЕ (`_reset_gate`): `brief`/`brief_verdict`
  удаляются, ставится `awaiting_correction`, этап остаётся тем же, а в чат уходит
  отдельная строка «🔁 Вводные не подтверждены» — молча начинать этап заново нельзя.
- `validation` → `done`: `verdict == "accepted"` (приёмка результата). Отказ
  (`changes_requested`) — это откат назад к работе, см. ниже.

У `asset` гейта на `planning` НЕТ намеренно: это справочный разбор, а не
рекомендация по структуре, и подтверждение трёх параметров было бы формальностью
ради автомата. При добавлении сценария решай так же — по тому, является ли
результат рекомендацией, а не «чтобы было одинаково».

**Откат назад В этап с гейтом снимает его гейт** (`_roll_back` чистит
`sequential_keys` и ключ гейта целевого этапа): если пользователь вернулся править
вводные, прежнее подтверждение сводки больше ничего не значит, иначе задача уехала
бы вперёд по устаревшему согласию. Остальные ключи целевого этапа, как и раньше,
сохраняются.

**ВПЕРЁД автомат двигается САМ, как только ключи собраны — поле `"stage"` из ответа
модели на это не влияет.** Так сделано не из вкуса: когда переход требовал явной
просьбы модели, она заполняла `verdict: accepted`, но оставляла в `"stage"` прежний
этап — агент говорил пользователю «задача завершена», а состояние навсегда
залипало на проверке (`stage: "validation"`, строка состояния продолжала ждать
ответа). Направление перехода в этом автомате всегда однозначно выводится из
данных, так что спрашивать его у модели незачем. Не возвращай сюда «модель сама
решает, пора ли дальше».

**НАЗАД — единственный переход по явному сигналу**: с этапа проверки направление
задаёт `verdict` (`changes_requested` → назад к работе), с остальных этапов —
более ранний допустимый этап в поле `"stage"` (например, пользователь хочет
поправить уже собранные параметры). Откат (`_roll_back`) снимает обязательные
ключи этапов ПОСЛЕ целевого, но НЕ самого целевого — возвращаются, чтобы поправить
один параметр, а не чтобы заново отвечать на все вопросы этапа. Поскольку условия
целевого этапа при этом остаются выполненными, откат выставляет флаг
`awaiting_correction`, который блокирует движение вперёд, пока не придёт правка
хотя бы одного ключа текущего этапа. Без этого флага автоматический переход вперёд
отменял бы откат в тот же миг — сохраняй эту пару «не чистим ответы + флаг» при
доработке. Ручной `/smart_agent_task_stage` назад идёт через тот же `_roll_back` по
той же причине.

**Словарь ключей фиксирован, и принимаются только ключи ТЕКУЩЕГО ЭТАПА**
(`task_state.normalize_data_updates`), остальные отбрасываются. Два разных повода, и
оба нужно сохранять:
1. Иначе модель на каждом ходу изобретала бы новое имя для того же параметра
   (`horizon`/«горизонт»/`time_horizon`), условие перехода никогда бы не выполнилось
   и задача зависла бы навсегда.
2. Иначе можно ПЕРЕПРЫГНУТЬ этап, заполнив ключи следующего: например, вернуть
   `allocation` вместе с `verdict: accepted` и уехать из работы прямо в `done` —
   не задав пользователю вопроса о приёмке и не дав сработать ревизору инвариантов
   (проверка вызывается на этапе проверки). Именно поэтому фильтр — по этапу, а не
   по сценарию; цикл движения вперёд в `apply_state_response` при этом можно
   оставить: данных на второй шаг в одном ходу взяться уже неоткуда.

На РУЧНУЮ запись (`/smart_agent_task_set`) фильтр не распространяется: там ключ
задаёт человек. Ключи-гейты дополнительно ограничены закрытым списком значений
(`CLOSED_VALUE_KEYS`: `verdict` — `accepted`/`changes_requested`, `brief_verdict` —
`confirmed`/`corrected`), поэтому произвольный текст вроде «пользователь вроде
согласен» отбрасывается, а не трактуется на глаз. А ключи гейта принимаются ещё и
ТОЛЬКО В ПОРЯДКЕ (`TaskStage.sequential_keys`): сводку нельзя проговорить до сбора
вводных, а подтвердить её нельзя до того, как она проговорена — без этого правила
модель закрыла бы гейт первым же ходом, и он перестал бы что-либо гарантировать.

**Кто двигает автомат.** Отдельный технический вызов LLM после КАЖДОГО ответа агента
(`SmartAgent.update_task_state`, `response_format={"type": "json_object"}`, свой
системный промпт) — по тому же образцу, что `Agent._update_facts()` у стратегии
Sticky Facts, включая один повтор с удвоенным `max_tokens` при обрезанном JSON.
Порядок важен: командный слой сначала отправляет ответ пользователю и только потом
делает этот вызов (`_task_service_lines` в `agents/smart_agent_command.py`) — иначе
пользователь ждал бы два запроса к API подряд, прежде чем увидеть хоть что-то.
Сбой вызова или невалидный JSON не трогают ни состояние, ни счётчик
`processed_pairs`, поэтому неразобранные пары копятся в бэклоге и достаются
следующей попытке (тот же принцип, что `facts_processed_pairs` у `Agent`).

**Старт задачи** возможен и явной командой, и автоматически — по намерению
пользователя в диалоге («хочу составить портфель»). Автоматический старт
срабатывает ТОЛЬКО когда активной задачи нет: молча подменять незавершённую задачу
новой автомат не должен, для смены сценария есть `/smart_agent_task_start`. Когда
задачи нет, вместо полного разбора состояния идёт укороченный вызов-детектор
(`_maybe_start_task`, `AGENT_TASK_START_SYSTEM_PROMPT`) — он случается после каждого
разового вопроса, и гонять ради него полный промпт незачем.

**Пауза — отдельный флаг (`paused`), ортогональный этапу**, а не ещё один этап:
поставить на паузу можно на любом этапе. Пока задача на паузе, технический вызов не
делается вообще, а основная модель получает указание не продолжать задачу (на другие
вопросы она отвечает как обычно). `/smart_agent_task_resume` печатает сводку «где
остановились», собранную ЦЕЛИКОМ из сохранённого состояния
(`task_state.describe_state`) — без обращения к LLM: это и есть проверяемая
реализация требования «продолжение без повторных объяснений».

**Данные задачи (`data`) живут до конца задачи, а не этапа.** Ключи предыдущих
этапов сознательно не вычищаются при переходе вперёд: на них держатся блок «Уже
собрано» в контексте каждого ответа (иначе на этапе работы модель не знает цели и
горизонта, собранных на профилировании), сводка после паузы (`describe_state`
строится целиком из состояния), откат назад (`_roll_back` чистит только этапы ПОСЛЕ
целевого — возвращаются поправить один параметр) и итог задачи
(`archive_fact`/`result_payload` читают ключи рабочих этапов). МЕЖДУ задачами
ничего не наследуется: оба пути старта (`start_task` и автостарт в
`_maybe_start_task`) заменяют слот целиком на `new_task()` с пустым `data`, а
`finish_task` обнуляет его. Единственное, что переживает задачу, — один компактный
факт в долговременной памяти.

**Завершение.** Как только автомат выставляет `stage="done"`, итог задачи
автоматически переносится в долговременную память ОДНИМ компактным фактом
(`task_state.archive_fact`, `source: "auto"`) — не по факту на каждый ключ `data`:
память ограничена 20 фактами с вытеснением, и подробный дамп одной задачи выбил бы
оттуда всё остальное. `/smart_agent_task_done` после этого просто очищает слот
задачи, как и раньше. Завершённая задача при этом НЕ блокирует следующую: для
автостарта `update_task_state` считает `done`-задачу отсутствующей, и новая задача
просто занимает слот (её итог уже в долговременной памяти) — иначе после каждого
завершения пользователю пришлось бы вручную вызывать `/smart_agent_task_done`,
прежде чем бот согласится начать новую работу. До этой команды завершённая задача
остаётся в слоте вместе со своим `data`, и оно продолжает уходить в контекст (с
инструкцией этапа `done` «новых шагов по ней не предлагай») — намеренно, чтобы агент
мог сослаться на только что выданный результат; автоматически слот после
архивирования не чистится.

**Наблюдаемость и предохранители.** После каждого ответа в чат уходит строка
состояния (`task_state.state_line`: сценарий, этап, чего ждём), а строки о событиях
(старт, смена этапа, сброс подтверждения вводных, отклонённая попытка,
архивирование) — только когда событие произошло.

**Отклонённые попытки видны, а не молчаливы.** Всё, что автомат НЕ сделал из
предложенного моделью, возвращается из `apply_state_response` в
`StateChange.rejected` (причины на русском: «переход … не разрешён из текущего
этапа», «`verdict` — не пункт этапа «формирование портфеля»», «значение … не из
списка», «сводка вводных — рано: сначала нужно собрать …») и печатается в чат
строкой «⛔ Автомат отклонил: …». Без этого «автомат не пустил» выглядело бы для
пользователя как проигнорированная просьба, а проверить поведение автомата можно
было бы только по логам. Те же записи, вместе с состоявшимися переходами, копятся в
КОРОТКОЙ ИСТОРИИ внутри задачи (`transitions`, последние `TRANSITIONS_MAX = 10`
записей, `record_transition`, у каждой — кто её сделал: автомат/вручную/откат/
инварианты/отказ на гейте/отклонено) и печатаются в `/smart_agent_task_show` и в
сводке после паузы. Это диагностика, а не журнал аудита — не превращай её в полный
лог и не выноси лимит в `.env`.

`/smart_agent_task_stage <этап>` переводит этап вручную, если автомат ошибся или
застрял. Что он проверяет, а что нет (`task_state.manual_transition_block`):
- **граф переходов** — соблюдается, иначе команда отклоняется и попытка пишется в
  историю;
- **заполненность обычных обязательных ключей** — НЕ проверяется, иначе застрявшую
  задачу нельзя было бы сдвинуть вообще (у пользователя остались бы только
  `/smart_agent_task_start` с потерей собранного и `/smart_agent_task_done`). При
  переводе ВПЕРЁД с несобранными пунктами команда печатает предупреждение со
  списком несобранного и оговоркой, что сам автомат такой переход не сделал бы;
- **гейт текущего этапа** — соблюдается, и это не симметрично предыдущему пункту
  намеренно: гейт не «условие по данным», а РЕШЕНИЕ ЧЕЛОВЕКА, и предохранитель,
  который его подменяет, обнуляет весь смысл гейта («нельзя начать работу без
  подтверждённых вводных» превращается в «нельзя, если не попросить по-другому»).
  Отказ называет и ключ, и явный путь: ответить в диалоге или
  `/smart_agent_task_set <ключ-гейта> <значение>` — там ключ указывает сам человек,
  то есть решение остаётся его и сказано прямо. Назад гейт не мешает никогда:
  вернуться и переделать можно всегда, а откат в этап с гейтом его же и снимает.

Отдельного тумблера «выключить автоматику» нет: выключение слоя `working`
(`/smart_agent_toggle working`) замораживает автомат целиком, т.к. выключенный слой
не участвует ни в сборке контекста, ни в записи.

**Денежные суммы не собираются ни на одном этапе ни в одном сценарии** — портфель
описывается только долями в процентах (см. «Правила предметной области» ниже). Это
прописано и в инструкциях этапов, и в промптах технического вызова; при добавлении
нового сценария или ключа сверяйся с этим правилом.

#### Профили персонализации

На чат может быть заведено НЕСКОЛЬКО именованных профилей (например,
«Консервативный»/«Агрессивный») — `SmartAgent._profiles: {имя: {"meta": {...},
"short_term": [...], "working": ..., "long_term": [...]}}` плюс указатель активного
(`SmartAgent._active_profile`). Это ТОТ ЖЕ ПРИНЦИП, что именованные ветки диалога
Branching у `Agent` (см. «Управление контекстом агента» выше), применённый к
независимой модели памяти `SmartAgent`: переключение профиля (`switch_profile`)
переключает разом диалог, задачу и факты — не только предпочтения. Новый профиль
(`create_profile`) стартует с ПУСТЫМИ `short_term`/`working`/`long_term` — не
наследует их от другого профиля, тот же принцип, что новая ветка Branching не
наследует facts/summary от ветки-источника.

`meta` — явно заданные пользователем предпочтения персонализации, `PROFILE_FIELDS` в
`agents/smart_agent.py`: `style` (стиль общения), `experience_level` (уровень опыта),
`format` (формат ответа), `risk_tolerance` (отношение к риску), `horizon` (горизонт
интересов), `interests` (интересующие темы/активы), `excluded_topics` (что не
затрагивать). Все поля — качественные предпочтения стиля общения, НИКОГДА не
финансовые/личные данные — при добавлении нового поля профиля сверяйся с «Правилами
предметной области» ниже. `meta` заполняется ТОЛЬКО явно пользователем — анкетой при
создании профиля (по одному вопросу на поле, можно пропустить словом «пропустить») в
`agents/smart_agent_command.py` (`_handle_new_profile_name`/
`_handle_profile_field_answer`, общий код для обоих путей анкеты ниже) или точечно
командой `/smart_agent_profile_set <ключ> <значение>` (`SmartAgent.update_profile_field`)
— НИКАКАЯ LLM-эвристика не решает за пользователя, что положить в профиль, тот же
принцип, что и у `working`/`long_term`.

Подключение к каждому запросу — `meta` уходит отдельным системным сообщением
(`SmartAgent._profile_meta_message()`), первым в порядке сборки контекста (профиль ->
долговременная память -> рабочая задача -> краткосрочный диалог, от самого
общего/стабильного к самому свежему), только если хотя бы одно поле заполнено. Текст
сообщения ЯВНО ограничивает влияние профиля стилем/форматом/объёмом ответа и
проговаривает, что он не отменяет обязательные предупреждения основного
`system_prompt` — это принципиально: персонализация не должна становиться обходным
путём для пользовательского системного промпта (см. «Ограничения безопасности»
ниже), а `risk_tolerance: агрессивный`, например, не должен быть поводом модели
убрать дисклеймеры о неопределённости рынка. Само подключение слоя `profile`
переключается `/smart_agent_toggle profile` наравне с тремя остальными слоями (общая
на весь чат настройка `enabled_layers`, не per-profile) — выключение прячет `meta` из
контекста, не удаляя её.

Автопоказ выбора профиля — если у чата НЕТ активного профиля, ему НЕОТКУДА взять
контекст для трёх слоёв памяти, поэтому:
- `/smart_agent` (`smart_agent_command`) при входе без активного профиля показывает
  пикер (кнопки: все существующие профили + «➕ Новый профиль») ВМЕСТО немедленного
  перехода в цикл вопросов — состояния `PROFILE_PICK`/`WAITING_PROFILE_NAME`/
  `WAITING_PROFILE_FIELD` того же `ConversationHandler`, ведущие в `WAITING_QUESTION`
  только после выбора/создания профиля.
- `/smart_agent_profile_delete <имя>`, удалив АКТИВНЫЙ профиль
  (`SmartAgent.delete_profile` явно ставит `active_profile = None`), сразу же
  показывает тот же пикер — чат не остаётся в подвешенном состоянии до следующей
  команды.
- Если активный профиль удалили ПРЯМО ПОСЕРЕДИНЕ диалога (чат в `WAITING_QUESTION`),
  `smart_agent_receive_question` проверяет это перед вызовом `ask()` и тоже
  показывает пикер, а не позволяет напороться на защитный `RuntimeError` в
  `SmartAgent.ask()` (тот рассчитан на программную ошибку вызывающего кода, а не на
  штатный пользовательский сценарий).

`/smart_agent_profile` (ОТДЕЛЬНЫЙ `ConversationHandler`, не часть `/smart_agent`) даёт
управлять профилем в ЛЮБОЙ момент, даже посреди диалога — по тому же принципу, что
`/agent_switch_branch` работает независимо от режима `/agent`. Пикер и анкета этого
пути ведут в `ConversationHandler.END` (подтверждение), а не в `WAITING_QUESTION` —
там просто нет диалога вопросов, в который возвращаться. Обе анкеты (эта и анкета
`/smart_agent` выше) используют ОБЩИЙ код сбора полей
(`_handle_new_profile_name`/`_handle_profile_field_answer` в
`agents/smart_agent_command.py`, транзитное состояние — в `context.user_data`), но
СВОИ inline-кнопки: `_START_PROFILE_CALLBACK_PREFIX` (`sa_start:`) для пикера
`/smart_agent`, `_MANAGE_PROFILE_CALLBACK_PREFIX` (`sa_manage:`) — для
`/smart_agent_profile`/`/smart_agent_profile_delete`. РАЗНЫЕ префиксы — не
стилистический выбор: если бы оба пути использовали один и тот же `callback_data`,
`ConversationHandler` одного пикера мог бы перехватить нажатие кнопки, показанной
другим (см. ниже про порядок регистрации). Обработчик кнопок manage-пикера
(`smart_agent_manage_pick_callback`) зарегистрирован ЕЩЁ и как entry point своего
`ConversationHandler`-а — иначе кнопки на пикере, отправленном
`/smart_agent_profile_delete` ВНЕ какого-либо диалога, было бы некому обработать.

Порядок регистрации в `main.py` ВАЖЕН:
`build_smart_agent_profile_conversation_handler()` регистрируется ПЕРЕД
`build_smart_agent_conversation_handler()`. Оба `ConversationHandler`-а могут
одновременно отслеживать состояние одного чата (PTB это позволяет — независимые
трекеры), и пока анкета `/smart_agent_profile` активна (ожидает имя/поле — обычный
текст), именно она должна первой перехватывать очередное текстовое сообщение, а не
`WAITING_QUESTION` команды `/smart_agent` — иначе ответ на вопрос анкеты ушёл бы
агенту как обычный вопрос пользователя.

Миграция старого формата — файлы `AGENT_MEMORY_DIR/<chat_id>.json`, созданные ДО
появления профилей (плоские `short_term`/`working`/`long_term` без `profiles`), при
загрузке (`SmartAgent._load_state()`) оборачиваются в один профиль (имя
`"Профиль по умолчанию"`), который СРАЗУ становится активным — чтобы уже работающие
чаты не прерывались выбором профиля на пустом месте после обновления бота. Пикер
показывается только чатам, у которых действительно нет активного профиля (новый чат
или профиль был явно удалён).

`enabled_layers` (`{"profile": bool, "invariants": bool, "short_term": bool,
"working": bool, "long_term": bool}`, все пять `True` по умолчанию) — ОБЩАЯ НА ВЕСЬ
ЧАТ (не per-profile) настройка того, какие слои участвуют в СБОРКЕ контекста
конкретного вызова (`SmartAgent._build_context_messages()`), переключается
`/smart_agent_toggle <profile|invariants|short|working|long>`
(`SmartAgent.set_layer_enabled`) БЕЗ удаления данных слоя — это инструмент проверки
влияния слоя на ответ (задать один вопрос с разными комбинациями включённых слоёв и
сравнить ответы), а не способ очистки. Инварианты — единственный слой, выключение
которого снимает ещё и проверку результата задачи, см. «Инварианты smart-агента».
`SmartAgent.reset_all()` (`/smart_agent_reset`) очищает три слоя памяти АКТИВНОГО
ПРОФИЛЯ (не сам профиль, его `meta`, его инварианты, другие профили и не
`enabled_layers`) — тот же
принцип, что `Agent.reset()` не трогает выбранную стратегию, это настройка режима
работы, а не часть очищаемых данных; для удаления профиля целиком есть отдельная
явная команда `/smart_agent_profile_delete`.

`SmartAgent.get_last_context_messages()` возвращает ровно те системные сообщения, что
реально ушли в LLM на последний `ask()` — на этом построена наблюдаемость: команда
`/smart_agent_show` печатает профиль, инварианты, все три слоя как есть, их статус
включено/выключено и последний собранный контекст — так видно и что попало в каждый
слой, и что из этого реально дошло до модели.

Как и `Agent`, `SmartAgent` использует `main_client`/`MAIN_MODEL`/`SYSTEM_PROMPT` (см.
«Ограничения безопасности» про то, почему слои памяти и профили не нарушают запрет на
runtime-переключение модели/провайдера/промпта — они влияют только на сборку
контекста для уже выбранного провайдера) и не перехватывает исключения OpenAI SDK в
`ask()` — перевод в сообщение на русском делает `agents/smart_agent_command.py`, тем
же паттерном, что `agent_command.py`/`handle_message`.

## Правила предметной области: инвестиционные рекомендации

Это не универсальный чат-бот, а сервис, отвечающий на вопросы о деньгах и инвестициях —
к содержимому ответов и к обработке пользовательских данных нужно относиться внимательнее,
чем в обычном проекте:

- Бот не является лицензированным финансовым советником, и это должно быть явно видно
  пользователю (в `/start`, `/help` и/или в `SYSTEM_PROMPT`, а не только в одном месте
  «для галочки»). Любая доработка текста ответов не должна убирать эту оговорку.
- Ответы не должны звучать как гарантия результата («точно вырастет», «гарантированная
  доходность X%») — формулировки должны отражать неопределённость рынка. Это в первую очередь
  задаётся через `SYSTEM_PROMPT`: при его изменении проверяй, что осторожные формулировки и
  дисклеймер сохраняются, а не только тон/стиль ответа.
- Не запрашивать и не хранить чувствительные финансовые и личные данные пользователя (номера
  счетов/карт, суммы на счетах, паспортные данные, ИНН и т.п.). Бот работает с вопросами общего
  характера, а не с персональным финансовым профилем — если в задаче просят добавить сбор таких
  данных, это отдельное архитектурное решение, а не мелкая доработка, и его стоит уточнить у
  пользователя явно, а не делать по умолчанию.
- Текущий stateless-режим основного потока (`handle_message`, без истории диалога) —
  сознательный выбор именно потому, что финансовые вопросы пользователя не должны
  накапливаться и храниться без явной необходимости. Персонализация (профили
  `/smart_agent_profile*`, см. «Профили персонализации» в разделе «Архитектура») —
  РЕАЛИЗОВАННОЕ, явно обсуждённое и запрошенное пользователем исключение из этого
  правила, ограниченное `/smart_agent` (не затрагивает `handle_message`/`/agent`) и
  качественными предпочтениями стиля общения (стиль, формат, риск как декларируемая
  склонность, горизонт, интересы) — НЕ персональным финансовым профилем (портфель,
  суммы, счета). Рабочая задача `/smart_agent` (сценарии `portfolio`/`asset`/`review`,
  см. «Конечный автомат рабочей задачи») — такое же явно запрошенное исключение, и
  оно намеренно ограничено качественными параметрами и ДОЛЯМИ В ПРОЦЕНТАХ: ни один
  этап ни одного сценария не собирает денежные суммы, поэтому сценарий вида
  «накопить сумму X к сроку» в реестр не добавлялся. Любое дальнейшее расширение
  полей профиля, набора сценариев или персонализации в других частях бота — то же
  архитектурное решение, что и раньше, и должно обсуждаться отдельно, а не
  добавляться как побочный эффект другой задачи.
- Оценка соответствия конкретному законодательству (лицензирование инвестиционного
  консультирования, требования к дисклеймерам в конкретной юрисдикции и т.д.) — ответственность
  владельца продукта, а не то, что можно вывести из кода; Claude Code не должен утверждать
  правовое соответствие, а должен по умолчанию сохранять осторожные формулировки и дисклеймеры.

## Ограничения безопасности, которые нужно сохранять

- Никаких админ/владельческих команд, никакого переключения модели или провайдера во
  время выполнения, никакого пользовательского системного промпта — не добавляй команды
  или параметры, позволяющие пользователю чата менять `MAIN_CLIENT`/`MAIN_MODEL`
  (основной поток) или `SYSTEM_PROMPT`. Смена `MAIN_CLIENT`/`MAIN_MODEL` — решение
  оператора бота через `.env` и перезапуск процесса, а не рантайм-настройка. Это не
  относится к фиксированным кнопкам-сценариям `/research_models` — там пользователь
  выбирает один из заранее заданных research-сценариев, а не произвольную модель или промпт.
  Аналогично не относится к `/agent_context` (переключение одной из 4 фиксированных
  стратегий управления контекстом `/agent`, см. «Управление контекстом агента») и к
  `/agent_checkpoint`/`/agent_branch`/`/agent_switch_branch` — все они меняют, как
  собирается контекст для уже выбранного `MAIN_CLIENT`/`MAIN_MODEL`/`SYSTEM_PROMPT`,
  а не сам провайдер/модель/системный промпт. Не относится и к `/smart_agent_toggle`
  (включение/выключение слоя памяти в сборке контекста, см. «Управление памятью
  smart-агента») — по тому же принципу. Также не относится к профилям персонализации
  `/smart_agent_profile*` (несколько именованных профилей со своими предпочтениями
  стиля/формата/риска и т.п., см. «Профили персонализации» там же) — это НЕ
  пользовательский системный промпт: `meta` профиля уходит отдельным системным
  сообщением, а не подменяет собой `SYSTEM_PROMPT`, и её текст явно ограничивает
  влияние профиля стилем/форматом/объёмом ответа, не отменяя обязательные
  предупреждения. При дальнейшей доработке персонализации не позволяй профилю
  переопределять сам `SYSTEM_PROMPT`, провайдера или модель — только то, как
  подаётся уже сформированный ответ. Не относится и к инвариантам
  (`/smart_agent_invariant_add`/`/smart_agent_invariant_show`/
  `/smart_agent_invariant_remove`, см. «Инварианты smart-агента»): они СУЖАЮТ то,
  что агент готов предложить, но не могут ослабить `SYSTEM_PROMPT` — системное
  сообщение со списком инвариантов прямо запрещает отменять ими обязательные
  предупреждения и осторожные формулировки, и этот пункт нужно сохранять при любой
  доработке текста, иначе слой превратится в пользовательский системный промпт.
- История сообщений не сохраняется и не передаётся между запросами — не добавляй
  функциональность памяти/истории без явного запроса пользователя (см. также раздел выше про
  инвестиционные данные — здесь это не только архитектурное, но и доменное ограничение).
  **Исключение** — команда `/agent` (`agents/agent.py`, `agents/agent_command.py`): по
  явному запросу пользователя она хранит историю диалога (по веткам, см. «Управление
  контекстом агента») в JSON на диске (`AGENT_HISTORY_DIR`, см. «Конфигурация») и
  переживает перезапуск бота — это единственное место в проекте с таким поведением,
  оно не распространяется ни на `handle_message` (`main.py`), ни на research-режимы.
  Пользователь может посмотреть свою историю командой `/agent_history` и очистить её
  командой `/agent_reset` (очищает все ветки/чекпоинты/facts/summary разом). При
  дальнейшей доработке `/agent` сохраняй этот принцип (история — только там, видимая
  и явно очищаемая пользователем), а не расширяй память на остальной бот молча.
  `/agent_compare` (`agents/compare_command.py`) — то же исключение, применённое
  трижды: по явному запросу пользователя (вход в `/agent_compare`) три
  дополнительных файла истории на чат, по одному на сравниваемую стратегию;
  очищаются разом командой `/agent_compare_reset`, отдельно от `/agent_reset`.
  `/smart_agent` (`agents/smart_agent.py`, `agents/smart_agent_command.py`) — то же
  исключение, но НЕЗАВИСИМОЕ от `/agent`/`/agent_compare` (свой класс, свой файл
  памяти в `AGENT_MEMORY_DIR`, см. «Управление памятью smart-агента»): по явному
  запросу пользователя хранит память тремя явно разделёнными слоями (в разрезе
  именованных профилей персонализации, которые пользователь тоже заводит явно).
  Изначально все три слоя `/smart_agent` писались ТОЛЬКО по явной команде
  пользователя. Сейчас это верно не для всех: по отдельному, явно обсуждённому
  запросу пользователя проекта `working` и `long_term` пополняются ещё и
  АВТОМАТИЧЕСКИ — техническим вызовом LLM после каждого ответа агента, который
  ведёт конечный автомат задачи и извлекает факты из диалога (см. «Конечный автомат
  рабочей задачи» выше). Это осознанное сужение прежней гарантии, а не недосмотр;
  взамен действуют: явный запрет на чувствительные данные в промптах самого
  технического вызова, видимость всего происходящего (строка состояния после
  каждого ответа, `/smart_agent_show`, `/smart_agent_long_show` с пометкой
  происхождения факта) и полностью ручное управление поверх автоматики
  (`/smart_agent_task_stage`, `/smart_agent_forget`, `/smart_agent_task_set`,
  `/smart_agent_toggle`). Слой ИНВАРИАНТОВ (жёсткие ограничения, которые агент не
  имеет права нарушать) сужения не касается вовсе: он пишется ТОЛЬКО явными
  командами пользователя, автоматической записи в него нет — при доработке не
  распространяй на него автоизвлечение, тем же правилом, что и на `meta` профиля.
  Профиль (`meta`) остаётся строго явным: пишется только
  анкетой или `/smart_agent_profile_set`, и НИКАКАЯ эвристика не должна решать за
  пользователя, что в него положить — при доработке не распространяй автоматическую
  запись на профиль. Ни один слой не фильтрует ввод, поэтому ответственность за
  чувствительные финансовые/личные данные остаётся на пользователе (см. «Правила
  предметной области» выше), а тексты команд должны об этом явно предупреждать.
- Групповые чаты исключены через фильтр обработчика; сохраняй этот фильтр при добавлении
  новых обработчиков текстовых сообщений.
