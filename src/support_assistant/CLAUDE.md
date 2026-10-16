# support_assistant/ — режим «ассистент поддержки» (`/support`)

Корневые правила — в `/CLAUDE.md`. Спеки — `openspec/specs/support-assistant/spec.md` и
`openspec/specs/support-crm-mock/spec.md` (после архивации изменения `add-support-assistant`),
обоснования — `openspec/changes/*add-support-assistant/design.md`.

Режим отвечает о работе САМОГО бота: по фрагментам FAQ и документации (отдельный индекс) и по
открытым обращениям (тикетам) ЭТОГО пользователя из CRM через MCP. Это не инвестиционный режим:
вопросы об инвестициях в нём отклоняются, дисклеймер и осторожные формулировки основного
`SYSTEM_PROMPT` не ослабляются. Устроен по образцу `dev_assistant/` (общие чанкер, сборка индекса,
механика поиска и сессии MCP переиспользуются импортом).

## Модули

- `rules.py` — ЧИСТОЕ: команда, тексты режима и фиксированные ответы без вызова модели, лимиты полей
  тикета (`TICKET_FIELD_MAX_CHARS`, `TICKET_EVENTS_MAX`). Не импортирует `config`.
- `sources.py` — фиксированный перечень: `README.md` и `docs/support/**/*.md`; `CLAUDE.md`, спеки,
  исходники и `data/` — нет.
- `indexer.py` + `cli.py` — `make support-index` (`python -m support_assistant.cli`): полная
  пересборка в `SUPPORT_INDEX_DIR`, стратегия `markdown`; чтение, чанкер и метаданные — из
  `dev_assistant.indexer`, атомарную замену даёт `rag.index_store.build_index()`.
- `search.py` — `SupportSearchConfig` + обёртка над `dev_assistant.search.retrieve`; исходы `FOUND` /
  `NOT_FOUND` / `NO_INDEX` / `FAILED`. Без переписывания вопроса и второго этапа отбора.
- `crm_session.py` — запуск CRM (`build_crm_server_params`), `SUPPORT_TOOL_ALLOWLIST`,
  `strip_user_param`/`force_user_param`, `CrmTools`, `open_crm_tools`, предзагрузка обращений
  (`preload_tickets`, `select_open_tickets`, `normalize_ticket`).
- `prompts.py` — системное сообщение поддержки, слой CRM (обращения / нет / недоступна / не
  настроена), сборка сообщений (`build_messages`), источники (`format_sources`).
- `core.py` — `SupportAssistant.ask(chat_id, user_id, question, should_cancel)`; Telegram не
  импортирует. Матрица исходов — design.md, решение 7.
- `command.py` — `/support`, вопросы, `/cancel`, `build_support_handlers()`.
- Вне пакета: `mock_servers/support_crm/` (мок CRM: `crm_store.py` — чистая выборка, `server.py` —
  MCP на `MCPServer` из `mcp` 2.x, `data/*.json` — вымышленные данные). `src/` мок НЕ импортирует,
  бот запускает его отдельным процессом (`sys.executable server.py --demo-user-id <id>`).

## Что сохранять

- **Идентификатор пользователя подставляет код.** `telegram_user_id` — id ПОЛЬЗОВАТЕЛЯ
  (`update.effective_user.id`), не чата; удалён из схем инструментов и перезаписывается в каждом
  вызове, даже если модель его прислала. Отправитель `None` — отказ без поиска, модели и CRM. Вторая
  граница — сервер CRM: только данные запрошенного id, чужой тикет неотличим от несуществующего,
  неизвестный пользователь получает пустой профиль. Настоящая CRM обязана обеспечить то же.
- **Модели — только `get_my_profile`, `list_my_tickets`, `get_ticket`** (`SUPPORT_TOOL_ALLOWLIST` +
  `require_read_only=True`). Новый инструмент (особенно изменяющий) — отдельное решение владельца.
- **Обращения запрашивает код на каждый вопрос** (не модель): открытые (не `closed`), самые свежие
  первыми, не больше `SUPPORT_TICKETS_LIMIT`, поля усечены. Подаются отдельным системным сообщением
  как ДАННЫЕ; тексты тикетов — недоверенный ввод (prompt injection), угловые скобки в них
  заменяются (`prompts._safe`). Мок содержит тикет с «инструкцией» для проверки; не убирай.
- **Недоступная CRM не блокирует ответ по справке**: модель получает «НЕДОСТУПНЫ», пользователь —
  предупреждение. Если справка ничего не нашла, а подавать нечего — фиксированный текст БЕЗ вызова
  модели; нет индекса / сорван поиск — тоже без модели.
- **Сообщения.** `SYSTEM_PROMPT` — первым и без изменений; системное сообщение поддержки — ОТДЕЛЬНОЕ,
  с запретом инвестиционных рекомендаций и оговоркой, что оно не ослабляет основной промпт (тест
  проверяет — не убирай). Пользователь не выбирает модель, провайдера или промпт.
- **Источники строит код** по фактически поданным фрагментам. Номера тикетов в ответе называет
  модель, код их не проверяет.
- **История — только память процесса** (`SUPPORT_HISTORY_TURNS` пар итогов без фрагментов, данных
  обращений и вызовов инструментов); на диск не пишется, очищается `/cancel` и перезапуском; пишется
  ПОСЛЕ проверки отмены. Данные CRM на стороне бота не хранятся.
- **Очередь**: вид задачи `support` (`llm_queue/rules.py`), интерактивный приоритет; `ask()` — в
  потоке под слотом, отмена кооперативна; `/cancel` снимает вопросы чата.
- **Взаимоисключение**: режим отмечает чат в `agents/active_mode.py` (`SUPPORT_MODE`) — `/dev` не
  открывается поверх него; `/agent*`/`/smart_agent` внутри диалога отклоняются подсказкой выйти;
  `/dev` отклоняет `/support` внутри своего диалога. Обработчик регистрируется в `main.py` ПОСЛЕ
  `/dev` и ДО `/agent*`; вне флага `RESEARCH`; строка в `/help` с неизменной оговоркой о советнике.
- **Данные наружу**: текст вопроса — серверу эмбеддингов и основной модели; вместе с ним модели —
  фрагменты справки и открытые обращения пользователя. При облачном `MAIN_CLIENT` это покидает
  машину оператора (для локальной обработки — `MAIN_CLIENT=ollama`). Реальная CRM с персональными
  данными — отдельное решение владельца проекта; мок и FAQ их не содержат.

## Конфигурация (`config.py`, `.env.example`)

`SUPPORT_INDEX_DIR` (`data/support_index`), `SUPPORT_TOP_K` (5), `SUPPORT_MIN_SCORE` (0.0 — не
откалибровано), `SUPPORT_HISTORY_TURNS` (6), `SUPPORT_TICKETS_LIMIT` (5), `SUPPORT_CRM_MOCK` (true),
`SUPPORT_DEMO_USER_ID` (пусто — демо-данные никому; не положительное целое — запуск останавливается),
`MCP_SUPPORT_COMMAND`/`MCP_SUPPORT_ARGS` (внешняя CRM, приоритетнее мока; `shlex.split`, без shell).
Нет ни команды, ни мока — режим работает только по FAQ. Общие: `MCP_TIMEOUT_SECONDS`,
`MCP_MAX_TOOL_STEPS`, `MCP_TOOL_RESULT_MAX_CHARS`, `RAG_SEARCH_TIMEOUT_SECONDS`.

## Тесты

`tests/test_support_assistant_{rules,sources,indexer,search,crm,prompts,core}.py` и
`tests/test_support_crm_mock.py` — без сети, Telegram и LLM (поиск, CRM-сессия и вызов модели
подставляются фейками). Обработчики `command.py`, живой мок по stdio и Ollama — вручную.
