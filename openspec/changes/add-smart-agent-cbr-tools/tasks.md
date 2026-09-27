# Tasks

## 1. Конфигурация

- [ ] 1.1 Добавить в `src/config.py` `MCP_CBR_RATES_ENABLED` — булев флаг по образцу
      `RESEARCH`/`PRICE_WATCH` (`_TRUE_VALUES`/`_FALSE_VALUES`, по умолчанию `false`),
      с проверкой значения в `_validate_config()` (тот же паттерн, что `_RESEARCH_RAW`/
      `_PRICE_WATCH_RAW`). Никакого каталога/пути — у источника его нет. Проверить:
      `python3 -m py_compile src/config.py`; бот стартует без переменной и без
      предупреждений; недопустимое значение (`MCP_CBR_RATES_ENABLED=maybe`) завершает
      процесс с понятным сообщением, как `RESEARCH`.

## 2. Механика запуска источника без каталога (`mcp_integration/market_session.py`)

- [ ] 2.1 Заменить поля `MarketSource.directory: str`/`.program: str` на
      `.params: StdioServerParameters` (design.md, решение 1). Обновить
      `open_market_tools()`: принимает `params: StdioServerParameters` вместо
      `(directory, program)`, остальная сигнатура (`timeout, max_result_chars,
      allowlist`) не меняется по смыслу. `build_server_params(directory,
      program="mcp-moex")` НЕ менять (существующие тесты на неё продолжают проходить
      как есть) — теперь он используется ТОЛЬКО как один из способов получить
      `params` для `MarketSource`, не как параметр самого `open_market_tools()`.
      Проверить: существующие тесты `test_market_tools.py` на `MarketTools`/
      `MultiMarketTools`/`open_market_tools` (через прямые вызовы с явно
      сконструированными `params`) проходят без потери покрытия.
- [ ] 2.2 Добавить `build_uvx_server_params(package: str) -> StdioServerParameters`
      — `StdioServerParameters(command="uvx", args=[package])`, без shell. Проверить:
      юнит-тест на команду.
- [ ] 2.3 Ввести `SOURCE_CBR = "cbr"`, `CBR_PACKAGE = "atomno-mcp-cbr-rates"`,
      `CBR_MODEL_TOOL_ALLOWLIST = frozenset({"get_rate", "history_rates", "key_rate",
      "inflation", "statistics"})`. Проверить: тест на состав allowlist.
- [ ] 2.4 Добавить `require_read_only: bool = True` в `MarketSource` (значение по
      умолчанию сохраняет поведение MOEX/Bybit без изменений вызовов их
      конструктора). Обновить `is_model_tool(tool, allowlist, require_read_only=True)`
      — при `False` не проверяет `is_read_only(tool)`, оставляя проверку `allowlist`
      обязательной. Обновить `MarketTools.__init__` (принимает и прокидывает тот же
      параметр в фильтрацию) и `open_market_tools()`/`open_multi_market_tools()`
      (прокидывают `source.require_read_only`). Проверить: тесты (design.md, решение
      2) — источник без пометки, но с `require_read_only=False` и именем в allowlist,
      получает инструмент; источник БЕЗ этого флага (по умолчанию) с той же
      непомеченной ситуацией — не получает (регресс-тест на существующее поведение
      MOEX/Bybit); инструмент вне allowlist не проходит НЕЗАВИСИМО от
      `require_read_only`.

## 3. Слой и правила (`agents/market_tools.py`)

- [ ] 3.1 Добавить `SOURCE_CBR` в `SOURCE_LABELS` (короткая подпись, например «CBR»)
      и `SOURCE_DESCRIPTIONS` (например: «Банк России — официальные курсы валют,
      ключевая ставка, инфляция (справочные данные, не рыночные цены)»). Проверить:
      `_context_head()` включает CBR в перечисление, когда он в списке источников.
- [ ] 3.2 Добавить в `AvailableSource` третье поле (например `has_reference_notes:
      bool`, независимое от `analytics_available` — design.md, решение 3) и функцию
      `_cbr_notes_block()` — абзац CBR: курс не рыночный, называть дату курса из
      результата, предел `history_rates` (не более года за вызов, повторные вызовы
      для длиннее), не путать `statistics`(снимок) с детальным запросом. `_ANALYTICS_
      BLOCKS`/цикл в `build_context_message()` добавляют абзац CBR по этому полю, БЕЗ
      условия на инструменты анализа (в отличие от MOEX/Bybit). Проверить: тесты —
      абзац CBR показывается всегда при наличии источника, независимо от
      `analytics_available`; абзац MOEX/Bybit не регрессировал (снапшот-тест текста);
      абзац не отменяет обязательные предупреждения (существующая проверка на всех
      абзацах).
- [ ] 3.3 Проверить `describe_status()`/предупреждения (`build_unavailable_context_message`/
      `build_disabled_context_message`/`unavailable_warning`) работают с CBR без
      изменений сигнатур (они уже принимают список подписей — просто добавляется
      третья). Проверить: тест с CBR в списке недоступных/выключенных источников.

## 4. `SmartAgent` (`agents/smart_agent.py`)

- [ ] 4.1 Добавить `self._cbr_enabled: bool` (из `MCP_CBR_RATES_ENABLED`, параметр
      конструктора по образцу `mcp_moex_dir`/`mcp_bybit_dir`, для тестируемости).
      Проверить: `py_compile` + конструирование `SmartAgent` с флагом true/false.
- [ ] 4.2 `_configured_sources()`: добавить ветку CBR (design.md, решение 4) —
      `MarketSource(SOURCE_CBR, label, build_uvx_server_params(CBR_PACKAGE),
      CBR_MODEL_TOOL_ALLOWLIST, require_read_only=False)`, добавляется в список ТОЛЬКО
      если `self._cbr_enabled`. Проверить: с флагом выключенным CBR отсутствует в
      списке; с флагом включённым присутствует наряду с настроенными MOEX/Bybit.
- [ ] 4.3 `get_tools_status()`: добавить запись для CBR (`STATUS_NOT_CONFIGURED`, если
      флаг выключен, иначе — последний известный статус, тот же принцип, что у
      MOEX/Bybit по `_source_dirs`). Проверить: живой прогон — с выключенным флагом
      статус CBR «не настроен», с включённым — обновляется после вопроса.
- [ ] 4.4 Убедиться, что `_run_tool_loop()`/сборка `AvailableSource` для CBR передаёт
      новое поле (`has_reference_notes=True` для CBR, `False` для MOEX/Bybit) и что
      это не требует данных про `analytics_available` источника CBR (у него его нет,
      `MultiMarketTools.analytics_available_for("cbr")` естественно вернёт `False`,
      так как у источника нет инструментов из `REF_PARAMS`). Проверить: живой прогон —
      абзац CBR показывается в собранном контексте LLM (`get_last_context_messages()`).

## 5. Тесты

- [ ] 5.1 Добавить в `tests/test_market_tools.py` тесты на: `build_uvx_server_params`,
      `MarketSource.params`-based `open_market_tools`, `require_read_only=False`
      (позитивный и регрессионный негативный случай), `CBR_MODEL_TOOL_ALLOWLIST`,
      абзац CBR в `build_context_message` (показывается без `analytics_available`,
      не показывается при отсутствии источника), отсутствие ref-цепочки у CBR
      (`MultiMarketTools.analytics_available_for` возвращает `False`, ссылки CBR не
      возникают, так как у него нет `get_price_history`-подобного инструмента).
      Проверить: `pytest tests/` проходит целиком.
- [ ] 5.2 Убедиться, что рефакторинг `MarketSource`/`open_market_tools` (задача 2.1)
      не сломал существующие тесты MOEX/Bybit — обновить только те вызовы, что
      напрямую конструируют `MarketSource`/`open_market_tools` со старыми полями
      `directory`/`program`. Проверить: `pytest tests/test_market_tools.py -v` без
      падений; `ruff check src/` не показывает новых замечаний относительно
      снятого перед этим изменением baseline (сравнить `git stash`/`git stash pop`).

## 6. Проверка на живом сервере и ручной прогон

- [ ] 6.1 Прогнать `uvx atomno-mcp-cbr-rates` вручную (уже проверено при подготовке
      предложения — используется как smoke-проверка ПОСЛЕ реализации, а не только
      до неё): подключиться и вызвать `get_rate`/`history_rates`/`key_rate`/
      `inflation`/`statistics` на реальных данных, сверить, что: `get_rate` без
      `on_date` действительно возвращает последнюю опубликованную дату (не
      сегодняшнюю, если сегодня выходной), `history_rates` действительно отклоняет
      диапазон длиннее ~366 дней (сверить точный текст ошибки сервера — он должен
      дойти до модели как есть, тем же принципом, что и ошибки MOEX/Bybit).
- [ ] 6.2 Полный ручной прогон бота с CBR включённым (`MCP_CBR_RATES_ENABLED=true`)
      наряду с MOEX/Bybit: `/smart_agent` → вопрос про курс доллара (модель вызывает
      `cbr__get_rate`, называет дату курса, отличает её от рыночной цены), вопрос про
      ключевую ставку/инфляцию, вопрос, требующий данных сразу от CBR и MOEX/Bybit в
      одном сообщении (например, «курс доллара по ЦБ и цена SBER») — сверить с
      delta-сценариями `smart-agent-market-tools`. Затем `/smart_agent_show` — статус
      CBR отдельной строкой. Затем прогон с флагом выключенным — поведение идентично
      состоянию до этого изменения.

## 7. Документация

- [ ] 7.1 Добавить `MCP_CBR_RATES_ENABLED` в `.env.example` рядом с `MCP_BYBIT_DIR` —
      булев флаг, без каталога, с пояснением, что источник ставится и запускается
      автоматически через `uvx` при первом вопросе.
- [ ] 7.2 Обновить `CLAUDE.md` (разделы «Конфигурация», список файлов
      `mcp_integration/market_session.py`/`agents/market_tools.py`, «Инструменты
      рыночных данных smart-агента» в «Архитектура»): третий источник без каталога,
      обобщение `MarketSource` на `params`, исключение из `read_only_hint` (явно, с
      формулировкой «почему это не общее правило»), абзац CBR не через
      `analytics_available`. Обновить README.md (описание слоя `tools` для
      пользователя) аналогично, упомянув, что курс ЦБ — официальный, не биржевой.
