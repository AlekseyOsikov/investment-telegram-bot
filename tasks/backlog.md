# Пул задач для execution loop (День 5)

Пул утверждён владельцем проекта. База — тег `exec-loop-baseline` (ветка `exec-loop-base`): в код
заранее подсажены баги и расхождения в документации. Задачи независимы, порядок — по номеру.

Правила для исполнителя: работай по `CLAUDE.md` и `docs/code-style.md`; в рамках одной задачи
правь только то, что она требует; тесты не ослабляй и не удаляй; один коммит на задачу. Критерий
«сделано» — выход 0 у `tasks/check.sh <ID>` (запускает исполнитель сам: он проверяет приёмку
задачи и полный `pytest`). Файл `tasks/check.sh` не читай и не правь.

Статус и результаты — в `tasks/log.md` (ведёт раннер, не исполнитель).

| ID | Тип | Профиль | Сложность |
|---|---|---|---|
| T01 | bug | bug-fix | лёгкая |
| T02 | bug | bug-fix | лёгкая |
| T03 | bug | bug-fix | лёгкая |
| T04 | bug (без падающего теста) | bug-fix | средняя |
| T05 | bug (без падающего теста) | bug-fix | средняя |
| T06 | test | test-writer | лёгкая |
| T07 | test | test-writer | лёгкая |
| T08 | test | test-writer | средняя |
| T09 | test | test-writer | средняя |
| T10 | feat | фича | лёгкая |
| T11 | feat | фича | лёгкая |
| T12 | feat | фича | лёгкая |
| T13 | bug / refactor | bug-fix | средняя |
| T14 | refactor | фича | средняя |
| T15 | refactor (ruff) | фича | средняя |
| T16 | docs | doc-sync | лёгкая |
| T17 | docs | doc-sync | лёгкая |
| T18 | docs | doc-sync | лёгкая |

---

## T01 — bug: нулевой id пользователя принимается в списке `/dev`
**Симптом:** `pytest tests/test_dev_assistant_rules.py` падает на `test_invalid_elements_raise[0]`.
**Сделано, когда:** тест зелёный, сам тест не менялся, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T01`

## T02 — bug: склонение «11 минут» (и 111, 211…)
**Симптом:** `pytest tests/test_price_watch_texts.py` падает на `test_every_text_declines_units`.
**Сделано, когда:** тест зелёный, сам тест не менялся, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T02`

## T03 — bug: порядок новостей в дайджесте
**Симптом:** падает `tests/test_news_digest_rules.py::test_sort_by_importance_then_freshness_with_unanalyzed_last`;
`ruff` по `src/news_digest/rules.py` даёт замечание.
**Сделано, когда:** тест зелёный и не менялся, `ruff check src/news_digest/rules.py` чисто,
полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T03`

## T04 — bug: лимит сообщений дайджеста уходит в минус
**Симптом (отчёт пользователя):** если за сутки отправлено больше сообщений, чем лимит
(например, лимит снизили в `.env`), бот считает «осталось» отрицательным числом, и отбор срочных
новостей ведёт себя странно. Тест на этот случай отсутствует.
**Сделано, когда:** ошибка исправлена, добавлен регрессионный тест, который падает на старом коде,
полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T04`

## T05 — bug: `/queue_cancel 0` выбирает вопрос
**Симптом (отчёт пользователя):** номер вопроса в `/queue_cancel` нумеруется с 1, но `0`
почему-то попадает в список. Теста на этот случай нет.
**Сделано, когда:** ошибка исправлена, добавлен регрессионный тест, который падает на старом коде,
полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T05`

## T06 — test: покрыть вход и сборку обработчика `/research_chunking_compare`
Функции `chunking_compare_command` и `build_chunking_compare_conversation_handler` в
`src/research/chunking_compare.py` не покрыты.
**Сделано, когда:** покрытие `src/research/chunking_compare.py` ≥ 57 %, тесты в
`tests/test_research_chunking_compare.py` без сети и Telegram, `src/` не менялся, полный `pytest`
зелёный.
**Проверка:** `tasks/check.sh T06`

## T07 — test: покрыть `research/_shared.py` (обработчик отмены)
**Сделано, когда:** покрытие `src/research/_shared.py` = 100 %, тесты в
`tests/test_research_shared.py`, `src/` не менялся, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T07`

## T08 — test: покрыть `bot_sender` и `start_scheduler` в `price_watch/scheduler.py`
**Сделано, когда:** покрытие `src/price_watch/scheduler.py` ≥ 99 %, тесты в
`tests/test_price_watch_scheduler.py` без сети и сна, `src/` не менялся, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T08`

## T09 — test: покрыть `/research_chunking_stats`
**Сделано, когда:** покрытие `src/research/chunking_stats.py` ≥ 95 %, новый
`tests/test_research_chunking_stats.py` (проверь, нет ли такого файла), без сети и Telegram,
`src/` не менялся, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T09`

## T10 — feat: `llm_queue.rules.format_duration(seconds: float) -> str`
Чистая функция (в бота пока не подключается). Округляет до целых секунд: `0` → «0 с»,
`45.4` → «45 с», `60` → «1 мин», `125` → «2 мин 5 с», отрицательное → «0 с».
**Сделано, когда:** функция есть, тесты в `tests/test_llm_queue_rules.py`, `ruff` по изменённым
файлам не добавил замечаний, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T10`

## T11 — feat: `news_digest.rules.count_by_importance(items: list[dict]) -> dict[str, int]`
Чистая функция. Ключи — все коды из `IMPORTANCE_LEVELS` в их порядке (значение 0, если нет таких
новостей) и последним ключ `"unanalyzed"` для новостей без разбора (`analysis` пуст или `None`);
новость с неизвестной важностью считается неразобранной.
**Сделано, когда:** функция есть, тесты в `tests/test_news_digest_rules.py`, `ruff` не добавил
замечаний, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T11`

## T12 — feat: `dev_assistant.rules.format_user_ids(ids: Iterable[int]) -> str`
Чистая функция, обратная `parse_user_ids`: `[1, 2, 3]` → `"1,2,3"`, дубли схлопываются с
сохранением порядка, пусто → `""`. Должно выполняться
`parse_user_ids(format_user_ids(ids)) == tuple(dict.fromkeys(ids))`.
**Сделано, когда:** функция есть, тесты в `tests/test_dev_assistant_rules.py`, `ruff` не добавил
замечаний, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T12`

## T13 — bug/refactor: `every_text` принимает не-ASCII цифры
`price_watch/texts.py` разбирает период шаблоном `\d+`, а `parse.py` — только ASCII-цифрами:
`every_text("١٥m")` возвращает «раз в 15 минут» вместо значения «как есть».
**Сделано, когда:** `every_text("١٥m") == "١٥m"`, остальное поведение не изменилось, добавлен
тест, падающий на старом коде, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T13`

## T14 — refactor: общий доступ к важности новости в `news_digest/rules.py`
`item.get("analysis")` + `.get("importance")` повторяется в нескольких функциях. Вынеси приватный
помощник `_importance(item) -> str | None` и используй его везде, где повторяется. Поведение не
меняется.
**Сделано, когда:** выражение `analysis.get("importance")` встречается в `rules.py` ровно один раз,
`ruff check src/news_digest/rules.py` чисто, тесты не менялись, полный `pytest` зелёный.
**Проверка:** `tasks/check.sh T14`

## T15 — refactor: замечания `ruff` в трёх файлах
Устрани замечания `ruff` в `src/rag/cli.py`, `src/rag/extraction.py`, `src/providers/main_client.py`
без изменения поведения (порядок импортов, неиспользуемое и т. п.; `noqa` — только с причиной).
**Сделано, когда:** `ruff check` по этим файлам чисто, `compileall -q src` без ошибок, полный
`pytest` зелёный.
**Проверка:** `tasks/check.sh T15`

## T16 — docs: `src/price_watch/CLAUDE.md` расходится с кодом
Сверь описание планировщика с `src/price_watch/scheduler.py` и исправь только документацию.
**Сделано, когда:** значения в тексте совпадают с константами кода, изменён только этот файл.
**Проверка:** `tasks/check.sh T16`

## T17 — docs: `src/llm_queue/CLAUDE.md` расходится с кодом
Сверь описание статистики длительностей с `src/llm_queue/rules.py` и исправь только документацию.
**Сделано, когда:** значения в тексте совпадают с кодом, изменён только этот файл.
**Проверка:** `tasks/check.sh T17`

## T18 — docs: `src/dev_assistant/CLAUDE.md` расходится с кодом
Сверь раздел «Конфигурация» со значениями по умолчанию в `src/config.py` и `.env.example`;
исправь только документацию.
**Сделано, когда:** все перечисленные значения по умолчанию совпадают с `config.py`, изменён
только этот файл.
**Проверка:** `tasks/check.sh T18`
