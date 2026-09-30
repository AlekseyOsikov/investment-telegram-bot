# research/ — технические режимы исследования API

Корневые правила (домен, безопасность, конфигурация) — в `/CLAUDE.md`. Эти режимы не
участвуют в обычном потоке сообщений и включаются флагом `RESEARCH` (см. корневой файл).

## Модули

- `_shared.py` — общий каркас: `run_scenario`/`api_error_to_message` (перехват ошибок OpenAI SDK
  → русское сообщение), `extract_usage`/`sum_usage`/`format_scenario_stats`,
  `content_or_reasoning_fallback`, `build_cancel_handler`. Изменения обработки ошибок API или
  статистики вноси сюда, а не в три режима по отдельности; специфику режима (сценарии,
  форматирование, кнопки) оставляй в его модуле.
- `constraints.py` (`/research_constraints`), `reasoning.py` (`/research_reasoning`),
  `temperature.py` (`/research_temperature`) — идут через `main_client`/`MAIN_MODEL`. Тексты и
  ошибки параметризованы `MAIN_CLIENT_LABEL`/`MAIN_API_KEY_ENV_VAR` — не возвращай
  захардкоженное «DeepSeek».
- `models.py` (`/research_models`) — сравнение 4 моделей DeepSeek/Kimi; от `MAIN_CLIENT` не
  зависит, не использует `run_scenario`/`api_error_to_message` (сообщения провайдер-нейтральны,
  статистика шире), но использует остальное из `_shared.py`.
- `chunking_stats.py` (`/research_chunking_stats`, обычный `CommandHandler`) и
  `chunking_compare.py` (`/research_chunking_compare`, `ConversationHandler` с одним циклическим
  состоянием до `/cancel`) — только ЧИТАЮТ индекс из `rag/`; без индекса отвечают понятным
  сообщением, а не падают. Compare показывает топ-`RAG_COMPARE_TOP_K` чанков каждой стратегии
  ОТДЕЛЬНЫМ сообщением на стратегию: оценка, заголовок, метаданные, текст (отделён от метаданных
  строкой `--------`). См. `src/rag/CLAUDE.md`.

## `/research_constraints`

`ConversationHandler` на 4 состояния (вопрос → один из 4 сценариев кнопками → для 3 и 4 ещё
параметр текстом). Найдено эмпирически на DeepSeek (на Kimi не проверялось — не выдавай за
универсальное свойство API):
- Сценарий 2 (формат ответа): `json_schema` не поддерживается (400) — используй
  `{"type": "json_object"}`, поля описывай текстом; не возвращай `json_schema`.
- Сценарий 3 (`max_tokens`): значение от пользователя уходит в API как есть (ограничение на
  стороне API, не постобработкой). На reasoning-моделях лимит может целиком уйти на
  размышления, `content` пуст при `finish_reason == "length"` — `content_or_reasoning_fallback`
  подставляет обрезанный `reasoning_content`; сохраняй.
- Сценарий 4 (`stop`): до `CONSTRAINTS_MAX_STOP_WORDS` (4) слов через запятую.

Обозначения сообщений (сохраняй): 👉 — ждёт выбора/ввода; 📊 — результат сценария; 📈 —
статистика (`finish_reason`, токены) отдельным сообщением после результата (кроме ошибки API).

## `/research_temperature`, `/research_models`

- `temperature`: 2 состояния, нейтральный промпт, `thinking` отключён
  (`extra_body={"thinking": {"type": "disabled"}}`). Сценарии 1–4 — фиксированные 0 / 0.7 / 1.2 / 2;
  сценарий 5 — параллельно 1–4 (`ThreadPoolExecutor`) + сравнение через `main_client` по
  точности, креативности, разнообразию.
- `models`: `MODEL_CATALOG` — список `{"id", "label", "client", ...цены}`, у каждой модели свой
  клиент; `call_model()` принимает `client` и `model_id` явно. Идентификаторы моделей приходят из
  `providers/*_client.py` (из `.env`); при замене меняй элемент каталога, а не `MAIN_MODEL`.
  `thinking` НЕ отключается. Статистика включает время и стоимость (цены в каталоге
  иллюстративные, не официальные). Сценарий 5: параллельно 1–4, затем сравнение ОДНИМ запросом
  через `main_client`/`MAIN_MODEL` (не через модель каталога) по качеству, скорости,
  ресурсоёмкости; статистика передаётся модели в самом запросе.
