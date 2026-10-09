# Tasks

## 1. Конфигурация

- [x] 1.1 В `src/config.py` добавить `ollama` в `_SUPPORTED_MAIN_CLIENTS`, перейти с тернарников на словари для `MAIN_CLIENT_LABEL` («Ollama») и `MAIN_API_KEY_ENV_VAR` (для `ollama` — `OLLAMA_BASE_URL`); дефолт `MAIN_MODEL` для `ollama` — пустая строка. Проверка: `python3 -c` с `MAIN_CLIENT=ollama MAIN_MODEL=x` показывает метку «Ollama»; `python3 -m compileall -q src` проходит
- [x] 1.2 В `_validate_config()` завершать запуск с понятной ошибкой, если `MAIN_CLIENT=ollama` и `MAIN_MODEL` пуст; сообщение о недопустимом `MAIN_CLIENT` перечисляет три значения. Проверка: запуск с пустой моделью завершается с кодом 1 и сообщением; с `MAIN_CLIENT=foo` — тоже

## 2. Клиенты

- [x] 2.1 В `src/providers/main_client.py` выбирать `ollama_client` при `MAIN_CLIENT == "ollama"` (словарь вместо тернарника), обновить докстринг. Проверка: при `MAIN_CLIENT=ollama` `main_client is ollama_client`, при остальных значениях поведение прежнее
- [x] 2.2 В `deepseek_client.py` и `kimi_client.py` сделать предупреждение об отсутствии ключа нейтральным (не утверждать «основной поток работает через Kimi/DeepSeek»), логику обязательности ключа не менять. Проверка: при `MAIN_CLIENT=ollama` без обоих ключей процесс запускается (доходит до проверки `TELEGRAM_BOT_TOKEN`), в журнале два предупреждения

## 3. Тексты ошибок и вызовы

- [x] 3.1 В `src/main.py` и `src/research/_shared.py` (`api_error_to_message`) для `ollama` заменить подсказку при `APIConnectionError` на «проверьте, что Ollama запущен и модель загружена», не упоминать API-ключ; категории исключений остаются раздельными. Проверка: unit-тест на `api_error_to_message(APIConnectionError(...), "Ollama", ...)` — в тексте нет «ключ», есть «Ollama»
- [x] 3.2 Grep по `src/` на параметры вызовов, специфичные для облака (`extra_body`, `thinking`, `reasoning_content`), в путях основного потока, `Agent`, `SmartAgent`, research-режимов; убедиться, что в Ollama они не отправляются (или обернуть условием по `MAIN_CLIENT`). Проверка: результат grep записан в PR/сообщение, при необходимости — правка и тест чистой функции
- [x] 3.3 Обновить строку лога запуска и приветствие в `main.py` (уже используют `MAIN_CLIENT_LABEL`/`MAIN_MODEL`); убедиться, что оговорка «не лицензированный финансовый советник» осталась. Проверка: чтением `/start`/`/help`-текстов

## 4. Тесты

- [x] 4.1 Добавить `tests/test_main_client_config.py`: чистая функция выбора метки/дефолтов/валидации (вынести из `config.py`, если иначе не тестируется без побочных эффектов) — случаи `deepseek`, `kimi`, `ollama` с моделью, `ollama` без модели, недопустимое значение. Проверка: `make test` зелёный

## 5. Документация и ручная проверка

- [x] 5.1 Обновить `.env.example` (`MAIN_CLIENT=ollama`, `MAIN_MODEL` обязателен, общий `OLLAMA_BASE_URL`, `num_ctx`/`OLLAMA_KEEP_ALIVE`, предупреждение про tool calling), `README.md` (заголовок, раздел выбора провайдера, пример `.env`, лог запуска) и `CLAUDE.md` (карта `providers/`, раздел «Конфигурация», обязательность ключа, канал данных при удалённом `OLLAMA_BASE_URL`). Проверка: grep по «deepseek|kimi» в этих файлах — нет утверждений, что провайдеров ровно два
- [x] 5.2 Ручная проверка с локальным Ollama: `MAIN_CLIENT=ollama MAIN_MODEL=<модель>` — вопрос в личном чате получает ответ; остановка Ollama даёт сообщение про подключение к Ollama; `/agent` и `/smart_agent` отвечают; со слоем `tools` на модели без поддержки инструментов бот не падает. Проверка: наблюдаемое поведение зафиксировано в сообщении по итогам
