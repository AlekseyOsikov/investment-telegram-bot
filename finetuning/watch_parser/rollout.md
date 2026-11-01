# Выкат модели-парсера /watch в прод

Как довести дообученную модель `qwen25-watch-v3` (Qwen2.5-1.5B-Instruct + LoRA) до работы в
боте, замерить её стоимость, переключать между тюненой и базовой без переката сервиса и
откатиться. Код бота — `src/price_watch/nl_parse.py` (перенос промпта и guard из `scripts/`).

## 1. Где что лежит

| Что | Путь | В git |
|---|---|---|
| LoRA-адаптер | `finetuning/artifacts/watch_parser/adapter_v3/` | нет |
| Слитая HF-модель | `finetuning/artifacts/watch_parser/merged_v3/` (генерируется) | нет |
| GGUF (F16/Q8/Q4) | `finetuning/artifacts/watch_parser/gguf/` (генерируется) | нет |
| llama.cpp | `/media/alexey/DATA/llama.cpp/` | нет |
| Volume моделей Ollama | `/media/alexey/DATA/ollama/models/` | нет |
| Modelfile | `finetuning/watch_parser/Modelfile.watch-v3` | да |

База: `Qwen/Qwen2.5-1.5B-Instruct` (HF-кэш уже есть на машине), в Ollama — `qwen2.5:1.5b-instruct-fp16`.
Ollama в Docker, контейнер `ollama`, порт 11434.

## 2. Сборка: слияние + GGUF + квантование

```bash
cd finetuning/watch_parser

# 1. слить адаптер с базой
python scripts/merge_adapter.py \
  --adapter ../artifacts/watch_parser/adapter_v3 --out ../artifacts/watch_parser/merged_v3

# 2. F16 GGUF
python /media/alexey/DATA/llama.cpp/convert_hf_to_gguf.py \
  ../artifacts/watch_parser/merged_v3 \
  --outfile ../artifacts/watch_parser/gguf/qwen25-watch-v3-f16.gguf --outtype f16

# 3. квантование (нужен собранный llama-quantize)
python /media/alexey/DATA/llama.cpp/llama-quantize \
  ../artifacts/watch_parser/gguf/qwen25-watch-v3-f16.gguf \
  ../artifacts/watch_parser/gguf/qwen25-watch-v3-q8_0.gguf q8_0
python /media/alexey/DATA/llama.cpp/llama-quantize \
  ../artifacts/watch_parser/gguf/qwen25-watch-v3-f16.gguf \
  ../artifacts/watch_parser/gguf/qwen25-watch-v3-q4_k_m.gguf q4_k_m
```

## 3. Загрузка в Ollama

Modelfile для каждого варианта — копия `Modelfile.watch-v3` с заменой `FROM` на свой GGUF.
Загружаем под ВЕРСИОНИРОВАННЫМИ именами и заводим СТАБИЛЬНЫЙ алиас, на который смотрит бот:

```bash
cp ../artifacts/watch_parser/gguf/qwen25-watch-v3-*.gguf /media/alexey/DATA/ollama/models/
# по одному Modelfile на вариант (FROM /root/.ollama/<файл>.gguf), затем:
docker exec ollama ollama create qwen25-watch-v3:q4_k_m -f /root/.ollama/Modelfile.watch-v3-q4
docker exec ollama ollama create qwen25-watch-v3:q8_0   -f /root/.ollama/Modelfile.watch-v3-q8

# стабильный алиас (бот зовёт именно его), по умолчанию — тюненая:
docker exec ollama ollama cp qwen25-watch-v3:q8_0 qwen25-watch-prod
```

## 4. Замеры

Размер — `ls -l` GGUF. Память — `docker exec ollama ollama ps` после загрузки. Скорость и время
загрузки — `scripts/ollama_eval.py` (латентность) и холодный старт (первый запрос после
`ollama stop`). Замерено на этой машине (Ollama 0.21 в Docker, AMD iGPU Radeon 780M, `num_ctx 4096`):

| Вариант | Размер GGUF | Память (`ollama ps`) | Скорость (ток/с) | Тёплая латентность | Холодный старт |
|---|---:|---:|---:|---:|---:|
| F16 | 2.9 ГБ | 3.5 ГБ | 23 | ~1.2 с | 4.3 с |
| Q8_0 | 1.6 ГБ | 2.1 ГБ | 44 | ~0.70 с | 3.0 с |
| Q4_K_M | 0.94 ГБ | 1.4 ГБ | 64 | ~0.51 с | 2.3 с |

Качество с guard (как в проде, `parse_free_text` в `src/price_watch/nl_parse.py`):

| Вариант | eval (60) exact | holdout (30) exact | error_acc |
|---|---:|---:|---:|
| F16 | 100.0% | 96.7% | 100% |
| Q8_0 | 100.0% | 96.7% | 100% |
| Q4_K_M | 95.0% | 96.7% | 100% |

**Вывод:** Q8_0 — без потерь (метрики совпадают с F16) при −40% размера и ~1.8× скорости. Q4_K_M
экономит ещё ~40%, но на eval теряет 3 примера (перепутанные периоды/дубль тикера, которые guard
не чинит) — берём Q4_K_M только ради максимальной экономии памяти. **Рекомендация для прода —
Q8_0.**

## 5. Проверка перед выкатом

1. Собрать и проверить данные: `make data check` (схема, дубли, контаминация, совместимость с
   `parse_watch_args` бота — `scripts/check_bot_compat.py`).
2. Прогон на holdout с guard: `python scripts/ollama_eval.py --eval data/holdout.jsonl --model <имя>`.
   Порог выката — exact ≥ 90% с guard (текущая F16: 96.7%).
3. Контрольные фразы вручную: `python scripts/ollama_ask.py --guard "Следи за Сбером каждые 15 минут, сводку раз в час."`.
4. Детерминированность: один и тот же ввод трижды → одинаковый вывод (в `nl_parse.py` задан `seed=42`).

## 6. Подключение к боту

В `.env`: `WATCH_PARSER_MODEL=qwen25-watch-prod` и перезапуск бота. Строгий синтаксис `/watch`
продолжает работать; свободный текст разбирает модель, поверх — guard. `WATCH_PARSER_MODEL=` —
функция выключена, поведение как раньше.

## 7. Переключение тюненая/база без переката сервиса

Бот всегда зовёт стабильный алиас `qwen25-watch-prod`, поэтому переключение — это смена того, на
что алиас указывает, БЕЗ рестарта бота (только выгрузка загруженной модели):

```bash
# на базу (qwen2.5:1.5b-instruct-fp16 + few-shot)
docker exec ollama ollama cp qwen2.5:1.5b-instruct-fp16 qwen25-watch-prod
docker exec ollama ollama stop qwen25-watch-prod     # чтобы следующий запрос подхватил новый тег

# обратно на тюненую
docker exec ollama ollama cp qwen25-watch-v3:q8_0 qwen25-watch-prod
docker exec ollama ollama stop qwen25-watch-prod
```

Это переключение ОПЕРАТОРА (через Ollama), а не команда пользователя — соответствует запрету на
рантайм-смену модели в `CLAUDE.md`. Сравнение на живых фразах — прогнать `ollama_ask.py`/бот на
одном наборе фраз до и после флипа.

## 8. Откат

- **Откат модели**: `docker exec ollama ollama cp <предыдущий-вариант> qwen25-watch-prod` + `stop`.
- **Откат функции**: очистить `WATCH_PARSER_MODEL` в `.env` и перезапустить бот — `/watch`
  возвращается к строгому синтаксису и подсказке.

> Правки промпта/guard в `scripts/spec.py`/`guard.py` переносить в `src/price_watch/nl_parse.py`
> вручную (граница `src/` ↔ `finetuning/`): после переноса прогнать
> `tests/test_price_watch_nl_parse.py`.
