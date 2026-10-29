# Watch parser LoRA: свободная просьба → JSON для `/watch`

Узкая задача для дообучения Qwen2.5-1.5B-Instruct: превратить обычную русскую
просьбу в строгий JSON, который бот сможет выполнить как команду `/watch`.

```text
«Следи за Сбером каждые 15 минут, сводку присылай раз в час.»
→ {"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}

«Следи за SBER, сводку присылай раз в час.»
→ {"ok":false,"error":"missing_poll_interval"}
```

## Почему задача подходит

- **Узкая и повторяющаяся.** Один тип запроса, один формат ответа, 20 тикеров.
- **Проверяемый результат.** Эталон — JSON; сравнение точное, без «на глаз».
- **Есть реальная польза.** Сейчас `/watch` требует строгий синтаксис
  (`/watch SBER GAZP 15m 1h`), а пользователи пишут свободным текстом.
- **Есть где ошибиться.** Разговорные периоды, падежи, несколько тикеров,
  обратный порядок ролей, исправления, мусорные запросы.

## Что считается верным ответом

Успех:

```json
{"ok":true,"secids":["SBER","GAZP"],"poll_interval":"15m","report_interval":"1h"}
```

- `secids` — заглавные тикеры, без повторов, по алфавиту;
- `poll_interval` / `report_interval` — целое число + `m|h|d`;
- минуты 1–1440, часы 1–24, дни 1–7;
- единицы не переводятся: 60 минут → `60m`, 90 минут → `90m`.

Ошибка:

```json
{"ok":false,"error":"missing_report_interval"}
```

Коды: `empty`, `not_watch`, `missing_tickers`, `missing_intervals`,
`missing_poll_interval`, `missing_report_interval`, `bad_interval`,
`bad_ticker`, `ambiguous`.

Полные правила — в `scripts/spec.py` (системный промпт и валидатор).

## Структура

```text
data/
  v1/train.jsonl, valid.jsonl   # 270 / 30
  v2/, v3/                      # train/valid следующих версий
  eval.jsonl                    # dev-eval, 60 запросов
  holdout.jsonl                 # holdout, 30 запросов
scripts/
  spec.py                    # правила, тикеры, системный промпт, валидатор
  build_dataset*.py          # генерация train/valid для v1/v2/v3
  make_eval.py, make_holdout.py
  check_dataset.py           # схема, дубли, контаминация, баланс
  check_bot_compat.py        # совместимость с parse_watch_args бота
  eval_model.py              # прогон eval через HF-модель
  guard.py                   # детерминированная страховка
  apply_guard_results.py     # guard поверх сохранённых ответов
  compare_*.py               # сводные таблицы
  merge_adapter.py           # adapter + base → merged HF
  ollama_eval.py, ollama_ask.py
training/
  train_lora.py              # LoRA-дообучение под Colab T4
  colab_v1.ipynb, colab_v2.ipynb, colab_v3.ipynb
results/
  baseline/                  # base_none, base_prompt
  v1/, v2/, v3/              # метрики, run_params, loss по версиям
  compare/                   # сводные сравнения
report.md                    # полный отчёт
Modelfile.watch-v3
Makefile
../artifacts/watch_parser/   # adapter_v3; merged/GGUF генерируются
```

## Локально: собрать и проверить данные

Нужен только Python 3.10+ без GPU и без torch:

```bash
cd finetuning/watch_parser
python scripts/build_dataset.py
python scripts/make_eval.py
python scripts/check_dataset.py
python scripts/check_bot_compat.py
```

Проверка должна завершиться кодом 0. Она печатает баланс, топ почти-дублей
и контаминацию eval → train. Текущий результат: точных совпадений eval с
train/valid — 0, пар с Jaccard ≥ 0.85 — 0. Отдельно проверяется, что все
285 успешных эталонов превращаются в команду, которую принимает настоящий
`price_watch.parse.parse_watch_args` из проекта бота.

## Colab T4: дообучение и сравнение

Готовые ноутбуки: `training/colab_v1.ipynb`, `training/colab_v2.ipynb`,
`training/colab_v3.ipynb`. В Colab загрузите папку проекта (или склонируйте
репозиторий) и выполните:

```python
!pip -q install -U "transformers>=4.56" "peft>=0.13" accelerate matplotlib
!pip -q uninstall -y torchao
# Runtime -> Restart session
```

Затем:

```bash
python training/train_lora.py \
  --train data/v1/train.jsonl \
  --valid data/v1/valid.jsonl \
  --out /content/adapter_watch_json \
  --run-dir /content/run_watch_json
```

Дообучение на 270 примерах, 2 эпохи, T4 — ориентировочно 8–15 минут.

Три прогона eval:

```bash
# 1. База без инструкций
python scripts/eval_model.py --prompt-mode none \
  --out results/baseline/base_none.json

# 2. База с проработанным промптом (system/fewshot)
python scripts/eval_model.py --prompt-mode fewshot \
  --out results/baseline/base_prompt.json

# 3. Дообученная модель с тем же промптом
python scripts/eval_model.py \
  --adapter /content/adapter_watch_json \
  --prompt-mode fewshot \
  --out results/v1/tuned_prompt.json
```

Сводка:

```bash
python scripts/compare_results.py \
  --base-none results/baseline/base_none.json \
  --base-prompt results/baseline/base_prompt.json \
  --tuned results/v1/tuned_prompt.json \
  --out results/v1/compare.md
```

Для быстрого пилота добавьте `--limit 30` к первым двум прогонам. Если база
с промптом уже даёт ≥ 90% exact match, задача слишком лёгкая — это тоже
результат, и стоит перейти к более сложной задаче (новости → JSON или RAG).

### Результаты на dev-eval (60 запросов, Colab T4)

| Метрика | База без промпта | База с few-shot | v1 | v2 | v2 + guard | v3 | v3 + guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | 0.0% | 46.7% | 71.7% | 83.3% | 98.3% | 85.0% | **100.0%** |
| schema_ok | 0.0% | 78.3% | 90.0% | 95.0% | 100.0% | 96.7% | **100.0%** |
| ticker_f1 | 0.0% | 98.1% | 98.9% | 97.8% | 100.0% | 95.9% | **100.0%** |
| poll_acc | 0.0% | 86.7% | 97.8% | 97.8% | 100.0% | 97.8% | **100.0%** |
| report_acc | 0.0% | 88.9% | 97.8% | 95.6% | 97.8% | 97.8% | **100.0%** |
| error_acc | 0.0% | 6.7% | 26.7% | 46.7% | 100.0% | 60.0% | **100.0%** |

По категориям (exact):

| Категория | v2 | v2 + guard | v3 | v3 + guard |
|---|---:|---:|---:|---:|
| simple | 100% | 100% | 100% | 100% |
| multi | 100% | 100% | 90% | 100% |
| colloquial | 93% | 93% | 93% | 100% |
| invalid | 40% | 100% | 60% | 100% |
| correction | 80% | 100% | 80% | 100% |
| role_order | 100% | 100% | 100% | 100% |
| out_of_scope | 60% | 100% | 60% | 100% |

### Честный holdout (30 новых запросов)

| Метрика | v3 | v3 + guard |
|---|---:|---:|
| exact | 76.7% | **96.7%** |
| schema_ok | 93.3% | **100.0%** |
| ticker_f1 | 100.0% | 100.0% |
| poll_acc | 94.4% | 94.4% |
| report_acc | 100.0% | 100.0% |
| error_acc | 58.3% | **100.0%** |

Единственная ошибка v3+guard — коррекция периода: «сводка раз в 3 часа,
опрос каждые 30 минут. Хотя нет, сводку лучше раз в 6 часов» — модель берёт
старую сводку `3h` как poll. Это известное ограничение, не подогнанное.

Файлы сгруппированы в `results/baseline/`, `results/v1/`, `results/v2/`,
`results/v3/`, `results/compare/`; описание — в `results/README.md`.

### V2: что изменилось

- `multi` 70% → 100%: сортировка и дедупликация тикеров.
- `colloquial` 80% → 93%: «четверть часа» больше не `1h`.
- `invalid` 10% → 40%: `missing_intervals`, `bad_interval`, часть `ambiguous`.
- `error_acc` 26.7% → 46.7%.
- Регрессия: `correction` 100% → 80% на #49 «SBER и SBER» — v2 приняла дубль
  за `ambiguous`.

### Guard: страховка на однозначные случаи

`scripts/guard.py` исправляет пустой ввод, пропущенный тикер/период, лишние
периоды, неизвестный тикер, дубли тикеров и согласованность `secids`.
На v2: **exact 83.3% → 98.3%**, `error_acc` 46.7% → 100%, регрессий 0.
Остался только #24 («дважды в сутки»).

### V3: результаты

606 примеров, train/valid — 544/62, дообучение 13 мин 49 с. Model-only:
**85.0% exact, 60.0% error_acc**. V3+guard: **100% на dev-eval**, но это
оптимистично — guard настраивался по dev. На честном holdout из 30 новых
запросов: **96.7% exact, 100% error_acc**, одна ошибка — коррекция периода.

Команды воспроизведения (результаты уже лежат в `results/`):

```bash
python training/train_lora.py \
  --train data/v3/train.jsonl \
  --valid data/v3/valid.jsonl \
  --out /content/adapter_watch_json_v3 \
  --run-dir /content/run_watch_json_v3

python scripts/eval_model.py \
  --adapter /content/adapter_watch_json_v3 \
  --prompt-mode fewshot \
  --out results/v3/tuned_prompt.json
python scripts/eval_model.py \
  --adapter /content/adapter_watch_json_v3 \
  --prompt-mode fewshot --guard \
  --out results/v3/tuned_prompt_guard.json

python scripts/compare_runs.py \
  --base results/baseline/base_prompt.json \
  --run v1=results/v1/tuned_prompt.json \
  --run v2=results/v2/tuned_prompt.json \
  --run "v2+guard=results/v2/tuned_prompt_guard.json" \
  --run v3=results/v3/tuned_prompt.json \
  --run "v3+guard=results/v3/tuned_prompt_guard.json" \
  --out results/compare/compare_all.md
```

## Локальная Ollama

Адаптер v3 слит с базой и загружен в Docker-Ollama как **`qwen25-watch-v3:latest`**
(GGUF F16, 3.1 ГБ). `merged_v3/` и GGUF не хранятся в репозитории — они
воспроизводятся командами ниже. В `../artifacts/watch_parser/` остаётся
только `adapter_v3/`; шаблон — `Modelfile.watch-v3`.

```bash
python scripts/merge_adapter.py \
  --adapter ../artifacts/watch_parser/adapter_v3 --out ../artifacts/watch_parser/merged_v3

python llama.cpp/convert_hf_to_gguf.py ../artifacts/watch_parser/merged_v3 \
  --outfile ../artifacts/watch_parser/gguf/qwen25-watch-v3-f16.gguf --outtype f16

cp ../artifacts/watch_parser/gguf/qwen25-watch-v3-f16.gguf /media/alexey/DATA/ollama/models/
cp Modelfile.watch-v3 /media/alexey/DATA/ollama/models/Modelfile.watch-v3
docker exec ollama ollama create qwen25-watch-v3 \
  -f /root/.ollama/Modelfile.watch-v3

python scripts/ollama_eval.py --eval data/eval.jsonl \
  --model qwen25-watch-v3 --out results/v3/ollama.json

# один вопрос вручную
python scripts/ollama_ask.py "Следи за Сбером каждые 15 минут, сводку раз в час."
python scripts/ollama_ask.py --guard "Следи за SBER и SBER каждые 15 минут, сводку раз в час."
```

| Набор | Вариант | exact HF | exact Ollama | error_acc Ollama |
|---|---|---:|---:|---:|
| dev, 60 | v3 | 85.0% | 85.0% | 60.0% |
| dev, 60 | v3 + guard | 100.0% | 100.0% | 100.0% |
| holdout, 30 | v3 | 76.7% | 76.7% | 58.3% |
| holdout, 30 | v3 + guard | 96.7% | 96.7% | 100.0% |

Слияние и конвертация поведение не изменили. Важно: Ollama по умолчанию
ставит `repeat_penalty=1.1`, из-за чего few-shot ломается; в `ollama_eval.py`
выставлены `repeat_penalty=1.0`, `top_k=1`, `top_p=1.0`, `num_ctx=4096`.

## Дизайн эксперимента

| Прогон | Системный промпт | Что показывает |
|---|---|---|
| база без промпта | нет | что модель делает «из коробки» |
| база с промптом | `SYSTEM_PROMPT` + 2 few-shot | сильный baseline, который нужно превзойти |
| дообученная | тот же промпт | вклад датасета v1/v2/v3 |

Метрики:

- `exact` — объект совпал с эталоном;
- `schema_ok` — ответ прошёл `validate_output`;
- `json_strict` — весь ответ является JSON без вырезания;
- `json_extractable` — JSON удалось вытащить из текста;
- `ticker_f1` — точность набора тикеров;
- `poll_acc`, `report_acc` — точность периодов;
- `error_acc` — точность кода ошибки на «плохих» запросах.

Отдельно смотрите `by_category`: simple / multi / colloquial / invalid /
correction / role_order / out_of_scope. Это показывает, где именно тюнинг
помог, а где сломал поведение.

## Что уже сделано с данными

- v1: 300 примеров; v2: 480 (50/50 success/invalid); v3: 606.
- v2 добавила 180 жёстких негативов из успешных запросов; v3 — 126 точечных
  примеров под оставшиеся ошибки.
- Train/valid — стратифицированно 90/10, dev-eval — отдельные 60,
  holdout — ещё 30 запросов.
- Метки построены генератором по слотам, а не языковой моделью.
- Проверены схема, точные дубли, почти-дубли (Jaccard 5-грамм), контаминация
  eval → train/valid (Jaccard 8-грамм), баланс категорий, кодов ошибок,
  тикеров и единиц периода.
- Ручной просмотр: см. `scripts/check_dataset.py`, раздел «5 случайных train/eval»;
  при ревью просмотрите 30 случайных примеров и отметьте ошибки в `report.md`.
- Для holdout проверено: 0 точных совпадений с train/valid/eval, пар с
  Jaccard ≥ 0.85 — 0.

## Ограничения

- Данные синтетические: формулировки собраны шаблонами и парафразами, а не
  взяты из реальных логов `/watch` (в проекте бота таких логов пока нет).
- 20 тикеров — маленький словарь; на реальном рынке тикеров тысячи.
- Dev-eval 60 и holdout 30 — наблюдения, а не статистически значимая оценка.
- Guard настраивался по dev-eval, а затем по holdout; 100% на dev и 96.7%
  на holdout — оценки разработки, а не чистый тест. Для финальной цифры
  нужен третий независимый набор.
- Один прогон, жадная генерация; разброс между запусками не измерялся.
- Коррекции периода («хотя нет, лучше…») guard не чинит — это известное
  ограничение, видно на holdout #24.

## Что проверено

- [x] узкая задача с проверяемым результатом;
- [x] сформулировано, что считается верным ответом;
- [x] eval-набор с эталонными ответами;
- [x] baseline без инструкций и с проработанным промптом;
- [x] 300 / 480 / 606 примеров в формате `messages`;
- [x] train/valid, eval не попал в train;
- [x] чистка: дубли, почти-дубли, ручной просмотр, баланс, единый формат;
- [x] дообучение и сравнение v1/v2/v3 + holdout;
- [x] выводы «лучше / прежним / хуже» — в `report.md` и `results/compare/compare_all.md`.
