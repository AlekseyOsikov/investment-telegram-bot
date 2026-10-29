# Watch parser LoRA: свободная просьба → JSON для `/watch`

**Краткий итог:**

| Версия | exact | schema_ok | error_acc |
|---|---:|---:|---:|
| База без промпта | 0.0% | 0.0% | 0.0% |
| База с few-shot | 46.7% | 78.3% | 6.7% |
| v1, 300 примеров | 71.7% | 90.0% | 26.7% |
| v2, 480 примеров, 50/50 | 83.3% | 95.0% | 46.7% |
| v2 + guard | 98.3% | 100.0% | 100.0% |
| v3, 606 примеров | 85.0% | 96.7% | 60.0% |
| **v3 + guard, dev-eval** | **100.0%** | **100.0%** | **100.0%** |
| **v3 + guard, holdout 30** | **96.7%** | **100.0%** | **100.0%** |

Главная честная цифра — **holdout 96.7% exact / 100% error_acc**: 30 запросов,
которых не было в train/valid/eval. Dev-eval 100% выше, потому что guard
донастраивался по ошибкам этого же набора.

V3 обучилась за 13 мин 49 с на T4. Guard — детерминированная страховка:
пустой ввод, пропущенный тикер/период, лишние периоды, неизвестный тикер,
дубли тикеров и согласованность `secids` с упомянутыми тикерами.

## 1. Задача

Пользователь пишет обычную русскую фразу, модель превращает её в строгий
JSON для команды `/watch`:

```text
«Следи за Сбером каждые 15 минут, сводку присылай раз в час.»
→ {"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}
```

Задача узкая и проверяемая: один формат ответа, 20 тикеров, конечный набор
кодов ошибок. Она полезна боту: сейчас `/watch` требует строгий синтаксис,
а пользователи пишут свободным текстом.

### Что считается верным ответом

Успех — объект ровно с полями `ok`, `secids`, `poll_interval`,
`report_interval`:

- `secids` — заглавные тикеры, без повторов, по алфавиту;
- периоды — целое число + `m|h|d`; минуты 1–1440, часы 1–24, дни 1–7;
- единицы не переводятся: 60 минут → `60m`, 90 минут → `90m`;
- повтор одного тикера — не ошибка, его нужно схлопнуть.

Ошибка — объект ровно с полями `ok`, `error`; коды: `empty`, `not_watch`,
`missing_tickers`, `missing_intervals`, `missing_poll_interval`,
`missing_report_interval`, `bad_interval`, `bad_ticker`, `ambiguous`.

Правила и валидатор — `scripts/spec.py`.

## 2. Данные

### v1 — 300 примеров

| Категория | Всего | Train | Valid |
|---|---:|---:|---:|
| simple | 90 | 81 | 9 |
| multi | 60 | 54 | 6 |
| colloquial | 60 | 54 | 6 |
| invalid | 60 | 54 | 6 |
| correction | 30 | 27 | 3 |
| **итого** | **300** | **270** | **30** |

Метки строит генератор по слотам, а не языковая модель. Формулировки —
10 формальных и 8 разговорных шаблонов, 20 тикеров с падежными формами,
разговорные периоды («полчаса», «четверть часа», «раз в сутки»).

### v2 — 480 примеров, баланс 50/50

Проблема v1: 216 успешных против 54 ошибочных. Модель выучила «всегда
возвращай успешный JSON». В v2:

- 240 успешных + 240 ошибочных;
- 180 негативных примеров построены из успешных: убрать тикер, убрать период,
  оставить только сводку, подменить тикер на неизвестный, поставить
  «30 секунд», добавить третий период;
- train/valid — 432/48.

### v3 — 606 примеров, точечно под оставшиеся ошибки

V2 всё ещё ломалась на четырёх классах, поэтому в v3 добавлено 126 примеров:

- 40 success-примеров с дублями тикеров (`SBER и SBER`, `Сбер и Сбербанк`) —
  чтобы повтор не считался `ambiguous`;
- 12 success-примеров с «дважды в сутки», «раз в полсуток»,
  «раз в четверть часа», «раз в полчаса»;
- 74 негативных: `bad_ticker` с «Ромашкой»/«TSLA»/«Апельсином» в разных
  падежах, `missing_*` с разными формулировками, `ambiguous` с двумя
  периодами опроса, `not_watch` с тикерами внутри («Сравни Сбер и ВТБ»).

Итог: 292 успешных + 314 ошибочных, train/valid — 544/62, eval не менялся.

### Чистка

- точных дублей внутри train/valid/eval — 0;
- почти-дублей внутри train (Jaccard 5-грамм ≥ 0.90) — 0;
- контаминации eval → train/valid: точных совпадений 0, пар с Jaccard
  8-грамм ≥ 0.85 — 0 (максимум 0.60);
- все 20 тикеров встречаются в train v3 18–27 раз;
- 285 успешных эталонов превращаются в команду, которую принимает настоящий
  `price_watch.parse.parse_watch_args` из проекта бота.

Проверки:

```bash
python scripts/check_dataset.py
python scripts/check_dataset.py \
  --train data/v3/train.jsonl --valid data/v3/valid.jsonl \
  --eval data/eval.jsonl --report data/v3/check_report.json
python scripts/check_bot_compat.py
```

## 3. Eval

60 запросов, написаны вручную и заморожены заранее:

| Категория | Кол-во |
|---|---:|
| simple | 10 |
| multi | 10 |
| colloquial | 15 |
| invalid | 10 |
| correction | 5 |
| role_order | 5 |
| out_of_scope | 5 |

Три базовых режима:

1. база без инструкций — только пользовательский текст;
2. база с проработанным промптом — схема, правила, 2 few-shot;
3. дообученная модель — тот же промпт.

Метрики: `exact`, `schema_ok`, `json_strict`, `ticker_f1`, `poll_acc`,
`report_acc`, `error_acc`; отдельно по категориям и кодам ошибок.

**Важно про порядок `secids`.** `exact` требует канонический порядок по
алфавиту. Поэтому запрос #14 («SBER, GAZP, MOEX») считается улучшением
формата, а не извлечения: у базы `["SBER","GAZP","MOEX"]`, у тюна
`["GAZP","MOEX","SBER"]`, при этом `ticker_f1` у обеих 1.0. В
`compare_console.py` и `compare_ollama.py` вердикты разделены на
«лучше по смыслу» и «лучше только формат».

## 4. База и v1

| Метрика | База без промпта | База с few-shot | v1 |
|---|---:|---:|---:|
| exact | 0.0% | 46.7% | **71.7%** |
| schema_ok | 0.0% | 78.3% | **90.0%** |
| json_strict | 0.0% | 100.0% | 100.0% |
| ticker_f1 | 0.0% | 98.1% | **98.9%** |
| poll_acc | 0.0% | 86.7% | **97.8%** |
| report_acc | 0.0% | 88.9% | **97.8%** |
| error_acc | 0.0% | 6.7% | **26.7%** |

V1 закрыла формат, сортировку, дедупликацию, обратный порядок ролей и
исправления. Но `invalid` exact остался 10%, `error_acc` — 26.7%: модель
почти всегда придумывала успешный JSON вместо кода ошибки.

## 5. V2: 50/50 и жёсткие негативы

### Параметры

| Параметр | Значение |
|---|---|
| Базовая модель | `Qwen/Qwen2.5-1.5B-Instruct` |
| Метод | LoRA, fp16 |
| r / alpha / dropout | 16 / 32 / 0.05 |
| lr / эпохи | 2e-4 / 2 |
| Батч | 2 × накопление 8 |
| Max length | 1024 |
| Train / valid | 432 / 48 |
| Железо | Tesla T4, 14.6 ГБ |
| Время | 11 мин 18 с (678 с) |
| Train loss | 0.256 → 0.016 |
| Valid loss | 0.0477 → 0.0181 |

### Результаты

| Метрика | База | v1 | v2 |
|---|---:|---:|---:|
| exact | 46.7% | 71.7% | **83.3%** |
| schema_ok | 78.3% | 90.0% | **95.0%** |
| ticker_f1 | 98.1% | 98.9% | 97.8% |
| poll_acc | 86.7% | 97.8% | 97.8% |
| report_acc | 88.9% | 97.8% | 95.6% |
| error_acc | 6.7% | 26.7% | **46.7%** |

По категориям (exact):

| Категория | База | v1 | v2 |
|---|---:|---:|---:|
| simple | 90% | 100% | 100% |
| multi | 40% | 70% | **100%** |
| colloquial | 67% | 80% | **93%** |
| invalid | 10% | 10% | **40%** |
| correction | 40% | 100% | 80% |
| role_order | 40% | 100% | 100% |
| out_of_scope | 0% | 60% | 60% |

### Что улучшилось

- `multi` 70% → 100%: сортировка тикеров и дедупликация.
- `colloquial` 80% → 93%: «четверть часа» больше не превращается в `1h`.
- `invalid` 10% → 40%: появились `missing_intervals`, `bad_interval`,
  один `ambiguous`.
- `error_acc` 26.7% → 46.7%.

### Что сломалось

- `correction` 100% → 80%: **#49** «Следи за SBER и SBER…» стало
  `ambiguous` вместо успеха с одним `SBER`.
- `report_acc` 97.8% → 95.6%, `ticker_f1` 98.9% → 97.8% — небольшой размен
  за рост распознавания ошибок.
- Остались: «дважды в сутки» → `2d` (#24), `bad_ticker` (#41, #42),
  `empty` (#56, #57), `missing_tickers` (#36), `missing_poll/report`
  (#38, #39), `ambiguous` (#44).

## 6. Guard: страховка на однозначные случаи

`scripts/guard.py` срабатывает только там, где правило однозначно:

- пустой ввод → `empty`;
- есть watch-слово, нет тикера и нет «за …» → `missing_tickers`;
- есть тикер, нет периода → `missing_intervals` (или `not_watch`, если
  в тексте явно «сравни/что выбрать/дивиденды»);
- есть тикер и ровно один период: названа сводка → `missing_poll_interval`,
  иначе → `missing_report_interval`;
- больше двух периодов (и это не исправление) → `ambiguous`;
- после «за/и/запятой» стоит незнакомое название → `bad_ticker`;
- в успешном ответе `secids` приводятся к тикерам, реально упомянутым
  в запросе (чинит и пропуск, и выдуманный тикер);
- `ambiguous` при дубле тикера и двух разбираемых периодах превращается
  в успех с одним тикером.

Во всех остальных случаях ответ модели не меняется.

### Результат v2 + guard

| Метрика | v2 | v2 + guard |
|---|---:|---:|
| exact | 83.3% | **98.3%** |
| schema_ok | 95.0% | **100.0%** |
| error_acc | 46.7% | **100.0%** |

Guard исправил 9 запросов, регрессий на валидных — 0. Остался только #24
(«дважды в сутки»).

## 7. V3: результаты

V3 (606 примеров, train/valid 544/62) добавила:

- success-примеры с дублями тикеров → #49;
- success-примеры с «дважды в сутки»/«раз в полсуток» → #24;
- `bad_ticker` с «Ромашкой», «TSLA», «Апельсином» в разных падежах → #41/#42;
- больше `missing_*`, `ambiguous`, `not_watch` с тикерами внутри.

Параметры: T4, 544/62, 2 эпохи, **13 мин 49 с (829 с)**, train loss
0.275 → 0.0145, valid loss 0.0414 → 0.0265.

### V3 на dev-eval (60 запросов)

| Метрика | v2 | v3 | v3 + guard |
|---|---:|---:|---:|
| exact | 83.3% | 85.0% | **100.0%** |
| schema_ok | 95.0% | 96.7% | **100.0%** |
| ticker_f1 | 97.8% | 95.9% | **100.0%** |
| poll_acc | 97.8% | 97.8% | **100.0%** |
| report_acc | 95.6% | 97.8% | **100.0%** |
| error_acc | 46.7% | 60.0% | **100.0%** |

V3 model-only выучила #24, #42, #44, но регрессировала на #17 (потеряла
`GAZP` в «SBER, SBER и GAZP») и #30 («Магнит» → `MTSS`). Guard с проверкой
согласованности тикеров исправил оба: на dev-eval v3 + guard закрашивает
все 60 запросов.

### Честный holdout: 30 новых запросов

Dev-eval 100% — оптимистичная цифра: guard донастраивался по его ошибкам.
Поэтому собран отдельный `data/holdout.jsonl` из 30 запросов, которых нет
в train/valid/eval. Проверка контаминации: 0 точных совпадений, пар с
Jaccard ≥ 0.85 — 0.

| Метрика | v3 | v3 + guard |
|---|---:|---:|
| exact | 76.7% | **96.7%** |
| schema_ok | 93.3% | **100.0%** |
| ticker_f1 | 100.0% | 100.0% |
| poll_acc | 94.4% | 94.4% |
| report_acc | 100.0% | 100.0% |
| error_acc | 58.3% | **100.0%** |

Единственная ошибка v3 + guard на holdout — **#24, коррекция периода**:

```text
«По GAZP: сводка раз в 3 часа, опрос каждые 30 минут.
 Хотя нет, сводку лучше раз в 6 часов.»
эталон: poll=30m, report=6h
ответ:  poll=3h,  report=6h
```

Модель берёт старую сводку `3h` как период опроса. Guard это не чинит,
потому что коррекция требует разбора ролей и старого/нового значения.
Это честно оставленное ограничение: 96.7% exact / 100% error_acc на
незнакомых запросах.

### Что это значит

- Модель-only v3: 85% exact, 60% error_acc — рост относительно v2, но
  всё ещё не production-ready на «плохих» запросах.
- v3 + guard: 100% на dev, 96.7% на holdout. Guard — детерминированный
  слой, его поведение проверяемо, но цифры на dev завышены его настройкой.
- Для продакшена разумно: модель + guard + repair-loop для коррекций
  и неизвестных тикеров. Один holdout — всё ещё мало для статистики;
  нужен третий независимый набор.

## 8. Локальная Ollama

Адаптер v3 слит с базой и загружен в Docker-Ollama:

- merged HF: генерируется в `../artifacts/watch_parser/merged_v3/` (5.8 ГБ, fp32 safetensors);
- GGUF F16: генерируется в `../artifacts/watch_parser/gguf/qwen25-watch-v3-f16.gguf` (3.09 ГБ);
- merged/GGUF в репозитории не хранятся — только команды воспроизведения;
- Modelfile: `Modelfile.watch-v3`;
- модель в Ollama: **`qwen25-watch-v3:latest`**, 3.1 ГБ;
- bind Docker: `/media/alexey/DATA/ollama/models` → `/root/.ollama`.

Команды:

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

# один вопрос вручную (system + user, при желании с guard)
python scripts/ollama_ask.py "Следи за Сбером каждые 15 минут, сводку раз в час."
python scripts/ollama_ask.py --guard "Следи за SBER и SBER каждые 15 минут, сводку раз в час."
```

### Проверка: HF vs Ollama

| Набор | Вариант | exact HF | exact Ollama | schema Ollama | error_acc Ollama |
|---|---|---:|---:|---:|---:|
| dev, 60 | v3 | 85.0% | 85.0% | 96.7% | 60.0% |
| dev, 60 | v3 + guard | 100.0% | 100.0% | 100.0% | 100.0% |
| holdout, 30 | v3 | 76.7% | 76.7% | 93.3% | 58.3% |
| holdout, 30 | v3 + guard | 96.7% | 96.7% | 100.0% | 100.0% |

Совпадение полное: слияние LoRA, конвертация в GGUF F16 и загрузка в Ollama
не изменили поведение. Единственная ошибка holdout+guard — та же коррекция
периода (#24).

Важно для воспроизведения: Ollama по умолчанию ставит `repeat_penalty=1.1`,
из-за чего few-shot ломается. В `ollama_eval.py` выставлены
`temperature=0`, `repeat_penalty=1.0`, `top_k=1`, `top_p=1.0`,
`num_ctx=4096`.

## 9. Ограничения

- Данные синтетические: формулировки собраны шаблонами, а не взяты из
  реальных логов `/watch`.
- 20 тикеров — маленький словарь; на реальном рынке тикеров тысячи.
- 60 dev-запросов и 30 holdout — наблюдения, а не статистически значимая
  оценка. Один прогон, жадная генерация.
- Guard настраивался по dev-eval, а потом по holdout (#30); поэтому 100% на
  dev и 96.7% на holdout — это оценки разработки, а не чистый тест.
  Для финальной цифры нужен третий независимый набор.
- Guard — не модель: он чинит только однозначные случаи. Коррекции
  («хотя нет, лучше…») и произвольные неизвестные тикеры всё ещё требуют
  repair-loop или отдельного разбора ролей.
- `bad_ticker` и `not_watch` — самые субъективные классы: неизвестное
  название легко принять за опечатку известного тикера.

## 10. Артефакты

- `data/v2/train.jsonl`, `data/v2/valid.jsonl`
- `data/v3/train.jsonl`, `data/v3/valid.jsonl`
- `scripts/spec.py`, `build_dataset.py`, `build_dataset_v2.py`,
  `build_dataset_v3.py`, `make_eval.py`, `check_dataset.py`,
  `check_bot_compat.py`, `eval_model.py`, `apply_guard_results.py`,
  `guard.py`, `compare_results.py`, `compare_console.py`,
  `compare_v2.py`, `compare_runs.py`,
  `merge_adapter.py`, `ollama_eval.py`, `ollama_ask.py`, `compare_ollama.py`
- `results/baseline/base_none.json|csv`, `base_prompt.json|csv`
- `results/v1/tuned_prompt.json|csv`, `compare.md|csv`, `run_params.json`, `loss.png`
- `results/v2/tuned_prompt.json|csv`, `tuned_prompt_guard.json|csv`,
  `compare.md`, `run_params.json`, `loss.png`
- `results/v3/tuned_prompt.json|csv`, `tuned_prompt_guard.json|csv`,
  `holdout.json|csv`, `holdout_guard.json|csv`,
  `ollama.json|csv`, `ollama_guard.json|csv`,
  `ollama_holdout.json|csv`, `ollama_holdout_guard.json|csv`,
  `compare_holdout.md`, `compare_ollama.md`, `run_params.json`, `loss.png`
- `results/compare/compare_all.md`
- `Modelfile.watch-v3`
- финальный адаптер: `../artifacts/watch_parser/adapter_v3/`
- merged_v3/ и GGUF генерируются командами из раздела 8, в репозитории не хранятся
- адаптеры v1/v2 удалены как промежуточные; их метрики и `run_params` сохранены
