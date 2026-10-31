# Дистилляция: маленькая модель учится на ответах большой

Эксперимент по заданию «Дистилляция»: берём задачу, где маленькая модель
(`qwen2.5:1.5b`) уступает большой (`qwen2.5:7b`), собираем ответы большой модели
на НЕРАЗМЕЧЕННЫХ входах, чистим их и дообучаем маленькую. Затем сравниваем три
варианта — база / дистиллированная / учитель — по качеству, скорости и стоимости
и показываем, где ученик уже заменяет учителя, а где ещё ошибается.

Полный отчёт и выводы — [`report.md`](report.md).

## Задача

Та же, что в `watch_parser`: свободная русская просьба → строгий JSON для
`/watch`. Разница принципиальная: в `watch_parser` метки строил слот-генератор,
а здесь метки ставит **учитель** (большая модель). Тестовый набор переиспользуется
из `watch_parser` (`eval.jsonl` 60 + `holdout.jsonl` 30) и в обучение не входит.

## Три плеча (один тестовый набор, одинаковый промпт)

| Вариант | Модель | Роль |
|---|---|---|
| `base` | `qwen2.5:1.5b-instruct-fp16` + few-shot | ученик до обучения |
| `distilled` | `qwen25-distill-v1` (LoRA на метках учителя) | ученик после |
| `golden` | `qwen25-golden-v1` (LoRA на строго отфильтрованных метках) | верхняя граница чистки |
| `teacher` | `qwen2.5:7b` + few-shot | учитель |

## Структура

```text
distillation/
  README.md, report.md
  Makefile, Modelfile.distill, Modelfile.golden
  data/
    make_inputs.py        # -> corpus.jsonl (неразмеченные входы + золотой оракул для QA)
    corpus.jsonl          # 511 свежих входов, не пересекаются с eval/holdout/v1..v3
    teacher_raw.jsonl     # ответы учителя + тайминги (генерируется)
    train_distill.jsonl / valid_distill.jsonl   # метки учителя, фильтр по форме
    train_golden.jsonl / valid_golden.jsonl     # метки, совпавшие со смыслом
    filter_report.json    # сколько ответов учителя отброшено
  scripts/
    ollama_client.py      # вызов Ollama + токены/тайминги
    pricing.py            # токены -> $ (справочные облачные тарифы)
    teacher_label.py      # учитель размечает corpus
    filter_labels.py      # проверка ответов учителя и сборка train/valid
    eval_three.py         # прогон базы/учителя/дистиллированной на тесте
    compare.py            # сводная таблица + анализ «зон замены»
    compare_ollama.py     # живое сравнение ответов в консоли (качество/скорость/стоимость)
  training/
    train_lora.py         # LoRA под Colab T4 (--variant distill|golden)
    colab_distill.ipynb   # ноутбук Colab: оба обучения + упаковка
  results/                # замеры eval/holdout + compare.md
  ../artifacts/distillation/   # adapter_*, merged_*, gguf/ (вне git)
```

## Переиспользование `watch_parser`

`spec.py` (правила задачи, `SYSTEM_PROMPT`, `validate_output`, `FEWSHOT`),
`metrics.py` и генераторы `build_dataset*.py` остаются единым источником правды в
`watch_parser/` — скрипты подключают их через `sys.path` (см. `make_inputs.py`),
а не копируют. Слияние адаптера — `watch_parser/scripts/merge_adapter.py`
(запускается локально с явными путями).

## Локально: собрать корпус и разметить учителем

Нужен только Python 3.10+ (без GPU/torch) и запущенный Docker-Ollama:

```bash
cd finetuning/distillation
make corpus      # data/make_inputs.py -> data/corpus.jsonl
make label       # teacher_label.py -> data/teacher_raw.jsonl (~20 мин на CPU)
make filter      # filter_labels.py -> train/valid_distill|golden
```

## Colab T4: дообучение (два адаптера)

Проще всего — ноутбук [`training/colab_distill.ipynb`](training/colab_distill.ipynb):
он сам загрузит `distillation.zip`, обучит оба варианта и упакует адаптеры.
Вручную — загрузите папку `distillation/` в Colab и выполните:

```python
!pip -q install -U "transformers>=4.56" "peft>=0.13" accelerate matplotlib
!pip -q uninstall -y torchao
# Runtime -> Restart session
```

```bash
python training/train_lora.py --variant distill --out /content/adapter_distill \
  --run-dir /content/run_distill
python training/train_lora.py --variant golden --out /content/adapter_golden \
  --run-dir /content/run_golden
```

Скачайте `adapter_distill/` и `adapter_golden/` (файлы адаптера + `run_params.json`).

## Локально: слить, сконвертировать, загрузить и сравнить

```bash
# слить адаптеры
python watch_parser/scripts/merge_adapter.py --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/distillation/adapter_distill --out artifacts/distillation/merged_distill
python watch_parser/scripts/merge_adapter.py --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/distillation/adapter_golden --out artifacts/distillation/merged_golden

# GGUF (путь к llama.cpp — как в watch_parser README)
python llama.cpp/convert_hf_to_gguf.py artifacts/distillation/merged_distill \
  --outfile artifacts/distillation/gguf/qwen25-distill-v1-f16.gguf --outtype f16
python llama.cpp/convert_hf_to_gguf.py artifacts/distillation/merged_golden \
  --outfile artifacts/distillation/gguf/qwen25-golden-v1-f16.gguf --outtype f16

# загрузить в Ollama
cp artifacts/distillation/gguf/*.gguf /media/alexey/DATA/ollama/models/
cp Modelfile.distill /media/alexey/DATA/ollama/models/Modelfile.distill
cp Modelfile.golden /media/alexey/DATA/ollama/models/Modelfile.golden
docker exec ollama ollama create qwen25-distill-v1 -f /root/.ollama/Modelfile.distill
docker exec ollama ollama create qwen25-golden-v1 -f /root/.ollama/Modelfile.golden

# замеры и сравнение
make eval-all    # eval_three.py --variants base,teacher,distilled,golden
make compare     # compare.py -> results/compare.md
```

### Живое сравнение в консоли

Печатает ответы каждой модели на проверочные вопросы и итог «качество / скорость /
стоимость запроса». Модели прогоняются по очереди целиком, поэтому скорость
сопоставима с `results/compare.md`:

```bash
make compare-live
python3 scripts/compare_ollama.py --limit 10
python3 scripts/compare_ollama.py --variants base,distilled,teacher
python3 scripts/compare_ollama.py --category invalid --only-diff
```

Результат попутно пишется в `results/compare_live.json`.

## Метрики

- **Качество:** `exact`, `schema_ok`, `ticker_f1`, `poll_acc`, `report_acc`,
  `error_acc` (из `watch_parser/metrics.py`), по категориям.
- **Скорость:** медианная латентность (с), токенов/с (из `eval_duration`).
- **Стоимость:** фактические токены входа/выхода → $ по справочным тарифам
  (`scripts/pricing.py`); $/1M выходных токенов и $/запрос.
- **Зоны замены:** попарно «ученик vs учитель» — заменяет / ещё ошибается /
  общая ошибка / ученик лучше.

## Ограничения

- Учитель (`qwen2.5:7b`) систематически нормализует единицы (60 минут → `1h`,
  сутки → `24h`), нарушая строгое правило spec «не переводить единицы»; это
  видно в `filter_report.json` и учитывается в отчёте.
- Корпус синтетический (слот-генератор из `watch_parser`), не реальные логи.
- 90 тестовых примеров — наблюдения, а не статистика; один прогон, жадная
  генерация.
- «Стоимость» — оценка по справочным облачным тарифам, а не реальный счёт
  (всё гоняется локально на CPU).
