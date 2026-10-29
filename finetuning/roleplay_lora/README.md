# Roleplay LoRA: стиль и границы роли

Дообучение небольшой модели на готовом датасете, чтобы она держалась роли
персонажа, отвечала коротко и не выходила за границы знаний.

- База: `Qwen/Qwen2.5-1.5B-Instruct`
- Метод: LoRA, fp16, T4
- Датасет: [`asd-ml/roleplay_boundaries_processed`](https://huggingface.co/datasets/asd-ml/roleplay_boundaries_processed)
- Обучение: 1934 примера, 2 эпохи, 54 мин 54 с
- Сравнение: 20 запросов «до/после»

Полный отчёт — [`report.md`](report.md), карточка данных — [`dataset_card.md`](dataset_card.md).

## Файлы

```text
roleplay_lora/
  report.md                  # полный отчёт
  dataset_card.md            # датасет, фильтрация, split, лицензия
  roleplay_lora_colab.ipynb  # ноутбук обучения
  Modelfile                  # ChatML Qwen для Ollama
  results/
    compare_20.json|csv      # 20 запросов: база vs адаптер
    compare_20_for_review.csv
    ollama_20.json           # повторный прогон через Ollama
    run_params.json          # параметры запуска и лосс
    loss.png
  scripts/
    merge.py                 # adapter + base → merged HF
    ollama_run.py            # 20 запросов через Ollama
```

Адаптер, merged-модель и GGUF — тяжёлые артефакты, они лежат вне git:

```text
finetuning/artifacts/roleplay_lora/
  adapter_roleplay_ru/       # LoRA-адаптер, ~82 МБ
  merged/                    # появится после scripts/merge.py
  qwen25-roleplay-f16.gguf   # появится после конвертации
```

## Воспроизведение

1. Обучение — `roleplay_lora_colab.ipynb` в Colab T4. Параметры и лосс —
   `results/run_params.json`.
2. Слияние адаптера с базой:

```bash
python scripts/merge.py
# или явно:
python scripts/merge.py \
  --adapter ../artifacts/roleplay_lora/adapter_roleplay_ru \
  --out ../artifacts/roleplay_lora/merged
```

3. Конвертация в GGUF F16 (`llama.cpp` клонируется отдельно, в git не входит):

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  ../artifacts/roleplay_lora/merged \
  --outfile ../artifacts/roleplay_lora/qwen25-roleplay-f16.gguf \
  --outtype f16
```

4. Загрузка в Ollama:

```bash
cp ../artifacts/roleplay_lora/qwen25-roleplay-f16.gguf /media/alexey/DATA/ollama/models/
cp Modelfile /media/alexey/DATA/ollama/models/Modelfile.roleplay
docker exec ollama ollama create qwen25-roleplay -f /root/.ollama/Modelfile.roleplay
```

5. Прогон 20 запросов:

```bash
python scripts/ollama_run.py
```

## Результат

В отчёте: 10 запросов стали лучше, 5 — смешанно, 5 — хуже. Адаптер выучил
манеру и границы роли, но появились регрессы вне роли и на требованиях
к формату. Подробности и таблица — в `report.md`.

## Ограничения

- Датасет англоязычный с русскими вставками; отбирались строки с долей
  кириллицы ≥ 0.5.
- Лицензия исходного датасета в карточке не указана — проверить у источника.
- Один прогон, 20 запросов, жадная генерация; это наблюдения, а не статистика.
