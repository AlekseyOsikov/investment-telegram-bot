# Границы файнтюнинга: что тюн чинит, а что нет

Экспериментальный проект: отвечает на вопросы «добавляет ли тюн знания»,
«что модель теряет», «где ловится переобучение» и «на каком типе задач тюн
даёт больше всего» — на той же базе `Qwen2.5-1.5B-Instruct` и тем же пайплайне
LoRA, что `roleplay_lora` и `watch_parser`.

Полный отчёт и выводы — [`report.md`](report.md).

## Что измеряем

| Вопрос задания | Как измеряется | Данные / скрипты |
|---|---|---|
| Добавляет ли тюн знания | факты, которых база не знает; тюн vs «факты в промпте» | `data/facts.py`, `scripts/eval_facts.py` |
| Что модель потеряла | задачи вне домена: база vs дообученная | `data/out_of_domain.py`, `scripts/eval_out_of_domain.py` |
| Переобучение | train/valid-лосс по шагам, чекпоинты по эпохам | `training/train_facts.py` |
| Что тюн даёт лучше всего | три типа задач: формат / стиль / факты | `scripts/compare_three.py` |

## Факты

- **Синтетические** — вымышленные «ЗаряПром» (ZRPR), «индекс Вепрецова»,
  фонд «Стабильный берег»: уникальные термины исключают утечку из предобучения,
  проверка по `keys` детерминированная.
- **Реальные** — факты 2026 года из корпуса `data/source` (после даты знаний
  модели), для связки с RAG.

Три «плеча» на каждый факт: `base_none` (база без фактов), `base_facts`
(база + факт в промпте — то, что на проде делает RAG), `tuned` (дообученная
модель без фактов в промпте). По две формулировки на факт — обучающая
(memorization) и перефразировка (generalization).

## Структура

```text
boundaries/
  README.md, report.md
  Makefile, Modelfile.facts
  data/
    facts.py            # источник фактов (train/valid/paraphrases/judge/keys)
    build_facts.py      # -> train.jsonl, valid.jsonl, eval_facts.json
    out_of_domain.py    # -> out_of_domain.json (задачи вне домена)
    style_eval.py       # -> style_eval.json (стиль и тон)
  scripts/
    ollama_chat.py      # общий вызов Ollama
    judge.py            # слепой judge по чек-листу (DeepSeek/Ollama)
    eval_facts.py       # прогон фактов (три плеча)
    eval_out_of_domain.py
    eval_style.py
    compare_three.py    # сводная таблица трёх типов
    compare_console.py  # консольное сравнение: ответы моделей по трём типам
    merge_adapter.py    # adapter + base → merged HF
  training/
    train_facts.py      # LoRA с eval по шагам и чекпоинтами по эпохам
    colab_boundaries.ipynb
  results/              # сгенерированные замеры
  ../artifacts/boundaries/   # adapter_facts, merged, gguf (вне git)
```

## Локально: собрать данные и прогнать eval

Нужен только Python 3.10+ (без GPU и torch) и запущенный Docker-Ollama.
Judge использует DeepSeek (`DEEPSEEK_API_KEY` из корневого `.env`).

```bash
cd finetuning/boundaries
make data

# факты: база без фактов / база + факт в промпте
make facts-none
make facts-prompt

# задачи вне домена: база
make ood-base

# стиль: база без роли / база + роль / roleplay-тюн
make style-none
make style-prompt
make style-tuned

# сводная таблица трёх типов
make compare
```

## Консольное сравнение моделей

`scripts/compare_console.py` печатает проверочные вопросы и ответы двух моделей
рядом по трём типам задач (как `watch_parser/scripts/compare_ollama.py`) и
выносит вердикт по каждому:

- **формат** — база + few-shot vs `qwen25-watch-v3` (точное совпадение JSON);
- **стиль** — база + ролевой промпт vs `qwen25-roleplay` (краткость; с
  `--judge` — покрытие критериев);
- **факты** — база + факт в промпте vs `qwen25-facts-v1` (ключевые значения;
  с `--judge` — покрытие фактов).

```bash
python scripts/compare_console.py                  # компактно: по 2 на группу
python scripts/compare_console.py --all            # весь набор (формат — 60)
python scripts/compare_console.py --task facts --per-group 1
python scripts/compare_console.py --task style --judge
python scripts/compare_console.py --only-diff --max-chars 200
python scripts/compare_console.py --show-system --limit 1
```

## Colab T4: обучение фактов и переобучение

`training/colab_boundaries.ipynb` — дообучение на фактах с контролем
переобучения:

```bash
python training/train_facts.py --epochs 10 \
  --out /content/adapter_facts --run-dir /content/run_facts
```

Скрипт считает valid-лосс каждые `--eval-steps` (по умолчанию 5) шагов и
сохраняет адаптер каждую эпоху — по графику `results/run_params.json`
+ `loss.png` видно, где train падает, а valid перестаёт (переобучение), и
можно взять чекпоинт «до» и «после».

После обучения — слить, сконвертировать в GGUF и загрузить в Ollama:

```bash
python scripts/merge_adapter.py \
  --adapter ../artifacts/boundaries/adapter_facts --out ../artifacts/boundaries/merged_facts

python /path/to/llama.cpp/convert_hf_to_gguf.py ../artifacts/boundaries/merged_facts \
  --outfile ../artifacts/boundaries/gguf/qwen25-facts-f16.gguf --outtype f16

cp ../artifacts/boundaries/gguf/qwen25-facts-f16.gguf /media/alexey/DATA/ollama/models/
cp Modelfile.facts /media/alexey/DATA/ollama/models/Modelfile.facts
docker exec ollama ollama create qwen25-facts-v1 -f /root/.ollama/Modelfile.facts
```

Затем прогнать дообученную модель:

```bash
make facts-tuned
make ood-tuned
```

### Результат из Colab — `boundaries_results.zip`

Ноутбук сам прогоняет дообученную модель через HF-адаптер (`scripts/eval_hf.py`,
без слияния и GGUF) и кладёт результаты в `results/facts/tuned.json` и
`results/ood/tuned.json`. После обучения он скачивает `boundaries_results.zip`;
он повторяет пути репозитория, поэтому восстанавливается одной командой в корне
репозитория:

```bash
unzip -o boundaries_results.zip -d .
```

Это кладёт на место:

- `finetuning/boundaries/results/run_params.json` и `loss.png` — лосс train vs
  valid по шагам (по ним виден шаг расхождения, `eval_loss_min_step`);
- `finetuning/boundaries/results/facts/tuned.json`, `results/ood/tuned.json` —
  замеры дообученной модели (если в ноутбуке запускали eval);
- `finetuning/artifacts/boundaries/adapter_facts/` — финальный (переобученный)
  LoRA-адаптер;
- `finetuning/artifacts/boundaries/checkpoint_best/` — лучший по valid чекпоинт
  (только файлы адаптера, без `optimizer.pt`/`scheduler.pt`).

Сравнить «лучший vs переобученный»:

```bash
python scripts/eval_hf.py --task facts --adapter ../artifacts/boundaries/checkpoint_best \
  --out results/facts/tuned_best.json
python scripts/eval_hf.py --task facts --adapter ../artifacts/boundaries/adapter_facts \
  --out results/facts/tuned.json
```

После распаковки — `make compare` (свободная таблица трёх типов задач).

Если eval в ноутбуке не запускали, дообученную модель можно прогнать и через
Ollama: слить адаптер, сконвертировать в GGUF, загрузить как `qwen25-facts-v1`
(команды выше), затем `make facts-tuned` и `make ood-tuned`.

## Ограничения

- Выборки малы (22 факта-вопроса, 26 задач вне домена, 8 стилевых промптов) —
  это наблюдения, а не статистика. Один прогон, жадная генерация.
- Judge — та же «способность к оценке», что у `research/rag_compare`:
  слепая оценка по чек-листу, возможен шум; детерминированную проверку дают
  `keys` (факты) и `keywords` (вне домена).
- «Тюн на фактах» обучен в Colab (T4), eval прогнан локально на CPU
  (`eval_hf.py`); база и «факты в промпте» — через Ollama.
- Общие уровни забывания: узкий факт-тюн вне домена не деградировал, широкий
  roleplay — деградировал на общих знаниях (0.667 → 0.5); выборки малы.
