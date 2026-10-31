# Обучение в Colab (T4)

Дообучаем `Qwen2.5-1.5B-Instruct` на двух вариантах меток (дистилляция):

- `distill` — метки учителя, отфильтрованные по форме (`train_distill.jsonl`);
- `golden` — метки, совпавшие со смыслом (`train_golden.jsonl`).

Готовый ноутбук — [`colab_distill.ipynb`](colab_distill.ipynb): загрузите его в
Colab, включите T4 и выполните ячейки по порядку (он сам попросит
`distillation.zip`, обучит оба адаптера и упакует результат с путями репозитория).
Шаги ниже — то же самое вручную.

## 1. Подготовить данные локально

```bash
cd finetuning/distillation
make corpus && make label && make filter   # или по отдельности
make zip                                   # finetuning/distillation.zip
```

## 2. Загрузить в Colab

Загрузите `distillation.zip`, распакуйте:

```python
!unzip -o distillation.zip -d /content
```

Установите зависимости и перезапустите среду:

```python
!pip -q install -U "transformers>=4.56" "peft>=0.13" accelerate matplotlib
!pip -q uninstall -y torchao
# Runtime -> Restart session
```

## 3. Обучение

```bash
cd /content/finetuning/distillation

python training/train_lora.py --variant distill \
  --out /content/adapter_distill --run-dir /content/run_distill

python training/train_lora.py --variant golden \
  --out /content/adapter_golden --run-dir /content/run_golden
```

Каждое обучение ~10–20 минут (меток ~450–500, 2 эпохи, T4). `run_params.json` и
`loss.png` сохраняются в `--run-dir`.

## 4. Скачать результат

Скачайте каталоги адаптеров:

- `/content/adapter_distill/` → локально `finetuning/artifacts/distillation/adapter_distill/`
- `/content/adapter_golden/` → локально `finetuning/artifacts/distillation/adapter_golden/`
- `/content/run_distill/run_params.json`, `loss.png` → `finetuning/distillation/results/`
- `/content/run_golden/run_params.json`, `loss.png` → `finetuning/distillation/results/`

## 5. Локально: слияние → GGUF → Ollama → eval

См. раздел «Локально: слить, сконвертировать, загрузить и сравнить» в
`../README.md`. Итог: `make eval-all && make compare` даёт `results/compare.md`.

## Быстрая проверка качества адаптера прямо в Colab (опционально)

До слияния можно прогнать `watch_parser/scripts/eval_model.py` на адаптере — но
для этого в Colab нужна и папка `watch_parser/` (импорты `spec`/`metrics`/`guard`).
Проще довериться локальному прогону через Ollama (см. выше), где метрики скорости
и стоимости сравнимы с базой и учителем.
