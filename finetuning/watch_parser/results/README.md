# Результаты

## Dev-eval (60 запросов)

| Метрика | База без промпта | База с few-shot | v1 | v2 | v2+guard | v3 | v3+guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | 0.0% | 46.7% | 71.7% | 83.3% | 98.3% | 85.0% | 100.0% |
| schema_ok | 0.0% | 78.3% | 90.0% | 95.0% | 100.0% | 96.7% | 100.0% |
| ticker_f1 | 0.0% | 98.1% | 98.9% | 97.8% | 100.0% | 95.9% | 100.0% |
| poll_acc | 0.0% | 86.7% | 97.8% | 97.8% | 100.0% | 97.8% | 100.0% |
| report_acc | 0.0% | 88.9% | 97.8% | 95.6% | 97.8% | 97.8% | 100.0% |
| error_acc | 0.0% | 6.7% | 26.7% | 46.7% | 100.0% | 60.0% | 100.0% |

## Holdout (30 новых запросов)

| Метрика | v3 | v3+guard |
|---|---:|---:|
| exact | 76.7% | **96.7%** |
| schema_ok | 93.3% | **100.0%** |
| ticker_f1 | 100.0% | 100.0% |
| poll_acc | 94.4% | 94.4% |
| report_acc | 100.0% | 100.0% |
| error_acc | 58.3% | **100.0%** |

Единственная ошибка v3+guard — #24, коррекция периода: модель берёт старую
сводку `3h` как poll. Это известное ограничение.

## Локальная Ollama (`qwen25-watch-v3:latest`)

| Набор | Вариант | exact HF | exact Ollama | error_acc Ollama |
|---|---|---:|---:|---:|
| dev, 60 | v3 | 85.0% | 85.0% | 60.0% |
| dev, 60 | v3+guard | 100.0% | 100.0% | 100.0% |
| holdout, 30 | v3 | 76.7% | 76.7% | 58.3% |
| holdout, 30 | v3+guard | 96.7% | 96.7% | 100.0% |

Файлы: `v3/ollama.json|csv`, `v3/ollama_guard.json|csv`,
`v3/ollama_holdout.json|csv`, `v3/ollama_holdout_guard.json|csv`,
`v3/compare_ollama.md`.

## Файлы

```text
baseline/
  base_none.json|csv        # база без промпта
  base_prompt.json|csv      # база с few-shot
v1/
  tuned_prompt.json|csv
  compare.md|csv
  run_params.json
  loss.png
v2/
  tuned_prompt.json|csv
  tuned_prompt_guard.json|csv
  compare.md
  run_params.json
  loss.png
v3/
  tuned_prompt.json|csv
  tuned_prompt_guard.json|csv
  holdout.json|csv
  holdout_guard.json|csv
  ollama.json|csv
  ollama_guard.json|csv
  ollama_holdout.json|csv
  ollama_holdout_guard.json|csv
  compare_holdout.md
  compare_ollama.md
  run_params.json
  loss.png
compare/
  compare_all.md
```
