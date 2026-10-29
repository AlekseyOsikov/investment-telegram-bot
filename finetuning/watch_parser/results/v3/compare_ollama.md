# Локальная Ollama-модель: проверка merged-адаптера

Модель: `qwen25-watch-v3:latest` (GGUF F16, 3.1 ГБ), собрана из `merged_v3` (LoRA v3 + Qwen2.5-1.5B-Instruct).

| набор | HF | Ollama | exact HF | exact Ollama | schema Ollama | error_acc Ollama |
|---|---|---|---:|---:|---:|---:|
| dev | HF v3 | Ollama v3 | 85.0% | 85.0% | 96.7% | 60.0% |
| dev | HF v3+guard | Ollama v3+guard | 100.0% | 100.0% | 100.0% | 100.0% |
| holdout | HF v3 | Ollama v3 | 76.7% | 76.7% | 93.3% | 58.3% |
| holdout | HF v3+guard | Ollama v3+guard | 96.7% | 96.7% | 100.0% | 100.0% |

Совпадение по exact и error_acc полное: слияние LoRA, конвертация в GGUF F16 и загрузка в Ollama не изменили поведение модели.

Важно: для Ollama выставлялись `temperature=0`, `repeat_penalty=1.0`, `top_k=1`, `top_p=1.0`, `num_ctx=4096`. Со штатным `repeat_penalty=1.1` few-shot ломается.
