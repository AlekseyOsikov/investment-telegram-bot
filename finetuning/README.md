# Fine-tuning: дообучение локальных моделей

Отдельный подпроект репозитория для дообучения локальных моделей: LoRA,
подготовка данных, eval, слияние адаптеров, GGUF и Ollama. Бот его
**не импортирует**; `src/` и `tests/` бота не зависят от `finetuning/`.
Это та же граница, что у `scripts/` и `src/research/`.

## Проекты

| Проект | Назначение | Модель | Статус | Отчёт |
|---|---|---|---|---|
| `roleplay_lora` | Roleplay: стиль и границы роли | Qwen2.5-1.5B + LoRA | готово | [`roleplay_lora/report.md`](roleplay_lora/report.md) |

Другие проекты добавляются по мере готовности.

## Структура

```text
finetuning/
  README.md
  requirements.txt           # зависимости ML, отдельно от requirements.txt бота
  roleplay_lora/
    README.md
    report.md
    dataset_card.md
    roleplay_lora_colab.ipynb
    Modelfile
    results/
    scripts/
  artifacts/                 # тяжёлые артефакты, в git не попадают
    README.md
    roleplay_lora/
      adapter_roleplay_ru/
```

Общий код (train/eval/merge/Ollama) будет вынесен в `common/`, когда
появится второй проект. Пока `roleplay_lora` самодостаточен: его скрипты
лежат в `roleplay_lora/scripts/`.

## Артефакты и git

**В git:** код, notebook, отчёты, карточки данных, небольшие JSONL,
`results/*.json|csv|md`, `loss*.png`, `run_params*.json`, `Modelfile`.

**Не в git:**

- `artifacts/` — адаптеры, merged-модели, GGUF;
- `*.gguf`, `merged*/`, `adapter_model.safetensors`, `model.safetensors`;
- `.venv/`, локальные клоны `llama.cpp/`, HF-кэш.

Правила — в корневом `.gitignore`. Для передачи тяжёлых артефактов
использовать release assets, внешний диск или Git LFS (адаптеры ~80 МБ;
merged/GGUF — гигабайты, LFS для них нежелателен).

## Зависимости

```bash
python -m venv .venv-finetuning
. .venv-finetuning/bin/activate
pip install -r finetuning/requirements.txt
```

`llama.cpp` для конвертации в GGUF клонируется отдельно и в git не входит.

## Как добавить новый проект

1. Создать `finetuning/<project>/`.
2. Положить `README.md`, `report.md`, `results/`, `scripts/` или notebook.
3. Тяжёлые веса — в `finetuning/artifacts/<project>/`.
4. Добавить строку в таблицу проектов выше.
5. Если код общий с другим проектом — вынести в `finetuning/common/`,
   а не копировать.
