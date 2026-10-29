# Артефакты fine-tuning

Каталог для тяжёлых файлов, которые **не коммитятся** (см. корневой
`.gitignore`). Здесь лежат только результаты обучения и экспорта; код и
отчёты — в `finetuning/<project>/`.

## roleplay_lora

```text
artifacts/roleplay_lora/
  adapter_roleplay_ru/       # LoRA-адаптер, ~82 МБ
  merged/                    # появится после scripts/merge.py
  qwen25-roleplay-f16.gguf   # появится после конвертации
```

Модель в Ollama: `qwen25-roleplay`.

## watch_parser

```text
artifacts/watch_parser/
  adapter_v3/                # финальный LoRA-адаптер, ~82 МБ
```

`merged_v3/` и GGUF здесь не хранятся: они воспроизводятся из `adapter_v3`
и базы командами из `watch_parser/README.md`. Готовая модель остаётся в
Ollama как `qwen25-watch-v3` (blob-хранилище Docker).

## Правила

- Не добавлять сюда `.venv`, `llama.cpp`, HF-кэш.
- Для передачи артефактов использовать release assets/внешний диск.
- Если артефакт нужен в git — только Git LFS и только адаптеры, не
  merged/GGUF.
- Контрольные суммы и параметры запуска хранить в `results/` рядом с отчётом.
