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

## Правила

- Не добавлять сюда `.venv`, `llama.cpp`, HF-кэш.
- Для передачи артефактов использовать release assets/внешний диск.
- Если артефакт нужен в git — только Git LFS и только адаптеры, не
  merged/GGUF.
- Контрольные суммы и параметры запуска хранить в `results/` рядом с отчётом.
