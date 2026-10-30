# Результаты проекта «Границы файнтюнинга»

Каталоги и файлы результатов eval (генерируются `make …`, см. Makefile).

```text
results/
  facts/
    base_none.json     # база без фактов (coverage + keys)
    base_facts.json    # база + факт в промпте (baseline RAG)
    tuned.json         # дообученная facts-модель, финальная (переобучена)
    tuned_best.json    # дообученная facts-модель, чекпоинт шага 15 (min valid)
  ood/
    base.json          # база на задачах вне домена
    watch_v3.json      # узкий тюн (формат) вне домена
    roleplay.json      # широкий тюн (стиль) вне домена
    tuned.json         # facts-модель вне домена
  style/
    base_none.json     # база без роли
    base_prompt.json   # база + ролевой промпт
    tuned.json         # roleplay-тюн
  run_params.json      # параметры обучения фактов и лосс
  loss.png             # train vs valid по шагам
```

## Формат JSON

Каждый eval-файл — объект с полями `config`, `overall`, `by_*`, `results`
(построчно: вопрос, ответ модели, метрики). Метрики:

- **факты** — `keys_ok` (все `keys` в ответе) и `coverage` (judge-покрытие
  `judge_facts`);
- **вне домена** — `ok` (все `keywords` в ответе), срез по категориям;
- **стиль** — `short` (≤ 2 предложений и ≤ 45 слов) и `coverage`
  (judge-покрытие критериев стиля).

## Обучение фактов (Colab)

`run_params.json`/`loss.png` появляются после прогона
`training/colab_boundaries.ipynb` (скачиваются в `boundaries_results.zip` и
восстанавливаются `unzip -o boundaries_results.zip -d .` из корня репозитория;
адаптер при этом попадает в `../artifacts/boundaries/adapter_facts/`). В
`run_params.json`: `loss_history` (train по шагам), `eval_loss_history` (valid по
шагам) и `eval_loss_min_step` — шаг минимума valid, после которого начинается
переобучение.
