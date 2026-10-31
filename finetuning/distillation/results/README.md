# Результаты замеров проекта «Дистилляция»

Все файлы генерируются скриптами; руками не правятся.

## `eval/`, `holdout/`

`<variant>.json` — прогон варианта (`base` / `teacher` / `distilled` / `golden`)
на тестовом наборе, от `eval_three.py`. Структура:

- `quality` — `exact`, `schema_ok`, `ticker_f1`, `poll_acc`, `report_acc`,
  `error_acc`, `json_strict`, `json_extractable`;
- `by_category` — те же метрики по категориям
  (`simple`/`multi`/`colloquial`/`invalid`/`correction`/`role_order`/`out_of_scope`);
- `speed` — медианная латентность (с), токенов/с, суммарные токены входа/выхода;
- `results` — построчный разбор (каждый пример с метриками и таймингами).

## `compare.md` / `compare.json`

Сводная таблица «качество/скорость/стоимость» по вариантам (объединённо
eval+holdout) и попарный анализ «ученик vs учитель» (зоны замены), от `compare.py`.

## `compare_live.json`

Тот же разбор, но с живым прогоном моделей и полными ответами на каждый вопрос
(от `compare_ollama.py` / `make compare-live`): `summary` по вариантам и
`results` построчно.

## `data/` (корень проекта)

- `filter_report.json` — сколько ответов учителя прошло форму/смысл, сколько
  отброшено, размеры train/valid обоих вариантов;
- `corpus_stats.json` — баланс категорий и кодов ошибок корпуса.

## `results/` после Colab

- `run_distill_params.json`, `run_distill_loss.png` — обучение варианта `distill`;
- `run_golden_params.json`, `run_golden_loss.png` — обучение варианта `golden`.

## Как читать «стоимость»

`pricing.py` переводит фактические токены входа/выхода в $ по справочным тарифам
(ориентировочно, см. docstring модуля). Реального счёта нет — всё локально.
