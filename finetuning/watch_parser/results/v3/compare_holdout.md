# Holdout v3: 30 новых запросов

Набор не участвовал в train/valid/eval; guard проверялся на нём после разработки на dev, поэтому это тоже не идеально чистый тест, но лучше, чем dev.

| метрика | v3 | v3+guard |
|---|---:|---:|
| exact | 76.7% | 96.7% |
| schema_ok | 93.3% | 100.0% |
| ticker_f1 | 100.0% | 100.0% |
| poll_acc | 94.4% | 94.4% |
| report_acc | 100.0% | 100.0% |
| error_acc | 58.3% | 100.0% |

## Ошибки v3+guard

- **#24 [correction]** По GAZP: сводка раз в 3 часа, опрос каждые 30 минут. Хотя нет, сводку лучше раз в 6 часов.
  - эталон: `{"ok": true, "secids": ["GAZP"], "poll_interval": "30m", "report_interval": "6h"}`
  - ответ: `{"ok":true,"secids":["GAZP"],"poll_interval":"3h","report_interval":"6h"}`
