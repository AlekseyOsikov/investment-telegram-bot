# Сравнение v1 и v2 на одном eval (60 запросов)

| метрика | база с промптом | v1 | v2 |
|---|---|---|---|
| exact | 46.7% | 71.7% | 83.3% |
| schema_ok | 78.3% | 90.0% | 95.0% |
| ticker_f1 | 98.1% | 98.9% | 97.8% |
| poll_acc | 86.7% | 97.8% | 97.8% |
| report_acc | 88.9% | 97.8% | 95.6% |
| error_acc | 6.7% | 26.7% | 46.7% |

## Exact match по категориям

| категория | база с промптом | v1 | v2 |
|---|---|---|---|
| colloquial | 66.7% | 80.0% | 93.3% |
| correction | 40.0% | 100.0% | 80.0% |
| invalid | 10.0% | 10.0% | 40.0% |
| multi | 40.0% | 70.0% | 100.0% |
| out_of_scope | 0.0% | 60.0% | 60.0% |
| role_order | 40.0% | 100.0% | 100.0% |
| simple | 90.0% | 100.0% | 100.0% |

## Точность по кодам ошибок (error_acc)

| код | всего | база с промптом | v1 | v2 |
|---|---:|---|---|---|
| ambiguous | 2 | 0/2 | 0/2 | 1/2 |
| bad_interval | 1 | 0/1 | 0/1 | 1/1 |
| bad_ticker | 2 | 0/2 | 0/2 | 0/2 |
| empty | 2 | 0/2 | 0/2 | 0/2 |
| missing_intervals | 1 | 0/1 | 0/1 | 1/1 |
| missing_poll_interval | 1 | 0/1 | 0/1 | 0/1 |
| missing_report_interval | 1 | 0/1 | 0/1 | 0/1 |
| missing_tickers | 1 | 0/1 | 0/1 | 0/1 |
| not_watch | 4 | 1/4 | 4/4 | 4/4 |

## v1 → v2

- стало лучше по exact: **8**
- стало хуже по exact: **1**
- верны у обеих: 42
- неверны у обеих: 9

### Где v2 лучше

- **#17 [multi]** SBER, SBER и GAZP — следи каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["GAZP", "SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - v1: `{"ok":true,"secids":["SBER","GAZP"],"poll_interval":"15m","report_interval":"1h"}`
  - v2: `{"ok":true,"secids":["GAZP","SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#18 [multi]** Следи за TATN и MGNT каждые 30 минут, сводку раз в 90 минут.
  - эталон: `{"ok": true, "secids": ["MGNT", "TATN"], "poll_interval": "30m", "report_interval": "90m"}`
  - v1: `{"ok":true,"secids":["TATN","MGNT"],"poll_interval":"30m","report_interval":"90m"}`
  - v2: `{"ok":true,"secids":["MGNT","TATN"],"poll_interval":"30m","report_interval":"90m"}`
- **#20 [multi]** За NVTK, PLZL, YDEX, AFLT и IRAO следи каждые 15 минут, сводка раз в 3 дня.
  - эталон: `{"ok": true, "secids": ["AFLT", "IRAO", "NVTK", "PLZL", "YDEX"], "poll_interval": "15m", "report_interval": "3d"}`
  - v1: `{"ok":true,"secids":["AFLT","NVTK","PLZL","IRAO","YDEX"],"poll_interval":"15m","report_interval":"3d"}`
  - v2: `{"ok":true,"secids":["AFLT","IRAO","NVTK","PLZL","YDEX"],"poll_interval":"15m","report_interval":"3d"}`
- **#21 [colloquial]** Следи за Сбером каждые четверть часа, сводку присылай раз в час.
  - эталон: `{"ok": true, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - v1: `{"ok":true,"secids":["SBER"],"poll_interval":"1h","report_interval":"1h"}`
  - v2: `{"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#30 [colloquial]** Татнефть и Магнит — каждые 90 минут, сводка раз в 7 дней.
  - эталон: `{"ok": true, "secids": ["MGNT", "TATN"], "poll_interval": "90m", "report_interval": "7d"}`
  - v1: `{"ok":true,"secids":["MTSS","TATN"],"poll_interval":"90m","report_interval":"7d"}`
  - v2: `{"ok":true,"secids":["MGNT","TATN"],"poll_interval":"90m","report_interval":"7d"}`
- **#37 [invalid]** Следи за SBER и GAZP.
  - эталон: `{"ok": false, "error": "missing_intervals"}`
  - v1: `{"ok":true,"secids":["GAZP","SBER"],"poll_interval":"30m","report_interval":"1h"}`
  - v2: `{"ok":false,"error":"missing_intervals"}`
- **#40 [invalid]** Следи за SBER каждые 30 секунд, сводку раз в час.
  - эталон: `{"ok": false, "error": "bad_interval"}`
  - v1: `{"ok":true,"secids":["SBER"],"poll_interval":"30s","report_interval":"1h"}`
  - v2: `{"ok":false,"error":"bad_interval"}`
- **#43 [invalid]** Следи за SBER каждые 15 минут, 30 минут и сводку раз в час.
  - эталон: `{"ok": false, "error": "ambiguous"}`
  - v1: `{"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"30m"}`
  - v2: `{"ok":false,"error":"ambiguous"}`

### Где v2 хуже

- **#49 [correction]** Следи за SBER и SBER каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - v1: `{"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}`
  - v2: `{"ok":false,"error":"ambiguous"}`
