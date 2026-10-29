# Сравнение прогонов на одном eval (60 запросов)

| метрика | база | v1 | v2 | v2+guard | v3 | v3+guard |
|---|---|---|---|---|---|---|
| exact | 46.7% | 71.7% | 83.3% | 98.3% | 85.0% | 100.0% |
| schema_ok | 78.3% | 90.0% | 95.0% | 100.0% | 96.7% | 100.0% |
| ticker_f1 | 98.1% | 98.9% | 97.8% | 100.0% | 95.9% | 100.0% |
| poll_acc | 86.7% | 97.8% | 97.8% | 100.0% | 97.8% | 100.0% |
| report_acc | 88.9% | 97.8% | 95.6% | 97.8% | 97.8% | 100.0% |
| error_acc | 6.7% | 26.7% | 46.7% | 100.0% | 60.0% | 100.0% |

## Exact match по категориям

| категория | база | v1 | v2 | v2+guard | v3 | v3+guard |
|---|---|---|---|---|---|---|
| colloquial | 66.7% | 80.0% | 93.3% | 93.3% | 93.3% | 100.0% |
| correction | 40.0% | 100.0% | 80.0% | 100.0% | 80.0% | 100.0% |
| invalid | 10.0% | 10.0% | 40.0% | 100.0% | 60.0% | 100.0% |
| multi | 40.0% | 70.0% | 100.0% | 100.0% | 90.0% | 100.0% |
| out_of_scope | 0.0% | 60.0% | 60.0% | 100.0% | 60.0% | 100.0% |
| role_order | 40.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| simple | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Точность по кодам ошибок

| код | всего | база | v1 | v2 | v2+guard | v3 | v3+guard |
|---|---:|---|---|---|---|---|---|
| ambiguous | 2 | 0/2 | 0/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| bad_interval | 1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| bad_ticker | 2 | 0/2 | 0/2 | 0/2 | 2/2 | 1/2 | 2/2 |
| empty | 2 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 |
| missing_intervals | 1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| missing_poll_interval | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |
| missing_report_interval | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |
| missing_tickers | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |
| not_watch | 4 | 1/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |

## v3 → v3+guard

- стало лучше по exact: **9**
- стало хуже по exact: **0**
- верны у обеих: 51
- неверны у обеих: 0

### Где лучше

- **#17 [multi]** SBER, SBER и GAZP — следи каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["GAZP", "SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - v3: `{"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}`
  - v3+guard: `{"ok":true,"secids":["GAZP","SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#30 [colloquial]** Татнефть и Магнит — каждые 90 минут, сводка раз в 7 дней.
  - эталон: `{"ok": true, "secids": ["MGNT", "TATN"], "poll_interval": "90m", "report_interval": "7d"}`
  - v3: `{"ok":true,"secids":["MTSS","TATN"],"poll_interval":"90m","report_interval":"7d"}`
  - v3+guard: `{"ok":true,"secids":["MGNT","TATN"],"poll_interval":"90m","report_interval":"7d"}`
- **#36 [invalid]** Следи каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": false, "error": "missing_tickers"}`
  - v3: `{"ok":true,"secids":["GAZP"],"poll_interval":"15m","report_interval":"1h"}`
  - v3+guard: `{"ok":false,"error":"missing_tickers"}`
- **#38 [invalid]** Следи за SBER, сводку присылай раз в час.
  - эталон: `{"ok": false, "error": "missing_poll_interval"}`
  - v3: `{"ok":true,"secids":["SBER"],"poll_interval":"1h","report_interval":"1h"}`
  - v3+guard: `{"ok":false,"error":"missing_poll_interval"}`
- **#39 [invalid]** Следи за SBER каждые 15 минут.
  - эталон: `{"ok": false, "error": "missing_report_interval"}`
  - v3: `{"ok":true,"secids":["SBER"],"poll_interval":"15m"}`
  - v3+guard: `{"ok":false,"error":"missing_report_interval"}`
- **#41 [invalid]** Следи за Ромашкой каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": false, "error": "bad_ticker"}`
  - v3: `{"ok":true,"secids":["ROSH"],"poll_interval":"15m","report_interval":"1h"}`
  - v3+guard: `{"ok":false,"error":"bad_ticker"}`
- **#49 [correction]** Следи за SBER и SBER каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - v3: `{"ok":false,"error":"ambiguous"}`
  - v3+guard: `{"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#56 [out_of_scope]** 
  - эталон: `{"ok": false, "error": "empty"}`
  - v3: `{"ok":false,"error":"missing_report_interval"}`
  - v3+guard: `{"ok":false,"error":"empty"}`
- **#57 [out_of_scope]**    
  - эталон: `{"ok": false, "error": "empty"}`
  - v3: `{"ok":false,"error":"missing_tickers"}`
  - v3+guard: `{"ok":false,"error":"empty"}`

### Где хуже

нет
