# Сравнение на eval-наборе (60 запросов)

| метрика | база без промпта | база с промптом | дообученная |
|---|---|---|---|
| exact | 0.0% | 46.7% | 71.7% |
| schema_ok | 0.0% | 78.3% | 90.0% |
| json_strict | 0.0% | 100.0% | 100.0% |
| json_extractable | 0.0% | 100.0% | 100.0% |
| ticker_f1 | 0.0% | 98.1% | 98.9% |
| poll_acc | 0.0% | 86.7% | 97.8% |
| report_acc | 0.0% | 88.9% | 97.8% |
| error_acc | 0.0% | 6.7% | 26.7% |

## Exact match по категориям

| категория | база без промпта | база с промптом | дообученная |
|---|---|---|---|
| colloquial | 0.0% | 66.7% | 80.0% |
| correction | 0.0% | 40.0% | 100.0% |
| invalid | 0.0% | 10.0% | 10.0% |
| multi | 0.0% | 40.0% | 70.0% |
| out_of_scope | 0.0% | 0.0% | 60.0% |
| role_order | 0.0% | 40.0% | 100.0% |
| simple | 0.0% | 90.0% | 100.0% |

## Где стало лучше

Всего: 15

- **#3 [simple]** Хочу получать сводки по LKOH раз в день, а цены проверять каждые 10 минут.
  - эталон: `{"ok": true, "secids": ["LKOH"], "poll_interval": "10m", "report_interval": "1d"}`
  - база: `{"ok":true,"secids":["LKOH"],"poll_interval":"1d","report_interval":"10m"}`
  - тюн: `{"ok":true,"secids":["LKOH"],"poll_interval":"10m","report_interval":"1d"}`
- **#11 [multi]** Следи за SBER и GAZP каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["GAZP", "SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - база: `{"ok":true,"secids":["SBER","GAZP"],"poll_interval":"15m","report_interval":"1h"}`
  - тюн: `{"ok":true,"secids":["GAZP","SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#13 [multi]** Добавь в отслеживание: Сбер, Газпром, Лукойл. Проверка каждые 10 минут, сводка раз в 3 часа.
  - эталон: `{"ok": true, "secids": ["GAZP", "LKOH", "SBER"], "poll_interval": "10m", "report_interval": "3h"}`
  - база: `{"ok":true,"secids":["SBER","GAZP","LKOH"],"poll_interval":"10m","report_interval":"3h"}`
  - тюн: `{"ok":true,"secids":["GAZP","LKOH","SBER"],"poll_interval":"10m","report_interval":"3h"}`
- **#14 [multi]** По SBER, GAZP и MOEX: опрос каждые 5 минут, сводка раз в 6 часов.
  - эталон: `{"ok": true, "secids": ["GAZP", "MOEX", "SBER"], "poll_interval": "5m", "report_interval": "6h"}`
  - база: `{"ok":true,"secids":["SBER","GAZP","MOEX"],"poll_interval":"5m","report_interval":"6h"}`
  - тюн: `{"ok":true,"secids":["GAZP","MOEX","SBER"],"poll_interval":"5m","report_interval":"6h"}`
- **#28 [colloquial]** РусГидро, ФосАгро и Сибур — опрос каждые четверть часа, сводка раз в 12 часов.
  - эталон: `{"ok": true, "secids": ["HYDR", "PHOR", "SIBN"], "poll_interval": "15m", "report_interval": "12h"}`
  - база: `{"ok":true,"secids":["PHOR","PLZL","SIBN"],"poll_interval":"15m","report_interval":"12h"}`
  - тюн: `{"ok":true,"secids":["HYDR","PHOR","SIBN"],"poll_interval":"15m","report_interval":"12h"}`
- **#33 [colloquial]** Полюс и Яндекс проверяй каждые 30 минут, отчёт раз в 12 часов.
  - эталон: `{"ok": true, "secids": ["PLZL", "YDEX"], "poll_interval": "30m", "report_interval": "12h"}`
  - база: `{"ok":true,"secids":["PLZL","VTBR"],"poll_interval":"30m","report_interval":"12h"}`
  - тюн: `{"ok":true,"secids":["PLZL","YDEX"],"poll_interval":"30m","report_interval":"12h"}`
- **#47 [correction]** По GAZP: сводка раз в час, опрос каждые 15 минут. Хотя нет, сводку лучше раз в 2 часа.
  - эталон: `{"ok": true, "secids": ["GAZP"], "poll_interval": "15m", "report_interval": "2h"}`
  - база: `{"ok":true,"secids":["GAZP"],"poll_interval":"1h","report_interval":"2h"}`
  - тюн: `{"ok":true,"secids":["GAZP"],"poll_interval":"15m","report_interval":"2h"}`
- **#48 [correction]** Следи за sber, GAZP и sber каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["GAZP", "SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - база: `{"ok":true,"secids":["SBER","GAZP"],"poll_interval":"15m","report_interval":"1h"}`
  - тюн: `{"ok":true,"secids":["GAZP","SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#49 [correction]** Следи за SBER и SBER каждые 15 минут, сводку раз в час.
  - эталон: `{"ok": true, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h"}`
  - база: `{"ok":true,"secids":["SBER","SBER"],"poll_interval":"15m","report_interval":"1h"}`
  - тюн: `{"ok":true,"secids":["SBER"],"poll_interval":"15m","report_interval":"1h"}`
- **#51 [role_order]** По ROSN: сводку присылай раз в 2 часа, цены проверяй каждые 30 минут.
  - эталон: `{"ok": true, "secids": ["ROSN"], "poll_interval": "30m", "report_interval": "2h"}`
  - база: `{"ok":true,"secids":["ROSN"],"poll_interval":"2h","report_interval":"30m"}`
  - тюн: `{"ok":true,"secids":["ROSN"],"poll_interval":"30m","report_interval":"2h"}`

## Где стало хуже

Всего: 0

