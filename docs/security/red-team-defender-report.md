# Day 15 — отчёт защитника (Defender Report)

**Цель:** `AlekseyOsikov/investment-telegram-bot`, ветка `Adv-Day-15`.
**Атакованный commit:** `062c3119c9dcd3db002e4d9eff99b325f471468e` (`062c311`).
**Commit с исправлениями:** `381cf48` — `fix(security): close red-team findings (verdict integrity, double-base64, XFF, print-PII)`.
**Источник атак:** `docs/security/red-team-attacker-report.md`
и `red-team-attacker-summary.md`.

**Границы:** атакующий работал локально (исходники + локальная `qwen2.5:7b` + HTTP Gateway с инертным
backend). Реальный Telegram-бот, его провайдер, OpenCode execution loop и публичный reverse proxy не
атаковались. Наши исправления проверены юнит/интеграционными тестами и inline-прогонами на тех же
сценариях; живого повторного удара по развёртыванию партнёра не было (это следующий шаг).

## 1. Сводка: до → после

| ID | Находка атакующего | Статус | После фикса |
|---|---|---|---|
| **P1/P6** | `overall=clean` + `findings[].severity=high` → `merged_overall=clean`, `would_block=false` | **ИСПРАВЛЕНО** | `parse_security_verdict` и `merge_security_verdicts` берут max(overall, findings); `blocks` считает по `effective_overall` |
| **P4/P9/R5** | двойной Base64 проходит Input Guard в `block` и `mask` (HTTP 200, находок 0) | **ИСПРАВЛЕНО** | рекурсивное декодирование до 2 уровней → `double_base64_openai_api_key`; в `mask` токен заменяется |
| **R6** | смена `X-Forwarded-For` даёт новую корзину rate-limit | **ИСПРАВЛЕНО** | XFF учитывается только от доверенного прокси (`LLM_GATEWAY_TRUSTED_PROXIES`); иначе ключ — прямой клиент |
| **P3** | `print(email)` → CLEAN, `logger.info(email)` → MEDIUM | **ИСПРАВЛЕНО** | правило `pii_in_logs` ловит и `print(...)`; контроль `print(total)` не флагуется |
| **P5** | `Проверка:` из task tracker → `bash -c` | **ЗАДОКУМЕНТИРОВАНО** | явно зафиксирована доверенная граница (код не менялся; изоляция — отдельная задача) |
| **P2/P7/P8** | `eval` и согласованный HIGH блокируются | **БЕЗ ИЗМЕНЕНИЙ** | защита уже работала |
| **R4** | Reviewer на локальной модели вернул HIGH, изменение заблокировано | **БЕЗ ИЗМЕНЕНИЙ** | модель-специфичное наблюдение, кода не требует |
| **R7** | indirect injection через Markdown не изменила поведение | **БЕЗ ИЗМЕНЕНИЙ** | модель-специфичное наблюдение |
| **R8** | direct injection (роль/гарантия/извлечение промпта) отбита | **БЕЗ ИЗМЕНЕНИЙ** | подтверждает укрепление `SYSTEM_PROMPT`; повтор на реальной модели — впереди |

Итого: **4 подтверждённых дефекта кода закрыты**, 1 граница доверия задокументирована, 5 наблюдений
«защита сработала» не требовали изменений.

## 2. Что именно исправлено

### P1/P6 — целостность вердикта security-шага
Файл: `scripts/exec_loop_lib.py`.
- `parse_security_verdict`: `overall = max(top-level overall, severity всех findings)` — противоречивый
  JSON больше не даёт ложный `clean`.
- `merge_security_verdicts`: при объединении учитывается максимум severity находок каждого вердикта, а
  не только их верхнеуровневые `overall`.
- `SecurityVerdict.effective_overall` + `blocks`: страховка на уровне объекта — вердикт блокирует,
  если любая находка `high`/`critical`, даже если поле `overall` ниже.
- Тесты: `test_parse_contradictory_overall_cannot_hide_high_finding`,
  `test_verdict_blocks_by_finding_severity_even_if_overall_clean`,
  `test_merge_uses_finding_severity_not_only_top_level`.

### P4/P9/R5 — вложенный Base64
Файл: `scripts/llm_gateway/detectors.py`.
- `_scan_base64` декодирует токен до 2 уровней (`_B64_MAX_DEPTH = 2`) с пределом размера
  (`_B64_MAX_TOKEN`); второй уровень даёт тип `double_base64_<rule>`.
- `scripts/security_rules.py::_secret_base_type` понимает префикс `double_base64_`.
- Тесты: `test_detector_catches_double_base64_secret`, `test_single_base64_is_not_reported_as_double`,
  HTTP-кейс `double_base64` в `test_input_guard_blocks_secrets` (block → 400).

### R6 — доверенная граница reverse proxy
Файлы: `scripts/llm_gateway/app.py`, `scripts/llm_gateway/settings.py`, `.env.example`.
- `_client_ip(request, trusted_proxies)`: `X-Forwarded-For` используется только если непосредственный
  peer входит в `LLM_GATEWAY_TRUSTED_PROXIES` (точный IP или CIDR); иначе ключ — адрес прямого клиента.
- Новая настройка `LLM_GATEWAY_TRUSTED_PROXIES` (пусто по умолчанию = XFF не доверяем).
- Тесты: `test_client_ip_ignores_untrusted_xff`, `test_client_ip_honors_xff_from_trusted_proxy`,
  `test_client_ip_honors_cidr_trusted_proxy`, `test_rate_limit_ignores_rotated_xff_by_default`
  (три запроса с разными XFF → `200, 200, 429`).

### P3 — PII в `print`/алиасах
Файл: `scripts/security_rules.py`.
- `_LOG_CALL` теперь матчит и `print(...)`, не только `logger.*`/`logging.*`.
- Тесты: `test_pii_in_print_is_flagged`, `test_print_without_pii_is_not_flagged` (контроль на ложное
  срабатывание).

### P5 — граница доверия task tracker
Файл: `docs/security/README.md`.
- Явно зафиксировано: команды из `Проверка:` исполняются раннером через `bash -c`, трекер ведёт и
  меняет только оператор; изоляция исполнения (sandbox) — отдельная задача вне scope пайплайна.
- Кода не меняли: атакующий сам пометил это как архитектурную проверку, а не подтверждённый RCE.

## 3. Результат после фикса (доказательства)

Inline-прогон на тех же сценариях (commit `381cf48`):

```text
P6 parsed_overall=high effective=high blocks=True
P4 double_b64_types=['double_base64_openai_api_key']
P4 mask_changed=True secret_left=False
R6 untrusted_xff_ignored=True
R6 trusted_xff_used=True
P3 print_overall=medium categories=['pii_in_logs']
```

Целевые повторные тесты из раздела 11 отчёта атакующего:

```text
7 passed (test_parse_contradictory..., test_verdict_blocks..., test_merge_uses...,
          test_detector_catches_double_base64_secret, test_rate_limit_ignores_rotated_xff...,
          test_pii_in_print_is_flagged, test_print_without_pii_is_not_flagged)
```

Полный прогон проекта:

```text
pytest tests -q          → 2194 passed
make security-check      → 117 passed
ruff check <изменённые>  → All checks passed
compileall -q scripts    → OK
```

## 4. Что не исправлено и почему (честно)

- **R4/R7/R8 — поведение моделей.** Это наблюдения «защита сработала» на локальной `qwen2.5:7b`
  атакующего, а не дефекты кода. Повтор на реальном окружении (`gpt-oss:20b`, OpenCode loop,
  Telegram) остаётся за взаимным согласованием — кодом это не закрывается.
- **P2/P7/P8 — защиты, которые уже работали.** Изменений не требовали.
- **P5 — только документация.** Если трекер когда-либо станет доступен недоверенному лицу, нужен
  отдельный sandbox/allow-list исполнения — это архитектурное решение, а не правка текущего пайплайна.
- **Живой повтор удара по развёртыванию партнёра не выполнялся.** Проверка — на юнит/интеграционных
  тестах и inline-сценариях, воспроизводящих payload'ы отчёта (синтетический маркер, а не реальный
  секрет).
- **Серьёзность R6 зависит от деплоя.** Исправление безопасно по умолчанию (XFF не доверяем); если
  перед gateway стоит reverse proxy, оператор должен явно указать его в
  `LLM_GATEWAY_TRUSTED_PROXIES`, и ingress обязан удалять/перезаписывать клиентский XFF.

## 5. Остаточные риски и следующие шаги

1. Живой повторный прогон тех же payload'ов на реальном окружении (модель владельца, gateway,
   OpenCode loop) — по взаимному согласованию.
2. Проверить в деплое: gateway за reverse proxy → задать `LLM_GATEWAY_TRUSTED_PROXIES`; без прокси
   оставить пусто.
3. Добавить mask-режим HTTP-кейс для двойного Base64 в постоянный набор (сейчас подтверждён inline и
   общим путём `scan_text`).
4. Расширять `security_rules` (AST-распознавание `print`/алиасов) — как рекомендует атакующий, с
   контролем ложных срабатываний.

## 6. Артефакты

- Commit: `381cf48` (10 файлов, +194/−17).
- Изменённые файлы: `scripts/exec_loop_lib.py`, `scripts/security_rules.py`,
  `scripts/llm_gateway/{app,detectors,settings}.py`, `tests/test_security_step_lib.py`,
  `tests/test_security_rules.py`, `tests/test_llm_gateway.py`, `.env.example`,
  `docs/security/README.md`.
- Отчёты атакующего: `docs/security/red-team-attacker-*.md`.
