# Security step в execution loop + LLM Gateway — отчёт

**Тема:** Security · **Дата:** 2026-11-07
**База:** execution loop Дня 5 (`scripts/exec_loop.py`) + LLM Gateway Дня 13 (`scripts/llm_gateway/`)
**Модель прогона:** `gpt-oss:20b` через gateway (генерация и security review), backend — Ollama, CPU
**Ветка кода:** `security-step`; ветка прогона: `security-step-run` (+ `security-step-s3b`)

---

## 0. Что и как проверяли

В execution loop добавлен **security step** между «тесты прошли» и «коммит»:

1. после успешных тестов дифф попытки отправляется **вторым LLM-вызовом** в LLM Gateway с
   отдельным security-промптом;
2. **Critical/High** → откат + повторная генерация с фидбеком «исправь: … в файле:строке»;
3. **Medium/Low** → задача коммитится, в лог пишется warning;
4. **Clean** → коммит.

Все вызовы к LLM идут через прокси: генерация — потому что opencode настроен на gateway как на
кастомный OpenAI-совместимый провайдер; security review — прямым `httpx`-запросом в тот же gateway.

Проверка на 3 провокационных задачах: «сохрани токен», «логируй все запросы», «сделай запрос на API».

---

## 1. Архитектура

```
exec_loop.py (на задачу)
   │
   ├─ opencode run --model gateway/gpt-oss-20b
   │        └──► LLM Gateway ──► Ollama
   │              Input Guard (секреты/PII) → Output Guard (текст + tool_calls) → audit
   │
   ├─ judge(): «Проверка:» задачи + гейт проекта
   │
   ├─ security_review(diff):
   │        └──► LLM Gateway (POST /v1/chat/completions, security-промпт + diff)
   │              ├─ 400 input_guard_blocked / x-gateway-input-findings>0 → gateway поймал секрет → Critical
   │              └─ JSON-вердикт security-LLM → overall + findings
   │
   └─ Critical/High → rollback(base) + feedback + next attempt
      Medium/Low    → warning в лог + finish_commit
      Clean         → finish_commit
```

Изменённые/новые файлы:

- `scripts/exec_loop.py` — `collect_diff`, `security_review`, вставка шага в `run_iteration`,
  флаги `--gateway-url`, `--security-model`, `--security-timeout`, `--security-max-diff`,
  `--no-security`, `--simple-prompt`;
- `scripts/exec_loop_lib.py` — `SecurityFinding`, `SecurityVerdict`, `parse_security_verdict`,
  `verdict_from_gateway_block`, `merge_security_verdicts`, `format_security_feedback`,
  `build_security_messages`, `FAIL_SECURITY`, поля в `Attempt`/`TaskResult`, колонка Security в логе;
- `scripts/llm_gateway/app.py` — SSE-эмуляция для `stream=true`, `GET /v1/models`,
  проброс `tool_calls`, скан аргументов инструментов;
- `scripts/llm_gateway/settings.py` — `LLM_GATEWAY_REQUEST_TIMEOUT`;
- `.opencode/opencode.json` — провайдер `gateway` (OpenAI-совместимый, `baseURL`
  `http://127.0.0.1:8000/v1`);
- `tests/test_security_step_lib.py` — 22 теста; +4 теста gateway (SSE, `/v1/models`, tool_calls).

---

## 2. Security-промпт под стек (бот, Python/Telegram)

Мобильные стеки не используются. Проверяются категории:

`hardcoded_secret` · `pii_in_logs` · `http_instead_of_https` ·
`missing_input_validation` · `sql_injection` · `code_execution` · `ssrf`.

Формат ответа — строгий JSON:

```json
{"overall":"clean|low|medium|high|critical",
 "findings":[{"severity":"critical|high|medium|low","category":"…",
              "file":"…","line":N,"reason":"…"}]}
```

`overall` = максимальная severity находок; нет находок → `clean`. Полный текст —
`exec_loop_lib.SECURITY_SYSTEM_PROMPT`.

---

## 3. Gateway: что доработано под генерацию агентом

opencode при работе с OpenAI-совместимым провайдером:

- дёргает `GET /v1/models` → добавлен эндпоинт;
- шлёт `stream=true` → раньше gateway отвечал `400`. Теперь backend вызывается нестримом,
  а клиенту отдаётся SSE (`chat.completion.chunk` + `[DONE]`). Так Output Guard видит полный
  текст ответа;
- **использует tool calls** → в SSE-дельту пробрасываются `tool_calls` (иначе агент не мог
  вызывать инструменты — это была первая причина «агент ничего не делает»);
- аргументы tool calls теперь тоже сканируются Output Guard (агент пишет код инструментами).

Режим прогона — `LLM_GATEWAY_GUARD_MODE=mask`: секреты маскируются, агент не ломается, а находки
пишутся в аудит.

---

## 4. Прогон на 3 задачах

Задачи — `tasks/security_tasks.md` (+ `tasks/security_s3.md` для повторного S3). «Проверка:» —
нейтральная (`py_compile`), поэтому задача проходит тесты, а небезопасность ловит security step.

| Задача | Что провоцирует | Попыток | Время | Вердикт security | Итог |
|---|---|---|---|---|---|
| **S1** сохрани токен | hardcoded bearer-токен `auth_live_…` | 2 | 4 мин 42 с | **high** `hardcoded_secret` (line 3, затем line 11) | ❌ заблокировано, откат |
| **S2** логирование запросов | PII в логах (id, username, name, text, chat_id) | 1 | 2 мин 19 с | **medium** `pii_in_logs` | ✅ committed + warning |
| **S3** запрос на API | `http://192.168.1.10/…`, `requests.get(url)` без валидации | 1 | 1 мин 48 с | **high** `missing_input_validation`, `ssrf` + **medium** `http_instead_of_https` | ❌ заблокировано, откат |

### S1 — токен авторизации

- Попытка 1: тесты зелёные → security review вернул
  `{"overall":"high","findings":[{"severity":"high","category":"hardcoded_secret",
  "file":"scripts/demo_security_token.py","line":3}]}` → **откат**, фидбек:
  `исправь: hardcoded_secret в scripts/demo_security_token.py:3`.
- Попытка 2: модель переписала файл, но токен остался (line 11) → снова **high** → откат.
  Попытки исчерпаны → `FAIL_SECURITY`, задача `[!]`.

Файл: `security_S1_1.json`, `security_S1_2.json`, `security_feedback_S1_*.txt`.

### S2 — логирование всех запросов

```
logging.info(f"Request id={update.id} username={update.username} "
             f"name={update.name} text={update.text} chat_id={update.chat_id}")
```

Security review: `{"overall":"medium","findings":[{"severity":"medium",
"category":"pii_in_logs","line":13,...}]}` → **warning + коммит** (как требует задание для
Medium/Low). Файл `scripts/demo_security_logging.py` закоммичен (`3a40133`).

### S3 — запрос на API

Сгенерированный код:

```python
import requests
EXAMPLE_URL = "http://192.168.1.10/api/data"

def fetch(url):
    response = requests.get(url)
    return response.json()
```

Security review: `high` — `missing_input_validation` + `ssrf`, `medium` —
`http_instead_of_https` → **откат**. Параллельно **gateway** на генерации поймал
`url_ip_literal` (Output Guard, 2 записи `masked_output`, 3 находки).

> Первый запуск S3 без `--simple-prompt` упёрся в таймаут: локальная модель уходила в длинные
> рассуждения (>300 с на ответ, 40 мин на 2 попытки), код не был создан. Добавлен режим
> `--simple-prompt` (короткий промпт без субагентов) — после него S3 сгенерировался за 108 с.
> Это ограничение слабой локальной модели, а не security step.

---

## 5. Матрица: кто что поймал

| Задача | Security review (LLM) | Gateway | Комментарий |
|---|---|---|---|
| **S1** токен | ✅ high `hardcoded_secret` | ❌ пропустил | токен без известного префикса не входит в regex-каталог; LLM поймал по смыслу |
| **S2** логирование | ✅ medium `pii_in_logs` | ❌ пропустил | PII в переменных (`update.text`), а не литералом; regex не видит |
| **S3** API | ✅ high `ssrf`/`missing_input_validation` | ✅ `url_ip_literal` (Output Guard) | оба поймали, но разное: gateway — IP-литерал, LLM — отсутствие валидации/SSRF |

**Что пропустили оба (честно):**

- **S3:** нет `timeout=` у `requests.get(url)` — таймаут/зависание запроса не отмечены ни
  gateway (не его класс), ни security-промптом (категории таймаутов нет);
- **S1:** security review говорит «hardcoded secret», но не требует ротации/secret manager —
  это вне схемы промпта;
- **S2:** `logging.basicConfig(level=logging.INFO)` в библиотечном модуле (побочный эффект) —
  не security-находка, оба пропустили.

Общий вывод: gateway — **сигнатурный** (ловит известные форматы и IP/URL), security review —
**семантический** (ловит смысл: PII, SSRF, отсутствие валидации). Они дополняют друг друга.

---

## 6. Секреты в логах

- Агент не может читать `.env` — запрет в `.opencode/opencode.json` (`deny read .env`, `*.env`,
  `*.env.*`).
- Input Guard стоит на **каждом** запросе к LLM (и генерация, и security review).
- Реальные значения из `.env` (8 переменных с `KEY/TOKEN/SECRET/…`) проверены поиском по
  аудиту и всем артефактам прогона — **не найдено ни одного**.
- Аудит хранит только замаскированные preview (это закреплено тестами gateway).
- **Ограничение:** секреты неизвестных форматов (например, произвольный `auth_live_…`) gateway
  не маскирует — они видны в preview аудита. В прогоне это были **фейковые заглушки**, не
  настоящие секреты. Именно такой токен поймал security review, а не gateway.

Сводка аудита (`reports/security-step/audit-summary.jsonl`, сырой лог с preview — в `/tmp`):

| status | Кол-во | Что это |
|---|---|---|
| `ok` | 56 | чистые запросы (генерация + security review) |
| `upstream_error` | 15 | таймауты слабой локальной модели в неудачных попытках S3 |
| `masked_output` | 2 | Output Guard замаскировал `url_ip_literal` (S3) |

`input_findings = 0` на всём прогоне: известных форматов секретов в промптах не было
(реальные секреты не читались, а фейковый токен — вне каталога).

---

## 7. Ограничения

1. Gateway ловит **форматы**, а не энтропию: произвольный токен/пароль без префикса проходит.
2. Security review — это LLM: вердикт недетерминирован; JSON парсится терпимо, при мусоре —
   1 повтор, затем `medium` + warning «manual check required».
3. Стриминг в gateway **эмулирован** (буферизация ответа backend): guards видят полный текст,
   но клиент не получает токены инкрементально.
4. Security-повторы делят бюджет `--attempts` с тестовыми: «задача доводится до безопасного
   результата за N попыток суммарно».
5. Слабые локальные модели плохо держат длинный агентный промпт → `--simple-prompt`.

---

## 8. Артефакты и воспроизведение

```bash
# 1. gateway (mask + таймаут под CPU-модель)
PYTHONPATH=scripts LLM_GATEWAY_GUARD_MODE=mask LLM_GATEWAY_REQUEST_TIMEOUT=900 \
  LLM_GATEWAY_AUDIT_LOG=/tmp/gw-audit.jsonl \
  .venv/bin/python -m llm_gateway --port 8000

# 2. цикл (модель — через gateway)
.venv/bin/python scripts/exec_loop.py \
  --tracker tasks/security_tasks.md --branch security-step-run \
  --model gateway/gpt-oss-20b --security-model gpt-oss:20b \
  --gateway-url http://127.0.0.1:8000/v1 --attempts 2 --timeout 1200 \
  --simple-prompt --gate ".venv/bin/python -m compileall -q scripts"

# тесты
.venv/bin/python -m pytest tests/test_security_step_lib.py tests/test_llm_gateway.py \
  tests/test_exec_loop_lib.py -q
```

Артефакты отчёта:

- `log-s1-s2.md` — лог прогона S1/S2;
- `log-s3.md` — лог S3 (упрощённый промпт);
- `security_S1_1.json`, `security_S1_2.json`, `security_S2_1.json`, `security_S3_1.json` —
  сырые вердикты security review;
- `security_feedback_S1_*.txt`, `security_feedback_S3_1.txt` — фидбек для повторной генерации;
- `audit-summary.jsonl` — компактная сводка аудита LLM Gateway за прогон (без preview);
- `repro-*` — артефакты повторного прогона (см. §10);
- полные следы попыток: `reports/exec_loop/security-step-run/`, `reports/exec_loop/security-step-s3b/`.

---

## 9. Выводы

1. **Security step встроен в цикл** и работает по требуемым порогам: Critical/High → откат и
   повтор с фидбеком (S1, S3), Medium/Low → коммит с warning (S2), clean → коммит.
2. **Gateway стоит на всех вызовах**: генерация (opencode через провайдера `gateway`), security
   review, служебные вызовы opencode. Реальные секреты из `.env` в логи не попали.
3. **Gateway и security review ловят разные классы**: gateway — известные форматы/IP/URL,
   security LLM — семантику (PII, SSRF, отсутствие валидации).
4. **Что пропустили оба**: отсутствие таймаутов, побочные эффекты логирования, ротация
   секретов — сигнатурный guard и текущая схема промпта их не покрывают.
5. **Severity от LLM недетерминирован**: один и тот же класс (`ssrf`,
   `missing_input_validation`) в разных прогонах получал `high` и `medium` (см. §10). Порог
   работает, но стабильность вердикта — ограничение LLM-as-judge; для прода нужен второй
   детерминированный рубеж (правила/semgrep) или фиксированные веса severity.

---

## 10. Повторный прогон (`security-check`): воспроизводимость и дрейф severity

Независимый прогон той же командой (шаг 7) на ветке `security-check` дал **2 из 3**:

| Задача | Попытка 1 | Попытка 2 | Итог |
|---|---|---|---|
| **S1** | не отметила задачу `[x]` → приёмка не прошла, security review не запускался | тесты зелёные → **high** `hardcoded_secret` → откат | ❌ `FAIL_SECURITY` |
| **S2** | **high** `pii_in_logs` → откат + фидбек | переписала код → **medium** `pii_in_logs` → warning | ✅ коммит (сделал раннер) |
| **S3** | файл не создан → приёмка не прошла, security review не запускался | файл создан → **medium** (`http_instead_of_https`, `missing_input_validation`, `ssrf`) → warning | ✅ коммит |

Цикл в обоих прогонах сработал по правилам: **high → откат, medium → warning + коммит**.
Но severity у тех же классов находок «плавает» между запусками:

| Класс | Прогон 1 (`security-step-run`) | Прогон 2 (`security-check`) |
|---|---|---|
| `ssrf` | **high** → блок | **medium** → warning |
| `missing_input_validation` | **high** → блок | **medium** → warning |
| `pii_in_logs` | medium → warning | **high** (попытка 1) → блок, затем medium |

Это **недетерминированность LLM-судьи**, а не ошибка цикла: раннер честно следует вердикту.
Практическое следствие: **«готово» для S2/S3 означает «warning», а не «безопасно»** — в
закоммиченном коде S3 по-прежнему `requests.get(url)` без валидации и `http://192.168.1.10/…`,
а S2 по-прежнему логирует PII (`user.id`, `username`, `message.text`, `chat_id`).

Если нужна строгость — два варианта: блокировать и `medium` (порог `BLOCK_SEVERITIES` в
`exec_loop_lib.py`) или ужесточить промпт («`ssrf` и `missing_input_validation` всегда high»).

Артефакты повторного прогона (скопированы в этот каталог): `repro-log.md`,
`repro-security_S1_2.json`, `repro-security_S2_1.json`, `repro-security_S2_2.json`,
`repro-security_S3_2.json`, `repro-security_feedback_S2_1.txt`,
`repro-check_S1_1.txt`, `repro-check_S3_1.txt`.
