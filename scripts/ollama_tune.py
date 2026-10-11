"""Одноразовая оснастка подбора параметров локальной модели Ollama под задачу бота (не часть
бота; спека — `openspec/changes/add-ollama-tuning-script/`): сравнение «до/после» по качеству,
скорости и ресурсам в терминале. Код бота (`src/`) не меняется, `.env`, `MAIN_MODEL` и рабочий
`SYSTEM_PROMPT` скрипт не трогает; найденные значения оператор переносит сам.

Запуск из корня проекта (нужны .env бота, запущенный Ollama с базовыми моделями и индекс RAG):
    python scripts/ollama_tune.py run                      # make tune
    python scripts/ollama_tune.py run --no-judge --repeats 1
    python scripts/ollama_tune.py run --only ctx8k think-low --questions 3
    python scripts/ollama_tune.py report                   # make tune-report: таблица отчёта
    python scripts/ollama_tune.py report 3                 # ответы всех вариантов на вопрос 3

Рекомендуемый порядок (ответ на CPU занимает 30–120 с, поэтому сетку делят на этапы):
  1) скорость и ресурсы без судьи: `--no-judge --repeats 1`, файл с сеткой контекста,
     рассуждения, квантований; 2) качество — только финалисты, с судьёй и 2–3 повторами;
     3) варианты промпта.
Перед подбором проверьте окружение контейнера Ollama: лимиты `--cpus`/памяти docker, потоки,
`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION` — лимит контейнера занижает скорость сильнее любой
настройки модели. Параметры сервера скрипт менять не может: опишите их в поле `environment` файла
вариантов, и они попадут в отчёт.

Файл вариантов (`data/ollama_tune/variants.json`, не коммитится):
    {"environment": "OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_NUM_PARALLEL=1",
     "variants": [
       {"name": "base",      "base": "qwen3:30b-a3b", "baseline": true},
       {"name": "ctx8k",     "base": "qwen3:30b-a3b", "params": {"num_ctx": 8192}},
       {"name": "think-low", "base": "qwen3:30b-a3b", "reasoning_effort": "low"},
       {"name": "prompt-v2", "base": "qwen3:30b-a3b", "system_prompt": "…"}]}
`params` — параметры Modelfile (временный тег создаётся и удаляется скриптом; `num_predict` там
не работает — бот шлёт `max_tokens`, и он перекрывает его, поэтому лимит ответа — поле
`max_output_tokens`, аналог MAX_OUTPUT_TOKENS в `.env`);
`reasoning_effort` (low|medium|high) — параметр ТОЛЬКО запроса: в боте не переносится без
доработки (вариант помечается в отчёте); `system_prompt` — экспериментальный промпт одного варианта.
Вопросы — `data/ollama_tune/questions.json` (формат `RAG_COMPARE_QUESTIONS_FILE`), при его
отсутствии — набор `RAG_COMPARE_QUESTIONS_FILE`.

Условия прогона как у `/research_rag_models`: поиск по индексу ОДИН раз на вопрос, одинаковые
фрагменты всем вариантам и повторам, переписывание вопроса и рыночные данные выключены, память
каждого ответа изолирована, порядок «повтор → варианты по очереди». Перед каждым обращением вариант
загружается отдельным пустым вызовом (время загрузки меряется отдельно и в скорость не входит).
Судье (`RAG_MODELS_JUDGE`) уходят вопросы, ответы и тексты цитируемых фрагментов корпуса; при
облачном судье они покидают машину — строка об этом есть в таблице. Отчёт с ответами пишется в
`data/ollama_tune/reports/` (вне git). Код выхода: 0 — готово; 1 — прогон прерван ошибкой или
автоостановкой; 2 — ошибка конфигурации (формат файлов, нет модели на сервере, нет индекса);
130 — остановлено оператором (частичный отчёт сохранён).

API Ollama, на которое опирается скрипт (проверено на 0.21.0, docker, CPU):
    POST /api/create  {"model": "tunetmp-…", "from": "qwen3:30b-a3b",
                       "parameters": {"num_ctx": 8192}, "stream": false}
                       -> {"status": "success"}
    POST /api/generate {"model": "tunetmp-…", "keep_alive": -1}  # загрузка
                       -> {"done": true, "done_reason": "load", "response": ""}
    GET  /api/ps -> {"models": [{"name": "tunetmp-…:latest", "size": 19045644288,
                                 "size_vram": 0, "context_length": 8192, ...}]}
    POST /api/generate {"model": …, "keep_alive": 0}             # выгрузка
                       -> {"done_reason": "unload"}
    DELETE /api/delete {"model": …}
Порядок очистки важен: удаление тега ЗАГРУЖЕННОЙ модели оставляет её в памяти, и выгрузить её по
имени уже нельзя — сначала выгрузка, потом удаление.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama_tune_lib as lib  # noqa: E402, I001
from research import rag_compare_eval as ev  # noqa: E402
from research import rag_models_eval as rme  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "ollama_tune")
DEFAULT_VARIANTS = os.path.join(DATA_DIR, "variants.json")
DEFAULT_QUESTIONS = os.path.join(DATA_DIR, "questions.json")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "reports")
DEFAULT_REPEATS = 2

EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_INTERRUPTED = 0, 1, 2, 130

LOAD_TIMEOUT_SECONDS = 900.0
ADMIN_TIMEOUT_SECONDS = 120.0


class ConfigError(Exception):
    """Ошибка конфигурации прогона: сообщение оператору, код выхода 2."""


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Ollama: теги, загрузка, память
# --------------------------------------------------------------------------- #


class OllamaAdmin:
    """Нативное API Ollama (не `/v1`): теги, загрузка/выгрузка, `/api/ps`."""

    def __init__(self, base_url: str) -> None:
        import httpx

        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._http = httpx.Client(base_url=root, timeout=ADMIN_TIMEOUT_SECONDS)

    def list_models(self) -> list[str]:
        response = self._http.get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def create(self, tag: str, base: str, params: dict) -> None:
        response = self._http.post(
            "/api/create", json={"model": tag, "from": base, "parameters": params, "stream": False}
        )
        response.raise_for_status()

    def load(self, model: str) -> None:
        response = self._http.post(
            "/api/generate",
            json={"model": model, "keep_alive": -1},
            timeout=LOAD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    def unload(self, model: str) -> None:
        response = self._http.post("/api/generate", json={"model": model, "keep_alive": 0})
        response.raise_for_status()

    def delete(self, model: str) -> None:
        response = self._http.request("DELETE", "/api/delete", json={"model": model})
        response.raise_for_status()

    def ps(self) -> dict:
        response = self._http.get("/api/ps")
        response.raise_for_status()
        return response.json()


def has_model(available: list[str], name: str) -> bool:
    """Модель есть на сервере: имя без тега Ollama понимает как `:latest`."""
    return name in available or (":" not in name and f"{name}:latest" in available)


def drop_model(admin: OllamaAdmin, name: str) -> None:
    """Выгружает и удаляет тег (порядок важен, см. докстринг); ошибки очистку не прерывают."""
    for step in (admin.unload, admin.delete):
        try:
            step(name)
        except Exception as exc:  # noqa: BLE001 — очистка best-effort, сообщаем и идём дальше
            print(f"  (очистка {name}: {type(exc).__name__}: {exc})", flush=True)


def cleanup_leftovers(admin: OllamaAdmin) -> int:
    """Удаляет временные теги прежних прерванных прогонов (по общему префиксу)."""
    leftovers = [m for m in admin.list_models() if lib.is_temp_tag(m)]
    for name in leftovers:
        drop_model(admin, name)
    return len(leftovers)


# --------------------------------------------------------------------------- #
# Клиент-обёртка: параметр уровня запроса и запись usage/finish_reason
# --------------------------------------------------------------------------- #


class RecordingClient:
    """Тонкая обёртка клиента OpenAI: добавляет `reasoning_effort` варианта в каждый вызов и пишет
    `usage` и `finish_reason` каждого обращения в `calls`. Один экземпляр — на один ответ."""

    def __init__(self, client, reasoning_effort: str | None = None) -> None:
        self._client = client
        self._reasoning_effort = reasoning_effort
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self._reasoning_effort:
            extra = dict(kwargs.get("extra_body") or {})
            extra["reasoning_effort"] = self._reasoning_effort
            kwargs["extra_body"] = extra
        response = self._client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        finish = getattr(choices[0], "finish_reason", None) if choices else None
        self.calls.append(lib.call_record(getattr(response, "usage", None), finish))
        return response


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


class Runtime:
    """Состояние прогона: администрирование Ollama, теги вариантов, текущая загруженная модель и
    ресурсы по тегам."""

    def __init__(self, admin: OllamaAdmin, variants: list[lib.Variant], run_id: str) -> None:
        self.admin = admin
        self.variants = variants
        self.tags = {
            v.name: (lib.temp_tag(run_id, v.name) if v.needs_tag else v.base) for v in variants
        }
        self.created: list[str] = []
        self.loaded: str | None = None
        self.by_tag: dict[str, dict] = {}

    def create_tags(self) -> None:
        for v in self.variants:
            if v.needs_tag:
                self.admin.create(self.tags[v.name], v.base, v.params)
                self.created.append(self.tags[v.name])

    def ensure_loaded(self, variant: lib.Variant) -> tuple[float | None, str | None]:
        """Загружает модель варианта (прежнюю выгружает). Возвращает (секунды загрузки или None,
        причина сбоя). Уже загруженная модель не перезагружается."""
        tag = self.tags[variant.name]
        if self.loaded == tag:
            return None, None
        info = self.by_tag.setdefault(
            tag, {"load_seconds": [], "size": None, "size_vram": None, "context_length": None}
        )
        try:
            if self.loaded is not None:
                self.admin.unload(self.loaded)
                self.loaded = None
            started = time.monotonic()
            self.admin.load(tag)
            seconds = time.monotonic() - started
            entry = lib.parse_ps_entry(self.admin.ps(), tag)
        except Exception as exc:  # noqa: BLE001 — сбой загрузки — сбой обращений варианта
            info["error"] = f"{type(exc).__name__}: {exc}"
            return None, f"модель не загрузилась: {type(exc).__name__}: {exc}"
        self.loaded = tag
        info["load_seconds"].append(round(seconds, 1))
        if entry:
            info.update(entry)
        return seconds, None

    def resources(self) -> dict[str, dict]:
        return {v.name: dict(self.by_tag.get(self.tags[v.name], {})) for v in self.variants}

    def cleanup(self) -> None:
        if self.loaded is not None:
            try:
                self.admin.unload(self.loaded)
            except Exception as exc:  # noqa: BLE001
                print(f"  (выгрузка {self.loaded}: {type(exc).__name__}: {exc})", flush=True)
            self.loaded = None
        for tag in self.created:
            drop_model(self.admin, tag)
        self.created = []


def _build_result(answer, elapsed: float, rag_context):
    """ModeResult из ответа SmartAgent (то же отображение, что у rag_compare.run_mode)."""
    return ev.ModeResult(
        answer=answer.text,
        elapsed=elapsed,
        prompt_tokens=answer.context_tokens,
        completion_tokens=answer.response_tokens,
        llm_calls=answer.llm_calls,
        warnings=list(answer.warnings),
        rag_sources=[
            {"title": s.title, "chunk_index": s.chunk_index, "score": round(s.score, 3),
             "chunk_id": s.chunk_id}
            for s in answer.rag_sources
        ],
        search_failed=rag_context.FAILURE_WARNING in answer.warnings,
        citations=[
            {"quote": c.quote, "title": c.source.title, "chunk_index": c.source.chunk_index,
             "chunk_id": c.source.chunk_id}
            for c in answer.citations
        ],
        citations_written=answer.citations_written,
        citations_unverified=answer.citations_unverified,
        abstained=answer.abstained,
        fragments=ev.unique_fragments(answer.citations),
    )


class TuneBudget:
    """Бюджет времени вопроса: как `QuestionBudget`, но загрузка модели его не расходует."""

    def __init__(self, rc, seconds: float, call_limit: float) -> None:
        self._inner = rc.QuestionBudget(seconds)
        self._extra = 0.0
        self._call_limit = call_limit

    def extend(self, seconds: float) -> None:
        self._extra += seconds

    def remaining(self) -> float:
        return self._inner.remaining() + self._extra

    def expired(self) -> bool:
        return self.remaining() <= 0

    def call_timeout(self) -> float:
        return max(min(self._call_limit, self.remaining()), 0.001)


def answer_variant(question, variant, tag, materials, budget, ctx) -> tuple[object, list[dict]]:
    """Один ответ варианта по готовым материалам: свежий изолированный SmartAgent с клиентом-
    обёрткой. Возвращает (ModeResult, записи обращений). Сбой провайдера — причина в результате."""
    from agents import rag_context
    from agents.smart_agent import LAYER_RAG, SmartAgent

    recorder = RecordingClient(ctx.answer_client, variant.reasoning_effort)
    limit = budget.call_timeout()
    kwargs = {}
    if variant.system_prompt is not None:
        kwargs["system_prompt"] = variant.system_prompt  # без варианта — рабочий SYSTEM_PROMPT
    if variant.max_output_tokens is not None:
        kwargs["max_output_tokens"] = variant.max_output_tokens  # без варианта — MAX_OUTPUT_TOKENS
    with tempfile.TemporaryDirectory(prefix="ollama_tune_") as tmp:
        agent = SmartAgent(
            "ollama_tune",
            client=recorder,
            model=tag,
            timeout=limit,
            memory_dir=tmp,
            mcp_moex_dir="",
            mcp_bybit_dir="",
            cbr_enabled=False,
            rewrite_backend=None,
            **kwargs,
        )
        agent.create_profile("eval", {})
        agent.set_layer_enabled(LAYER_RAG, True)
        started = time.monotonic()
        try:
            answer = ev.call_with_deadline(
                lambda: agent.ask(question.question, materials=materials), limit
            )
        except ev.DeadlineExceeded:
            return (
                ev.ModeResult(
                    error=f"модель не ответила за {limit:.0f} с (превышен предел ожидания)",
                    elapsed=time.monotonic() - started,
                ),
                recorder.calls,
            )
        except Exception as exc:  # noqa: BLE001 — сбой одного ответа не роняет прогон
            return (
                ev.ModeResult(
                    error=ctx.rc._error_reason(exc, ctx.local_backend),  # noqa: SLF001
                    elapsed=time.monotonic() - started,
                ),
                recorder.calls,
            )
        elapsed = time.monotonic() - started
    return _build_result(answer, elapsed, rag_context), recorder.calls


def judge_answer(question, mode, budget, ctx) -> None:
    """Оценка ответа судьёй по тем же правилам, что у `/research_rag_models`. Сбой судьи —
    `judge_error`, а не сбой генерации. Судья не получает имя варианта и номер повтора."""
    if ctx.judge_backend is None or mode.answer is None or mode.error:
        return
    if question.expect_abstain:
        return  # зачёт — abstained, оценки по фактам нет
    if budget.expired():
        mode.judge_error = ev.QUESTION_TIMEOUT_REASON
    else:
        mode.verdict, mode.judge_error = ctx.rc.judge(
            question, mode.answer, budget, ctx.judge_backend
        )
    if mode.citations:
        if budget.expired():
            mode.citation_judge_error = ev.QUESTION_TIMEOUT_REASON
        else:
            mode.citation_verdict, mode.citation_judge_error = ctx.rc.judge_citations(
                question,
                mode.answer,
                [c["quote"] for c in mode.citations],
                mode.fragments,
                budget,
                ctx.judge_backend,
            )


def evaluate_question(
    run, calls, runtime, repeats, budget_seconds, cold_seen, ctx, progress
) -> None:
    """Вопрос целиком: общий поиск, затем «повтор → вариант → загрузка → ответ → судья». Результаты
    пишутся в переданные `run` (QuestionRun) и `calls` (записи обращений по вариантам) по мере
    готовности — при остановке посреди вопроса частичный отчёт не теряет уже полученные ответы."""
    question = run.question
    budget = TuneBudget(ctx.rc, budget_seconds, ctx.request_timeout)
    materials, sources, seconds, error = ctx.rag_models.search_once(question, budget)
    run.search_seconds = seconds
    if materials is None:
        run.search_error = error or ev.SEARCH_FAILED_REASON
        return
    run.search_sources = sources

    for repeat in range(1, repeats + 1):
        for variant in runtime.variants:
            tag = runtime.tags[variant.name]
            progress(f"повтор {repeat} из {repeats} · {variant.name} · загрузка")
            load_seconds, load_error = runtime.ensure_loaded(variant)
            if load_seconds:
                budget.extend(load_seconds)
            cold = variant.name not in cold_seen
            if load_error:
                result, recorded = ev.ModeResult(error=load_error), []
            else:
                progress(f"повтор {repeat} из {repeats} · {variant.name} · ответ")
                result, recorded = answer_variant(question, variant, tag, materials, budget, ctx)
                if ctx.judge_backend is not None and result.answer and not result.error:
                    progress(f"повтор {repeat} из {repeats} · {variant.name} · оценка")
                    judge_answer(question, result, budget, ctx)
            if result.llm_calls > 0 or result.error:
                cold_seen.add(variant.name)
            else:
                cold = False
            attempt = rme.Attempt(variant.name, repeat, result, cold=cold)
            run.attempts.setdefault(variant.name, []).append(attempt)
            calls[variant.name].append(recorded)
            tps = attempt.tokens_per_second
            print(
                f"    {variant.name:<14} повтор {repeat}: "
                + (f"{tps:.1f} ток/с, {result.elapsed:.0f} с" if tps and not result.error
                   else (result.error or "скорость неизвестна"))
                + (" (холодное)" if cold else "")
                + (" · обрезан" if lib.is_truncated(recorded) else ""),
                flush=True,
            )


# --------------------------------------------------------------------------- #
# Отчёт на диске
# --------------------------------------------------------------------------- #


def save_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    name = lib.report_file_name(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path = os.path.join(out_dir, name)
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    return path


def load_report(out_dir: str) -> tuple[dict, str] | None:
    """Последний отчёт каталога. None — отчётов нет; повреждённый файл — ConfigError."""
    try:
        name = lib.latest_report_name(os.listdir(out_dir))
    except FileNotFoundError:
        return None
    if name is None:
        return None
    path = os.path.join(out_dir, name)
    try:
        with open(path, encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"не удалось прочитать отчёт {path}: {exc}") from exc
    if not lib.is_tune_report(report):
        raise ConfigError(f"отчёт {path} другого вида или версии формата")
    return report, path


# --------------------------------------------------------------------------- #
# Режимы
# --------------------------------------------------------------------------- #


def cmd_report(args) -> int:
    try:
        loaded = load_report(args.out_dir)
    except ConfigError as exc:
        print(f"Ошибка: {exc}")
        return EXIT_CONFIG
    if loaded is None:
        print(f"Отчётов нет в {args.out_dir}. Сначала выполните: make tune")
        return EXIT_CONFIG
    report, path = loaded
    report["path"] = path
    if args.number is None:
        print(lib.format_summary(report))
        return EXIT_OK
    text = lib.format_question_detail(report, args.number)
    if text is None:
        numbers = ", ".join(str(n) for n in lib.question_numbers(report)) or "нет"
        print(f"Вопроса с номером {args.number} в отчёте нет. Доступные номера: {numbers}.")
        return EXIT_CONFIG
    print(text)
    return EXIT_OK


def _prepare(args):
    """Всё, что можно проверить до обращений к моделям: файлы вариантов и вопросов, настройки,
    индекс, судья, сервер Ollama и базовые модели. Возвращает контекст прогона."""
    try:
        with open(args.variants, encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"файл вариантов не найден: {args.variants} (формат — в докстринге скрипта)"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"не удалось прочитать файл вариантов: {exc}") from exc
    try:
        variant_set = lib.select_variants(lib.parse_variants(raw), args.only or [])
    except lib.VariantFileError as exc:
        raise ConfigError(f"файл вариантов: {exc}") from exc

    # Дальше нужны .env бота и клиенты провайдеров: импорт только после проверки формата.
    import config
    from providers.deepseek_client import DEEPSEEK_API_KEY, DEEPSEEK_MODEL_FLASH
    from providers.kimi_client import KIMI_API_KEY, KIMI_MODEL_K3
    from providers.ollama_client import ollama_client
    from rag import index_store
    from research import rag_compare as rc
    from research import rag_models

    questions_path = args.questions_file
    if questions_path is None:
        questions_path = (
            DEFAULT_QUESTIONS if os.path.isfile(DEFAULT_QUESTIONS)
            else config.RAG_COMPARE_QUESTIONS_FILE
        )
    try:
        all_questions = ev.load_questions(questions_path)
    except ev.QuestionSetError as exc:
        raise ConfigError(f"набор вопросов {questions_path}: {exc}") from exc
    selection = lib.pick_questions(all_questions, args.questions)
    if selection.error:
        raise ConfigError(f"набор вопросов {questions_path}: {selection.error}")

    if not index_store.index_exists(config.RAG_SMART_AGENT_STRATEGY, config.RAG_INDEX_DIR):
        raise ConfigError(
            f"индекс стратегии «{config.RAG_SMART_AGENT_STRATEGY}» не построен — выполните "
            "`make index`."
        )

    judge_spec = judge_backend = None
    if not args.no_judge:
        defaults = {"deepseek": DEEPSEEK_MODEL_FLASH, "kimi": KIMI_MODEL_K3}
        try:
            judge_spec = rme.parse_judge_spec(config.RAG_MODELS_JUDGE, defaults)
        except rme.ModelSettingError as exc:
            raise ConfigError(f"{exc} Либо запустите с --no-judge.") from exc
        missing = rme.missing_api_keys(
            [judge_spec], {"deepseek": bool(DEEPSEEK_API_KEY), "kimi": bool(KIMI_API_KEY)}
        )
        if missing:
            raise ConfigError("не задан ключ провайдера судьи: " + ", ".join(missing))
        judge_backend = rag_models.build_backend(judge_spec)

    admin = OllamaAdmin(config.OLLAMA_BASE_URL)
    try:
        available = admin.list_models()
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"сервер Ollama недоступен ({config.OLLAMA_BASE_URL}): {type(exc).__name__}: {exc}"
        ) from exc
    absent = sorted({v.base for v in variant_set.variants if not has_model(available, v.base)})
    if absent:
        raise ConfigError(
            "на сервере Ollama нет моделей: " + ", ".join(absent)
            + ". Загрузите их (`ollama pull <имя>`) — скрипт моделей не загружает."
        )

    local_backend = rc.LlmBackend(ollama_client.with_options(max_retries=0), "", "ollama")
    return SimpleNamespace(
        config=config,
        rc=rc,
        rag_models=rag_models,
        variant_set=variant_set,
        selection=selection,
        questions_path=questions_path,
        judge_spec=judge_spec,
        judge_backend=judge_backend,
        admin=admin,
        answer_client=local_backend.client,
        local_backend=local_backend,
        request_timeout=args.request_timeout or config.REQUEST_TIMEOUT_SECONDS,
    )


def _progress(stage: str) -> None:
    """Строка этапа поверх себя — только в терминале (в файле/пайпе мусор из возвратов каретки)."""
    if sys.stdout.isatty():
        print(f"  … {stage:<60}", end="\r", flush=True)


def cmd_run(args) -> int:
    try:
        ctx = _prepare(args)
    except ConfigError as exc:
        print(f"Ошибка: {exc}")
        return EXIT_CONFIG

    config = ctx.config
    variants = list(ctx.variant_set.variants)
    questions = list(ctx.selection.questions)
    repeats = args.repeats
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime = Runtime(ctx.admin, variants, run_id)

    for note in ctx.selection.notes:
        print(f"ℹ️  {note}")
    print(
        f"Варианты: {', '.join(v.name for v in variants)}; вопросов: {len(questions)}, "
        f"повторов: {repeats}, ответов ≈ {len(questions) * len(variants) * repeats}."
    )
    if ctx.judge_spec is not None:
        print(
            "Судье уходят вопросы, ответы и тексты цитируемых фрагментов корпуса: "
            + rme.judge_data_line(ctx.judge_spec.to_dict())
        )
    else:
        print("Оценка качества отключена (--no-judge): судья не вызывается.")

    started_at = now_utc()
    runs: list[rme.QuestionRun] = []
    recorded: list[dict[str, list[list[dict]]]] = []
    stop_reason: str | None = None
    status = ev.STATUS_COMPLETED
    exit_code = EXIT_OK
    cold_seen: set[str] = set()
    # Локальный ответ идёт минутами: бюджета «на двоих» из настроек сравнения моделей не хватает
    # на сетку вариантов, поэтому он не меньше предела обращения на каждый ответ (и на оценку).
    answers = len(variants) * repeats
    budget_seconds = max(
        rme.question_budget_seconds(
            config.RAG_COMPARE_QUESTION_TIMEOUT_SECONDS, len(variants), repeats
        ),
        ctx.request_timeout * answers * (1 if ctx.judge_spec is None else 2),
    )

    try:
        removed = cleanup_leftovers(ctx.admin)
        if removed:
            print(f"Удалено временных тегов прошлых прогонов: {removed}.")
        runtime.create_tags()
        for index, question in enumerate(questions, start=1):
            print(f"\n[{index}/{len(questions)}] вопрос {question.number}: {question.question}")

            run = rme.QuestionRun(question=question)
            run_calls: dict[str, list[list[dict]]] = {v.name: [] for v in variants}
            runs.append(run)
            recorded.append(run_calls)
            evaluate_question(
                run, run_calls, runtime, repeats, budget_seconds, cold_seen, ctx, _progress
            )
            if run.search_error:
                print(f"    поиск не удался, вопрос исключён: {run.search_error}")
            limit = config.RAG_COMPARE_MAX_CONSECUTIVE_FAILURES
            if rme.should_stop(runs, limit):
                stop_reason = (
                    f"автоостановка: {limit} вопросов подряд завершились сбоем (все обращения "
                    "провалились или поиск недоступен)"
                )
                break
    except KeyboardInterrupt:
        stop_reason = "остановлено оператором (Ctrl+C)"
        exit_code = EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 — сохраняем частичный отчёт и завершаемся ошибкой
        stop_reason = f"прогон прерван ошибкой: {type(exc).__name__}: {exc}"
        exit_code = EXIT_FAILED
        print(f"\nОшибка: {stop_reason}")
    finally:
        runtime.cleanup()

    if stop_reason and exit_code == EXIT_OK:
        exit_code = EXIT_FAILED
    if stop_reason:
        status = ev.STATUS_STOPPED

    settings = {
        "strategy": config.RAG_SMART_AGENT_STRATEGY,
        "top_k": config.RAG_TOP_K,
        "min_score": config.RAG_MIN_SCORE,
        "candidates": config.RAG_CANDIDATES,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "rewrite": "off",
        "repeats": repeats,
        "judge": ctx.judge_spec.to_dict() if ctx.judge_spec else None,
        "environment": ctx.variant_set.environment,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "questions_file": os.path.basename(ctx.questions_path),
    }
    report = lib.build_report(
        started_at=started_at,
        finished_at=now_utc(),
        settings=settings,
        variants=variants,
        resources=runtime.resources(),
        questions=[lib.attach_calls(r.to_dict(), c) for r, c in zip(runs, recorded, strict=True)],
        planned=len(questions),
        repeats=repeats,
        status=status,
        stop_reason=stop_reason,
    )
    try:
        path = save_report(report, args.out_dir)
        report["path"] = path
    except OSError as exc:
        print(f"Не удалось сохранить отчёт: {exc}")
        exit_code = exit_code or EXIT_FAILED
    print("\n" + lib.format_summary(report))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="прогон вариантов и таблица «до/после»")
    run.add_argument("--variants", default=DEFAULT_VARIANTS, help="файл вариантов (JSON)")
    run.add_argument("--questions-file", default=None, help="набор вопросов (JSON)")
    run.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="повторов на вариант")
    run.add_argument("--questions", type=int, default=None,
                     help="сколько вопросов по корпусу взять (плюс один вне корпуса)")
    run.add_argument("--only", nargs="+", default=None, metavar="ИМЯ",
                     help="только эти варианты (исходный остаётся всегда)")
    run.add_argument("--request-timeout", type=float, default=None,
                     help="предел обращения к модели, с (по умолчанию REQUEST_TIMEOUT_SECONDS)")
    run.add_argument("--no-judge", action="store_true", help="без оценки качества судьёй")
    run.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="каталог отчётов (вне git)")

    report = sub.add_parser("report", help="таблица сохранённого отчёта без обращений к моделям")
    report.add_argument("number", nargs="?", type=int, default=None,
                        help="номер вопроса: ответы всех вариантов рядом")
    report.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="каталог отчётов")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "report":
        return cmd_report(args)
    if args.repeats < 1:
        print("Ошибка: --repeats должен быть не меньше 1.")
        return EXIT_CONFIG
    if args.questions is not None and args.questions < 1:
        print("Ошибка: --questions должен быть не меньше 1.")
        return EXIT_CONFIG
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
