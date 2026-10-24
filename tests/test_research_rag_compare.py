"""Тесты чистых частей research/rag_compare.py: флаг и прогресс единственного прогона,
остановка, бюджет времени вопроса, причины сбоев, параметры режимов поиска, подписи этапов,
хранение отчётов на диске (tmp_path), настройки отчёта и тексты итога.

Без сети, Telegram и модели. Не покрыто (нужны модель, SmartAgent, слот очереди или Telegram):
run_mode, _judge_call*, _judge_data, judge, judge_citations, _answer_and_judge,
evaluate_question, build_mode_agents, evaluate_retrieval, _run_comparison,
_ProgressReporter, _edit_progress, _send_parts, обработчики команд и reply_stop_result.
"""

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from research import rag_compare as rc
from research import rag_compare_eval as ev
from research import rag_modes_eval as rm


@pytest.fixture(autouse=True)
def clean_run_state():
    """Состояние прогона — глобальное: до и после каждого теста сбрасывается."""
    rc.release_run()
    rc.set_progress_done(0)
    yield
    rc.release_run()
    rc.set_progress_done(0)


class FakeTask:
    """Минимальная задача: запоминает, что её отменили."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> bool:
        self.cancelled = True
        return True


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1")


def _status_error(cls, code: int):
    response = httpx.Response(code, request=_request())
    return cls("сбой", response=response, body=None)


# --- флаг «прогон идёт» и прогресс -----------------------------------------------


def test_acquire_run_takes_the_flag_once_and_resets_progress():
    rc.set_progress_done(5)
    assert rc.is_run_in_progress() is False
    assert rc.acquire_run(7) is True
    assert rc.is_run_in_progress() is True
    assert rc.run_progress() == (0, 7)
    assert rc.acquire_run(3) is False
    assert rc.run_progress() == (0, 7)  # отказ не трогает прогресс идущего прогона


def test_progress_done_is_reported_and_release_frees_the_flag():
    rc.acquire_run(4)
    rc.set_progress_done(2)
    assert rc.run_progress() == (2, 4)
    rc.release_run()
    assert rc.is_run_in_progress() is False
    assert rc.acquire_run(1) is True


def test_release_clears_stop_state_and_task():
    rc.acquire_run(2)
    task = FakeTask()
    rc.set_run_task(task)
    assert rc.request_stop() == "stopping"
    rc.release_run()
    assert rc.current_stop_reason() is None
    assert rc.request_stop() == "idle"


def test_acquire_run_clears_stale_stop_reason_and_finalizing():
    rc.acquire_run(2)
    rc.set_run_task(FakeTask())
    rc.request_stop()
    rc.set_finalizing()
    rc._run_in_progress = False  # прогон «закончился» без release_run
    assert rc.acquire_run(2) is True
    assert rc.current_stop_reason() is None
    rc.set_run_task(FakeTask())
    assert rc.request_stop() == "stopping"  # finalizing сброшен


# --- остановка ----------------------------------------------------------------


def test_stop_when_nothing_runs_is_idle():
    assert rc.request_stop() == "idle"


def test_stop_without_task_is_idle_even_if_flag_is_set():
    rc.acquire_run(2)
    assert rc.request_stop() == "idle"


def test_stop_cancels_task_and_records_reason_then_reports_requested():
    rc.acquire_run(2)
    task = FakeTask()
    rc.set_run_task(task)
    assert rc.request_stop() == "stopping"
    assert task.cancelled is True
    assert rc.current_stop_reason() == "по команде пользователя"
    assert rc.request_stop() == "requested"


def test_stop_during_finalizing_does_not_cancel():
    rc.acquire_run(2)
    task = FakeTask()
    rc.set_run_task(task)
    rc.set_finalizing()
    assert rc.request_stop() == "finalizing"
    assert task.cancelled is False
    assert rc.current_stop_reason() is None


def test_every_stop_result_has_a_distinct_key():
    results = set()
    results.add(rc.request_stop())  # idle
    rc.acquire_run(1)
    rc.set_run_task(FakeTask())
    results.add(rc.request_stop())  # stopping
    results.add(rc.request_stop())  # requested
    rc.release_run()
    rc.acquire_run(1)
    rc.set_run_task(FakeTask())
    rc.set_finalizing()
    results.add(rc.request_stop())  # finalizing
    assert results == {"idle", "stopping", "requested", "finalizing"}


def test_background_task_reference_is_kept_until_done():
    async def scenario():
        gate = asyncio.Event()

        async def work():
            await gate.wait()

        task = asyncio.get_running_loop().create_task(work())
        rc.track_background_task(task)
        assert task in rc._background_tasks
        gate.set()
        await task
        await asyncio.sleep(0)
        assert task not in rc._background_tasks

    asyncio.run(scenario())


# --- бюджет времени вопроса ---------------------------------------------------


def _fake_clock(monkeypatch, start: float = 1000.0) -> list[float]:
    now = [start]
    monkeypatch.setattr(rc.time, "monotonic", lambda: now[0])
    return now


def test_budget_remaining_decreases_with_time(monkeypatch):
    now = _fake_clock(monkeypatch)
    budget = rc.QuestionBudget(120)
    assert budget.remaining() == 120
    now[0] += 30
    assert budget.remaining() == 90


def test_budget_expires_exactly_at_deadline(monkeypatch):
    now = _fake_clock(monkeypatch)
    budget = rc.QuestionBudget(10)
    now[0] += 9.9
    assert budget.expired() is False
    now[0] += 0.1
    assert budget.expired() is True
    now[0] += 5
    assert budget.expired() is True


def test_budget_call_timeout_is_capped_by_request_timeout(monkeypatch):
    _fake_clock(monkeypatch)
    monkeypatch.setattr(rc, "REQUEST_TIMEOUT_SECONDS", 60)
    assert rc.QuestionBudget(1000).call_timeout() == 60


def test_budget_call_timeout_is_remainder_when_smaller(monkeypatch):
    now = _fake_clock(monkeypatch)
    monkeypatch.setattr(rc, "REQUEST_TIMEOUT_SECONDS", 60)
    budget = rc.QuestionBudget(100)
    now[0] += 70
    assert budget.call_timeout() == 30


def test_budget_call_timeout_is_never_zero_or_negative(monkeypatch):
    now = _fake_clock(monkeypatch)
    budget = rc.QuestionBudget(5)
    now[0] += 500
    assert budget.call_timeout() == 0.001


# --- причины сбоев ------------------------------------------------------------


def test_error_reason_translates_sdk_errors_with_main_provider_label():
    text = rc._error_reason(_status_error(AuthenticationError, 401))
    assert rc.MAIN_CLIENT_LABEL in text
    assert "авторизации" in text


def test_error_reason_uses_label_of_the_backend_provider():
    backend = rc.LlmBackend(client=None, model="m", provider="kimi")
    text = rc._error_reason(_status_error(RateLimitError, 429), backend)
    assert "Kimi" in text
    assert "перегружен" in text


def test_error_reason_for_backend_of_main_provider_uses_main_label():
    backend = rc.LlmBackend(client=None, model="m", provider=rc.MAIN_CLIENT)
    text = rc._error_reason(_status_error(APIStatusError, 500), backend)
    assert rc.MAIN_CLIENT_LABEL in text


def test_error_reason_for_timeout_and_connection_errors():
    assert "не ответил вовремя" in rc._error_reason(APITimeoutError(request=_request()))
    assert "подключиться" in rc._error_reason(APIConnectionError(request=_request()))


def test_error_reason_for_unexpected_exception_has_type_and_message():
    assert rc._error_reason(ValueError("плохое значение")) == "ValueError: плохое значение"


def test_error_reason_for_unexpected_exception_without_message_is_stripped():
    assert rc._error_reason(KeyError()) == "KeyError:"


# --- параметры режимов поиска --------------------------------------------------


def _set_filter_config(monkeypatch, *, margin, min_chars, per_doc, top_k=3, candidates=8):
    monkeypatch.setattr(rc, "RAG_TOP_K", top_k)
    monkeypatch.setattr(rc, "RAG_CANDIDATES", candidates)
    monkeypatch.setattr(rc, "RAG_RELATIVE_MARGIN", margin)
    monkeypatch.setattr(rc, "RAG_MIN_CHUNK_CHARS", min_chars)
    monkeypatch.setattr(rc, "RAG_MAX_PER_DOC", per_doc)


def test_filter_steps_are_working_settings_when_any_step_is_on(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2)
    steps, reference = rc._filter_steps()
    assert reference is False
    assert steps == {
        "candidates": 8,
        "relative_margin": 0.2,
        "min_chunk_chars": 50,
        "max_per_doc": 2,
    }


def test_filter_steps_fall_back_to_reference_when_all_neutral(monkeypatch):
    _set_filter_config(monkeypatch, margin=1.0, min_chars=0, per_doc=3, candidates=3)
    steps, reference = rc._filter_steps()
    assert reference is True
    assert steps == rm.REFERENCE_FILTER


def test_baseline_mode_has_neutral_steps_and_keeps_rewrite_off(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2, top_k=4)
    kwargs = rc._mode_agent_kwargs(rm.MODE_BASELINE)
    assert kwargs == {
        "rag_candidates": 4,
        "rag_relative_margin": 1.0,
        "rag_min_chunk_chars": 0,
        "rag_max_per_doc": 4,
        "rewrite_backend": None,
    }


def test_filter_mode_uses_working_steps_without_rewrite(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2)
    kwargs = rc._mode_agent_kwargs(rm.MODE_FILTER)
    assert kwargs["rag_candidates"] == 8
    assert kwargs["rag_relative_margin"] == 0.2
    assert kwargs["rag_min_chunk_chars"] == 50
    assert kwargs["rag_max_per_doc"] == 2
    assert kwargs["rewrite_backend"] is None


def test_rewrite_mode_keeps_rewrite_backend_but_neutral_steps(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2, top_k=3)
    kwargs = rc._mode_agent_kwargs(rm.MODE_REWRITE)
    assert "rewrite_backend" not in kwargs
    assert kwargs["rag_relative_margin"] == 1.0
    assert kwargs["rag_candidates"] == 3


def test_rewrite_filter_mode_has_filter_and_rewrite(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2)
    kwargs = rc._mode_agent_kwargs(rm.MODE_REWRITE_FILTER)
    assert "rewrite_backend" not in kwargs
    assert kwargs["rag_relative_margin"] == 0.2


def test_concat_mode_is_neutral_without_rewrite(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2)
    kwargs = rc._mode_agent_kwargs(rm.MODE_CONCAT)
    assert kwargs["rewrite_backend"] is None
    assert kwargs["rag_relative_margin"] == 1.0


def test_reference_filter_gives_working_agent_kwargs_for_filter_mode(monkeypatch):
    _set_filter_config(monkeypatch, margin=1.0, min_chars=0, per_doc=3, candidates=3)
    kwargs = rc._mode_agent_kwargs(rm.MODE_FILTER)
    assert kwargs["rag_candidates"] == rm.REFERENCE_FILTER["candidates"]
    assert kwargs["rag_max_per_doc"] == rm.REFERENCE_FILTER["max_per_doc"]


# --- этапы прогресса -----------------------------------------------------------


def test_variant_stage_keys():
    assert rc.variant_stage("filter", judge_stage=False) == "variant_answer:filter"
    assert rc.variant_stage("filter", judge_stage=True) == "variant_judge:filter"


def test_stage_label_for_retrieval_stages():
    assert rc.stage_label(rc.STAGE_REWRITE) == "переписывание вопроса"
    assert rc.stage_label(rc.STAGE_SEARCH) == "поиск по режимам"


def test_stage_label_for_variant_stages_uses_mode_titles():
    answer = rc.stage_label(rc.variant_stage(rm.MODE_FILTER, judge_stage=False))
    judge = rc.stage_label(rc.variant_stage(rm.MODE_FILTER, judge_stage=True))
    assert answer == f"ответ, режим поиска «{rm.MODE_TITLES[rm.MODE_FILTER]}»"
    assert judge == f"оценка ответа, режим поиска «{rm.MODE_TITLES[rm.MODE_FILTER]}»"


def test_stage_label_for_unknown_mode_key_shows_the_key():
    assert rc.stage_label("variant_answer:zzz") == "ответ, режим поиска «zzz»"


def test_stage_label_of_base_stages_is_none_so_ev_labels_apply():
    for stage in ev.STAGES:
        assert rc.stage_label(stage) is None
    assert rc.stage_label("совсем неизвестный") is None


# --- хранение отчётов ----------------------------------------------------------


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 1, 12, 30, 5, tzinfo=timezone.utc)


def _report() -> dict:
    return {"questions": [{"number": 1}], "planned": 1}


def test_save_report_writes_json_named_by_utc_time_and_leaves_no_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "datetime", _FixedDatetime)
    path = rc.save_report(_report(), str(tmp_path / "nested"))
    assert path.endswith("20261001T123005Z.json")
    assert json.loads(open(path, encoding="utf-8").read()) == _report()
    assert [p.name for p in (tmp_path / "nested").iterdir()] == ["20261001T123005Z.json"]


def test_save_report_keeps_cyrillic_readable(tmp_path):
    path = rc.save_report({"questions": [{"text": "вопрос"}]}, str(tmp_path))
    assert "вопрос" in open(path, encoding="utf-8").read()


def test_load_latest_report_roundtrip(tmp_path):
    rc.save_report(_report(), str(tmp_path))
    assert rc.load_latest_report(str(tmp_path)) == _report()


def test_load_latest_report_without_directory_is_none(tmp_path):
    assert rc.load_latest_report(str(tmp_path / "нет")) is None


def test_load_latest_report_in_empty_or_foreign_directory_is_none(tmp_path):
    assert rc.load_latest_report(str(tmp_path)) is None
    (tmp_path / "questions.json").write_text("[]", encoding="utf-8")
    assert rc.load_latest_report(str(tmp_path)) is None


def test_load_latest_report_picks_the_newest_by_name(tmp_path):
    (tmp_path / "20260101T000000Z.json").write_text(
        json.dumps({"questions": [], "tag": "old"}), encoding="utf-8"
    )
    (tmp_path / "20260201T000000Z.json").write_text(
        json.dumps({"questions": [], "tag": "new"}), encoding="utf-8"
    )
    assert rc.load_latest_report(str(tmp_path))["tag"] == "new"


def test_load_latest_report_broken_json_is_report_error(tmp_path):
    (tmp_path / "20260101T000000Z.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(rc.ReportError):
        rc.load_latest_report(str(tmp_path))


@pytest.mark.parametrize("content", ["[]", '{"questions": "x"}', '{"other": 1}', "5"])
def test_load_latest_report_unexpected_structure_is_report_error(tmp_path, content):
    (tmp_path / "20260101T000000Z.json").write_text(content, encoding="utf-8")
    with pytest.raises(rc.ReportError):
        rc.load_latest_report(str(tmp_path))


# --- настройки отчёта ----------------------------------------------------------


def test_settings_default_level_is_answers_and_records_working_config(monkeypatch):
    _set_filter_config(monkeypatch, margin=0.2, min_chars=50, per_doc=2)
    monkeypatch.setattr(rc, "rewrite_backend", None)
    settings = rc._settings()
    assert settings["level"] == rm.LEVEL_ANSWERS
    assert settings["citation_judge"] == ev.CITATION_JUDGE_FRAGMENTS
    assert settings["top_k"] == 3
    assert settings["candidates"] == 8
    assert settings["filter_reference"] is False
    assert settings["filter_steps"]["max_per_doc"] == 2
    assert settings["rewrite_provider"] == ""
    assert settings["rewrite_model"] == ""
    assert settings["rewrite_search_mode"] == ""
    assert settings["model"] == rc.MAIN_MODEL


def test_settings_with_rewrite_backend_names_provider_and_model(monkeypatch):
    backend = rc.LlmBackend(client=None, model="rw-model", provider="ollama")
    monkeypatch.setattr(rc, "rewrite_backend", backend)
    monkeypatch.setattr(rc, "REWRITE_SEARCH_MODE", "replace")
    settings = rc._settings(rm.LEVEL_SEARCH)
    assert settings["level"] == rm.LEVEL_SEARCH
    assert settings["rewrite_provider"] == "ollama"
    assert settings["rewrite_model"] == "rw-model"
    assert settings["rewrite_search_mode"] == "replace"


def test_settings_history_questions_use_reference_when_disabled(monkeypatch):
    monkeypatch.setattr(rc, "REWRITE_HISTORY_QUESTIONS", 0)
    settings = rc._settings()
    assert settings["history_questions"] == rm.REFERENCE_HISTORY_QUESTIONS
    assert settings["history_reference"] is True


def test_settings_history_questions_use_operator_value(monkeypatch):
    monkeypatch.setattr(rc, "REWRITE_HISTORY_QUESTIONS", 2)
    settings = rc._settings()
    assert settings["history_questions"] == 2
    assert settings["history_reference"] is False


def test_settings_are_json_serialisable():
    json.dumps(rc._settings())


def test_now_has_utc_timestamp_format(monkeypatch):
    monkeypatch.setattr(rc, "datetime", _FixedDatetime)
    assert rc._now() == "2026-10-01 12:30:05"


def test_shared_aliases_point_to_the_same_functions():
    assert rc.plural_questions is rc._plural_questions
    assert rc.run_settings is rc._settings
    assert rc.now_utc is rc._now
    assert rc.ProgressReporter is rc._ProgressReporter
    assert rc.send_parts is rc._send_parts


# --- тексты --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0 вопросов"),
        (1, "1 вопрос"),
        (2, "2 вопроса"),
        (4, "4 вопроса"),
        (5, "5 вопросов"),
        (11, "11 вопросов"),
        (12, "12 вопросов"),
        (14, "14 вопросов"),
        (21, "21 вопрос"),
        (22, "22 вопроса"),
        (101, "101 вопрос"),
        (111, "111 вопросов"),
    ],
)
def test_plural_questions(count, expected):
    assert rc._plural_questions(count) == expected


def test_initial_progress_for_search_level_names_mode_comparison():
    text = rc._initial_progress(rm.LEVEL_SEARCH, 6)
    assert "режимов поиска" in text
    assert "0 из 6" in text


def test_initial_progress_for_answers_level_is_the_common_progress_text():
    assert rc._initial_progress(rm.LEVEL_ANSWERS, 6) == ev.progress_text(0, 6)


def _question(number: int = 1) -> ev.Question:
    return ev.Question(
        number=number, question=f"Вопрос {number}?", kind="concept", facts=("a", "b"), sources=()
    )


def _answers_report(variant_modes=None) -> dict:
    mode = ev.ModeResult(
        answer="ответ", verdict=ev.Verdict(present=(True, False), contradicts=False)
    )
    result = ev.QuestionResult(question=_question(), off=mode, on=mode)
    return ev.build_report(
        started_at="2026-10-01 12:00:00",
        finished_at="2026-10-01 12:05:00",
        settings={"strategy": "structural", "top_k": 3, "min_score": 0.6, "model": "m"},
        results=[result],
        planned=1,
        extra={"variant_modes": variant_modes} if variant_modes else None,
    )


def _retrieval_report() -> dict:
    item = rm.QuestionRetrieval(
        question=_question(),
        modes={
            rm.MODE_BASELINE: rm.RetrievalResult(
                chunks=[{"title": "Doc1", "chunk_index": 0, "score": 0.8}], candidates=3
            )
        },
    )
    return rm.build_retrieval_report(
        started_at="2026-10-01 12:00:00",
        finished_at="2026-10-01 12:05:00",
        settings={"strategy": "structural", "top_k": 3, "min_score": 0.6, "model": "m"},
        results=[item],
        planned=1,
        modes=[rm.MODE_BASELINE],
        unavailable={},
    )


def test_report_texts_of_answers_report_are_summary_and_table():
    report = _answers_report()
    assert rc._report_texts(report) == [ev.format_summary(report), ev.format_table(report)]


def test_report_texts_of_answers_report_with_variants_add_modes_block():
    report = _answers_report(variant_modes=[rm.MODE_FILTER])
    texts = rc._report_texts(report)
    assert len(texts) == 3
    assert texts[1] == rm.format_variants_summary(report)
    assert "Режимы поиска в ответах" in texts[1]
    assert texts[0] == ev.format_summary(report)
    assert texts[2] == ev.format_table(report)


def test_report_texts_of_retrieval_report_are_retrieval_summary_and_table():
    report = _retrieval_report()
    assert rc._report_texts(report) == [
        rm.format_retrieval_summary(report),
        rm.format_retrieval_table(report),
    ]


# --- шаг прогона под слотом очереди (настоящий шлюз, обычная функция) -------------


def _gate():
    from llm_queue.gate import LlmGate

    return LlmGate(1, 1, 10, 1800, timeout_seconds=600)


def test_run_under_slot_returns_result_of_the_function_and_wait_time():
    async def scenario():
        result, waited = await rc.run_under_slot(_gate(), lambda a, b: a + b, 2, 3)
        assert result == 5
        assert waited >= 0.0

    asyncio.run(scenario())


def test_run_under_slot_releases_the_slot_after_the_step():
    async def scenario():
        gate = _gate()
        await rc.run_under_slot(gate, lambda: "первый")
        result, _ = await rc.run_under_slot(gate, lambda: "второй")
        assert result == "второй"

    asyncio.run(scenario())


def test_run_under_slot_propagates_error_of_the_function_and_frees_the_slot():
    async def scenario():
        gate = _gate()

        def broken():
            raise ValueError("сбой вопроса")

        with pytest.raises(ValueError, match="сбой вопроса"):
            await rc.run_under_slot(gate, broken)
        result, _ = await rc.run_under_slot(gate, lambda: "после сбоя")
        assert result == "после сбоя"

    asyncio.run(scenario())


def test_run_under_slot_waits_for_a_busy_slot_and_reports_the_wait():
    async def scenario():
        gate = _gate()
        holder_ready = asyncio.Event()
        holder_release = asyncio.Event()

        async def holder():
            async with gate.slot():
                holder_ready.set()
                await holder_release.wait()

        holder_task = asyncio.create_task(holder())
        await holder_ready.wait()
        step = asyncio.create_task(rc.run_under_slot(gate, lambda: "готово"))
        await asyncio.sleep(0.05)
        assert not step.done()  # слот занят — шаг ждёт
        holder_release.set()
        result, waited = await step
        await holder_task
        assert result == "готово"
        assert waited >= 0.04

    asyncio.run(scenario())
