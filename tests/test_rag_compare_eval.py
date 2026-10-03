"""Тесты чистых правил сравнения ответов с RAG и без него (research/rag_compare_eval.py).

Только чистые функции — ни сети, ни LLM, ни Telegram, ни SmartAgent. Прогон, обращения к
модели и Telegram проверяются вручную (см. tasks.md изменения add-rag-compare-command, 7.2).
"""

import json

import pytest

from research import rag_compare_eval as ev
from research.rag_compare_eval import (
    ModeResult,
    Question,
    QuestionResult,
    QuestionSetError,
    Verdict,
)


def _raw(**overrides) -> dict:
    item = {"question": "Что такое X?", "kind": "concept", "facts": ["a", "b"], "sources": ["Doc1"]}
    item.update(overrides)
    return item


def _question(number: int = 1, facts=("a", "b"), sources=("Doc1",), kind="concept") -> Question:
    return Question(number=number, question=f"Вопрос {number}?", kind=kind,
                    facts=tuple(facts), sources=tuple(sources))


def _mode(coverage: float | None = None, *, facts: int = 4, error=None, judge_error=None,
          rag_titles=(), search_failed=False, answer="ответ", contradicts=False,
          prompt=10, completion=5, elapsed=1.0) -> ModeResult:
    verdict = None
    if coverage is not None:
        found = round(coverage * facts)
        verdict = Verdict(present=tuple([True] * found + [False] * (facts - found)),
                          contradicts=contradicts)
    return ModeResult(
        answer=None if error else answer, error=error, elapsed=elapsed,
        prompt_tokens=prompt, completion_tokens=completion, llm_calls=1,
        rag_sources=[{"title": t, "chunk_index": 0, "score": 0.7} for t in rag_titles],
        search_failed=search_failed, verdict=verdict, judge_error=judge_error,
    )


def _result(off: ModeResult, on: ModeResult, number: int = 1, missing=()) -> QuestionResult:
    return QuestionResult(question=_question(number), off=off, on=on, missing_sources=list(missing))


# --- проверка набора ---------------------------------------------------------


def test_valid_set_is_parsed_with_numbers_from_one():
    questions = ev.parse_questions([_raw(), _raw(question="Что такое Y?")])
    assert [q.number for q in questions] == [1, 2]
    assert questions[0].facts == ("a", "b") and questions[0].sources == ("Doc1",)


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "не список",
        [_raw(question="  ")],
        [_raw(facts=[])],
        [_raw(facts=["a", ""])],
        [_raw(sources=[])],
        [_raw(sources="Doc1")],
        [_raw(kind="unknown")],
        ["не объект"],
    ],
)
def test_invalid_set_is_rejected(raw):
    with pytest.raises(QuestionSetError):
        ev.parse_questions(raw)


def test_question_file_is_loaded_from_disk(tmp_path):
    # Реальный набор вопросов в репозиторий не входит (привязан к корпусу оператора), поэтому
    # чтение проверяется на синтетическом файле.
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([_raw(), _raw(question="Что такое Y?", kind="post_cutoff")]),
                    encoding="utf-8")
    questions = ev.load_questions(str(path))
    assert [q.number for q in questions] == [1, 2]
    assert all(q.facts and q.sources for q in questions)


def test_missing_question_file_says_it_was_not_found(tmp_path):
    with pytest.raises(QuestionSetError, match="не найден"):
        ev.load_questions(str(tmp_path / "нет_такого.json"))


def test_non_json_question_file_is_a_set_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{не json", encoding="utf-8")
    with pytest.raises(QuestionSetError):
        ev.load_questions(str(bad))


# --- оценщик -----------------------------------------------------------------


def test_judge_request_has_question_numbered_facts_answer_and_no_mode_hint():
    content = ev.build_judge_user_content("Что такое X?", ["первый", "второй"], "текст ответа")
    assert "Что такое X?" in content and "1. первый" in content and "2. второй" in content
    assert "текст ответа" in content
    lowered = content.lower()
    for forbidden in ("rag", "материал", "справочн", "без rag", "с rag"):
        assert forbidden not in lowered


def test_parse_judge_response_counts_present_facts():
    verdict = ev.parse_judge_response(
        {"facts": [{"index": 1, "present": True}, {"index": 2, "present": False},
                   {"index": 3, "present": True}], "contradicts": False, "note": " мелочь "},
        3,
    )
    assert verdict.present == (True, False, True)
    assert verdict.coverage == pytest.approx(2 / 3)
    assert verdict.note == "мелочь"


def test_parse_judge_response_is_tolerant_to_gaps_extras_and_duplicates():
    verdict = ev.parse_judge_response(
        {"facts": [{"index": 2, "present": True}, {"index": 2, "present": False},
                   {"index": 9, "present": True}, {"index": "1", "present": True},
                   "мусор", {"index": True, "present": True}], "contradicts": "yes"},
        3,
    )
    # факт 2 — первая запись побеждает; 1 и 3 пропущены или некорректны → отсутствуют
    assert verdict.present == (False, True, False)
    assert verdict.contradicts is False  # только явный true


@pytest.mark.parametrize("data", [None, [], "x", {}, {"facts": "нет"}])
def test_unusable_judge_response_is_none(data):
    assert ev.parse_judge_response(data, 3) is None


def test_verdict_roundtrips_through_dict():
    verdict = Verdict(present=(True, False), contradicts=True, note="n")
    assert Verdict.from_dict(verdict.to_dict()) == verdict


# --- покрытие и сравнение ----------------------------------------------------


def test_compare_coverage_better_equal_worse():
    assert ev.compare_coverage(0.25, 0.75) == "better"
    assert ev.compare_coverage(0.5, 0.5) == "equal"
    assert ev.compare_coverage(1.0, 0.5) == "worse"


# --- источники и метрики поиска ---------------------------------------------


def test_source_hit_by_title_prefix_case_insensitive_with_several_sources():
    assert ev.source_hit(["Док01"], ["Док01. Чек-лист выбора инструментов"])
    assert ev.source_hit(["обл12"], ["ОБЛ12. Чек-лист"])
    assert ev.source_hit(["Док02", "Док01"], ["Док05. Скользящие", "Док01. Чек-лист"])
    assert not ev.source_hit(["Док01"], ["Док03. Основы диверсификации"])
    assert not ev.source_hit(["Док01"], [])


def test_sources_not_indexed_lists_prefixes_without_documents():
    titles = ["Док01. Чек-лист выбора инструментов", "Короткая заметка"]
    assert ev.sources_not_indexed(["Док01", "короткая"], titles) == []
    assert ev.sources_not_indexed(["Док04", "Док01"], titles) == ["Док04"]
    assert ev.sources_not_indexed(["Док04"], []) == ["Док04"]


def test_fired_and_hit_metrics():
    q = _question(sources=("Док01",))
    with_hit = _mode(1.0, rag_titles=["Док01. Чек-лист"])
    miss = _mode(1.0, rag_titles=["Док03. Основы диверсификации"])
    nothing = _mode(1.0)
    assert ev.fired(with_hit) and ev.hit(q, with_hit)
    assert ev.fired(miss) and not ev.hit(q, miss)
    assert not ev.fired(nothing) and not ev.hit(q, nothing)


# --- агрегат ------------------------------------------------------------------


def test_aggregate_means_deltas_and_verdict_counters():
    results = [
        _result(_mode(0.5), _mode(1.0, rag_titles=["Doc1. x"]), number=1),   # лучше
        _result(_mode(0.75), _mode(0.75, rag_titles=["Doc1. x"]), number=2),  # равно
        _result(_mode(1.0), _mode(0.5, rag_titles=["Doc1. x"]), number=3),   # хуже
    ]
    agg = ev.aggregate(results)
    assert agg["scored"] == 3 and agg["excluded"] == []
    assert agg["avg_off"] == pytest.approx(0.75) and agg["avg_on"] == pytest.approx(0.75)
    assert agg["delta"] == pytest.approx(0.0)
    assert (agg["better"], agg["equal"], agg["worse"]) == (1, 1, 1)


def test_unscored_questions_are_excluded_from_means_and_counters_with_reasons():
    results = [
        _result(_mode(0.5), _mode(1.0), number=1),
        _result(_mode(error="таймаут"), _mode(1.0), number=2),
        _result(_mode(0.5), _mode(judge_error="не JSON"), number=3),
        _result(_mode(0.5), _mode(0.5, search_failed=True), number=4),
    ]
    agg = ev.aggregate(results)
    assert agg["scored"] == 1 and agg["avg_off"] == pytest.approx(0.5)
    reasons = {e["number"]: e["reason"] for e in agg["excluded"]}
    assert "ответ без RAG" in reasons[2] and "таймаут" in reasons[2]
    assert "оценка с RAG" in reasons[3] and "не JSON" in reasons[3]
    assert reasons[4] == ev.SEARCH_FAILED_REASON
    assert agg["better"] + agg["equal"] + agg["worse"] == 1
    assert agg["search_failed"] == [4]


def test_search_metrics_use_only_answered_questions_and_not_missing_sources():
    results = [
        _result(_mode(0.5), _mode(1.0, rag_titles=["Doc1. x"]), number=1),   # сработал, попал
        _result(_mode(0.5), _mode(0.5), number=2),                            # не сработал
        _result(_mode(0.5), _mode(0.5, rag_titles=["Other. y"]), number=3),   # сработал, мимо
        _result(_mode(0.5), _mode(0.5), number=4, missing=["Doc1"]),  # документа нет в индексе
    ]
    agg = ev.aggregate(results)
    assert agg["on_answered"] == 4 and agg["fired"] == 2
    assert agg["hit"] == 1 and agg["hit_candidates"] == 3  # вопрос 4 не считается слабостью поиска
    assert agg["not_fired"] == [2, 4]
    assert agg["source_missing"] == [4]


def test_mode_without_any_judged_answer_makes_comparison_unavailable():
    results = [_result(_mode(error="сбой"), _mode(0.5), number=i) for i in (1, 2)]
    agg = ev.aggregate(results)
    assert agg["unavailable_modes"] == [ev.MODE_OFF]
    assert agg["avg_off"] is None and agg["delta"] is None
    summary = ev.format_summary({"questions": [r.to_dict() for r in results]})
    assert "Сравнение невозможно" in summary and "без RAG" in summary


def test_tokens_and_time_are_summed_per_mode_and_contradictions_counted():
    results = [
        _result(_mode(0.5, prompt=10, completion=5, elapsed=2.0, contradicts=True),
                _mode(1.0, prompt=100, completion=20, elapsed=3.0)),
    ]
    agg = ev.aggregate(results)
    assert (agg["tokens_off"], agg["tokens_on"]) == (15, 120)
    assert (agg["time_off"], agg["time_on"]) == (2.0, 3.0)
    assert (agg["contradicts_off"], agg["contradicts_on"]) == (1, 0)


# --- отчёт --------------------------------------------------------------------


def _report(results) -> dict:
    return {
        "finished_at": "2026-10-01 21:00:00",
        "settings": {"strategy": "structural", "top_k": 3, "min_score": 0.6, "model": "m"},
        "index_missing": [],
        "questions": [r.to_dict() for r in results],
    }


def test_summary_contains_coverage_counters_search_and_disclaimer():
    results = [
        _result(_mode(0.5), _mode(1.0, rag_titles=["Doc1. x"]), number=1),
        _result(_mode(0.5), _mode(0.5), number=2, missing=["Doc1"]),
        _result(_mode(error="сбой"), _mode(0.5), number=3),
    ]
    summary = ev.format_summary(_report(results))
    assert "без RAG: 50%" in summary and "с RAG: 75%" in summary and "+25 п.п." in summary
    assert "лучше: 1" in summary and "равно: 1" in summary
    assert "нет в индексе" in summary and "вопросах: 2" in summary
    assert "Исключено из расчёта: 1" in summary and "3 (" in summary
    assert "порог 0.6" in summary
    assert "недетерминирован" in summary


def test_table_has_one_entry_per_question_with_trend_and_search_marks():
    results = [
        _result(_mode(0.5), _mode(1.0, rag_titles=["Doc1. x"]), number=1),
        _result(_mode(0.5), _mode(0.5), number=2),
        _result(_mode(0.5), _mode(0.5, search_failed=True), number=3),
    ]
    table = ev.format_table(_report(results))
    assert "1. " in table and "2. " in table and "3. " in table
    assert "50% → 100% ▲ · поиск: попал" in table
    assert "поиск: не сработал" in table
    assert "поиск: сбой" in table


def test_question_detail_shows_both_answers_facts_marks_sources_and_notes():
    on = _mode(0.5, facts=2, rag_titles=["Doc1. x"], answer="ответ с RAG")
    on.verdict = Verdict(present=(True, False), contradicts=True, note="расхождение")
    results = [_result(_mode(0.0, facts=2, answer="ответ без"), on)]
    results[0].question = _question(facts=("факт один", "факт два"))
    detail = ev.format_question_detail(_report(results), 1)
    assert "ответ без" in detail and "ответ с RAG" in detail
    assert "✅ 1. факт один" in detail and "❌ 2. факт два" in detail
    assert "Doc1. x — фрагмент 0" in detail
    assert "противоречащее" in detail and "расхождение" in detail


def test_question_detail_for_unknown_number_is_none():
    assert ev.format_question_detail(_report([_result(_mode(0.5), _mode(0.5))]), 99) is None


def test_question_result_roundtrips_through_json():
    original = _result(_mode(0.5, rag_titles=["Doc1. x"]), _mode(1.0, rag_titles=["Doc1. x"]),
                       missing=["Doc2"])
    restored = QuestionResult.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()


def test_split_text_respects_limit_and_loses_nothing():
    text = "\n".join(f"строка {i} " + "x" * 30 for i in range(40))
    parts = ev.split_text(text, 200)
    assert all(len(p) <= 200 for p in parts) and len(parts) > 1
    assert "\n".join(parts) == text

    long_line = "я" * 450
    parts = ev.split_text(long_line, 200)
    assert all(len(p) <= 200 for p in parts)
    assert "".join(parts) == long_line


# --- проваленные вопросы, автоостановка, отчёт остановленного прогона -------------


def _ok(number: int = 1) -> QuestionResult:
    return _result(_mode(0.5), _mode(1.0), number=number)


def _failed(number: int = 1, how: str = "answer") -> QuestionResult:
    if how == "answer":
        return _result(_mode(error="таймаут"), _mode(1.0), number=number)
    if how == "judge":
        unusable = _mode(judge_error="оценщик вернул непригодный ответ")
        return _result(_mode(0.5), unusable, number=number)
    return _result(_mode(0.5), _mode(error=ev.QUESTION_TIMEOUT_REASON), number=number)


def test_question_failed_for_answer_judge_and_timeout_failures():
    assert not ev.question_failed(_ok())
    assert ev.question_failed(_failed(how="answer"))
    assert ev.question_failed(_failed(how="judge"))
    assert ev.question_failed(_failed(how="timeout"))


def test_search_failure_alone_is_not_a_failed_question():
    result = _result(_mode(0.5), _mode(0.5, search_failed=True))
    assert not ev.question_failed(result)


def test_consecutive_failures_counts_only_the_trailing_series():
    assert ev.consecutive_failures([]) == 0
    assert ev.consecutive_failures([_ok(1), _failed(2), _failed(3)]) == 2
    assert ev.consecutive_failures([_failed(1), _failed(2), _ok(3)]) == 0


def test_success_between_failures_resets_the_series():
    results = [_failed(1), _failed(2), _ok(3), _failed(4), _failed(5)]
    assert ev.consecutive_failures(results) == 2
    assert not ev.should_stop(results, 3)
    assert ev.should_stop(results + [_failed(6)], 3)


def test_should_stop_at_the_limit_and_never_for_zero_limit():
    results = [_ok(1), _failed(2), _failed(3), _failed(4)]
    assert ev.should_stop(results, 3) and not ev.should_stop(results, 4)
    assert not ev.should_stop(results, 0)


def test_build_report_has_status_planned_and_reason():
    results = [_ok(1), _ok(2)]
    report = ev.build_report(
        started_at="s", finished_at="f", settings={"strategy": "structural"}, results=results,
        planned=10, index_missing=["Б", "А"], status=ev.STATUS_STOPPED, stop_reason="по команде",
    )
    assert report["status"] == ev.STATUS_STOPPED and report["stop_reason"] == "по команде"
    assert report["planned"] == 10 and len(report["questions"]) == 2
    assert report["index_missing"] == ["А", "Б"]
    completed = ev.build_report(
        started_at="s", finished_at="f", settings={}, results=results, planned=2
    )
    assert completed["status"] == ev.STATUS_COMPLETED and completed["stop_reason"] is None


def test_summary_of_stopped_run_names_reason_and_processed_count():
    report = ev.build_report(
        started_at="s", finished_at="2026-10-02 10:00:00", settings={}, planned=10,
        results=[_ok(1), _ok(2), _failed(3), _failed(4), _failed(5)],
        status=ev.STATUS_STOPPED,
        stop_reason="автоостановка: 3 вопроса подряд без ответа провайдера",
    )
    summary = ev.format_summary(report)
    assert "Прогон остановлен: автоостановка: 3 вопроса подряд без ответа провайдера" in summary
    assert "Обработано 5 из 10 вопросов" in summary
    assert "Прогон остановлен: 2026-10-02 10:00:00" in summary
    # метрики — только по обработанным; исключённые названы
    assert "по 2 из 5 вопросов" in summary and "Исключено из расчёта: 3" in summary


def test_summary_of_completed_run_has_no_stop_line():
    report = ev.build_report(
        started_at="s", finished_at="f", settings={}, results=[_ok(1)], planned=1
    )
    summary = ev.format_summary(report)
    assert "остановлен" not in summary and "Прогон завершён: f" in summary


def test_report_written_before_stop_support_reads_as_completed():
    legacy = {"finished_at": "f", "settings": {}, "index_missing": [],
              "questions": [_ok(1).to_dict()]}
    summary = ev.format_summary(legacy)
    assert "остановлен" not in summary and "Прогон завершён" in summary


def test_summary_when_stopped_before_any_question_says_comparison_impossible():
    report = ev.build_report(started_at="s", finished_at="f", settings={}, results=[], planned=10,
                             status=ev.STATUS_STOPPED, stop_reason="по команде")
    summary = ev.format_summary(report)
    assert "Обработано 0 из 10" in summary and "Сравнение невозможно" in summary


def test_progress_text_shows_question_number_and_stage():
    plain = ev.progress_text(3, 10)
    assert "обработано 3 из 10" in plain
    staged = ev.progress_text(2, 10, ev.STAGE_ON_ANSWER, "Что такое X?")
    assert "Вопрос 3 из 10" in staged and "ответ с RAG" in staged and "Что такое X?" in staged
    assert "Обработано вопросов: 2" in staged


def test_every_stage_has_a_label():
    assert set(ev.STAGE_LABELS) == set(ev.STAGES)
    assert [ev.STAGE_LABELS[s] for s in ev.STAGES] == [
        "ответ без RAG", "оценка ответа без RAG", "ответ с RAG", "оценка ответа с RAG"
    ]


# --- жёсткий предел времени обращения по часам -------------------------------------


def test_call_with_deadline_returns_result_of_a_fast_call():
    assert ev.call_with_deadline(lambda: 42, 5) == 42


def test_call_with_deadline_propagates_the_exception_of_the_call():
    def failing():
        raise ValueError("сбой провайдера")

    with pytest.raises(ValueError, match="сбой провайдера"):
        ev.call_with_deadline(failing, 5)


def test_call_with_deadline_gives_up_on_a_stuck_call_without_waiting_for_it():
    import threading
    import time

    release = threading.Event()  # «провайдер» держит соединение, пока мы его не отпустим
    started = time.monotonic()
    with pytest.raises(ev.DeadlineExceeded):
        ev.call_with_deadline(lambda: release.wait(30), 0.3)
    # вернулись по дедлайну, а не после «ответа» зависшего вызова
    assert time.monotonic() - started < 5
    release.set()  # зависший поток-сирота дорабатывает и не мешает


def test_call_with_deadline_orphan_result_is_discarded():
    import threading
    import time

    done = threading.Event()

    def slow():
        time.sleep(0.4)
        done.set()
        return "поздний результат"

    with pytest.raises(ev.DeadlineExceeded):
        ev.call_with_deadline(slow, 0.05)
    assert done.wait(5)  # поток-сирота действительно дожил, но вызывающий его результат не получил


# --- история диалога в контрольном наборе (изменение add-rag-rewrite-dialog-context) ----------


def test_question_without_history_field_is_a_standalone_question():
    (question,) = ev.parse_questions([_raw()])
    assert question.history == ()


def test_valid_history_is_parsed_in_order_and_stripped():
    (question,) = ev.parse_questions([_raw(history=["  Как выбирать X?  ", "А Y?"])])
    assert question.history == ("Как выбирать X?", "А Y?")


def test_null_history_means_standalone():
    (question,) = ev.parse_questions([_raw(history=None)])
    assert question.history == ()


def test_history_must_be_a_list():
    with pytest.raises(QuestionSetError) as error:
        ev.parse_questions([_raw(), _raw(history="Как выбирать X?")])
    assert "вопрос 2" in str(error.value)


@pytest.mark.parametrize("bad", [[""], ["   "], ["нормально", ""], [5], [None]])
def test_history_with_empty_or_non_string_entry_is_rejected(bad):
    with pytest.raises(QuestionSetError) as error:
        ev.parse_questions([_raw(history=bad)])
    assert "вопрос 1" in str(error.value)


def test_history_survives_a_roundtrip_through_the_report_dict():
    question = Question(number=3, question="А Y?", kind="concept", facts=("a",),
                        sources=("Doc1",), history=("Как выбирать X?",))
    result = QuestionResult(question=question, off=_mode(), on=_mode())
    restored = QuestionResult.from_dict(result.to_dict())
    assert restored.question.history == ("Как выбирать X?",)


def test_old_report_without_history_still_loads():
    result = _result(_mode(0.5), _mode(0.75))
    data = result.to_dict()
    del data["history"]
    assert QuestionResult.from_dict(data).question.history == ()


def test_question_detail_shows_the_dialog_history():
    question = Question(number=1, question="А Y?", kind="concept", facts=("a",),
                        sources=("Doc1",), history=("Как выбирать X?", "А Z?"))
    result = QuestionResult(question=question, off=_mode(0.5), on=_mode(0.75))
    text = ev.format_question_detail({"questions": [result.to_dict()]}, 1)
    assert "История диалога" in text and "Как выбирать X? → А Z?" in text


def test_split_multi_turn_separates_questions_with_history_and_keeps_numbers():
    questions = ev.parse_questions([
        _raw(question="Самостоятельный?"),
        _raw(question="А продолжение?", history=["Первый вопрос?"]),
        _raw(question="Ещё самостоятельный?"),
    ])
    standalone, multi_turn = ev.split_multi_turn(questions)
    assert [q.number for q in standalone] == [1, 3]
    assert [q.number for q in multi_turn] == [2]


def test_split_multi_turn_of_a_set_without_history_skips_nothing():
    questions = ev.parse_questions([_raw(), _raw(question="Другой?")])
    standalone, multi_turn = ev.split_multi_turn(questions)
    assert standalone == questions and multi_turn == []


def test_split_multi_turn_of_a_set_where_every_question_has_history():
    questions = ev.parse_questions([_raw(history=["a?"]), _raw(history=["b?", "c?"])])
    standalone, multi_turn = ev.split_multi_turn(questions)
    assert standalone == [] and len(multi_turn) == 2
