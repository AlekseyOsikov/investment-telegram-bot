"""Тесты чистых правил сравнения режимов поиска (research/rag_modes_eval.py).

Только чистые функции — ни сети, ни LLM, ни Telegram, ни SmartAgent, ни индекса. Прогон по
режимам, переписывание и обращения к агенту проверяются вручную (см. tasks.md изменения
add-rag-rerank-and-rewrite, 6.2, 7.2). Названия документов в примерах — вымышленные.
"""

import pytest

from research import rag_compare_eval as ev
from research import rag_modes_eval as rm
from research.rag_modes_eval import QuestionRetrieval, RetrievalResult


def _question(number: int = 1, sources=("Облигации",)) -> ev.Question:
    return ev.Question(number=number, question=f"Вопрос {number}?", kind="corpus_specific",
                       facts=("a",), sources=tuple(sources))


def _mode(*titles: str, candidates: int = 10, seconds: float = 0.1, error=None,
          search_text=None) -> RetrievalResult:
    chunks = [{"title": t, "chunk_index": i, "score": 0.9 - 0.05 * i} for i, t in enumerate(titles)]
    return RetrievalResult(chunks=chunks, candidates=candidates, seconds=seconds, error=error,
                           search_text=search_text)


def _retrieval(number=1, sources=("Облигации",), missing=(), **modes) -> QuestionRetrieval:
    return QuestionRetrieval(question=_question(number, sources), modes=dict(modes),
                             missing_sources=list(missing))


# --- выбор режимов и уровня --------------------------------------------------


def test_parse_modes_empty_means_no_explicit_choice():
    assert rm.parse_modes("") == []
    assert rm.parse_modes("  ,  ") == []


def test_parse_modes_normalizes_case_order_and_duplicates():
    assert rm.parse_modes(" Rewrite , BASELINE, rewrite ") == [rm.MODE_BASELINE, rm.MODE_REWRITE]


def test_parse_modes_rejects_unknown_names():
    with pytest.raises(rm.ModeSettingError) as error:
        rm.parse_modes("baseline,magic")
    assert "magic" in str(error.value)


def test_parse_level():
    assert rm.parse_level(" Search ") == rm.LEVEL_SEARCH
    assert rm.parse_level("answers") == rm.LEVEL_ANSWERS
    with pytest.raises(rm.ModeSettingError):
        rm.parse_level("fast")


def test_resolve_empty_choice_search_level_means_all_modes():
    active, unavailable = rm.resolve_modes([], rm.LEVEL_SEARCH, rewrite_configured=True)
    assert active == list(rm.RETRIEVAL_MODES)
    assert unavailable == {}


def test_resolve_empty_choice_answers_level_means_no_extra_modes():
    assert rm.resolve_modes([], rm.LEVEL_ANSWERS, rewrite_configured=True) == ([], {})


def test_resolve_rewrite_modes_unavailable_without_rewrite_with_reason():
    active, unavailable = rm.resolve_modes([], rm.LEVEL_SEARCH, rewrite_configured=False)
    assert active == [rm.MODE_BASELINE, rm.MODE_FILTER, rm.MODE_CONCAT]
    assert set(unavailable) == {rm.MODE_REWRITE, rm.MODE_REWRITE_FILTER, rm.MODE_REWRITE_CONTEXT}
    assert all(reason == rm.REWRITE_NOT_CONFIGURED for reason in unavailable.values())


def test_resolve_explicit_choice_is_respected():
    active, unavailable = rm.resolve_modes(
        [rm.MODE_FILTER], rm.LEVEL_ANSWERS, rewrite_configured=False
    )
    assert active == [rm.MODE_FILTER] and unavailable == {}


def test_mode_flags():
    assert not rm.uses_rewrite(rm.MODE_BASELINE) and not rm.uses_filter(rm.MODE_BASELINE)
    assert rm.uses_filter(rm.MODE_FILTER) and not rm.uses_rewrite(rm.MODE_FILTER)
    assert rm.uses_rewrite(rm.MODE_REWRITE) and not rm.uses_filter(rm.MODE_REWRITE)
    assert rm.uses_rewrite(rm.MODE_REWRITE_FILTER) and rm.uses_filter(rm.MODE_REWRITE_FILTER)


# --- метрики поиска ------------------------------------------------------------


def test_first_hit_position_is_one_based_and_case_insensitive():
    assert rm.first_hit_position(("облигации",), ["Акции", "Облигации — основы"]) == 2
    assert rm.first_hit_position(("Облигации",), ["Акции", "Фонды"]) is None
    assert rm.first_hit_position(("Облигации",), []) is None


def test_irrelevant_count_counts_chunks_outside_expected_documents():
    assert rm.irrelevant_count(("Облигации",), ["Облигации 1", "Акции", "Фонды"]) == 2
    assert rm.irrelevant_count(("Облигации",), []) == 0


def test_aggregate_hit_rate_position_mrr_and_irrelevant_share():
    results = [
        _retrieval(1, baseline=_mode("Акции", "Облигации 1"),
                   rewrite=_mode("Облигации 1", "Облигации 2")),
        _retrieval(2, baseline=_mode("Акции", "Фонды"),
                   rewrite=_mode("Фонды", "Облигации 3")),
    ]
    agg = rm.aggregate_retrieval(results, [rm.MODE_BASELINE, rm.MODE_REWRITE])
    base, rewrite = agg[rm.MODE_BASELINE], agg[rm.MODE_REWRITE]
    assert (base["questions"], base["hits"], base["hit_rate"]) == (2, 1, 0.5)
    assert base["mean_position"] == 2.0
    assert base["mrr"] == pytest.approx(0.25)  # (1/2 + 0) / 2
    assert base["irrelevant_share"] == pytest.approx(3 / 4)
    assert (rewrite["hits"], rewrite["hit_rate"]) == (2, 1.0)
    assert rewrite["mean_position"] == 1.5
    assert rewrite["mrr"] == pytest.approx(0.75)  # (1 + 1/2) / 2
    assert rewrite["irrelevant_share"] == pytest.approx(1 / 4)


def test_aggregate_excludes_questions_whose_documents_are_not_indexed():
    results = [
        _retrieval(1, baseline=_mode("Облигации 1")),
        _retrieval(2, missing=("Крипта",), sources=("Крипта",), baseline=_mode("Акции")),
    ]
    agg = rm.aggregate_retrieval(results, [rm.MODE_BASELINE])[rm.MODE_BASELINE]
    assert agg["questions"] == 1 and agg["hit_rate"] == 1.0
    assert agg["irrelevant_share"] == 0.0


def test_aggregate_failed_search_is_counted_separately_not_as_miss():
    results = [
        _retrieval(1, baseline=_mode("Облигации 1")),
        _retrieval(2, baseline=_mode(error="Ollama недоступна")),
    ]
    agg = rm.aggregate_retrieval(results, [rm.MODE_BASELINE])[rm.MODE_BASELINE]
    assert (agg["questions"], agg["failed"], agg["hit_rate"]) == (1, 1, 1.0)


def test_aggregate_counts_empty_selection_and_handles_no_data():
    results = [_retrieval(1, filter=_mode())]
    agg = rm.aggregate_retrieval(results, [rm.MODE_FILTER, rm.MODE_BASELINE])
    assert agg[rm.MODE_FILTER]["empty"] == 1
    assert agg[rm.MODE_FILTER]["irrelevant_share"] is None
    assert agg[rm.MODE_FILTER]["mean_position"] is None
    assert agg[rm.MODE_BASELINE]["questions"] == 0
    assert agg[rm.MODE_BASELINE]["hit_rate"] is None


# --- отчёт и тексты -------------------------------------------------------------


def _report(results, modes, unavailable=None, **kwargs):
    return rm.build_retrieval_report(
        started_at="2026-10-02 10:00:00", finished_at="2026-10-02 10:01:00",
        settings={"strategy": "structural", "top_k": 3, "candidates": 10},
        results=results, planned=len(results), modes=modes,
        unavailable=unavailable or {}, **kwargs,
    )


def test_report_roundtrip_preserves_modes_rewrite_and_chunks():
    original = QuestionRetrieval(
        question=_question(3),
        modes={rm.MODE_REWRITE: _mode("Облигации 1", search_text="облигации срок")},
        rewrite_query="облигации срок", rewrite_seconds=1.5, missing_sources=["X"],
    )
    restored = QuestionRetrieval.from_dict(original.to_dict())
    assert restored.rewrite_query == "облигации срок" and restored.rewrite_seconds == 1.5
    assert restored.missing_sources == ["X"]
    assert restored.modes[rm.MODE_REWRITE].titles == ["Облигации 1"]
    assert restored.modes[rm.MODE_REWRITE].search_text == "облигации срок"


def test_report_is_marked_as_retrieval_kind():
    report = _report([_retrieval(1, baseline=_mode("Облигации 1"))], [rm.MODE_BASELINE])
    assert rm.is_retrieval_report(report)
    assert not rm.is_retrieval_report({"questions": []})


def test_summary_shows_delta_against_baseline_and_unavailable_modes():
    results = [
        _retrieval(1, baseline=_mode("Акции", "Фонды"), filter=_mode("Облигации 1")),
        _retrieval(2, baseline=_mode("Облигации 1"), filter=_mode("Облигации 2")),
    ]
    report = _report(results, [rm.MODE_BASELINE, rm.MODE_FILTER],
                     unavailable={rm.MODE_REWRITE: rm.REWRITE_NOT_CONFIGURED})
    text = rm.format_retrieval_summary(report)
    assert "Сравнение режимов поиска" in text
    assert "как до изменения" in text and "с отбором" in text
    assert "+50 п.п. к базовому" in text  # 50% -> 100%
    assert "«с переписыванием» недоступен" in text
    assert ev.DISCLAIMER in text


def test_summary_marks_stopped_run_and_excluded_questions():
    report = _report([_retrieval(1, missing=("Крипта",), baseline=_mode("Акции"))],
                     [rm.MODE_BASELINE], status=ev.STATUS_STOPPED, stop_reason="по команде")
    text = rm.format_retrieval_summary(report)
    assert "Прогон остановлен" in text and "по команде" in text
    assert "нет в индексе" in text
    assert "Вопросов в расчёте: 0 из 1" in text


def test_table_has_a_cell_per_mode():
    results = [_retrieval(1, baseline=_mode("Акции", "Облигации 1"), filter=_mode())]
    text = rm.format_retrieval_table(_report(results, [rm.MODE_BASELINE, rm.MODE_FILTER]))
    assert "#2" in text and "пусто" in text


def test_detail_lists_chunks_with_hit_marks_and_rewrite():
    item = _retrieval(1, baseline=_mode("Акции", "Облигации 1"))
    item.rewrite_query = "облигации критерии"
    item.rewrite_seconds = 0.8
    text = rm.format_retrieval_detail(_report([item], [rm.MODE_BASELINE]), 1)
    assert "Переписанный запрос" in text and "облигации критерии" in text
    assert "▫️ 1. Акции" in text and "✅ 2. Облигации 1" in text


def test_detail_unknown_number_returns_none():
    assert rm.format_retrieval_detail(_report([], []), 5) is None


# --- режимы поиска в сравнении ответов -----------------------------------------------


def _answer_mode(coverage, titles=()):
    found = round(coverage * 4)
    verdict = ev.Verdict(present=tuple([True] * found + [False] * (4 - found)))
    return ev.ModeResult(
        answer="ответ", verdict=verdict, elapsed=2.0,
        rag_sources=[{"title": t, "chunk_index": 0, "score": 0.7} for t in titles],
    )


def _answer_result(off, on, variants, number=1):
    return ev.QuestionResult(question=_question(number), off=off, on=on, variants=variants)


def test_variants_roundtrip_through_question_result_dict():
    result = _answer_result(_answer_mode(0.25), _answer_mode(0.5),
                            {rm.MODE_FILTER: _answer_mode(0.75, titles=("Облигации 1",))})
    restored = ev.QuestionResult.from_dict(result.to_dict())
    assert restored.variants[rm.MODE_FILTER].coverage == 0.75
    assert ev.QuestionResult.from_dict({**result.to_dict(), "variants": None}).variants == {}


def test_old_report_without_variants_still_loads():
    result = _answer_result(_answer_mode(0.25), _answer_mode(0.5), {})
    data = result.to_dict()
    del data["variants"]
    assert ev.QuestionResult.from_dict(data).variants == {}


def test_aggregate_variants_compares_against_no_rag():
    results = [
        _answer_result(_answer_mode(0.25), _answer_mode(0.5),
                       {rm.MODE_FILTER: _answer_mode(0.75, titles=("Облигации 1",))}, number=1),
        _answer_result(_answer_mode(0.5), _answer_mode(0.5),
                       {rm.MODE_FILTER: _answer_mode(0.25)}, number=2),
    ]
    agg = rm.aggregate_variants(results, [rm.MODE_FILTER])[rm.MODE_FILTER]
    assert agg["scored"] == 2
    assert agg["avg"] == pytest.approx(0.5) and agg["avg_off"] == pytest.approx(0.375)
    assert (agg["better"], agg["equal"], agg["worse"]) == (1, 0, 1)
    assert agg["hit"] == 1 and agg["fired"] == 1


def test_aggregate_variants_skips_failed_and_search_failed_answers():
    failed = ev.ModeResult(error="сбой")
    no_search = _answer_mode(0.5)
    no_search.search_failed = True
    results = [_answer_result(_answer_mode(0.25), _answer_mode(0.5),
                              {rm.MODE_FILTER: failed, rm.MODE_BASELINE: no_search})]
    agg = rm.aggregate_variants(results, [rm.MODE_FILTER, rm.MODE_BASELINE])
    assert agg[rm.MODE_FILTER]["scored"] == 0 and agg[rm.MODE_BASELINE]["scored"] == 0
    assert agg[rm.MODE_FILTER]["avg"] is None


def test_variants_summary_is_none_without_extra_modes_and_shows_unavailable():
    plain = {"questions": [], "variant_modes": []}
    assert rm.format_variants_summary(plain) is None
    report = {"questions": [], "variant_modes": [],
              "unavailable_modes": {rm.MODE_REWRITE: rm.REWRITE_NOT_CONFIGURED}}
    assert "«с переписыванием» недоступен" in rm.format_variants_summary(report)


def test_variants_summary_lists_each_mode():
    result = _answer_result(_answer_mode(0.25), _answer_mode(0.5),
                            {rm.MODE_FILTER: _answer_mode(0.75, titles=("Облигации 1",))})
    report = {"questions": [result.to_dict()], "variant_modes": [rm.MODE_FILTER]}
    text = rm.format_variants_summary(report)
    assert "с отбором" in text and "лучше: 1" in text


# --- остановка прогона по поиску ---------------------------------------------------------


def test_question_failed_only_when_every_mode_failed():
    all_failed = _retrieval(1, baseline=_mode(error="x"), filter=_mode(error="y"))
    some_ok = _retrieval(2, baseline=_mode(error="x"), filter=_mode("Облигации 1"))
    assert rm.question_failed(all_failed)
    assert not rm.question_failed(some_ok)
    assert not rm.question_failed(_retrieval(3))  # режимов нет — не сбой


def test_should_stop_counts_trailing_failures_and_resets_on_success():
    bad = lambda n: _retrieval(n, baseline=_mode(error="x"))  # noqa: E731
    good = lambda n: _retrieval(n, baseline=_mode("Облигации 1"))  # noqa: E731
    assert rm.should_stop([good(1), bad(2), bad(3)], 2)
    assert not rm.should_stop([bad(1), bad(2), good(3)], 2)
    assert not rm.should_stop([bad(1)], 2)
    assert not rm.should_stop([bad(1), bad(2)], 0)


def test_question_detail_includes_answers_in_extra_search_modes():
    result = _answer_result(_answer_mode(0.25), _answer_mode(0.5),
                            {rm.MODE_FILTER: _answer_mode(0.75, titles=("Облигации 1",))})
    report = {"questions": [result.to_dict()]}
    text = ev.format_question_detail(report, 1, rm.MODE_TITLES)
    assert "С RAG, режим поиска «с отбором»" in text
    assert "Облигации 1 — фрагмент" in text
    # без подписей режим показывается по ключу, прежние отчёты выводятся как раньше
    assert "режим поиска «filter»" in ev.format_question_detail(report, 1)
    plain = _answer_result(_answer_mode(0.25), _answer_mode(0.5), {})
    assert "режим поиска" not in ev.format_question_detail({"questions": [plain.to_dict()]}, 1)


# --- эталонные шаги отбора для режимов «с отбором» ----------------------------------------


def test_filter_steps_neutral_detection():
    assert rm.filter_steps_neutral(3, 1.0, 0, 3)
    assert rm.filter_steps_neutral(3, 1.0, 0, 5)
    assert not rm.filter_steps_neutral(3, 0.08, 0, 3)
    assert not rm.filter_steps_neutral(3, 1.0, 50, 3)
    assert not rm.filter_steps_neutral(3, 1.0, 0, 2)


def test_filter_settings_use_reference_values_when_operator_steps_are_off():
    settings, reference = rm.filter_settings(10, 3, 1.0, 0, 3)
    assert reference
    assert settings == rm.REFERENCE_FILTER


def test_filter_settings_keep_candidates_if_larger_than_reference():
    settings, reference = rm.filter_settings(25, 3, 1.0, 0, 3)
    assert reference and settings["candidates"] == 25
    assert settings["relative_margin"] == rm.REFERENCE_FILTER["relative_margin"]


def test_filter_settings_use_operator_values_when_any_step_is_on():
    settings, reference = rm.filter_settings(12, 3, 0.2, 0, 3)
    assert not reference
    assert settings == {
        "candidates": 12,
        "relative_margin": 0.2,
        "min_chunk_chars": 0,
        "max_per_doc": 3,
    }


def test_summary_mentions_reference_filter_origin():
    steps, reference = rm.filter_settings(10, 3, 1.0, 0, 3)
    report = rm.build_retrieval_report(
        started_at="-", finished_at="-",
        settings={"filter_steps": steps, "filter_reference": reference},
        results=[], planned=0, modes=[rm.MODE_BASELINE], unavailable={},
    )
    text = rm.format_retrieval_summary(report)
    assert "эталонные, у оператора шаги выключены" in text
    assert "отн. порог 0.08" in text


# --- режимы с контекстом диалога (изменение add-rag-rewrite-dialog-context) -----------------------


def _dialog_question(number: int, sources=("Облигации",), history=()) -> ev.Question:
    return ev.Question(number=number, question=f"Вопрос {number}?", kind="concept",
                       facts=("a",), sources=tuple(sources), history=tuple(history))


def _dialog_retrieval(number, history=(), sources=("Облигации",), **modes) -> QuestionRetrieval:
    return QuestionRetrieval(question=_dialog_question(number, sources, history), modes=dict(modes))


def test_new_modes_are_appended_after_the_existing_ones_in_order():
    assert rm.RETRIEVAL_MODES[:4] == (
        rm.MODE_BASELINE, rm.MODE_FILTER, rm.MODE_REWRITE, rm.MODE_REWRITE_FILTER)
    assert rm.RETRIEVAL_MODES[4:] == (rm.MODE_REWRITE_CONTEXT, rm.MODE_CONCAT)
    assert rm.MODE_TITLES[rm.MODE_REWRITE_CONTEXT] == "с контекстом диалога"
    assert rm.MODE_TITLES[rm.MODE_CONCAT] == "склейка"


def test_mode_flags_for_dialog_modes():
    assert rm.uses_rewrite(rm.MODE_REWRITE_CONTEXT) and not rm.uses_filter(rm.MODE_REWRITE_CONTEXT)
    assert rm.uses_history(rm.MODE_REWRITE_CONTEXT) and rm.uses_history(rm.MODE_CONCAT)
    assert not rm.uses_rewrite(rm.MODE_CONCAT) and not rm.uses_filter(rm.MODE_CONCAT)
    assert not rm.uses_history(rm.MODE_REWRITE) and not rm.uses_history(rm.MODE_BASELINE)


def test_parse_modes_accepts_dialog_modes_in_canonical_order():
    assert rm.parse_modes("concat, rewrite_context, baseline") == [
        rm.MODE_BASELINE, rm.MODE_REWRITE_CONTEXT, rm.MODE_CONCAT]


def test_concat_works_without_rewrite_but_context_mode_needs_it():
    active, unavailable = rm.resolve_modes(
        [rm.MODE_REWRITE_CONTEXT, rm.MODE_CONCAT], rm.LEVEL_SEARCH, rewrite_configured=False)
    assert active == [rm.MODE_CONCAT]
    assert unavailable == {rm.MODE_REWRITE_CONTEXT: rm.REWRITE_NOT_CONFIGURED}


def test_dialog_modes_are_unavailable_at_the_answers_level_with_a_reason():
    active, unavailable = rm.resolve_modes(
        [rm.MODE_FILTER, rm.MODE_REWRITE_CONTEXT, rm.MODE_CONCAT],
        rm.LEVEL_ANSWERS, rewrite_configured=True)
    assert active == [rm.MODE_FILTER]
    assert set(unavailable) == {rm.MODE_REWRITE_CONTEXT, rm.MODE_CONCAT}
    assert all(reason == rm.HISTORY_MODE_ANSWERS_REASON for reason in unavailable.values())


def test_history_questions_setting_uses_working_value_or_the_reference():
    assert rm.history_questions_setting(2) == (2, False)
    assert rm.history_questions_setting(0) == (rm.REFERENCE_HISTORY_QUESTIONS, True)
    assert rm.REFERENCE_HISTORY_QUESTIONS == 3


def test_concat_text_joins_the_last_history_question_with_the_current_one():
    assert rm.concat_text(["Как выбирать облигации?", "А ОФЗ?"], "А акции?") == "А ОФЗ? А акции?"
    assert rm.concat_text(("Как выбирать облигации?",), "А акции?") == (
        "Как выбирать облигации? А акции?")


def test_concat_text_without_history_is_the_question_itself():
    assert rm.concat_text([], "А акции?") == "А акции?"
    assert rm.concat_text((), "А акции?") == "А акции?"


def test_split_by_history_separates_multi_turn_from_standalone_questions():
    multi_item = _dialog_retrieval(1, history=["Как выбирать облигации?"])
    single_item = _dialog_retrieval(2)
    multi, single = rm.split_by_history([multi_item, single_item])
    assert multi == [multi_item] and single == [single_item]
    assert rm.is_multi_turn(multi_item) and not rm.is_multi_turn(single_item)


def test_dialog_report_fields_survive_a_roundtrip():
    item = _dialog_retrieval(3, history=["Как выбирать облигации?"],
                             rewrite_context=RetrievalResult(history_used=1, search_text="акции"))
    item.context_rewrite_query = "выбор акций"
    item.context_rewrite_seconds = 1.2
    item.context_rewrite_failure = None
    restored = QuestionRetrieval.from_dict(item.to_dict())
    assert restored.question.history == ("Как выбирать облигации?",)
    assert restored.context_rewrite_query == "выбор акций"
    assert restored.context_rewrite_seconds == 1.2
    assert restored.modes[rm.MODE_REWRITE_CONTEXT].history_used == 1


def test_old_retrieval_report_without_dialog_fields_still_loads():
    item = _dialog_retrieval(1, baseline=_mode("Облигации 1"))
    data = item.to_dict()
    for key in ("history", "context_rewrite"):
        del data[key]
    data["modes"]["baseline"].pop("history_used")
    restored = QuestionRetrieval.from_dict(data)
    assert restored.question.history == ()
    assert restored.context_rewrite_query is None
    assert restored.modes["baseline"].history_used == 0


def _dialog_report(results, modes, **settings):
    return rm.build_retrieval_report(
        started_at="-", finished_at="-", settings=settings, results=results,
        planned=len(results), modes=modes, unavailable={})


def test_summary_splits_metrics_into_multi_turn_and_standalone_groups():
    results = [
        _dialog_retrieval(1, history=["Как выбирать облигации?"],
                          baseline=_mode("Акции"), rewrite_context=_mode("Облигации 1")),
        _dialog_retrieval(2, baseline=_mode("Облигации 1"), rewrite_context=_mode("Облигации 1")),
    ]
    text = rm.format_retrieval_summary(
        _dialog_report(results, [rm.MODE_BASELINE, rm.MODE_REWRITE_CONTEXT]))
    assert "▶ Многоходовые вопросы (1)" in text and "▶ Самостоятельные вопросы (1)" in text
    multi_block = text.split("▶ Многоходовые вопросы (1)")[1].split("▶ Самостоятельные")[0]
    assert "попал в ожидаемый документ: 0 из 1" in multi_block  # baseline промахнулся
    assert "+100 п.п. к базовому" in multi_block  # контекст диалога попал


def test_summary_has_no_groups_when_there_is_only_one_kind_of_question():
    results = [_dialog_retrieval(1, baseline=_mode("Облигации 1"))]
    text = rm.format_retrieval_summary(_dialog_report(results, [rm.MODE_BASELINE]))
    assert "▶" not in text


def test_summary_reports_context_rewrite_statistics():
    item = _dialog_retrieval(1, history=["Как выбирать облигации?"], baseline=_mode("Акции"))
    item.context_rewrite_query = "выбор акций"
    item.context_rewrite_seconds = 2.0
    text = rm.format_retrieval_summary(_dialog_report([item], [rm.MODE_BASELINE]))
    assert "Переписывание с историей диалога: удалось 1 из 1" in text


def test_summary_settings_mention_the_reference_history_size():
    text = rm.format_retrieval_summary(_dialog_report(
        [], [rm.MODE_BASELINE], history_questions=3, history_reference=True))
    assert "эталонное" in text and "в режиме «с контекстом диалога» (эталонное): 3" in text
    working = rm.format_retrieval_summary(_dialog_report(
        [], [rm.MODE_BASELINE], history_questions=2, history_reference=False))
    assert "(рабочее): 2" in working


def test_detail_shows_history_and_both_rewrites():
    item = _dialog_retrieval(1, history=["Как выбирать облигации?", "А ОФЗ?"],
                             baseline=_mode("Акции"))
    item.rewrite_query = "акции"
    item.rewrite_seconds = 1.0
    item.context_rewrite_query = "выбор акций после облигаций"
    item.context_rewrite_seconds = 1.5
    text = rm.format_retrieval_detail(_dialog_report([item], [rm.MODE_BASELINE]), 1)
    assert "Как выбирать облигации? → А ОФЗ?" in text
    assert "Переписанный запрос (1.0 с): акции" in text
    assert "Переписанный запрос с историей (1.5 с): выбор акций после облигаций" in text


# --- Дополнение: тексты сводки, таблицы и деталей, блок режимов в ответах -------------------


def _hq(number: int, *, history=(), sources=("Облигации",)) -> ev.Question:
    return ev.Question(
        number=number,
        question=f"Вопрос {number}?",
        kind="corpus_specific",
        facts=("a",),
        sources=tuple(sources),
        history=tuple(history),
    )


def _rep(results, modes, **overrides) -> dict:
    report = rm.build_retrieval_report(
        started_at="2026-10-01 12:00:00",
        finished_at="2026-10-01 12:05:00",
        settings={"strategy": "structural", "top_k": 3, "min_score": 0.6},
        results=results,
        planned=len(results),
        modes=modes,
        unavailable={},
    )
    report.update(overrides)
    return report


def test_delta_pp_cases():
    assert rm._delta_pp(None, 0.5) == ""
    assert rm._delta_pp(0.5, None) == ""
    assert rm._delta_pp(0.75, 0.5) == ", +25 п.п. к базовому"
    assert rm._delta_pp(0.25, 0.5) == ", -25 п.п. к базовому"
    assert rm._delta_pp(0.5, 0.5) == ", как базовый"
    assert rm._delta_pp(0.504, 0.5) == ", как базовый"


def test_pct_and_number_show_missing_values():
    assert rm._pct(None) == "н/д"
    assert rm._pct(0.5) == "50%"
    assert rm._number(None, ".1f") == "н/д"
    assert rm._number(1.25, ".1f") == "1.2"


def test_mode_blocks_show_metrics_delta_to_baseline_and_failures():
    results = [
        _retrieval(1, **{rm.MODE_BASELINE: _mode("Другое"), rm.MODE_FILTER: _mode("Облигации А")}),
        _retrieval(2, **{rm.MODE_BASELINE: _mode("Облигации Б"), rm.MODE_FILTER: _mode(error="x")}),
    ]
    modes = [rm.MODE_BASELINE, rm.MODE_FILTER]
    aggregates = rm.aggregate_retrieval(results, modes)
    text = "\n".join(rm._mode_blocks(aggregates, modes))
    assert "• как до изменения" in text
    assert "• с отбором" in text
    assert "попал в ожидаемый документ: 1 из 2 (50%)" in text
    assert "попал в ожидаемый документ: 1 из 1 (100%, +50 п.п. к базовому)" in text
    assert "⚠️ поиск не удался на 1 вопросах" in text
    assert "время поиска:" in text


def test_mode_blocks_without_baseline_have_no_delta():
    results = [_retrieval(1, **{rm.MODE_FILTER: _mode("Облигации")})]
    aggregates = rm.aggregate_retrieval(results, [rm.MODE_FILTER])
    text = "\n".join(rm._mode_blocks(aggregates, [rm.MODE_FILTER]))
    assert "к базовому" not in text
    assert "как базовый" not in text


def test_mode_blocks_of_mode_without_answers_show_dashes():
    aggregates = rm.aggregate_retrieval([], [rm.MODE_BASELINE])
    text = "\n".join(rm._mode_blocks(aggregates, [rm.MODE_BASELINE]))
    assert "0 из 0 (н/д)" in text
    assert "MRR: н/д" in text


def test_settings_line_lists_known_keys_history_and_filter_steps():
    line = rm._settings_line(
        {
            "strategy": "structural",
            "top_k": 3,
            "candidates": 8,
            "min_score": 0.6,
            "relative_margin": 0.1,
            "min_chunk_chars": 50,
            "max_per_doc": 2,
            "embeddings_model": "emb",
            "rewrite_provider": "ollama",
            "rewrite_model": "rw",
            "rewrite_search_mode": "both",
            "history_questions": 3,
            "history_reference": True,
            "filter_steps": {
                "candidates": 10,
                "relative_margin": 0.08,
                "min_chunk_chars": 120,
                "max_per_doc": 2,
            },
            "filter_reference": True,
        }
    )
    assert line.startswith("Настройки: стратегия: structural, топ-K: 3, кандидатов: 8")
    assert "эмбеддинги: emb" in line
    assert "переписывание: ollama" in line
    assert "прошлых вопросов в режиме «с контекстом диалога» (эталонное): 3" in line
    assert "эталонные, у оператора шаги выключены" in line
    assert line.endswith(".")


def test_settings_line_working_origin_and_empty_values():
    assert rm._settings_line({}) == ""
    assert rm._settings_line({"strategy": "", "top_k": None}) == ""
    line = rm._settings_line(
        {"history_questions": 2, "filter_steps": {"candidates": 5}, "filter_reference": False}
    )
    assert "(рабочее): 2" in line
    assert "(рабочие)" in line


def test_summary_of_stopped_run_names_reason_and_counts():
    results = [_retrieval(1, **{rm.MODE_BASELINE: _mode("Облигации")})]
    report = _rep(results, [rm.MODE_BASELINE], status=ev.STATUS_STOPPED, stop_reason="по команде")
    report["planned"] = 5
    text = rm.format_retrieval_summary(report)
    assert "⛔ Прогон остановлен: по команде. Обработано 1 из 5 вопросов" in text
    assert "Прогон остановлен: 2026-10-01 12:05:00 (UTC)." in text
    report["stop_reason"] = None
    assert "причина не указана" in rm.format_retrieval_summary(report)


def test_summary_of_completed_run_has_no_stop_line_but_has_settings_and_disclaimer():
    results = [_retrieval(1, **{rm.MODE_BASELINE: _mode("Облигации")})]
    text = rm.format_retrieval_summary(_rep(results, [rm.MODE_BASELINE]))
    assert "⛔" not in text
    assert "Прогон завершён: 2026-10-01 12:05:00 (UTC)." in text
    assert "Настройки: стратегия: structural" in text
    assert "Вопросов в расчёте: 1 из 1." in text
    assert ev.DISCLAIMER in text


def test_summary_names_unavailable_modes_and_missing_index_documents():
    results = [
        _retrieval(1, **{rm.MODE_BASELINE: _mode("Облигации")}),
        _retrieval(2, missing=["Акции"], **{rm.MODE_BASELINE: _mode("Другое")}),
    ]
    report = _rep(
        results,
        [rm.MODE_BASELINE],
        unavailable_modes={rm.MODE_REWRITE: rm.REWRITE_NOT_CONFIGURED},
        index_missing=["Акции"],
    )
    text = rm.format_retrieval_summary(report)
    assert f"⚠️ Режим «с переписыванием» недоступен: {rm.REWRITE_NOT_CONFIGURED}." in text
    assert "Вопросов в расчёте: 1 из 2." in text
    assert "исключены из метрик) на вопросах: 2." in text
    assert "Не найдены в индексе: Акции." in text


def test_summary_splits_metrics_between_multi_turn_and_standalone_questions():
    single = QuestionRetrieval(
        question=_hq(1), modes={rm.MODE_BASELINE: _mode("Облигации")}
    )
    multi = QuestionRetrieval(
        question=_hq(2, history=("Раньше?",)), modes={rm.MODE_BASELINE: _mode("Другое")}
    )
    text = rm.format_retrieval_summary(_rep([single, multi], [rm.MODE_BASELINE]))
    assert "▶ Многоходовые вопросы (1)" in text
    assert "▶ Самостоятельные вопросы (1)" in text
    only_single = rm.format_retrieval_summary(_rep([single], [rm.MODE_BASELINE]))
    assert "▶" not in only_single


def test_summary_reports_rewrite_statistics_with_and_without_history():
    ok = QuestionRetrieval(
        question=_hq(1),
        modes={rm.MODE_BASELINE: _mode("Облигации")},
        rewrite_query="облигации", rewrite_seconds=1.0,
        context_rewrite_query="облигации снова", context_rewrite_seconds=3.0,
    )
    failed = QuestionRetrieval(
        question=_hq(2),
        modes={rm.MODE_BASELINE: _mode("Облигации")},
        rewrite_failure="таймаут", rewrite_seconds=3.0,
        context_rewrite_failure="таймаут", context_rewrite_seconds=1.0,
    )
    text = rm.format_retrieval_summary(_rep([ok, failed], [rm.MODE_BASELINE]))
    assert "Переписывание: удалось 1 из 2, среднее время 2.0 с." in text
    assert "Переписывание с историей диалога: удалось 1 из 2, среднее время 2.0 с." in text


def test_summary_adds_queue_line_when_users_were_served():
    results = [_retrieval(1, **{rm.MODE_BASELINE: _mode("Облигации")})]
    report = _rep(results, [rm.MODE_BASELINE], queue={"slot_wait_seconds": 20, "user_questions": 2})
    assert "Очередь:" in rm.format_retrieval_summary(report)
    assert "Очередь:" not in rm.format_retrieval_summary(_rep(results, [rm.MODE_BASELINE]))


def test_summary_ignores_unknown_mode_keys_of_the_report():
    results = [_retrieval(1, **{rm.MODE_BASELINE: _mode("Облигации")})]
    text = rm.format_retrieval_summary(_rep(results, [rm.MODE_BASELINE, "magic"]))
    assert "magic" not in text


def test_mode_cell_cases():
    question = _question(sources=("Облигации",))
    assert rm._mode_cell(question, None) == "—"
    assert rm._mode_cell(question, _mode(error="x")) == "сбой"
    assert rm._mode_cell(question, _mode()) == "пусто"
    assert rm._mode_cell(question, _mode("Акции", "Облигации А")) == "#2"
    assert rm._mode_cell(question, _mode("Акции")) == "мимо"


def test_table_has_header_cells_per_mode_and_missing_source_note():
    results = [
        _retrieval(1, **{rm.MODE_BASELINE: _mode("Облигации"), rm.MODE_FILTER: _mode("Акции")}),
        _retrieval(2, missing=["Облигации"], **{rm.MODE_BASELINE: _mode(error="x")}),
    ]
    text = rm.format_retrieval_table(_rep(results, [rm.MODE_BASELINE, rm.MODE_FILTER]))
    header = "📋 По вопросам: позиция первого попадания (как до изменения · с отбором)"
    assert text.startswith(header)
    assert "1. Вопрос 1?\n   #1 · мимо" in text
    assert "2. Вопрос 2?\n   сбой · — ⚠️ источника нет в индексе" in text


def test_table_cuts_long_titles():
    question = ev.Question(1, "я" * 60, "concept", ("a",), ("Облигации",))
    item = QuestionRetrieval(question=question, modes={rm.MODE_BASELINE: _mode("Облигации")})
    text = rm.format_retrieval_table(_rep([item], [rm.MODE_BASELINE]))
    assert "я" * 37 + "…" in text
    assert "я" * 38 not in text


def test_detail_of_unknown_number_is_none():
    assert rm.format_retrieval_detail(_rep([], [rm.MODE_BASELINE]), 3) is None


def test_detail_shows_question_sources_history_and_missing_documents():
    item = QuestionRetrieval(
        question=_hq(1, history=("Раз?", "Два?"), sources=("Облигации", "Акции")),
        modes={rm.MODE_BASELINE: _mode("Облигации")},
        missing_sources=["Акции"],
    )
    text = rm.format_retrieval_detail(_rep([item], [rm.MODE_BASELINE]), 1)
    assert "❓ Вопрос 1 (по специфике корпуса): Вопрос 1?" in text
    assert "Ожидаемые источники: Облигации; Акции" in text
    assert "История диалога (предыдущие вопросы): Раз? → Два?" in text
    assert "⚠️ Нет в индексе: Акции" in text


def test_detail_shows_rewrites_and_their_failures():
    done = QuestionRetrieval(
        question=_hq(1), modes={}, rewrite_query="облигации", rewrite_seconds=1.234,
        context_rewrite_query="облигации снова", context_rewrite_seconds=2.0,
    )
    text = rm.format_retrieval_detail(_rep([done], []), 1)
    assert "Переписанный запрос (1.2 с): облигации" in text
    assert "Переписанный запрос с историей (2.0 с): облигации снова" in text
    failed = QuestionRetrieval(
        question=_hq(1), modes={}, rewrite_failure="таймаут", context_rewrite_failure="отказ"
    )
    text = rm.format_retrieval_detail(_rep([failed], []), 1)
    assert "Переписывание не удалось: таймаут" in text
    assert "Переписывание с историей не удалось: отказ" in text


def test_detail_of_every_mode_state():
    item = QuestionRetrieval(
        question=_hq(1),
        modes={
            rm.MODE_BASELINE: _mode("Облигации А", "Акции", candidates=7),
            rm.MODE_FILTER: _mode(error="индекс повреждён"),
            rm.MODE_REWRITE: _mode(candidates=4),
        },
    )
    text = rm.format_retrieval_detail(
        _rep([item], [rm.MODE_BASELINE, rm.MODE_FILTER, rm.MODE_REWRITE, rm.MODE_CONCAT]), 1
    )
    assert "— Как до изменения —" in text
    assert "Кандидатов найдено: 7, в выдаче: 2." in text
    assert "✅ 1. Облигации А — фрагмент 0, близость 0.90" in text
    assert "▫️ 2. Акции — фрагмент 1, близость 0.85" in text
    assert "Сбой поиска: индекс повреждён" in text
    assert "Материалов нет (кандидатов найдено: 4)." in text
    assert "— Склейка —\nНет данных." in text


def _variants_report(variants: dict, modes=None, unavailable=None) -> dict:
    off = ev.ModeResult(answer="а", verdict=ev.Verdict(present=(True, False)))
    on = ev.ModeResult(answer="б", verdict=ev.Verdict(present=(True, True)))
    result = ev.QuestionResult(question=_hq(1), off=off, on=on, variants=variants)
    extra = {}
    if modes is not None:
        extra["variant_modes"] = modes
    if unavailable is not None:
        extra["unavailable_modes"] = unavailable
    return ev.build_report(
        started_at="a",
        finished_at="b",
        settings={},
        results=[result],
        planned=1,
        extra=extra or None,
    )


def test_variants_summary_is_none_without_extra_modes():
    assert rm.format_variants_summary(_variants_report({})) is None
    assert rm.format_variants_summary(_variants_report({}, modes=[], unavailable={})) is None


def test_variants_summary_for_scored_mode_shows_coverage_counters_and_time():
    mode = ev.ModeResult(
        answer="в",
        verdict=ev.Verdict(present=(True, True)),
        rag_sources=[{"title": "Облигации А", "chunk_index": 0, "score": 0.8}],
        elapsed=4.0,
    )
    text = rm.format_variants_summary(_variants_report({rm.MODE_FILTER: mode}, [rm.MODE_FILTER]))
    assert text.startswith("🔀 Режимы поиска в ответах")
    assert "• с отбором (по 1 вопросам)" in text
    assert "покрытие: 100% (без RAG: 50%)" in text
    assert "лучше: 1, равно: 0, хуже: 0" in text
    assert "поиск сработал в 1, попал в ожидаемый документ в 1; время ответов 4 с" in text


def test_variants_summary_for_mode_without_scored_answers_and_unavailable_mode():
    text = rm.format_variants_summary(
        _variants_report(
            {}, [rm.MODE_FILTER], {rm.MODE_REWRITE: rm.REWRITE_NOT_CONFIGURED}
        )
    )
    assert "• с отбором: нет оценённых ответов." in text
    assert f"⚠️ «с переписыванием» недоступен: {rm.REWRITE_NOT_CONFIGURED}." in text


def test_variants_summary_only_unavailable_modes_is_still_a_block():
    text = rm.format_variants_summary(
        _variants_report({}, [], {rm.MODE_REWRITE: "нет настройки"})
    )
    assert "недоступен: нет настройки." in text
