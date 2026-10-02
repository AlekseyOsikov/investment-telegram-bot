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
    assert active == [rm.MODE_BASELINE, rm.MODE_FILTER]
    assert set(unavailable) == {rm.MODE_REWRITE, rm.MODE_REWRITE_FILTER}
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
