"""Тесты чистых правил сравнения моделей с RAG (research/rag_models_eval.py).

Только чистые функции — ни сети, ни LLM, ни Telegram, ни SmartAgent. Прогон, обращения к
моделям и судье и Telegram проверяются вручную (tasks.md изменения add-rag-models-compare).
Названия документов и вопросы — вымышленные.
"""

import json

import pytest

from research import rag_compare_eval as ev
from research import rag_models_eval as rme
from research.rag_models_eval import (
    Attempt,
    ModelSettingError,
    ModelSpec,
    QuestionRun,
)

DEFAULTS = {"deepseek": "ds-default", "kimi": "kimi-default"}

LOCAL = ModelSpec("ollama", "gpt-oss:20b")
CLOUD = ModelSpec("deepseek", "ds-default")
JUDGE = ModelSpec("kimi", "kimi-default")


# --------------------------------------------------------------------------- #
# Разбор настроек (3.1)
# --------------------------------------------------------------------------- #


def test_parse_pairs_splits_on_first_colon():
    specs = rme.parse_model_specs("ollama:gpt-oss:20b, deepseek:", DEFAULTS)
    assert [(s.provider, s.model) for s in specs] == [
        ("ollama", "gpt-oss:20b"),
        ("deepseek", "ds-default"),
    ]
    assert specs[0].key == "ollama:gpt-oss:20b"
    assert specs[0].label == "ollama/gpt-oss:20b"


def test_parse_cloud_without_colon_uses_default():
    specs = rme.parse_model_specs("ollama:m, kimi", DEFAULTS)
    assert specs[1] == ModelSpec("kimi", "kimi-default")


def test_parse_cloud_falls_back_to_project_defaults():
    specs = rme.parse_model_specs("ollama:m,deepseek")
    assert specs[1].model  # умолчание проекта, не пустая строка


def test_parse_ollama_requires_model():
    with pytest.raises(ModelSettingError, match="ollama.*явно"):
        rme.parse_model_specs("ollama:,deepseek:", DEFAULTS)
    with pytest.raises(ModelSettingError, match="ollama.*явно"):
        rme.parse_model_specs("ollama,deepseek", DEFAULTS)


def test_parse_rejects_unknown_provider():
    with pytest.raises(ModelSettingError, match="провайдер"):
        rme.parse_model_specs("openai:gpt,deepseek:", DEFAULTS)


def test_parse_needs_at_least_two_models():
    with pytest.raises(ModelSettingError, match="не меньше 2"):
        rme.parse_model_specs("ollama:m", DEFAULTS)


def test_parse_empty_setting_means_not_configured():
    with pytest.raises(ModelSettingError, match="не настроено"):
        rme.parse_model_specs("  ", DEFAULTS)
    with pytest.raises(ModelSettingError, match="не настроено"):
        rme.parse_model_specs("", DEFAULTS)


def test_parse_rejects_duplicates():
    with pytest.raises(ModelSettingError, match="дважды"):
        rme.parse_model_specs("deepseek:a,ollama:m,deepseek:a", DEFAULTS)
    # Умолчание и явное значение — одна и та же модель.
    with pytest.raises(ModelSettingError, match="дважды"):
        rme.parse_model_specs("deepseek:,deepseek:ds-default", DEFAULTS)


def test_parse_same_provider_different_models_is_fine():
    specs = rme.parse_model_specs("ollama:a,ollama:b", DEFAULTS)
    assert [s.key for s in specs] == ["ollama:a", "ollama:b"]


def test_parse_judge():
    assert rme.parse_judge_spec("kimi:kimi-k3", DEFAULTS) == ModelSpec("kimi", "kimi-k3")
    assert rme.parse_judge_spec("kimi:", DEFAULTS) == JUDGE
    with pytest.raises(ModelSettingError, match="судья не настроен"):
        rme.parse_judge_spec("", DEFAULTS)
    with pytest.raises(ModelSettingError, match="одна пара"):
        rme.parse_judge_spec("kimi:a,deepseek:b", DEFAULTS)
    with pytest.raises(ModelSettingError, match="ollama.*явно"):
        rme.parse_judge_spec("ollama:", DEFAULTS)


def test_missing_api_keys():
    specs = [LOCAL, CLOUD, JUDGE]
    assert rme.missing_api_keys(specs, {"deepseek": True, "kimi": True}) == []
    assert rme.missing_api_keys(specs, {"deepseek": False, "kimi": True}) == ["DEEPSEEK_API_KEY"]
    assert rme.missing_api_keys(specs, {}) == ["DEEPSEEK_API_KEY", "KIMI_API_KEY"]
    # Локальной модели ключ не нужен.
    assert rme.missing_api_keys([LOCAL], {}) == []


# --------------------------------------------------------------------------- #
# Выбор вопросов (3.2)
# --------------------------------------------------------------------------- #


def _q(number: int, *, abstain: bool = False, history=()) -> ev.Question:
    return ev.Question(
        number=number,
        question=f"Вопрос {number}?",
        kind=ev.KIND_OUT_OF_CORPUS if abstain else "concept",
        facts=() if abstain else ("a", "b"),
        sources=() if abstain else ("Doc",),
        history=tuple(history),
        expect_abstain=abstain,
    )


def test_select_first_n_plus_one_abstain():
    questions = [_q(i) for i in range(1, 11)] + [_q(11, abstain=True)]
    selection = rme.select_questions(questions, 4)
    assert selection.error is None
    assert [q.number for q in selection.questions] == [1, 2, 3, 4, 11]
    assert selection.notes == ()


def test_select_skips_multi_turn_and_reports():
    questions = [_q(1), _q(2, history=("раньше?",)), _q(3), _q(4, abstain=True)]
    selection = rme.select_questions(questions, 5)
    assert [q.number for q in selection.questions] == [1, 3, 4]
    assert any("Многоходовые" in n and "2" in n for n in selection.notes)


def test_select_shortage_runs_with_available_and_notes():
    selection = rme.select_questions([_q(1), _q(2), _q(3, abstain=True)], 4)
    assert selection.error is None
    assert [q.number for q in selection.questions] == [1, 2, 3]
    assert any("меньше настроенных 4" in n for n in selection.notes)


def test_select_without_abstain_question_notes_it():
    selection = rme.select_questions([_q(1), _q(2)], 2)
    assert [q.number for q in selection.questions] == [1, 2]
    assert any("вне корпуса" in n for n in selection.notes)


def test_select_no_corpus_questions_is_error():
    selection = rme.select_questions([_q(1, abstain=True), _q(2, history=("x?",))], 4)
    assert selection.error and "нет самостоятельных вопросов" in selection.error
    assert selection.questions == ()


def test_question_budget_scales_with_models_and_repeats():
    assert rme.question_budget_seconds(120, 2, 1) == 120
    assert rme.question_budget_seconds(120, 2, 3) == 360


# --------------------------------------------------------------------------- #
# Метрики обращения и агрегата (3.3, 3.3a)
# --------------------------------------------------------------------------- #


def _result(
    *,
    tokens: int | None = 600,
    elapsed: float = 20.0,
    coverage: float | None = 1.0,
    facts: int = 2,
    error: str | None = None,
    judge_error: str | None = None,
    abstained: bool = False,
    citations: int = 1,
    unverified: bool = False,
    contradicts: bool = False,
    calls: int = 1,
) -> ev.ModeResult:
    verdict = None
    if coverage is not None:
        found = round(coverage * facts)
        verdict = ev.Verdict(
            present=tuple([True] * found + [False] * (facts - found)), contradicts=contradicts
        )
    return ev.ModeResult(
        answer=None if error else "ответ",
        error=error,
        elapsed=elapsed,
        prompt_tokens=100,
        completion_tokens=tokens,
        llm_calls=calls,
        verdict=verdict,
        judge_error=judge_error,
        abstained=abstained,
        citations=[{"quote": "q", "title": "Doc", "chunk_index": 0, "chunk_id": "c"}]
        * citations,
        citations_unverified=unverified,
    )


def _attempt(key: str, repeat: int, cold: bool = False, **kwargs) -> Attempt:
    return Attempt(key, repeat, _result(**kwargs), cold=cold)


def _run(number: int, attempts: list[Attempt], *, abstain: bool = False, **kwargs) -> QuestionRun:
    run = QuestionRun(question=_q(number, abstain=abstain), **kwargs)
    for a in attempts:
        run.attempts.setdefault(a.model_key, []).append(a)
    return run


def test_tokens_per_second_spec_example():
    assert rme.tokens_per_second(600, 20.0) == 30.0


def test_tokens_per_second_unknown_without_tokens_or_time():
    assert rme.tokens_per_second(None, 5.0) is None
    assert rme.tokens_per_second(0, 5.0) is None
    assert rme.tokens_per_second(100, 0.0) is None


def test_attempt_dict_roundtrip_keeps_speed_and_cold():
    attempt = _attempt("m", 2, cold=True, tokens=600, elapsed=20.0)
    data = attempt.to_dict()
    assert data["tokens_per_second"] == 30.0
    assert data["cold"] is True and data["repeat"] == 2
    again = Attempt.from_dict("m", json.loads(json.dumps(data)))
    assert again.repeat == 2 and again.cold and again.tokens_per_second == 30.0
    assert again.result.verdict.coverage == 1.0


def test_speed_excludes_cold_and_unknown():
    key = "m"
    run = _run(
        1,
        [
            _attempt(key, 1, cold=True, tokens=100, elapsed=50.0),  # 2 ток/с, холодный
            _attempt(key, 2, tokens=600, elapsed=20.0),  # 30
            _attempt(key, 3, tokens=600, elapsed=10.0),  # 60
            _attempt(key, 4, tokens=None),  # неизвестна
        ],
    )
    agg = rme.aggregate_model([run], key)
    assert agg["speed_median"] == 45.0
    assert agg["speed_min"] == 30.0 and agg["speed_max"] == 60.0
    assert agg["speed_n"] == 2
    assert agg["speed_unknown"] == 1
    assert agg["cold_tps"] == 2.0 and agg["cold_elapsed"] == 50.0


def test_aggregate_success_rate_and_failure_reasons():
    key = "m"
    attempts = [_attempt(key, i) for i in range(1, 9)]
    attempts.append(_attempt(key, 9, error="превышен предел ожидания", tokens=None))
    agg = rme.aggregate_model([_run(1, attempts)], key)
    assert (agg["answered"], agg["attempts"]) == (8, 9)
    assert agg["failures"] == {"превышен предел ожидания": 1}


def test_aggregate_quality_ignores_unjudged_and_counts_judge_errors_separately():
    key = "m"
    run = _run(
        1,
        [
            _attempt(key, 1, coverage=1.0),
            _attempt(key, 2, coverage=0.5, contradicts=True),
            _attempt(key, 3, coverage=None, judge_error="судья недоступен"),
        ],
    )
    agg = rme.aggregate_model([run], key)
    assert agg["avg_coverage"] == pytest.approx(0.75)
    assert agg["judged"] == 2 and agg["unjudged"] == 1
    assert agg["contradicts"] == 1
    # Сбой судьи — не сбой генерации.
    assert agg["answered"] == 3
    assert agg["judge_errors"] == {"судья недоступен": 1}


def test_aggregate_spread_between_repeats():
    key = "m"
    facts = 7
    run = _run(
        1,
        [
            _attempt(key, 1, coverage=3 / 7, facts=facts),
            _attempt(key, 2, coverage=5 / 7, facts=facts),
            _attempt(key, 3, coverage=7 / 7, facts=facts),
        ],
    )
    run.question = ev.Question(1, "в?", "concept", tuple("abcdefg"), ("D",))
    agg = rme.aggregate_model([run], key)
    low, high, total = agg["spread"][1]
    assert (round(low * total), round(high * total), total) == (3, 7, 7)


def test_aggregate_citations_and_abstain():
    key = "m"
    corpus = _run(
        1,
        [
            _attempt(key, 1, citations=1),
            _attempt(key, 2, citations=0, unverified=True),
        ],
    )
    out = _run(2, [_attempt(key, 1, abstained=True, coverage=None)], abstain=True)
    agg = rme.aggregate_model([corpus, out], key)
    assert agg["cit_eligible"] == 2
    assert agg["cit_quotes"] == 1 and agg["cit_unverified"] == 1
    assert (agg["abstain_correct"], agg["abstain_answers"]) == (1, 1)


def test_aggregate_counts_missing_citation_judge_separately():
    key = "m"
    attempt = _attempt(key, 1, citations=1)
    attempt.result.citation_judge_error = "лимит запросов"
    agg = rme.aggregate_model([_run(1, [attempt])], key)
    assert agg["cit_judge_missing"] == 1
    assert agg["answered"] == 1  # сбой судьи — не сбой генерации
    report = _report([_run(1, [attempt, _attempt(CLOUD.key, 1)])])
    assert "без оценки судьи — 1" in rme.format_summary(report)


def test_aggregate_false_abstain_counted_and_excluded_from_citations():
    key = "m"
    run = _run(1, [_attempt(key, 1, abstained=True, coverage=None)])
    agg = rme.aggregate_model([run], key)
    assert agg["false_abstain"] == 1 and agg["cit_eligible"] == 0


def test_aggregate_empty_lists():
    agg = rme.aggregate_model([], "m")
    assert agg["attempts"] == 0 and agg["avg_coverage"] is None
    assert agg["speed_median"] is None and agg["spread"] == {}
    agg = rme.aggregate_model([_run(1, [])], "m")
    assert agg["attempts"] == 0


def test_search_failed_question_is_excluded_from_all_metrics():
    key = "m"
    good = _run(1, [_attempt(key, 1), _attempt(key, 2)])
    broken = QuestionRun(question=_q(2), search_error="сервер эмбеддингов недоступен")
    before = rme.aggregate_model([good], key)
    after = rme.aggregate_model([good, broken], key)
    assert after == before
    excluded = rme.excluded_questions([good, broken])
    assert excluded == [{"number": 2, "reason": "сервер эмбеддингов недоступен"}]


def test_summary_names_excluded_question_reason():
    report = _report(
        [
            _run(1, [_attempt(LOCAL.key, 1), _attempt(CLOUD.key, 1)]),
            QuestionRun(question=_q(2), search_error="сервер эмбеддингов недоступен"),
        ]
    )
    text = rme.format_summary(report)
    assert "Исключено из сравнения" in text
    assert "2 (сервер эмбеддингов недоступен)" in text


# --------------------------------------------------------------------------- #
# Автоостановка (3.4)
# --------------------------------------------------------------------------- #


def _failed_run(number: int) -> QuestionRun:
    return _run(
        number,
        [
            _attempt(LOCAL.key, 1, error="нет связи", tokens=None),
            _attempt(CLOUD.key, 1, error="нет связи", tokens=None),
        ],
    )


def _ok_run(number: int) -> QuestionRun:
    return _run(number, [_attempt(LOCAL.key, 1), _attempt(CLOUD.key, 1)])


def test_all_attempts_failed_is_failed_question():
    assert rme.question_all_failed(_failed_run(1))


def test_one_model_failing_does_not_fail_question():
    run = _run(
        1,
        [_attempt(LOCAL.key, 1, error="нет связи", tokens=None), _attempt(CLOUD.key, 1)],
    )
    assert not rme.question_all_failed(run)


def test_judge_failure_is_not_generation_failure():
    run = _run(
        1,
        [
            _attempt(LOCAL.key, 1, coverage=None, judge_error="судья недоступен"),
            _attempt(CLOUD.key, 1, coverage=None, judge_error="судья недоступен"),
        ],
    )
    assert not rme.question_all_failed(run)


def test_search_failure_counts_towards_stop():
    run = QuestionRun(question=_q(1), search_error="сервер эмбеддингов недоступен")
    assert rme.question_all_failed(run)


def test_should_stop_after_consecutive_failures_only():
    assert rme.should_stop([_failed_run(1), _failed_run(2), _failed_run(3)], 3)
    assert not rme.should_stop([_failed_run(1), _failed_run(2)], 3)
    # Успешный вопрос сбрасывает счёт.
    assert not rme.should_stop([_failed_run(1), _failed_run(2), _ok_run(3)], 3)
    assert not rme.should_stop([_failed_run(1), _failed_run(2), _failed_run(3)], 0)
    assert rme.consecutive_failures([_ok_run(1), _failed_run(2), _failed_run(3)]) == 2


# --------------------------------------------------------------------------- #
# Отчёт (3.5)
# --------------------------------------------------------------------------- #


def _report(runs: list[QuestionRun], **kwargs) -> dict:
    return rme.build_report(
        started_at="2026-10-09 10:00:00",
        finished_at="2026-10-09 10:05:00",
        settings={"strategy": "structural", "top_k": 3, "min_score": 0.5, "repeats": 3},
        models=[LOCAL, CLOUD],
        judge=JUDGE,
        runs=runs,
        planned=kwargs.pop("planned", len(runs)),
        repeats=3,
        **kwargs,
    )


def test_report_name_does_not_match_compare_report_pattern():
    name = rme.report_file_name("20261009T100000Z")
    assert name == "20261009T100000Z_models.json"
    assert ev.latest_report_name([name]) is None
    assert rme.latest_models_report_name(["20261009T100000Z.json"]) is None


def test_latest_models_report_name_picks_latest_and_ignores_others():
    names = [
        "20261001T100000Z_models.json",
        "20261009T100000Z_models.json",
        "20261009T100000Z.json",
        "questions.json",
        "20261010T100000Z_models.json.tmp",
    ]
    assert rme.latest_models_report_name(names) == "20261009T100000Z_models.json"
    assert rme.latest_models_report_name([]) is None


def test_foreign_or_old_report_is_rejected():
    mine = _report([_ok_run(1)])
    assert rme.is_models_report(mine)
    assert not rme.is_models_report({"questions": [], "settings": {}})  # чужой вид
    assert not rme.is_models_report({**mine, "version": 0})
    assert not rme.is_models_report({**mine, "kind": "retrieval"})
    assert not rme.is_models_report({k: v for k, v in mine.items() if k != "models"})
    assert not rme.is_models_report([])


def test_report_keeps_all_repeats_of_all_models():
    runs = [
        _run(
            1,
            [_attempt(m.key, r) for r in (1, 2, 3) for m in (LOCAL, CLOUD)],
            search_sources=[{"title": "Doc", "chunk_index": 1, "score": 0.7}],
            search_seconds=2.0,
        )
    ]
    report = json.loads(json.dumps(_report(runs)))
    attempts = report["questions"][0]["attempts"]
    assert [len(attempts[m.key]) for m in (LOCAL, CLOUD)] == [3, 3]
    assert report["questions"][0]["search"]["seconds"] == 2.0
    assert report["judge"]["provider"] == "kimi"
    assert "fragments" not in json.dumps(report)  # тексты фрагментов не пишутся


def test_report_roundtrip_through_json():
    run = _ok_run(1)
    run.search_sources = [{"title": "Doc", "chunk_index": 2, "score": 0.6}]
    restored = QuestionRun.from_dict(json.loads(json.dumps(run.to_dict())))
    assert restored.question.number == 1
    assert restored.search_sources == run.search_sources
    assert len(restored.attempts[LOCAL.key]) == 1


# --------------------------------------------------------------------------- #
# Тексты (3.6)
# --------------------------------------------------------------------------- #


def _rich_report() -> dict:
    runs = [
        _run(
            1,
            [
                _attempt(LOCAL.key, 1, cold=True, tokens=100, elapsed=50.0),
                _attempt(CLOUD.key, 1, tokens=600, elapsed=4.0),
                _attempt(LOCAL.key, 2, tokens=300, elapsed=30.0),
                _attempt(CLOUD.key, 2, tokens=600, elapsed=5.0, coverage=0.5),
            ],
            search_sources=[{"title": "Doc", "chunk_index": 1, "score": 0.71}],
            search_seconds=2.0,
        ),
        _run(
            2,
            [
                _attempt(LOCAL.key, 1, abstained=True, coverage=None),
                _attempt(CLOUD.key, 1, coverage=None),
            ],
            abstain=True,
        ),
    ]
    return _report(runs)


def test_summary_has_settings_sample_and_per_model_metrics():
    text = rme.format_summary(_rich_report())
    assert "ollama/gpt-oss:20b" in text and "deepseek/ds-default" in text
    assert "kimi/kimi-default" in text  # судья
    assert "повторов каждой модели на вопрос — 3" in text
    assert "вопросов по корпусу — 1" in text and "вне корпуса — 1" in text
    for word in ("Качество:", "Скорость:", "Стабильность:", "Холодный старт"):
        assert word in text
    assert "10.0 ток/с" in text  # локальная модель без холодного: 300/30
    assert "победителя не объявляет" in text
    assert "токены рассуждений" in text.lower()


def test_summary_discloses_data_sent_to_cloud_judge():
    text = rme.format_summary(_rich_report())
    assert "передавались судье" in text and "Kimi" in text


def test_judge_line_for_local_judge():
    line = rme.judge_data_line({"provider": "ollama", "label": "ollama/judge:7b"})
    assert "локальный" in line and "не уходили" in line


def test_summary_marks_stopped_run():
    report = _report([_ok_run(1)], status=ev.STATUS_STOPPED, stop_reason="по команде", planned=4)
    text = rme.format_summary(report)
    assert "остановлен" in text and "1 из 4" in text


def test_detail_shows_all_repeats_side_by_side():
    text = rme.format_question_detail(_rich_report(), 1)
    assert text.index("повтор 1") < text.index("повтор 2")
    # Повтор 1 обеих моделей идёт раньше повтора 2.
    first_cloud = text.index("deepseek/ds-default, повтор 1")
    first_local = text.index("ollama/gpt-oss:20b, повтор 1")
    second_local = text.index("ollama/gpt-oss:20b, повтор 2")
    assert max(first_cloud, first_local) < second_local
    assert "(холодный старт)" in text
    assert "30.0 ток/с" in text or "20.0 ток/с" in text
    assert "токенов ответа" in text and "выз. модели" in text
    assert "✅" in text or "❌" in text
    assert "Общий поиск" in text and "Doc — фрагмент 1" in text


def test_detail_for_abstain_question_reports_refusal():
    text = rme.format_question_detail(_rich_report(), 2)
    assert "Отказ «не знаю»: да" in text and "нет ✗" in text


def test_detail_for_excluded_question():
    report = _report([QuestionRun(question=_q(1), search_error="индекс повреждён")])
    text = rme.format_question_detail(report, 1)
    assert "исключён из сравнения: индекс повреждён" in text


def test_detail_unknown_number_is_none():
    assert rme.format_question_detail(_rich_report(), 99) is None


def test_long_detail_splits_within_telegram_limit():
    report = _rich_report()
    long_run = _run(
        3,
        [_attempt(m.key, r) for r in (1, 2, 3) for m in (LOCAL, CLOUD)],
    )
    for attempts in long_run.attempts.values():
        for a in attempts:
            a.result.answer = "слово " * 800
    report["questions"].append(long_run.to_dict())
    text = rme.format_question_detail(report, 3)
    parts = ev.split_text(text, 4000)
    assert len(parts) > 1
    assert all(len(p) <= 4000 for p in parts)
    assert "".join(p.replace("\n", "") for p in parts) == text.replace("\n", "")


def test_question_numbers_are_set_numbers_and_shown_in_summary():
    report = _rich_report()  # вопросы с номерами 1 и 2
    report["questions"][1]["number"] = 9
    assert rme.question_numbers(report) == [1, 9]
    assert "номера из набора: 1, 9" in rme.format_summary(report)


# --------------------------------------------------------------------------- #
# Дополнение: ветки блока модели, цитаты судьи, строки обращения и сводка с исключениями
# --------------------------------------------------------------------------- #


def _with_citation_verdict(attempt: Attempt, verdict: str) -> Attempt:
    attempt.result.citation_verdict = ev.CitationVerdict(verdict)
    return attempt


def _block(runs: list[QuestionRun], model: ModelSpec = LOCAL) -> str:
    agg = rme.aggregate_model(runs, model.key)
    return "\n".join(rme._model_block(1, model.to_dict(), agg))


def test_aggregate_counts_citation_judge_verdicts_and_ignores_unknown_ones():
    attempts = [
        _with_citation_verdict(_attempt(LOCAL.key, 1), ev.CITATION_SUPPORTED),
        _with_citation_verdict(_attempt(LOCAL.key, 2), ev.CITATION_PARTIAL),
        _with_citation_verdict(_attempt(LOCAL.key, 3), "странный"),
        _attempt(LOCAL.key, 4),
    ]
    agg = rme.aggregate_model([_run(1, attempts)], LOCAL.key)
    assert agg["cit_verdicts"] == {
        ev.CITATION_SUPPORTED: 1,
        ev.CITATION_PARTIAL: 1,
        ev.CITATION_UNSUPPORTED: 0,
    }
    assert agg["cit_judge_missing"] == 1  # только обращение 4: у «странного» вердикт всё же есть


def test_settings_line_with_and_without_candidates():
    base = rme._settings_line({"strategy": "structural", "top_k": 3, "min_score": 0.5})
    assert base == (
        "Настройки поиска: стратегия structural, top-k 3, порог 0.5, "
        "переписывание вопроса выключено."
    )
    assert "кандидатов 8" in rme._settings_line({"candidates": 8})
    assert "стратегия ?, top-k ?, порог ?" in rme._settings_line({})


def test_model_block_shows_citation_verdicts_false_abstain_and_correct_refusal():
    runs = [
        _run(1, [_with_citation_verdict(_attempt(LOCAL.key, 1), ev.CITATION_SUPPORTED)]),
        _run(2, [_attempt(LOCAL.key, 1, abstained=True)]),
        _run(3, [_attempt(LOCAL.key, 1, abstained=True, coverage=None)], abstain=True),
    ]
    text = _block(runs)
    assert "Цитаты: проверенные в 1 из 1 ответов" in text
    assert "подтверждается — 1" in text
    assert "без оценки судьи" not in text
    assert "Вопрос вне корпуса: верный отказ в 1 из 1 ответов." in text
    assert "⚠️ Ложных отказов («не знаю») по корпусу: 1." in text


def test_model_block_names_citation_attempts_without_judge_verdict():
    text = _block([_run(1, [_attempt(LOCAL.key, 1)])])
    assert "без оценки судьи — 1" in text


def test_model_block_speed_variants():
    only_cold = _block([_run(1, [_attempt(LOCAL.key, 1, cold=True, tokens=100, elapsed=10)])])
    assert "нет данных вне холодного старта" in only_cold
    cold_line = "Холодный старт (первое обращение, не входит в статистику): 10.0 ток/с, 10 с."
    assert cold_line in only_cold

    no_tokens = _block([_run(1, [_attempt(LOCAL.key, 1, tokens=None)])])
    assert "Скорость: нет данных (токены ответа неизвестны или нет ответов)." in no_tokens
    assert "Скорость неизвестна (провайдер не вернул токены): 1." in no_tokens

    normal = _block([_run(1, [_attempt(LOCAL.key, 1, tokens=100, elapsed=10)])])
    assert "медиана 10.0 ток/с" in normal
    assert "(обращений: 1)" in normal


def test_model_block_failed_cold_start_and_failure_reasons_ranked():
    attempts = [
        _attempt(LOCAL.key, 1, cold=True, error="таймаут", coverage=None),
        _attempt(LOCAL.key, 2, error="таймаут", coverage=None),
        _attempt(LOCAL.key, 3, error="отказ сервера", coverage=None),
        _attempt(LOCAL.key, 4),
    ]
    text = _block([_run(1, attempts)])
    assert "Холодный старт: первое обращение не завершилось ответом." in text
    assert "Стабильность: ответ получен в 1 из 4 обращений." in text
    assert "Причины сбоев: таймаут ×2; отказ сервера ×1." in text


def test_model_block_unjudged_answers_with_and_without_reasons():
    with_reason = _block(
        [_run(1, [_attempt(LOCAL.key, 1, coverage=None, judge_error="судья недоступен")])]
    )
    assert "Не оценено судьёй: 1 (судья недоступен ×1)." in with_reason
    assert "нет ответов, оценённых судьёй" in with_reason
    without_reason = _block([_run(1, [_attempt(LOCAL.key, 1, coverage=None)])])
    assert "Не оценено судьёй: 1." in without_reason


def test_model_block_average_calls_and_spread_between_repeats():
    attempts = [
        _attempt(LOCAL.key, 1, coverage=1.0, calls=1),
        _attempt(LOCAL.key, 2, coverage=0.5, calls=2),
    ]
    steady = [_attempt(LOCAL.key, 1, coverage=1.0), _attempt(LOCAL.key, 2, coverage=1.0)]
    text = _block([_run(1, attempts), _run(2, steady)])
    assert "Вызовов модели на ответ в среднем: 1.2." in text
    assert "вопрос 1: 1–2 из 2; вопрос 2: 2 из 2" in text


def test_spread_text_formats_range_and_single_value():
    assert rme._spread_text({3: (0.5, 1.0, 2), 1: (1.0, 1.0, 2)}) == (
        "вопрос 1: 2 из 2; вопрос 3: 1–2 из 2"
    )
    assert rme._spread_text({}) == ""


def test_summary_lists_excluded_questions_search_time_and_queue():
    ok = _run(
        1,
        [_attempt(LOCAL.key, 1), _attempt(CLOUD.key, 1)],
        search_seconds=2.0,
    )
    excluded = _run(2, [], search_error="эмбеддинги недоступны")
    report = _report([ok, excluded])
    report["queue"] = {"slot_wait_seconds": 12, "user_questions": 2}
    text = rme.format_summary(report)
    assert "Не вошли в сравнение из-за сбоя поиска: 1." in text
    assert "к моделям не обращались): 1 — 2 (эмбеддинги недоступны)." in text
    assert "Поиск по индексу: один на вопрос, в среднем 2.0 с" in text
    assert "Очередь:" in text
    assert "(номера из набора: 1, 2)" in text


def test_summary_without_search_time_or_exclusions_has_no_such_lines():
    report = _report([_run(1, [_attempt(LOCAL.key, 1), _attempt(CLOUD.key, 1)])])
    text = rme.format_summary(report)
    assert "Исключено из сравнения" not in text
    assert "Поиск по индексу:" not in text
    assert "Очередь:" not in text


def test_summary_of_kimi_judge_notes_fixed_temperature():
    text = rme.format_summary(_rich_report())
    assert "температуру задаёт сам Kimi" in text


def test_attempt_lines_of_failed_attempt_with_and_without_error():
    question = _q(1)
    failed = Attempt(LOCAL.key, 2, ev.ModeResult(error="таймаут провайдера"), cold=True)
    lines = rme._attempt_lines("ollama/m", failed, question)
    assert lines[0] == "\n— ollama/m, повтор 2 (холодный старт) —"
    assert lines[1] == "Сбой ответа: таймаут провайдера"
    assert len(lines) == 2
    silent = Attempt(LOCAL.key, 1, ev.ModeResult())
    assert rme._attempt_lines("м", silent, question)[1] == "Сбой ответа: ответ не получен"


def test_attempt_lines_of_abstain_question_report_refusal_or_answer():
    question = _q(1, abstain=True)
    refusal = _attempt(LOCAL.key, 1, abstained=True, coverage=None)
    refused = rme._attempt_lines("м", refusal, question)
    assert refused[-1] == "Отказ «не знаю»: да ✓"
    answered = rme._attempt_lines("м", _attempt(LOCAL.key, 1, coverage=None), question)
    assert answered[-1] == "Отказ «не знаю»: нет ✗ (дан ответ)"


def test_attempt_lines_with_verdict_contradiction_note_and_missing_estimate():
    question = _q(1)
    attempt = _attempt(LOCAL.key, 1, coverage=0.5, contradicts=True)
    attempt.result.verdict = ev.Verdict(present=(True, False), contradicts=True, note="Спорно")
    text = "\n".join(rme._attempt_lines("м", attempt, question))
    assert "Оценка: 50% фактов." in text
    assert "✅ 1. a" in text and "❌ 2. b" in text
    assert "⚠️ Есть утверждение, противоречащее ожиданию." in text
    assert "Замечание судьи: Спорно" in text
    failed_judge = _attempt(LOCAL.key, 1, coverage=None, judge_error="нет JSON")
    no_verdict = "\n".join(rme._attempt_lines("м", failed_judge, question))
    assert "Оценка не получена: нет JSON" in no_verdict
    assert "Оценка не получена: нет данных" in "\n".join(
        rme._attempt_lines("м", _attempt(LOCAL.key, 1, coverage=None), question)
    )


def test_attempt_lines_show_speed_tokens_calls_and_unknown_tokens():
    question = _q(1)
    fast = _attempt(LOCAL.key, 1, tokens=200, elapsed=10)
    known = "\n".join(rme._attempt_lines("м", fast, question))
    expected = "Скорость: 20.0 ток/с · 200 токенов ответа, промпт 100 · 1 выз. модели · 10.0 с"
    assert expected in known
    unknown = "\n".join(rme._attempt_lines("м", _attempt(LOCAL.key, 1, tokens=None), question))
    assert "токены ответа неизвестны" in unknown
    assert "неизвестна" in unknown
