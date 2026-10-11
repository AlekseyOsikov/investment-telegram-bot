"""Тесты чистых правил оснастки подбора параметров Ollama (scripts/ollama_tune_lib.py).

Только чистые функции — ни сети, ни Ollama, ни LLM. Прогон и обращения к серверу проверяются
вручную (tasks.md изменения add-ollama-tuning-script). Вопросы, документы и цифры — вымышленные.
"""

from types import SimpleNamespace

import ollama_tune_lib as lib
import pytest

from research import rag_compare_eval as ev
from research import rag_models_eval as rme

# --------------------------------------------------------------------------- #
# Разбор файла вариантов (2.1)
# --------------------------------------------------------------------------- #


def _file(*variants, environment=""):
    return {"environment": environment, "variants": list(variants)}


BASE = {"name": "base", "base": "model-a:1b", "baseline": True}
CTX = {"name": "ctx8k", "base": "model-a:1b", "params": {"num_ctx": 8192, "temperature": 0.2}}


def test_parse_valid_variants():
    parsed = lib.parse_variants(_file(BASE, CTX, environment="OLLAMA_KV_CACHE_TYPE=q8_0"))
    assert [v.name for v in parsed.variants] == ["base", "ctx8k"]
    assert parsed.baseline.name == "base"
    assert parsed.variants[1].params == {"num_ctx": 8192, "temperature": 0.2}
    assert parsed.variants[1].needs_tag
    assert not parsed.variants[0].needs_tag
    assert parsed.environment == "OLLAMA_KV_CACHE_TYPE=q8_0"


def test_parse_requires_baseline():
    with pytest.raises(lib.VariantFileError, match="исходный"):
        lib.parse_variants(_file({**BASE, "baseline": False}, CTX))


def test_parse_rejects_two_baselines():
    with pytest.raises(lib.VariantFileError, match="только один"):
        lib.parse_variants(_file(BASE, {**CTX, "baseline": True}))


def test_parse_rejects_duplicate_names():
    with pytest.raises(lib.VariantFileError, match="повторяется"):
        lib.parse_variants(_file(BASE, {**CTX, "name": "base"}))


def test_parse_requires_two_variants():
    with pytest.raises(lib.VariantFileError, match="не меньше 2"):
        lib.parse_variants(_file(BASE))


def test_parse_rejects_unknown_param():
    bad = {**CTX, "params": {"num_cxt": 8192}}
    with pytest.raises(lib.VariantFileError, match="неизвестный параметр"):
        lib.parse_variants(_file(BASE, bad))


@pytest.mark.parametrize(
    "params",
    [{"num_ctx": "8192"}, {"num_ctx": True}, {"num_ctx": 0}, {"num_ctx": 4096.5}],
)
def test_parse_rejects_bad_param_values(params):
    with pytest.raises(lib.VariantFileError):
        lib.parse_variants(_file(BASE, {**CTX, "params": params}))


def test_parse_accepts_integral_float():
    parsed = lib.parse_variants(_file(BASE, {**CTX, "params": {"num_ctx": 8192.0}}))
    assert parsed.variants[1].params == {"num_ctx": 8192}


def test_parse_rejects_num_predict_overridden_by_request_limit():
    bad = {**CTX, "params": {"num_predict": 512}}
    with pytest.raises(lib.VariantFileError, match="max_output_tokens"):
        lib.parse_variants(_file(BASE, bad))


def test_parse_max_output_tokens_field():
    cap = {"name": "cap", "base": "model-a:1b", "max_output_tokens": 4000}
    parsed = lib.parse_variants(_file(BASE, cap))
    assert parsed.variants[1].max_output_tokens == 4000
    assert not parsed.variants[1].needs_tag          # тег не нужен — лимит идёт в запросе
    assert "MAX_OUTPUT_TOKENS" in lib.portability(parsed.variants[1])["text"]
    for bad in (0, -5, True, "4000", 10.5):
        with pytest.raises(lib.VariantFileError, match="max_output_tokens"):
            lib.parse_variants(_file(BASE, {**cap, "max_output_tokens": bad}))


def test_parse_marks_request_level_and_prompt_variants():
    effort = {"name": "think-low", "base": "model-a:1b", "reasoning_effort": "low"}
    prompt = {"name": "prompt-v2", "base": "model-a:1b", "system_prompt": "Отвечай кратко."}
    parsed = lib.parse_variants(_file(BASE, effort, prompt))
    assert parsed.variants[1].request_level
    assert not parsed.variants[1].needs_tag
    assert parsed.variants[2].experimental_prompt


def test_parse_rejects_reasoning_none():
    bad = {"name": "think-off", "base": "model-a:1b", "reasoning_effort": "none"}
    with pytest.raises(lib.VariantFileError, match="reasoning_effort"):
        lib.parse_variants(_file(BASE, bad))


@pytest.mark.parametrize("raw", [[], {}, {"variants": []}, {"variants": ["x"]}])
def test_parse_rejects_malformed_file(raw):
    with pytest.raises(lib.VariantFileError):
        lib.parse_variants(raw)


def test_parse_rejects_missing_base_and_bad_name():
    with pytest.raises(lib.VariantFileError, match="base"):
        lib.parse_variants(_file(BASE, {"name": "x"}))
    with pytest.raises(lib.VariantFileError, match="имя"):
        lib.parse_variants(_file(BASE, {**CTX, "name": "с пробелом"}))


def test_select_variants_keeps_baseline_and_order():
    third = {"name": "ctx16k", "base": "model-a:1b", "params": {"num_ctx": 16384}}
    parsed = lib.parse_variants(_file(BASE, CTX, third))
    chosen = lib.select_variants(parsed, ["ctx16k"])
    assert [v.name for v in chosen.variants] == ["base", "ctx16k"]
    assert lib.select_variants(parsed, []) is parsed


def test_select_variants_unknown_name():
    parsed = lib.parse_variants(_file(BASE, CTX))
    with pytest.raises(lib.VariantFileError, match="nope"):
        lib.select_variants(parsed, ["nope"])
    with pytest.raises(lib.VariantFileError, match="остался один"):
        lib.select_variants(parsed, ["base"])


def test_portability_levels():
    plain = lib.Variant("a", "m", params={"num_ctx": 8192})
    effort = lib.Variant("b", "m", reasoning_effort="low")
    prompt = lib.Variant("c", "m", system_prompt="Кратко.")
    assert lib.portability(plain)["level"] == "bot"
    assert "Modelfile" in lib.portability(plain)["text"]
    assert lib.portability(effort)["level"] == "request"
    assert "доработки бота" in lib.portability(effort)["text"]
    assert lib.portability(prompt)["level"] == "prompt"
    assert "SYSTEM_PROMPT" in lib.portability(prompt)["text"]
    both = lib.Variant("d", "m", reasoning_effort="low", system_prompt="Кратко.")
    assert lib.portability(both)["level"] == "request"


def test_temp_tag_prefix_and_detection():
    tag = lib.temp_tag("20261011T100000Z", "Ctx8K")
    assert tag.startswith(lib.TAG_PREFIX)
    assert lib.is_temp_tag(tag + ":latest")
    assert not lib.is_temp_tag("gpt-oss:20b")


def test_parse_ps_entry_matches_latest_suffix():
    ps = {"models": [{"name": "tunetmp-x:latest", "size": 10, "size_vram": 0,
                      "context_length": 4096}]}
    assert lib.parse_ps_entry(ps, "tunetmp-x") == {
        "size": 10, "size_vram": 0, "context_length": 4096
    }
    assert lib.parse_ps_entry(ps, "other") is None


# --------------------------------------------------------------------------- #
# Метрики и агрегация (2.2)
# --------------------------------------------------------------------------- #


def _question(number=1, *, abstain=False, facts=("факт 1", "факт 2")):
    return ev.Question(
        number=number,
        question=f"Вопрос {number}?",
        kind=ev.KIND_OUT_OF_CORPUS if abstain else "concept",
        facts=() if abstain else tuple(facts),
        sources=() if abstain else ("Документ А",),
        expect_abstain=abstain,
    )


def _attempt(key, repeat, *, tokens=100, elapsed=10.0, cold=False, coverage=None,
             error=None, prompt=1500, finish="stop", abstained=False):
    verdict = None
    if coverage is not None:
        verdict = ev.Verdict(present=tuple(i < coverage for i in range(2)))
    result = ev.ModeResult(
        answer=None if error else "Текст ответа.",
        error=error,
        elapsed=elapsed,
        prompt_tokens=prompt,
        completion_tokens=tokens,
        llm_calls=1,
        verdict=verdict,
        abstained=abstained,
    )
    return rme.Attempt(key, repeat, result, cold=cold), [
        {"prompt_tokens": prompt, "completion_tokens": tokens, "finish_reason": finish}
    ]


def _run(question, per_variant, search_error=None):
    """per_variant: {имя: [(Attempt, calls), ...]} → словарь вопроса отчёта с `calls`."""
    run = rme.QuestionRun(question=question, search_error=search_error)
    calls = {}
    for key, items in per_variant.items():
        run.attempts[key] = [a for a, _ in items]
        calls[key] = [c for _, c in items]
    return lib.attach_calls(run.to_dict(), calls)


def test_aggregate_speed_excludes_cold_and_reports_extremes():
    q = _run(_question(), {"v": [
        _attempt("v", 1, tokens=100, elapsed=20.0, cold=True),   # 5 ток/с — холодное
        _attempt("v", 2, tokens=100, elapsed=10.0),              # 10
        _attempt("v", 3, tokens=100, elapsed=5.0),               # 20
        _attempt("v", 4, tokens=90, elapsed=6.0),                # 15
    ]})
    agg = lib.aggregate_variant([q], "v")
    assert agg["speed_median"] == 15.0
    assert (agg["speed_min"], agg["speed_max"]) == (10.0, 20.0)
    assert agg["speed_n"] == 3
    assert agg["cold_count"] == 1
    assert agg["cold_tps"] == 5.0


def test_aggregate_unknown_speed_not_computed():
    a, calls = _attempt("v", 2, tokens=None)
    q = _run(_question(), {"v": [_attempt("v", 1, cold=True), (a, calls)]})
    agg = lib.aggregate_variant([q], "v")
    assert agg["speed_n"] == 0
    assert agg["speed_unknown"] == 1
    assert agg["speed_median"] is None


def test_aggregate_truncated_share_from_finish_reason():
    q = _run(_question(), {"v": [
        _attempt("v", 1, finish="length"),
        _attempt("v", 2, finish="stop"),
        _attempt("v", 3, finish="length"),
        _attempt("v", 4, finish="stop"),
    ]})
    agg = lib.aggregate_variant([q], "v")
    assert (agg["truncated"], agg["with_calls"]) == (2, 4)


def test_is_truncated_any_call():
    assert lib.is_truncated([{"finish_reason": "stop"}, {"finish_reason": "length"}])
    assert not lib.is_truncated([])
    assert not lib.is_truncated([{"finish_reason": "stop"}])


def test_call_record_reads_usage_and_tolerates_missing():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    assert lib.call_record(usage, "stop") == {
        "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"
    }
    assert lib.call_record(None, None)["prompt_tokens"] is None


def test_aggregate_prompt_tokens_and_resources():
    q = _run(_question(), {"v": [
        _attempt("v", 1, prompt=1000), _attempt("v", 2, prompt=3000), _attempt("v", 3, prompt=2000),
    ]})
    resources = {"load_seconds": [40.0, 44.0, 60.0], "size": 19_000_000_000,
                 "size_vram": 0, "context_length": 8192}
    agg = lib.aggregate_variant([q], "v", resources)
    assert agg["prompt_tokens_median"] == 2000
    assert agg["prompt_tokens_max"] == 3000
    assert agg["load_median"] == 44.0
    assert agg["context_length"] == 8192


def test_aggregate_excludes_questions_with_search_failure():
    ok = _run(_question(1), {"v": [_attempt("v", 1, finish="length")]})
    failed = _run(_question(2), {}, search_error="сервер эмбеддингов недоступен")
    agg = lib.aggregate_variant([ok, failed], "v")
    assert agg["with_calls"] == 1
    assert agg["attempts"] == 1


def test_aggregate_coverage_spread_between_repeats():
    q = _run(_question(), {"v": [
        _attempt("v", 1, coverage=2), _attempt("v", 2, coverage=1), _attempt("v", 3, coverage=2),
    ]})
    agg = lib.aggregate_variant([q], "v")
    assert agg["judged"] == 3
    assert agg["avg_coverage"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert agg["spread"] == {1: (0.5, 1.0, 2)}


def test_attach_calls_aligns_by_repeat_order_and_tolerates_gaps():
    q = _run(_question(), {"v": [_attempt("v", 1), _attempt("v", 2)]})
    q = lib.attach_calls(q, {"v": [[{"finish_reason": "stop"}]]})
    attempts = q["attempts"]["v"]
    assert attempts[0]["calls"] == [{"finish_reason": "stop"}]
    assert attempts[1]["calls"] == []


# --------------------------------------------------------------------------- #
# Отчёт и тексты (2.3)
# --------------------------------------------------------------------------- #


def _report(*, judge=True, status=ev.STATUS_COMPLETED, repeats=2):
    base = lib.Variant("base", "model-a:1b", baseline=True)
    fast = lib.Variant("ctx8k", "model-a:1b", params={"num_ctx": 8192})
    effort = lib.Variant("think-low", "model-a:1b", reasoning_effort="low")
    q1 = _run(_question(1), {
        "base": [_attempt("base", 1, tokens=100, elapsed=20.0, cold=True, coverage=1),
                 _attempt("base", 2, tokens=100, elapsed=20.0, coverage=1, finish="length")],
        "ctx8k": [_attempt("ctx8k", 1, tokens=100, elapsed=10.0, cold=True, coverage=2),
                  _attempt("ctx8k", 2, tokens=100, elapsed=10.0, coverage=2)],
        "think-low": [_attempt("think-low", 1, tokens=100, elapsed=5.0, cold=True),
                      _attempt("think-low", 2, tokens=100, elapsed=5.0)],
    })
    q2 = _run(_question(2), {}, search_error="сервер эмбеддингов недоступен")
    resources = {
        "base": {"load_seconds": [44.0], "size": 19_000_000_000, "size_vram": 0,
                 "context_length": 4096},
        "ctx8k": {"load_seconds": [46.0], "size": 20_000_000_000, "size_vram": 0,
                  "context_length": 8192},
        "think-low": {"load_seconds": [44.0], "size": 19_000_000_000, "size_vram": 0,
                      "context_length": 4096},
    }
    judge_dict = {"provider": "kimi", "label": "kimi/kimi-judge"} if judge else None
    return lib.build_report(
        started_at="2026-10-11T10:00:00Z",
        finished_at="2026-10-11T10:30:00Z",
        settings={"judge": judge_dict, "strategy": "structural", "repeats": repeats,
                  "environment": "OLLAMA_KV_CACHE_TYPE=q8_0"},
        variants=[base, fast, effort],
        resources=resources,
        questions=[q1, q2],
        planned=2,
        repeats=repeats,
        status=status,
        stop_reason="по Ctrl+C" if status == ev.STATUS_STOPPED else None,
    )


def test_report_roundtrip_markers():
    report = _report()
    assert lib.is_tune_report(report)
    assert report["kind"] == lib.KIND and report["version"] == lib.REPORT_VERSION
    assert not lib.is_tune_report({**report, "kind": "models"})
    assert not lib.is_tune_report({**report, "version": 99})
    assert not lib.is_tune_report("мусор")


def test_summary_shows_baseline_comparison_and_sample_size():
    text = lib.format_summary(_report())
    assert "← до" in text
    assert "вопросов — 1, повторов на вариант — 2" in text
    assert "(+100%)" in text          # ctx8k: 10 ток/с против 5 у исходного
    assert "Обрезано" in text and "1 из 2" in text
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in text
    assert "Исключено из сравнения" in text and "2 (" in text
    assert "ориентир" in text


def test_summary_marks_portability():
    text = lib.format_summary(_report())
    assert "доработки бота" in text    # think-low
    assert "Modelfile рабочего тега" in text


def test_summary_without_judge_has_no_quality_columns_and_no_judge_channel():
    text = lib.format_summary(_report(judge=False))
    assert "Покрытие" not in text
    assert "Противореч." not in text
    assert "Качество не оценивалось" in text
    assert "kimi" not in text


def test_summary_with_judge_names_provider_and_data_channel():
    text = lib.format_summary(_report(judge=True))
    assert "Покрытие" in text
    assert "kimi" in text.lower()
    assert "передавались" in text


def test_summary_partial_report_is_marked():
    text = lib.format_summary(_report(status=ev.STATUS_STOPPED))
    assert "⛔ Прогон остановлен" in text
    assert "по Ctrl+C" in text


def test_summary_reads_report_without_calls_field():
    report = _report()
    for q in report["questions"]:
        for items in q["attempts"].values():
            for a in items:
                a.pop("calls", None)
    text = lib.format_summary(report)    # старый/урезанный отчёт читается
    assert "← до" in text


def test_question_detail_and_unknown_number():
    report = _report()
    detail = lib.format_question_detail(report, 1)
    assert detail is not None
    assert "base" in detail and "ctx8k" in detail and "think-low" in detail
    assert lib.format_question_detail(report, 77) is None
    assert lib.question_numbers(report) == [1, 2]


def test_report_file_name_and_latest():
    names = ["20261010T093744Z_tune.json", "20261011T100000Z_tune.json",
             "20261011T110000Z_models.json", "questions.json", "x_tune.json"]
    assert lib.report_file_name("20261011T100000Z") == "20261011T100000Z_tune.json"
    assert lib.latest_report_name(names) == "20261011T100000Z_tune.json"
    assert lib.latest_report_name(["questions.json"]) is None


def test_pick_questions_all_independent_or_counted():
    qs = [
        _question(1),
        _question(2),
        ev.Question(3, "Продолжение?", "concept", ("ф",), ("Д",), history=("Вопрос 1?",)),
        _question(4, abstain=True),
    ]
    everything = lib.pick_questions(qs, None)
    assert [q.number for q in everything.questions] == [1, 2, 4]
    assert everything.notes
    counted = lib.pick_questions(qs, 1)
    assert [q.number for q in counted.questions] == [1, 4]
    empty = lib.pick_questions([qs[2]], None)
    assert empty.error
