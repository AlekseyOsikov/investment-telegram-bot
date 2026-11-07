"""Тесты security step (scripts/exec_loop_lib.py): разбор вердикта, severity, фидбек, слияние.

Без сети и git: проверяются чистые правила второго LLM-вызова.
"""

import exec_loop_lib as lib
import pytest

# ---------------------------------------------------------------------------
# Разбор вердикта security review
# ---------------------------------------------------------------------------


def test_parse_clean_verdict() -> None:
    verdict = lib.parse_security_verdict('{"overall": "clean", "findings": []}')
    assert verdict is not None
    assert verdict.overall == "clean" and verdict.findings == ()
    assert not verdict.blocks and verdict.warnings == []


def test_parse_high_verdict_with_findings() -> None:
    text = (
        '{"overall":"high","findings":[{"severity":"high",'
        '"category":"hardcoded_secret","file":"scripts/x.py","line":7,'
        '"reason":"token in code"}]}'
    )
    verdict = lib.parse_security_verdict(text)
    assert verdict is not None
    assert verdict.overall == "high" and verdict.blocks
    assert verdict.findings[0].category == "hardcoded_secret"
    assert verdict.findings[0].file == "scripts/x.py"
    assert verdict.findings[0].line == 7


def test_parse_verdict_from_markdown_fence() -> None:
    text = '```json\n{"overall": "low", "findings": []}\n```'
    verdict = lib.parse_security_verdict(text)
    assert verdict is not None and verdict.overall == "low"


def test_parse_overall_inferred_from_findings() -> None:
    # overall отсутствует — берём максимальную severity находок.
    text = '{"findings":[{"severity":"medium","category":"pii_in_logs"},' \
           '{"severity":"critical","category":"sql_injection"}]}'
    verdict = lib.parse_security_verdict(text)
    assert verdict is not None and verdict.overall == "critical"


def test_parse_invalid_returns_none() -> None:
    assert lib.parse_security_verdict("не JSON вовсе") is None
    assert lib.parse_security_verdict("") is None
    assert lib.parse_security_verdict("{битый json") is None


def test_parse_tolerates_unknown_severity() -> None:
    verdict = lib.parse_security_verdict(
        '{"overall":"weird","findings":[{"severity":"unknown","category":"x"}]}'
    )
    assert verdict is not None
    # Неизвестный overall выводится из находок; неизвестная severity находки → low.
    assert verdict.overall == "low" and verdict.findings[0].severity == "low"


@pytest.mark.parametrize(
    ("severity", "blocks"),
    [("critical", True), ("high", True), ("medium", False), ("low", False), ("clean", False)],
)
def test_block_threshold(severity: str, blocks: bool) -> None:
    assert lib.SecurityVerdict(severity).blocks is blocks


def test_warn_severities_are_reported() -> None:
    verdict = lib.SecurityVerdict(
        "medium",
        (
            lib.SecurityFinding("medium", "pii_in_logs"),
            lib.SecurityFinding("low", "missing_input_validation"),
            lib.SecurityFinding("high", "hardcoded_secret"),
        ),
    )
    assert {f.severity for f in verdict.warnings} == {"medium", "low"}


# ---------------------------------------------------------------------------
# Gateway-блок и слияние с LLM
# ---------------------------------------------------------------------------


def test_verdict_from_gateway_block_is_critical() -> None:
    verdict = lib.verdict_from_gateway_block(
        [{"type": "openai_api_key", "preview": "sk-***90"}]
    )
    assert verdict.overall == "critical" and verdict.blocks
    assert verdict.source == "gateway_input_guard"
    assert verdict.findings[0].category == "gateway:openai_api_key"
    assert "Input Guard" in verdict.findings[0].reason


def test_verdict_from_empty_gateway_block_still_critical() -> None:
    verdict = lib.verdict_from_gateway_block([])
    assert verdict.overall == "critical" and verdict.findings


def test_merge_takes_max_severity_and_concatenates_findings() -> None:
    gateway = lib.verdict_from_gateway_block([{"type": "jwt"}])
    llm = lib.SecurityVerdict("medium", (lib.SecurityFinding("medium", "pii_in_logs"),))
    merged = lib.merge_security_verdicts(gateway, llm)
    assert merged is not None
    assert merged.overall == "critical"
    assert merged.source == "gateway_input_guard+llm"
    assert {f.category for f in merged.findings} == {"gateway:jwt", "pii_in_logs"}


def test_merge_handles_missing_side() -> None:
    llm = lib.SecurityVerdict("low", (lib.SecurityFinding("low", "x"),))
    assert lib.merge_security_verdicts(None, llm) is llm
    gateway = lib.verdict_from_gateway_block([{"type": "email"}])
    assert lib.merge_security_verdicts(gateway, None) is gateway
    assert lib.merge_security_verdicts(None, None) is None


# ---------------------------------------------------------------------------
# Фидбек для повторной генерации
# ---------------------------------------------------------------------------


def test_feedback_names_file_and_line() -> None:
    verdict = lib.SecurityVerdict(
        "high",
        (
            lib.SecurityFinding(
                "high", "hardcoded_secret", "scripts/demo_security_token.py", 7, "токен в коде"
            ),
        ),
    )
    feedback = lib.format_security_feedback(verdict)
    assert "SECURITY REVIEW" in feedback
    assert "исправь: hardcoded_secret в scripts/demo_security_token.py:7" in feedback
    assert "токен в коде" in feedback


def test_feedback_without_position() -> None:
    verdict = lib.SecurityVerdict(
        "critical",
        (lib.SecurityFinding("critical", "gateway:openai_api_key", reason="секрет в диффе"),),
    )
    feedback = lib.format_security_feedback(verdict)
    assert "исправь: gateway:openai_api_key" in feedback
    assert ":0" not in feedback


# ---------------------------------------------------------------------------
# Промпт
# ---------------------------------------------------------------------------


def test_security_messages_include_stack_schema_and_diff() -> None:
    system, user = lib.build_security_messages("- old\n+ new")
    assert "security-ревьюер" in system
    assert "hardcoded_secret" in system and "pii_in_logs" in system
    assert "http_instead_of_https" in system and "sql_injection" in system
    assert "Python 3.11" in system and "Telegram" in system
    assert '"overall"' in system
    assert "```diff" in user and "+ new" in user


# ---------------------------------------------------------------------------
# Интеграция со структурами раннера
# ---------------------------------------------------------------------------


def _attempt(**kwargs) -> lib.Attempt:
    values = {"seconds": 1.0, "returncode": 0, "turns": 3, "changed_files": 1}
    values.update(kwargs)
    return lib.Attempt(**values)


def test_security_block_is_its_own_failure_class() -> None:
    blocked = _attempt(security_blocked=True, security_overall="high")
    assert lib.classify_failure([blocked]) == lib.FAIL_SECURITY


def test_render_log_shows_security_verdict() -> None:
    attempt = _attempt(
        security_overall="medium",
        security_findings=[{"severity": "medium", "category": "pii_in_logs"}],
    )
    result = lib.TaskResult("S2", [attempt], passed=True)
    result.security_warnings = [{"severity": "medium", "category": "pii_in_logs"}]
    text = lib.render_log("run", "model-x", [result])
    assert "Security" in text
    assert "medium: pii_in_logs" in text
    assert "warnings: 1" in text


def test_render_log_without_security_shows_dash() -> None:
    result = lib.TaskResult("T01", [_attempt()], passed=True)
    text = lib.render_log("run", "m", [result])
    assert "| — |" in text
