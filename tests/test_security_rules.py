"""Тесты детерминированного security-рубежа (scripts/security_rules.py).

Без сети и git: чистые правила по диффу.
"""

import base64

import security_rules as rules


def _diff(*lines: str) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    return (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def test_added_lines_tracks_file_and_line() -> None:
    assert rules.added_lines(_diff("x = 1")) == [("x.py", 1, "x = 1")]


def test_added_lines_ignores_removed_and_context() -> None:
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,2 +1,2 @@\n context\n-old\n+new\n"
    )
    assert rules.added_lines(diff) == [("x.py", 2, "new")]


def test_known_secret_is_critical() -> None:
    verdict = rules.scan_diff(_diff('TOKEN = "sk-proj-abc1234567890"'))
    assert verdict.overall == "critical" and verdict.source == "rules"
    finding = next(f for f in verdict.findings if f.category == "hardcoded_secret")
    assert finding.file == "x.py" and finding.line == 1


def test_base64_secret_is_critical() -> None:
    token = base64.b64encode(b"sk-proj-abc1234567890").decode()
    verdict = rules.scan_diff(_diff(f'DATA = "{token}"'))
    assert verdict.overall == "critical"
    assert any(f.category == "hardcoded_secret" for f in verdict.findings)


def test_entropy_secret_is_medium() -> None:
    verdict = rules.scan_diff(_diff('AUTH = "auth_live_9fK2mQ7pR4tX8zB1nV6c"'))
    assert verdict.overall == "medium"
    assert any(f.category == "possible_secret" for f in verdict.findings)


def test_code_execution_is_high() -> None:
    for line in (
        "os.system(cmd)",
        "eval(expr)",
        "subprocess.run(cmd, shell=True)",
        "pickle.loads(data)",
    ):
        verdict = rules.scan_diff(_diff(line))
        assert verdict.overall == "high", line
        assert any(f.category == "code_execution" for f in verdict.findings), line


def test_sql_injection_is_high() -> None:
    verdict = rules.scan_diff(_diff('cursor.execute(f"SELECT * FROM t WHERE id={x}")'))
    assert verdict.overall == "high"
    assert any(f.category == "sql_injection" for f in verdict.findings)


def test_parameterized_sql_is_not_flagged() -> None:
    verdict = rules.scan_diff(_diff('cursor.execute("SELECT * FROM t WHERE id=%s", (x,))'))
    assert not any(f.category == "sql_injection" for f in verdict.findings)


def test_ssrf_is_high() -> None:
    verdict = rules.scan_diff(_diff("response = requests.get(url)"))
    assert verdict.overall == "high"
    assert any(f.category == "ssrf" for f in verdict.findings)


def test_literal_url_is_not_ssrf() -> None:
    verdict = rules.scan_diff(_diff('requests.get("https://api.example.com/data")'))
    assert not any(f.category == "ssrf" for f in verdict.findings)


def test_http_url_is_medium_but_localhost_is_not() -> None:
    verdict = rules.scan_diff(_diff('URL = "http://example.com/api"'))
    assert any(f.category == "http_instead_of_https" for f in verdict.findings)
    local = rules.scan_diff(_diff('URL = "http://localhost:8000/api"'))
    assert not any(f.category == "http_instead_of_https" for f in local.findings)


def test_pii_in_logs_is_medium() -> None:
    verdict = rules.scan_diff(_diff('logger.info("request chat_id=%s", update.chat_id)'))
    assert any(f.category == "pii_in_logs" for f in verdict.findings)


def test_pii_in_print_is_flagged() -> None:
    # P3: print с PII раньше пропускался (правило ловило только logger.*).
    verdict = rules.scan_diff(_diff("print(user.email)"))
    assert any(f.category == "pii_in_logs" for f in verdict.findings)


def test_print_without_pii_is_not_flagged() -> None:
    verdict = rules.scan_diff(_diff("print(total)"))
    assert not any(f.category == "pii_in_logs" for f in verdict.findings)


def test_clean_diff_is_clean() -> None:
    verdict = rules.scan_diff(_diff("total = price * quantity"))
    assert verdict.overall == "clean" and verdict.findings == ()


def test_overall_is_max_severity() -> None:
    verdict = rules.scan_diff(
        _diff('logger.info("chat_id=%s", update.chat_id)', "os.system(cmd)")
    )
    assert verdict.overall == "high"
    assert {f.category for f in verdict.findings} >= {"pii_in_logs", "code_execution"}
