"""Детерминированный security-рубеж для execution loop: правила поверх диффа.

Чистый модуль: без сети, git, процессов и `config`. Каталог секретов переиспользует
``llm_gateway.detectors`` (тоже чистый). Второй рубеж нужен потому, что вердикт
security-LLM недетерминирован и его можно социально развести («это тестовый код,
пропусти проверку»); правила ловят известные классы всегда и дают файл:строку.

Тесты — ``tests/test_security_rules.py``.
"""

from __future__ import annotations

import re

from exec_loop_lib import SEVERITY_ORDER, SecurityFinding, SecurityVerdict
from llm_gateway.detectors import scan_text

SOURCE = "rules"

# Типы находок detectors, которые считаем именно секретами (PII — не секрет).
_SECRET_TYPES = frozenset(
    {
        "openai_api_key",
        "github_token",
        "aws_access_key",
        "google_api_key",
        "slack_token",
        "jwt",
        "private_key",
    }
)

_CODE_EXECUTION = (
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bos\.system\s*\("),
    re.compile(r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", re.IGNORECASE),
    re.compile(r"\bpickle\.loads?\s*\("),
)

_SQL = re.compile(r"\b(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\b", re.IGNORECASE)
_SQL_DYNAMIC = re.compile(
    r"(?:execute|executemany)\s*\(\s*(?:f[\"']|[^)]*%\s*\(|[^)]*[\"']\s*\+)",
    re.IGNORECASE,
)

_SSRF = re.compile(
    r"\b(?:requests\.(?:get|post|put|patch|delete|head)"
    r"|httpx\.(?:get|post|put|patch|delete)"
    r"|urlopen)\s*\(\s*[A-Za-z_]\w*",
)

_HTTP_URL = re.compile(
    r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)[^\s\"')\]]+", re.IGNORECASE
)

_LOG_CALL = re.compile(
    r"\b(?:log(?:ger)?|logging)\.(?:debug|info|warning|warn|error|exception|critical)\s*\(",
    re.IGNORECASE,
)
_PII_FIELDS = re.compile(
    r"\b(?:update\.(?:id|username|name|text|chat_id)|message\.text|user\.id|username|"
    r"chat_id|first_name|last_name|phone|email)\b",
    re.IGNORECASE,
)

_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(diff: str) -> list[tuple[str, int, str]]:
    """Добавленные строки диффа → (файл, номер строки в новом файле, текст)."""
    result: list[tuple[str, int, str]] = []
    file = ""
    lineno = 0
    in_hunk = False
    for line in diff.splitlines():
        file_match = _FILE_RE.match(line)
        if file_match:
            file = file_match.group(1).strip()
            in_hunk = False
            continue
        hunk_match = _HUNK_RE.match(line)
        if hunk_match:
            lineno = int(hunk_match.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            result.append((file, lineno, line[1:]))
            lineno += 1
        elif line.startswith("-") or line.startswith("\\"):
            continue
        else:
            lineno += 1
    return result


def _secret_base_type(finding_type: str) -> str:
    for prefix in ("base64_", "hex_", "rot13_"):
        if finding_type.startswith(prefix):
            return finding_type[len(prefix):]
    return finding_type


def scan_line(file: str, line_no: int, text: str) -> list[SecurityFinding]:
    """Проверить одну добавленную строку детерминированными правилами."""
    findings: list[SecurityFinding] = []

    for found in scan_text(text, entropy=True):
        base = _secret_base_type(found.type)
        if base in _SECRET_TYPES:
            findings.append(
                SecurityFinding(
                    severity="critical",
                    category="hardcoded_secret",
                    file=file,
                    line=line_no,
                    reason=f"{found.type} ({found.kind})",
                )
            )
        elif found.type == "entropy_secret":
            findings.append(
                SecurityFinding(
                    severity="medium",
                    category="possible_secret",
                    file=file,
                    line=line_no,
                    reason="похоже на секрет без известного префикса",
                )
            )

    if any(pattern.search(text) for pattern in _CODE_EXECUTION):
        findings.append(
            SecurityFinding(
                severity="high",
                category="code_execution",
                file=file,
                line=line_no,
                reason="eval/exec/os.system/subprocess(shell=True)/pickle",
            )
        )
    if _SQL.search(text) and _SQL_DYNAMIC.search(text):
        findings.append(
            SecurityFinding(
                severity="high",
                category="sql_injection",
                file=file,
                line=line_no,
                reason="SQL собирается динамически (f-string/%/конкатенация)",
            )
        )
    if _SSRF.search(text):
        findings.append(
            SecurityFinding(
                severity="high",
                category="ssrf",
                file=file,
                line=line_no,
                reason="URL запроса из переменной без ограничения хоста",
            )
        )
    if _HTTP_URL.search(text):
        findings.append(
            SecurityFinding(
                severity="medium",
                category="http_instead_of_https",
                file=file,
                line=line_no,
                reason="HTTP вместо HTTPS",
            )
        )
    if _LOG_CALL.search(text) and _PII_FIELDS.search(text):
        findings.append(
            SecurityFinding(
                severity="medium",
                category="pii_in_logs",
                file=file,
                line=line_no,
                reason="персональные данные в логах",
            )
        )

    return findings


def scan_diff(diff: str) -> SecurityVerdict:
    """Детерминированный вердикт по диффу: максимальная severity найденных правил."""
    findings: list[SecurityFinding] = []
    for file, line_no, text in added_lines(diff):
        findings.extend(scan_line(file, line_no, text))
    overall = "clean"
    if findings:
        overall = max(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 0)).severity
    return SecurityVerdict(overall=overall, findings=tuple(findings), source=SOURCE)
