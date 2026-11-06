"""Output Guard: проверка ответа модели перед отдачей пользователю.

Ловим четыре класса проблем:
1. Сгенерированные моделью секреты — тем же каталогом правил, что и Input Guard.
2. Утечку system prompt — по случайному canary-маркеру и по дословным фрагментам.
3. Подозрительные URL — IP-литералы, сокращатели, javascript:, data:, креды в URL.
4. Опасные команды — rm -rf, mkfs, pipe-to-shell, fork bomb, eval, shell=True и т.п.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .detectors import Finding, _preview, apply_redactions, scan_text

# ---------------------------------------------------------------------------
# Правила
# ---------------------------------------------------------------------------

SUSPICIOUS_URL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "url_ip_literal",
        re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/\S*)?", re.IGNORECASE),
    ),
    (
        "url_shortener",
        re.compile(
            r"https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|cutt\.ly)/\S+",
            re.IGNORECASE,
        ),
    ),
    ("url_javascript", re.compile(r"javascript:\S+", re.IGNORECASE)),
    ("url_data_html", re.compile(r"data:text/html[^\s)\"']*", re.IGNORECASE)),
    (
        "url_with_credentials",
        re.compile(r"https?://[^\s/@]+:[^\s/@]+@[^\s/]+", re.IGNORECASE),
    ),
    ("url_punycode", re.compile(r"https?://xn--\S+", re.IGNORECASE)),
)

DANGEROUS_COMMAND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cmd_rm_rf", re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(?:/|~|\*|/\*)", re.IGNORECASE)),
    ("cmd_mkfs", re.compile(r"\bmkfs(?:\.\w+)?\b", re.IGNORECASE)),
    ("cmd_dd", re.compile(r"\bdd\s+if=", re.IGNORECASE)),
    ("cmd_fork_bomb", re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE)),
    (
        "cmd_pipe_to_shell",
        re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sh|bash|zsh|python)\b", re.IGNORECASE),
    ),
    ("cmd_chmod_777", re.compile(r"\bchmod\s+777\b", re.IGNORECASE)),
    ("cmd_sudo", re.compile(r"\bsudo\s+\S+", re.IGNORECASE)),
    ("cmd_powershell_iex", re.compile(r"\b(?:Invoke-Expression|IEX)\b\s*[\(\s]", re.IGNORECASE)),
    ("cmd_eval", re.compile(r"\beval\s*\(", re.IGNORECASE)),
    ("cmd_os_system", re.compile(r"\bos\.system\s*\(", re.IGNORECASE)),
    (
        "cmd_shell_true",
        re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True", re.IGNORECASE),
    ),
)


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------


def scan_output(
    text: str,
    *,
    canary: str | None = None,
    system_prompt: str | None = None,
) -> list[Finding]:
    """Проверить ответ модели; вернуть список находок."""
    if not text:
        return []

    findings: list[Finding] = list(scan_text(text))

    if canary and canary in text:
        start = text.index(canary)
        findings.append(
            Finding(
                type="system_prompt_leak_canary",
                redaction="[REDACTED_SYSTEM_PROMPT]",
                preview=_preview(canary),
                kind="canary",
                start=start,
                end=start + len(canary),
            )
        )

    if system_prompt:
        for fragment in _fragments(system_prompt):
            if fragment in text:
                start = text.index(fragment)
                findings.append(
                    Finding(
                        type="system_prompt_leak_fragment",
                        redaction="[REDACTED_SYSTEM_PROMPT]",
                        preview=_preview(fragment),
                        kind="fragment",
                        start=start,
                        end=start + len(fragment),
                    )
                )
                break

    for name, pattern in SUSPICIOUS_URL_RULES:
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    type=name,
                    redaction="[REDACTED_URL]",
                    preview=_preview(match.group(0)),
                    kind="url",
                    start=match.start(),
                    end=match.end(),
                )
            )

    for name, pattern in DANGEROUS_COMMAND_RULES:
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    type=name,
                    redaction="[REDACTED_COMMAND]",
                    preview=_preview(match.group(0)),
                    kind="command",
                    start=match.start(),
                    end=match.end(),
                )
            )

    return _dedupe(findings)


def block_message(findings: Iterable[Finding]) -> str:
    """Безопасная замена ответа в block-режиме."""
    types = ", ".join(sorted({finding.type for finding in findings}))
    return f"⚠️ Ответ заблокирован Output Guard. Обнаружено: {types}."


def mask_output(text: str, findings: Iterable[Finding]) -> str:
    """Замаскировать найденное в ответе (mask-режим)."""
    return apply_redactions(text, findings)


# ---------------------------------------------------------------------------
# Внутренние помощники
# ---------------------------------------------------------------------------


def _fragments(system_prompt: str) -> Iterable[str]:
    """Достаточно длинные дословные фрагменты system prompt для проверки утечки."""
    for part in re.split(r"[.\n;]+", system_prompt):
        part = part.strip()
        if len(part) >= 40:
            yield part


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, int | None, int | None, str]] = set()
    result: list[Finding] = []
    for finding in findings:
        key = (finding.type, finding.start, finding.end, finding.preview)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result
