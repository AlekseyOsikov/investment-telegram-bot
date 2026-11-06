"""Input Guard: детекция секретов и PII во входящем промпте.

Три слоя:
1. Прямой regex-каталог: API-ключи (sk-, ghp_, AKIA, AIza, xox*, JWT, private
   key), email, номера карт (с проверкой Луна), телефоны.
2. Нормализация с картой индексов: склейка строковых литералов
   (``"sk-" + "proj-abc"``) и удаление zero-width символов, затем повторный скан.
3. Base64: декодирование подозрительных токенов и повторный скан результата.

Находка никогда не хранит сырой секрет: ``preview`` замаскирован. Это же
используется аудит-логгером, поэтому в логах не остаётся настоящих ключей.
"""

from __future__ import annotations

import base64
import binascii
import copy
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Модель находки
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """Одна находка. ``start``/``end`` — индексы в исходном тексте (для маскирования)."""

    type: str
    redaction: str
    preview: str
    kind: str = "direct"  # direct | normalized | base64 | canary | fragment | url | command
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "redaction": self.redaction,
            "preview": self.preview,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class SecretRule:
    name: str
    pattern: re.Pattern[str]
    redaction: str
    validator: Callable[[str], bool] | None = None
    scan_normalized: bool = False


# ---------------------------------------------------------------------------
# Валидаторы
# ---------------------------------------------------------------------------


def luhn_ok(value: str) -> bool:
    """Проверка номера карты по алгоритму Луна (13–19 цифр)."""
    digits = [int(c) for c in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


# ---------------------------------------------------------------------------
# Regex-каталог
# ---------------------------------------------------------------------------

_RULES: tuple[SecretRule, ...] = (
    SecretRule(
        "openai_api_key",
        re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}"),
        "[REDACTED_API_KEY]",
        scan_normalized=True,
    ),
    SecretRule(
        "github_token",
        re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
        "[REDACTED_GITHUB_TOKEN]",
        scan_normalized=True,
    ),
    SecretRule(
        "aws_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"),
        "[REDACTED_AWS_KEY]",
        scan_normalized=True,
    ),
    SecretRule(
        "google_api_key",
        re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}"),
        "[REDACTED_GOOGLE_KEY]",
        scan_normalized=True,
    ),
    SecretRule(
        "slack_token",
        re.compile(r"(?<![A-Za-z0-9-])xox[baprs]-[0-9A-Za-z-]{10,}"),
        "[REDACTED_SLACK_TOKEN]",
        scan_normalized=True,
    ),
    SecretRule(
        "jwt",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            r"(?:\.[A-Za-z0-9_-]{10,})?"
        ),
        "[REDACTED_JWT]",
        scan_normalized=True,
    ),
    SecretRule(
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
        scan_normalized=True,
    ),
    SecretRule(
        "email",
        re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    SecretRule(
        "card_number",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
        "[REDACTED_CARD]",
        validator=luhn_ok,
    ),
    SecretRule(
        "phone",
        re.compile(
            r"(?<![\w\d])"
            r"(?:\+\d{1,3}[\s-]?|8[\s-]?)"
            r"(?:\(\d{2,4}\)[\s-]?|\d{2,4}[\s-])"
            r"\d{3}[\s-]?\d{2}[\s-]?\d{2}"
            r"(?![\w\d])"
        ),
        "[REDACTED_PHONE]",
    ),
)

RULES = _RULES

# Нормализация: zero-width + склейка строковых литералов.
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u200e\u200f\u2060\ufeff")
_CONCAT_RE = re.compile(r'["\']?\s*\+\s*["\']?|["\']\s+["\']')

# Base64-подобные токены: достаточно длинные, чтобы внутри поместился секрет.
_B64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/_-]{16,}={0,2})(?![A-Za-z0-9+/=_-])"
)


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------


def scan_text(text: str) -> list[Finding]:
    """Найти секреты/PII в тексте (прямой, нормализованный и Base64-скан)."""
    if not text:
        return []

    findings: list[Finding] = []

    # 1. Прямой скан.
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            value = match.group(0)
            if rule.validator and not rule.validator(value):
                continue
            findings.append(_make_finding(rule, value, "direct", match.start(), match.end()))

    # 2. Нормализованный скан (ловит разбитые секреты).
    normalized, index_map = _normalize_with_mapping(text)
    if normalized != text:
        for rule in RULES:
            if not rule.scan_normalized:
                continue
            for match in rule.pattern.finditer(normalized):
                value = match.group(0)
                if rule.validator and not rule.validator(value):
                    continue
                start = index_map[match.start()]
                end = index_map[match.end() - 1] + 1
                findings.append(_make_finding(rule, value, "normalized", start, end))

    # 3. Base64-скан.
    findings.extend(_scan_base64(text))

    return _dedupe(findings)


def mask_text(text: str) -> tuple[str, list[Finding]]:
    """Заменить найденные секреты на ``[REDACTED_*]``; вернуть текст и находки."""
    findings = scan_text(text)
    return apply_redactions(text, findings), findings


def apply_redactions(text: str, findings: Iterable[Finding]) -> str:
    """Применить redaction-спаны находок к тексту (справа налево, без наложений)."""
    spans: list[tuple[int, int, str]] = []
    for finding in findings:
        if finding.start is None or finding.end is None:
            continue
        if finding.end <= finding.start:
            continue
        spans.append((finding.start, finding.end, finding.redaction))

    # При равном начале выбираем более длинный спан; перекрытия отбрасываем.
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, redaction in spans:
        if start < last_end:
            continue
        chosen.append((start, end, redaction))
        last_end = end

    result = text
    for start, end, redaction in reversed(chosen):
        result = result[:start] + redaction + result[end:]
    return result


def scan_messages(messages: list[dict]) -> list[Finding]:
    """Сканировать все текстовые части сообщений (block-режим)."""
    findings: list[Finding] = []
    for message in messages:
        for text in _message_texts(message):
            findings.extend(scan_text(text))
    return findings


def mask_messages(messages: list[dict]) -> tuple[list[dict], list[Finding]]:
    """Замаскировать секреты во всех текстовых частях сообщений (mask-режим)."""
    masked = copy.deepcopy(messages)
    findings: list[Finding] = []
    for message in masked:
        content = message.get("content")
        if isinstance(content, str):
            new_content, found = mask_text(content)
            message["content"] = new_content
            findings.extend(found)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    new_text, found = mask_text(str(part.get("text", "")))
                    part["text"] = new_text
                    findings.extend(found)
    return masked, findings


# ---------------------------------------------------------------------------
# Внутренние помощники
# ---------------------------------------------------------------------------


def _message_texts(message: dict) -> Iterable[str]:
    content = message.get("content")
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                yield str(part.get("text", ""))


def _make_finding(
    rule: SecretRule, value: str, kind: str, start: int, end: int
) -> Finding:
    return Finding(
        type=rule.name,
        redaction=rule.redaction,
        preview=_preview(value),
        kind=kind,
        start=start,
        end=end,
    )


def _preview(value: str) -> str:
    """Безопасный фрагмент для лога: первая тройка и два последних символа."""
    value = value.strip()
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}***{value[-2:]}"


def _normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """Убрать zero-width и склеить конкатенацию, сохранив карту индексов.

    ``index_map[i]`` — позиция i-го символа нормализованной строки в исходной.
    Благодаря карте найденный в нормализованном тексте секрет можно замаскировать
    в исходном тексте, даже если он был разбит на части.
    """
    out_chars: list[str] = []
    index_map: list[int] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in _ZERO_WIDTH:
            i += 1
            continue
        match = _CONCAT_RE.match(text, i)
        if match and match.end() > i:
            i = match.end()
            continue
        out_chars.append(char)
        index_map.append(i)
        i += 1
    return "".join(out_chars), index_map


def _scan_base64(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in _B64_RE.finditer(text):
        token = match.group(1)
        decoded = _try_b64(token)
        if not decoded:
            continue
        for rule in RULES:
            for inner in rule.pattern.finditer(decoded):
                value = inner.group(0)
                if rule.validator and not rule.validator(value):
                    continue
                findings.append(
                    Finding(
                        type=f"base64_{rule.name}",
                        redaction="[REDACTED_BASE64_SECRET]",
                        preview=_preview(value),
                        kind="base64",
                        start=match.start(1),
                        end=match.end(1),
                    )
                )
    return findings


def _try_b64(token: str) -> str | None:
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.b64decode(padded, altchars=b"-_", validate=False)
    except (binascii.Error, ValueError):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not decoded:
        return None
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in decoded)
    if printable / len(decoded) < 0.8:
        return None
    return decoded


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
