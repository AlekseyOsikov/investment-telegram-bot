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
import math
import re
from collections import Counter
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

# Hex-подобные токены (чётная длина, чтобы декодировались в байты).
_HEX_RE = re.compile(r"(?<![A-Za-z0-9])([0-9a-fA-F]{16,})(?![A-Za-z0-9])")

# Кандидаты в «секрет без известного префикса»: длинный токен из букв/цифр/_- .
_ENTROPY_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{24,80})(?![A-Za-z0-9_-])")

# Правила, которые считаются именно секретами (не PII): только их ищем в кодировках.
_SECRET_RULE_NAMES = frozenset(
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

_ROT13_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------


def scan_text(text: str, *, entropy: bool = False) -> list[Finding]:
    """Найти секреты/PII в тексте.

    Слои: прямой regex, нормализованный (разбитые секреты, в т.ч. склейка через комментарии),
    Base64, hex и ROT13. ``entropy=True`` добавляет эвристику «секрет без известного префикса».
    """
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

    # 3. Кодировки: Base64, hex, ROT13.
    findings.extend(_scan_base64(text))
    findings.extend(_scan_hex(text))
    findings.extend(_scan_rot13(text))

    # 4. Энтропия (опционально): только для спанов, не занятых известными правилами.
    if entropy:
        occupied = [(f.start, f.end) for f in findings if f.start is not None]
        for finding in scan_entropy(text):
            if not _overlaps(finding, occupied):
                findings.append(finding)

    return _dedupe(findings)


def mask_text(text: str, *, entropy: bool = False) -> tuple[str, list[Finding]]:
    """Заменить найденные секреты на ``[REDACTED_*]``; вернуть текст и находки."""
    findings = scan_text(text, entropy=entropy)
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


def scan_messages(messages: list[dict], *, entropy: bool = False) -> list[Finding]:
    """Сканировать все текстовые части сообщений (block-режим)."""
    findings: list[Finding] = []
    for message in messages:
        for text in _message_texts(message):
            findings.extend(scan_text(text, entropy=entropy))
    return findings


def mask_messages(
    messages: list[dict], *, entropy: bool = False
) -> tuple[list[dict], list[Finding]]:
    """Замаскировать секреты во всех текстовых частях сообщений (mask-режим)."""
    masked = copy.deepcopy(messages)
    findings: list[Finding] = []
    for message in masked:
        content = message.get("content")
        if isinstance(content, str):
            new_content, found = mask_text(content, entropy=entropy)
            message["content"] = new_content
            findings.extend(found)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    new_text, found = mask_text(str(part.get("text", "")), entropy=entropy)
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
        # Комментарии между частями секрета (sk-/* … */proj-…, sk-// …) не мешают склейке:
        # блок-комментарий пропускаем целиком, строчный — до конца строки.
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end != -1:
                i = end + 2
                continue
        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end == -1 else end
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


def _scan_hex(text: str) -> list[Finding]:
    """Hex-кодированный секрет: декодируем hex-токен и повторяем скан."""
    findings: list[Finding] = []
    for match in _HEX_RE.finditer(text):
        token = match.group(1)
        if len(token) % 2:
            token = token[:-1]
        try:
            decoded = bytes.fromhex(token).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if not _mostly_printable(decoded):
            continue
        for rule in RULES:
            if rule.name not in _SECRET_RULE_NAMES:
                continue
            for inner in rule.pattern.finditer(decoded):
                value = inner.group(0)
                if rule.validator and not rule.validator(value):
                    continue
                findings.append(
                    Finding(
                        type=f"hex_{rule.name}",
                        redaction="[REDACTED_HEX_SECRET]",
                        preview=_preview(value),
                        kind="hex",
                        start=match.start(1),
                        end=match.end(1),
                    )
                )
    return findings


def _scan_rot13(text: str) -> list[Finding]:
    """ROT13-кодированный секрет: rot13 сохраняет позиции, поэтому спан совпадает с исходным."""
    decoded = text.translate(_ROT13_TABLE)
    if decoded == text:
        return []
    findings: list[Finding] = []
    for rule in RULES:
        if rule.name not in _SECRET_RULE_NAMES:
            continue
        for match in rule.pattern.finditer(decoded):
            value = match.group(0)
            if rule.validator and not rule.validator(value):
                continue
            findings.append(
                Finding(
                    type=f"rot13_{rule.name}",
                    redaction="[REDACTED_ROT13_SECRET]",
                    preview=_preview(value),
                    kind="rot13",
                    start=match.start(),
                    end=match.end(),
                )
            )
    return findings


def scan_entropy(text: str) -> list[Finding]:
    """Токены без известного префикса, похожие на секрет (длина, набор символов, энтропия)."""
    findings: list[Finding] = []
    for match in _ENTROPY_RE.finditer(text):
        token = match.group(1)
        if not _looks_like_entropy_secret(token):
            continue
        findings.append(
            Finding(
                type="entropy_secret",
                redaction="[REDACTED_SECRET]",
                preview=_preview(token),
                kind="entropy",
                start=match.start(1),
                end=match.end(1),
            )
        )
    return findings


def _looks_like_entropy_secret(token: str) -> bool:
    """Длинный токен с буквами и цифрами, не hex-хэш, с высокой энтропией."""
    if not re.search(r"[A-Za-z]", token) or not re.search(r"\d", token):
        return False
    if re.fullmatch(r"[0-9a-fA-F]+", token):  # hex/хэш — не секрет
        return False
    classes = sum(bool(re.search(p, token)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[_-]"))
    return classes >= 3 and _shannon_entropy(token) >= 4.0


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _mostly_printable(value: str) -> bool:
    if not value:
        return False
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in value)
    return printable / len(value) >= 0.8


def _overlaps(finding: Finding, spans: list[tuple[int | None, int | None]]) -> bool:
    if finding.start is None or finding.end is None:
        return False
    for start, end in spans:
        if start is None or end is None:
            continue
        if finding.start < end and start < finding.end:
            return True
    return False


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
