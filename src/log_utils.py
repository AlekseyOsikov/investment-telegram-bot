"""Безопасное представление текста для логов (основной поток бота).

Логирование запроса/ответа (`MAIN_LOG_MESSAGES`) пишет пользовательский текст, поэтому перед
записью он усекается и очищается от очевидных PII (email, телефон, длинные номера — карты/счета)
и управляющих символов, чтобы не раздувать лог и не хранить чувствительные данные как есть.

Чистые функции, без сети и импорта `config`. Тесты — `tests/test_log_utils.py`.
"""

from __future__ import annotations

import re

DEFAULT_LOG_LIMIT = 500

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE_RE = re.compile(
    r"(?<![\w\d])(?:\+\d{1,3}[\s-]?|8[\s-]?)"
    r"(?:\(\d{2,4}\)[\s-]?|\d{2,4}[\s-])"
    r"\d{3}[\s-]?\d{2}[\s-]?\d{2}(?![\w\d])"
)


def preview_for_log(text: str, limit: int = DEFAULT_LOG_LIMIT) -> str:
    """Усечь и замаскировать PII для записи в лог.

    Маскируются email, номера карт/счетов (13–19 цифр) и телефоны; управляющие символы и переводы
    строк заменяются пробелами (одна строка лога). Длинный текст режется до ``limit`` с «…».
    """
    clean = _CONTROL.sub(" ", str(text or ""))
    clean = _CARD_RE.sub("[number]", clean)
    clean = _PHONE_RE.sub("[phone]", clean)
    clean = _EMAIL_RE.sub("[email]", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "…"
