"""Очистка недоверенного текста из каналов данных перед подачей модели.

Чистые функции без сети и без импорта `config`: удаляют скрытый и встроенный контент
(HTML-комментарии, `<script>`/`<style>`, элементы со скрытым стилем, payload markdown-ссылок,
zero-width-символы). Перенесено из исследовательской оснастки `scripts/indirect_injection.py`
(отчёт `reports/indirect-prompt-injection/`): промптового «данные ≠ инструкции» недостаточно против
content-poisoning — скрытый текст нужно вычищать кодом до модели.

Тесты — `tests/test_injection_guard.py`.
"""

from __future__ import annotations

import re

# Признаки «скрытого» стиля: белый текст, display:none, микро-шрифт, off-screen.
_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|color\s*:\s*(#fff(?:fff)?|white|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\))"
    r"|font-size\s*:\s*(0|1px)"
    r"|opacity\s*:\s*0"
    r"|position\s*:\s*absolute",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.DOTALL)


def sanitize_text(text: str) -> str:
    """Вычистить HTML-комментарии, скрытые элементы, zero-width и payload ссылок.

    Видимый обычный текст сохраняется. Возвращает строку без лишних пробелов по краям.
    """
    text = _SCRIPT_STYLE_RE.sub("", text)          # <script>/<style> целиком
    text = _COMMENT_RE.sub("", text)               # HTML-комментарии
    text = _strip_hidden_elements(text)            # белый текст / display:none и т.п.
    text = re.sub(r"<[^>]+>", "", text)            # все оставшиеся теги -> только текст
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)      # ссылки -> только якорный текст
    text = _ZERO_WIDTH_RE.sub("", text)            # zero-width символы
    return re.sub(r"[ \t]+", " ", text).strip()


def _strip_hidden_elements(html: str) -> str:
    """Удаляет элементы со скрытым стилем вместе с их содержимым."""
    out: list[str] = []
    i = 0
    n = len(html)
    while i < n:
        match = _TAG_RE.search(html, i)
        if not match:
            out.append(html[i:])
            break
        tag, attrs = match.group(1), match.group(2)
        if _HIDDEN_STYLE_RE.search(attrs):
            close_start = html.find("</" + tag, match.end())
            if close_start != -1:
                close_end = html.find(">", close_start)
                if close_end != -1:
                    out.append(html[i:match.start()])  # текст ДО скрытого элемента сохраняем
                    i = close_end + 1                  # скрытый элемент выброшен целиком
                    continue
        out.append(html[i:match.end()])
        i = match.end()
    return "".join(out)
