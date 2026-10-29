"""Метрики eval без torch: парсинг ответа, схема, exact, поля."""

from __future__ import annotations

import json
import re

from spec import validate_output


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def set_f1(expected: list[str], predicted: list[str]) -> float:
    expected_set, predicted_set = set(expected), set(predicted)
    if not expected_set and not predicted_set:
        return 1.0
    intersection = len(expected_set & predicted_set)
    if intersection == 0:
        return 0.0
    precision = intersection / len(predicted_set)
    recall = intersection / len(expected_set)
    return 2 * precision * recall / (precision + recall)


def score_row(expected: dict, raw: str) -> dict:
    try:
        strict_ok = isinstance(json.loads(raw.strip()), dict)
    except json.JSONDecodeError:
        strict_ok = False
    extracted = extract_json(raw)
    schema_ok = extracted is not None and not validate_output(extracted)
    exact = extracted == expected
    result = {
        "json_strict": strict_ok,
        "json_extractable": extracted is not None,
        "schema_ok": schema_ok,
        "exact": exact,
        "problems": validate_output(extracted) if extracted is not None else ["JSON не найден"],
    }
    if expected.get("ok"):
        # Если модель вернула ошибку вместо успеха, поля считаются нулевыми:
        # иначе poll_acc выглядел бы как точность только среди успешных ответов.
        if extracted is not None and extracted.get("ok") is True:
            result["ticker_f1"] = set_f1(expected.get("secids", []), extracted.get("secids", []))
            result["poll_acc"] = expected.get("poll_interval") == extracted.get("poll_interval")
            result["report_acc"] = expected.get("report_interval") == extracted.get("report_interval")
        else:
            result["ticker_f1"] = 0.0
            result["poll_acc"] = False
            result["report_acc"] = False
    if not expected.get("ok"):
        result["error_acc"] = extracted is not None and extracted.get("error") == expected.get("error")
    return result
