"""Тесты разбора и сборки файла памяти SmartAgent (agents/memory_state.py): чистые функции,
без файловой системы и LLM. Фиксируют совместимость со старыми файлами на диске."""

import json

from agents import memory_state as ms


def _parse(data) -> tuple:
    return ms.parse_state(data if isinstance(data, str) else json.dumps(data))


def test_corrupted_json_gives_empty_memory_and_flag():
    profiles, active, layers, corrupted = _parse("{не json")
    assert (profiles, active, corrupted) == ({}, None, True)
    assert layers == ms.default_enabled_layers()


def test_non_dict_root_is_empty_without_corruption_flag():
    assert _parse([1, 2]) == ({}, None, ms.default_enabled_layers(), False)


def test_flat_legacy_format_migrates_into_active_profile():
    profiles, active, _, _ = _parse(
        {"short_term": [{"role": "user", "content": "привет"}], "long_term": ["факт"]}
    )
    assert active == ms._MIGRATED_PROFILE_NAME
    profile = profiles[active]
    assert profile["short_term"] == [{"role": "user", "content": "привет"}]
    assert profile["long_term"] == [{"text": "факт", "source": "user", "created_at": ""}]


def test_empty_flat_format_has_no_profile():
    assert _parse({"short_term": [], "long_term": []})[:2] == ({}, None)


def test_active_profile_missing_from_profiles_is_dropped():
    profiles, active, _, _ = _parse({"profiles": {"А": {}}, "active_profile": "Б"})
    assert list(profiles) == ["А"]
    assert active is None


def test_non_dict_profile_entries_are_skipped():
    profiles, _, _, _ = _parse({"profiles": {"А": {}, "Б": "мусор"}})
    assert list(profiles) == ["А"]


def test_enabled_layers_unknown_and_non_bool_ignored_missing_default_on():
    _, _, layers, _ = _parse(
        {"profiles": {}, "enabled_layers": {"rag": False, "tools": "нет", "чужой": False}}
    )
    assert layers["rag"] is False
    assert layers["tools"] is True
    assert "чужой" not in layers
    assert set(layers) == set(ms.ALL_LAYERS)


def test_profile_without_invariants_gets_empty_list_and_full_meta():
    profile = ms.sanitize_profile({"meta": {"style": "кратко", "лишнее": "x"}})
    assert profile["invariants"] == []
    assert profile["meta"]["style"] == "кратко"
    assert set(profile["meta"]) == set(ms.PROFILE_FIELDS)
    assert profile["short_term"] == [] and profile["working"] is None


def test_sanitize_fact_variants():
    assert ms.sanitize_fact("  ") is None
    assert ms.sanitize_fact(5) is None
    assert ms.sanitize_fact({"text": " "}) is None
    assert ms.sanitize_fact({"text": "а", "source": "странный"})["source"] == ms.FACT_SOURCE_USER
    fact = ms.sanitize_fact({"text": "а", "source": "auto", "created_at": 7})
    assert fact == {"text": "а", "source": "auto", "created_at": ""}


def test_dump_parse_round_trip():
    profiles = {"А": ms.empty_profile()}
    profiles["А"]["long_term"] = [{"text": "т", "source": "auto", "created_at": "2026-01-01"}]
    layers = ms.default_enabled_layers()
    layers["rag"] = False
    got_profiles, active, got_layers, corrupted = ms.parse_state(
        ms.dump_state(profiles, "А", layers)
    )
    assert (got_profiles, active, got_layers, corrupted) == (profiles, "А", layers, False)
