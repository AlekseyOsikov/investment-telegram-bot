"""Тесты чистых частей research/rag_models.py: сборка бэкенда пары, проверка настроек оператора,
хранение отчётов на диске (tmp_path), настройки отчёта и текст прогресса.

Без сети, Telegram и модели. Не покрыто (нужны SmartAgent, модель, слот очереди или Telegram):
_new_agent, search_once, answer_and_judge, evaluate_question, _run_comparison, обработчики
команд.
"""

import json
from datetime import datetime, timezone

import pytest

from research import rag_compare as rc
from research import rag_models as rmod
from research import rag_models_eval as rme


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 1, 12, 30, 5, tzinfo=timezone.utc)


def _spec(provider: str = "ollama", model: str = "demo-model:7b") -> rme.ModelSpec:
    return rme.ModelSpec(provider, model)


def _set_operator_config(monkeypatch, compare: str, judge: str, keys: dict[str, bool]) -> None:
    monkeypatch.setattr(rmod, "RAG_MODELS_COMPARE", compare)
    monkeypatch.setattr(rmod, "RAG_MODELS_JUDGE", judge)
    monkeypatch.setattr(rmod, "_API_KEYS", keys)


def _models_report() -> dict:
    return rme.build_report(
        started_at="2026-10-01 12:00:00",
        finished_at="2026-10-01 12:05:00",
        settings={"top_k": 3},
        models=[_spec("ollama", "a"), _spec("deepseek", "b")],
        judge=_spec("kimi", "c"),
        runs=[],
        planned=0,
        repeats=1,
    )


# --- бэкенд пары ------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["ollama", "deepseek", "kimi"])
def test_build_backend_keeps_provider_and_model_and_disables_sdk_retries(provider):
    backend = rmod.build_backend(_spec(provider, "m-1"))
    assert backend.provider == provider
    assert backend.model == "m-1"
    assert backend.client.max_retries == 0


def test_build_backend_does_not_change_the_shared_client():
    before = rmod._CLIENTS["deepseek"].max_retries
    rmod.build_backend(_spec("deepseek", "m"))
    assert rmod._CLIENTS["deepseek"].max_retries == before


def test_build_backend_unknown_provider_is_a_key_error():
    with pytest.raises(KeyError):
        rmod.build_backend(_spec("неизвестный", "m"))


# --- настройки оператора ----------------------------------------------------------


def test_load_settings_parses_pairs_and_judge_with_defaults(monkeypatch):
    _set_operator_config(
        monkeypatch, "ollama:demo:7b,deepseek:", "kimi:", {"deepseek": True, "kimi": True}
    )
    specs, judge = rmod.load_settings()
    assert [s.provider for s in specs] == ["ollama", "deepseek"]
    assert specs[0].model == "demo:7b"  # разбор по первому двоеточию
    assert specs[1].model == rmod._DEFAULT_MODELS["deepseek"]
    assert judge == rme.ModelSpec("kimi", rmod._DEFAULT_MODELS["kimi"])


def test_load_settings_local_models_need_no_keys(monkeypatch):
    _set_operator_config(monkeypatch, "ollama:a,ollama:b", "ollama:j", {})
    specs, judge = rmod.load_settings()
    assert [s.key for s in specs] == ["ollama:a", "ollama:b"]
    assert judge.key == "ollama:j"


def test_load_settings_reports_missing_key_of_a_pair(monkeypatch):
    _set_operator_config(
        monkeypatch, "ollama:a,deepseek:", "ollama:j", {"deepseek": False, "kimi": True}
    )
    with pytest.raises(rme.ModelSettingError) as info:
        rmod.load_settings()
    assert "DEEPSEEK_API_KEY" in str(info.value)
    assert "KIMI_API_KEY" not in str(info.value)


def test_load_settings_reports_missing_key_of_the_judge_without_duplicates(monkeypatch):
    _set_operator_config(
        monkeypatch, "kimi:,kimi:other", "kimi:", {"deepseek": True, "kimi": False}
    )
    with pytest.raises(rme.ModelSettingError) as info:
        rmod.load_settings()
    assert str(info.value).count("KIMI_API_KEY") == 1


@pytest.mark.parametrize(
    ("compare", "judge"),
    [
        ("", "ollama:j"),  # сравнение не настроено
        ("ollama:a", "ollama:j"),  # одна модель
        ("ollama:a,ollama:a", "ollama:j"),  # дубликат
        ("ollama:a,ollama:b", ""),  # судья не настроен
        ("ollama:a,ollama:b", "ollama:j,ollama:k"),  # две пары у судьи
        ("ollama:,ollama:b", "ollama:j"),  # у ollama модель обязательна
    ],
)
def test_load_settings_rejects_invalid_operator_values(monkeypatch, compare, judge):
    _set_operator_config(monkeypatch, compare, judge, {"deepseek": True, "kimi": True})
    with pytest.raises(rme.ModelSettingError):
        rmod.load_settings()


# --- хранение отчётов -------------------------------------------------------------


def test_save_report_writes_models_file_named_by_utc_time(tmp_path, monkeypatch):
    monkeypatch.setattr(rmod, "datetime", _FixedDatetime)
    path = rmod.save_report(_models_report(), str(tmp_path / "reports"))
    assert path.endswith("20261001T123005Z_models.json")
    with open(path, encoding="utf-8") as file:
        assert json.load(file) == _models_report()
    assert [p.name for p in (tmp_path / "reports").iterdir()] == ["20261001T123005Z_models.json"]


def test_models_report_is_invisible_to_compare_report_reader(tmp_path):
    rmod.save_report(_models_report(), str(tmp_path))
    assert rc.load_latest_report(str(tmp_path)) is None


def test_load_latest_report_roundtrip(tmp_path):
    rmod.save_report(_models_report(), str(tmp_path))
    assert rmod.load_latest_report(str(tmp_path)) == _models_report()


def test_load_latest_report_without_directory_or_files_is_none(tmp_path):
    assert rmod.load_latest_report(str(tmp_path / "нет")) is None
    assert rmod.load_latest_report(str(tmp_path)) is None
    (tmp_path / "20260101T000000Z.json").write_text("{}", encoding="utf-8")
    assert rmod.load_latest_report(str(tmp_path)) is None


def test_load_latest_report_takes_the_newest(tmp_path):
    old, new = _models_report(), _models_report()
    old["stop_reason"], new["stop_reason"] = "старый", "новый"
    (tmp_path / "20260101T000000Z_models.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "20260301T000000Z_models.json").write_text(json.dumps(new), encoding="utf-8")
    assert rmod.load_latest_report(str(tmp_path))["stop_reason"] == "новый"


def test_load_latest_report_broken_file_is_report_error(tmp_path):
    (tmp_path / "20260101T000000Z_models.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(rc.ReportError):
        rmod.load_latest_report(str(tmp_path))


def test_load_latest_report_of_other_kind_or_version_is_report_error(tmp_path):
    report = _models_report()
    report["version"] = rme.REPORT_VERSION + 1
    (tmp_path / "20260101T000000Z_models.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(rc.ReportError):
        rmod.load_latest_report(str(tmp_path))


# --- настройки отчёта и прогресс ---------------------------------------------------


def test_settings_drop_main_flow_fields_and_add_comparison_fields():
    models = [_spec("ollama", "a"), _spec("deepseek", "b")]
    settings = rmod._settings(models, _spec("kimi", "c"), repeats=3)
    for key in ("level", "model", "rewrite_provider", "rewrite_model", "rewrite_search_mode"):
        assert key not in settings
    assert settings["rewrite"] == "off"
    assert settings["repeats"] == 3
    assert settings["models"] == ["ollama:a", "deepseek:b"]
    assert settings["judge"] == "kimi:c"
    assert "top_k" in settings and "strategy" in settings  # настройки поиска сохранены
    json.dumps(settings)


def test_settings_do_not_modify_the_shared_run_settings():
    rmod._settings([_spec()], _spec("kimi", "c"), repeats=1)
    assert "model" in rc.run_settings()


def test_progress_text_has_question_number_stage_done_count_and_text():
    text = rmod._progress_text(2, 5, 3, "повтор 1 из 2 · ollama/a · ответ", "Что такое X?")
    assert text == (
        "⏳ Сравнение моделей: вопрос 3 из 5 · повтор 1 из 2 · ollama/a · ответ\n"
        "Обработано вопросов: 2.\nЧто такое X?"
    )
