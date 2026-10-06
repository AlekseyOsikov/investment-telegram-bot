"""Тесты конвейера поиска материалов (agents/rag_search.py): порядок шагов, режимы
переписывания и сбои. Эмбеддинг, поиск и проверка индекса подставляются функциями —
сети, Ollama и индекса нет; клиент модели переписывания не используется (готовое
`rewrite` передаётся параметром)."""

from dataclasses import dataclass

from agents import rag_context, rag_search


@dataclass
class Chunk:
    chunk_id: str
    score: float
    text: str = "достаточно длинный текст фрагмента для отбора"
    title: str = "Док"
    author: str | None = None
    chunk_index: int = 0


def make_config(**overrides) -> rag_search.RagSearchConfig:
    values = dict(
        strategy="structural",
        index_dir="idx",
        top_k=3,
        min_score=0.5,
        search_timeout=1.0,
        candidates=10,
        relative_margin=1.0,
        min_chunk_chars=0,
        max_per_doc=100,
        rewrite_backend=None,
        rewrite_timeout=1.0,
        rewrite_search_mode="replace",
        rewrite_history_questions=0,
    )
    values.update(overrides)
    return rag_search.RagSearchConfig(**values)


class Recorder:
    """Подставные эмбеддинг и поиск: запоминают, по каким текстам искали."""

    def __init__(self, results_by_text=None):
        self.embedded: list[list[str]] = []
        self.results_by_text = results_by_text or {}

    def embed(self, texts, budget_seconds):
        self.embedded.append(list(texts))
        return list(texts)  # «вектор» = сам текст

    def search(self, strategy, index_dir, vector, limit):
        return self.results_by_text.get(vector, [Chunk("a", 0.9)])


def run(config=None, text="вопрос", rewrite=None, history=None, recorder=None, **kw):
    recorder = recorder or Recorder()
    result = rag_search.retrieve(
        config or make_config(),
        text,
        rewrite,
        history,
        embed=recorder.embed,
        search=recorder.search,
        index_exists=kw.pop("index_exists", lambda strategy, directory: True),
        **kw,
    )
    return result, recorder


def test_search_texts_modes():
    assert rag_search.search_texts("в", None, "both") == (["в"], False)
    assert rag_search.search_texts("в", "п", "replace") == (["п"], False)
    assert rag_search.search_texts("в", "п", "both") == (["в", "п"], True)


def test_rewrite_status():
    assert rag_search.rewrite_status("п", None) == rag_context.REWRITE_OK
    assert rag_search.rewrite_status(None, "сбой") == rag_context.REWRITE_FAILED
    assert rag_search.rewrite_status(None, None) == rag_context.REWRITE_OFF


def test_missing_index_is_silent_empty_result():
    result, recorder = run(index_exists=lambda strategy, directory: False)
    assert result.materials == rag_context.Materials()
    assert result.search_info is None and result.failure is None
    assert recorder.embedded == []


def test_plain_search_uses_original_question_only():
    result, recorder = run()
    assert recorder.embedded == [["вопрос"]]
    assert result.materials.searched is True
    assert [c.chunk_id for c in result.materials.chunks] == ["a"]
    info = result.search_info
    assert info.query is None and info.rewrite_status == rag_context.REWRITE_OFF
    assert (info.candidates, info.selected, info.history_used) == (1, 1, 0)


def test_ready_rewrite_replaces_search_text_and_counts_history():
    result, recorder = run(rewrite=("запрос", None), history=["раньше"])
    assert recorder.embedded == [["запрос"]]
    assert result.search_info.query == "запрос"
    assert result.search_info.rewrite_status == rag_context.REWRITE_OK
    assert result.search_info.history_used == 1


def test_both_mode_searches_both_texts_and_merges_candidates():
    recorder = Recorder(
        {
            "вопрос": [Chunk("a", 0.6)],
            "запрос": [Chunk("a", 0.9), Chunk("b", 0.7, text="другой текст фрагмента")],
        }
    )
    result, _ = run(
        make_config(rewrite_search_mode="both"), rewrite=("запрос", None), recorder=recorder
    )
    assert recorder.embedded == [["вопрос", "запрос"]]
    assert result.search_info.both is True
    assert [(c.chunk_id, c.score) for c in result.materials.chunks] == [("a", 0.9), ("b", 0.7)]


def test_failed_rewrite_searches_original_and_ignores_history():
    result, recorder = run(rewrite=(None, "таймаут"), history=["раньше"])
    assert recorder.embedded == [["вопрос"]]
    info = result.search_info
    assert info.rewrite_status == rag_context.REWRITE_FAILED
    assert info.rewrite_reason == "таймаут"
    assert info.history_used == 0


def test_default_history_used_only_when_history_not_given():
    calls = []

    def provider():
        calls.append(1)
        return ["прошлый"]

    # Без настроенной модели переписывания вызов не удаётся, но история запрошена один раз.
    run(default_history=provider)
    assert calls == [1]
    run(history=[], default_history=provider)
    assert calls == [1]


def test_embedding_failure_returns_warning_and_reason_without_raising():
    def boom(texts, budget_seconds):
        raise TimeoutError("нет ответа")

    result = rag_search.retrieve(
        make_config(), "в", embed=boom, index_exists=lambda s, d: True
    )
    assert result.materials.warning == rag_context.FAILURE_WARNING
    assert result.materials.searched is False and result.materials.chunks == []
    assert result.failure == "TimeoutError: нет ответа"
    assert result.search_info is None


def test_selection_threshold_applies_and_counts_dropped():
    recorder = Recorder({"вопрос": [Chunk("a", 0.9), Chunk("b", 0.1)]})
    result, _ = run(recorder=recorder)
    assert [c.chunk_id for c in result.materials.chunks] == ["a"]
    assert result.search_info.candidates == 2
    assert result.search_info.dropped[rag_context.DROP_THRESHOLD] == 1


def test_rewrite_query_without_backend_is_not_configured():
    assert rag_search.rewrite_query(make_config(), "в") == (None, None)


def test_call_with_deadline_returns_value_and_propagates_errors():
    assert rag_search.call_with_deadline(lambda: "ок", 1.0) == "ок"

    def boom():
        raise ValueError("x")

    try:
        rag_search.call_with_deadline(boom, 1.0)
    except ValueError as exc:
        assert str(exc) == "x"
    else:
        raise AssertionError("исключение должно пробрасываться")
