"""Тесты поиска по справке поддержки (support_assistant/search.py): порядок шагов и исходы.
Эмбеддинг, поиск и проверка индекса подставляются функциями — сети, Ollama и индекса нет."""

from dataclasses import dataclass

from support_assistant import search
from support_assistant.search import SupportSearchConfig


@dataclass
class Chunk:
    title: str
    score: float
    text: str = "текст"


def config(**kw):
    values = dict(index_dir="idx", top_k=3, min_score=0.0, search_timeout=2.0)
    values.update(kw)
    return SupportSearchConfig(**values)


class Fakes:
    def __init__(self, results=None, exists=True, embed_error=None):
        self.results = results if results is not None else []
        self.exists = exists
        self.embed_error = embed_error
        self.embedded: list[list[str]] = []
        self.budgets: list[float] = []
        self.searched: list[tuple] = []

    def embed(self, texts, budget_seconds=None):
        if self.embed_error:
            raise self.embed_error
        self.embedded.append(list(texts))
        self.budgets.append(budget_seconds)
        return [[1.0, 0.0]]

    def search(self, strategy, index_dir, vector, limit):
        self.searched.append((strategy, index_dir, limit))
        return self.results

    def index_exists(self, strategy, index_dir):
        return self.exists

    def run(self, cfg=None, question="почему нет сводок?"):
        return search.retrieve(
            cfg or config(),
            question,
            embed=self.embed,
            search=self.search,
            index_exists=self.index_exists,
        )


def test_found_uses_question_budget_and_support_index():
    fakes = Fakes([Chunk("a", 0.9), Chunk("b", 0.5)])
    result = fakes.run()
    assert result.outcome == search.FOUND
    assert [c.title for c in result.chunks] == ["a", "b"]
    assert fakes.embedded == [["почему нет сводок?"]]
    assert fakes.budgets == [2.0]
    assert fakes.searched == [("markdown", "idx", 3)]


def test_top_k_limits_results():
    fakes = Fakes([Chunk(str(i), 0.9 - i / 10) for i in range(6)])
    assert len(fakes.run(config(top_k=2)).chunks) == 2


def test_min_score_filters_and_zero_disables():
    results = [Chunk("a", 0.8), Chunk("b", 0.3), Chunk("c", -0.1)]
    assert [c.title for c in Fakes(results).run(config(min_score=0.5)).chunks] == ["a"]
    assert len(Fakes(results).run(config(min_score=0.0)).chunks) == 3


def test_nothing_above_threshold_is_not_found():
    result = Fakes([Chunk("a", 0.2)]).run(config(min_score=0.5))
    assert result.outcome == search.NOT_FOUND and result.chunks == ()


def test_empty_index_is_not_found():
    assert Fakes([]).run().outcome == search.NOT_FOUND


def test_missing_index_skips_embedding():
    fakes = Fakes(exists=False)
    assert fakes.run().outcome == search.NO_INDEX
    assert fakes.embedded == [] and fakes.searched == []


def test_embedding_failure_is_failed_with_reason():
    result = Fakes(embed_error=TimeoutError("бюджет")).run()
    assert result.outcome == search.FAILED
    assert "TimeoutError" in result.failure
