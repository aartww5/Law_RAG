from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_rag.retrievers.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_aggregates_weighted_query_scores() -> None:
    reranker = CrossEncoderReranker(model_name="demo-model")

    class FakeModel:
        def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
            assert pairs == [
                ("query one", "Law: Civil Code\nArticle: Article 1\nContent: doc one"),
                ("query one", "Law: Civil Code\nArticle: Article 2\nContent: doc two"),
                ("query two", "Law: Civil Code\nArticle: Article 1\nContent: doc one"),
                ("query two", "Law: Civil Code\nArticle: Article 2\nContent: doc two"),
            ]
            return [0.2, 0.8, 0.9, 0.1]

    reranker._model = FakeModel()
    docs = [
        {
            "canonical_id": "law:1",
            "content": "doc one",
            "metadata": {"law_name": "Civil Code", "article_id_cn": "Article 1"},
        },
        {
            "canonical_id": "law:2",
            "content": "doc two",
            "metadata": {"law_name": "Civil Code", "article_id_cn": "Article 2"},
        },
    ]

    scores = reranker.score([("query one", 1.0), ("query two", 0.5)], docs)

    assert scores["law:1"] == (0.2 + 0.9 * 0.5) / 1.5
    assert scores["law:2"] == (0.8 + 0.1 * 0.5) / 1.5


def test_cross_encoder_reranker_deduplicates_queries_before_scoring() -> None:
    reranker = CrossEncoderReranker(model_name="demo-model")

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
            self.calls += 1
            assert pairs == [("same query", "Article: Article 1\nContent: doc one")]
            return [0.4]

    fake_model = FakeModel()
    reranker._model = fake_model
    docs = [
        {
            "canonical_id": "law:1",
            "content": "doc one",
            "metadata": {"article_id_cn": "Article 1"},
        }
    ]

    scores = reranker.score([("same query", 1.0), ("same query", 0.5)], docs)

    assert fake_model.calls == 1
    assert scores["law:1"] == pytest.approx(0.4)
