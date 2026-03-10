import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_context_builder_keeps_exact_docs_and_deduplicates_by_canonical_id() -> None:
    assert importlib.util.find_spec("legal_rag.generation") is not None
    assert importlib.util.find_spec("legal_rag.generation.context_builder") is not None

    generation_module = importlib.import_module("legal_rag.generation.context_builder")
    types_module = importlib.import_module("legal_rag.types")

    build_answer_context = generation_module.build_answer_context
    RetrievedDoc = types_module.RetrievedDoc
    RetrievalResult = types_module.RetrievalResult
    RouteDecision = types_module.RouteDecision

    exact = RetrievalResult(
        docs=[RetrievedDoc("law:1", "条文一", {}, 1.0, {"exact": 1.0}, "exact_match")],
        confidence=1.0,
        latency_ms=0.0,
        reasons=["exact_article_match"],
        raw_signals={},
    )
    hybrid = RetrievalResult(
        docs=[
            RetrievedDoc("law:1", "条文一", {}, 0.9, {"hybrid": 0.9}, "hybrid"),
            RetrievedDoc("law:2", "条文二", {}, 0.8, {"hybrid": 0.8}, "hybrid"),
        ],
        confidence=0.9,
        latency_ms=0.0,
        reasons=["hybrid_ranked"],
        raw_signals={},
    )
    decision = RouteDecision("hybrid", False, 0.9, "hybrid_plus_exact", ["exact_article_match"])

    context = build_answer_context(
        "问题",
        decision,
        exact_result=exact,
        hybrid_result=hybrid,
        mini_result=None,
        max_articles=2,
    )

    assert [doc.canonical_id for doc in context.docs] == ["law:1", "law:2"]
