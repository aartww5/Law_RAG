import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_router_falls_back_to_mini_when_hybrid_margin_is_weak() -> None:
    assert importlib.util.find_spec("legal_rag.router") is not None
    assert importlib.util.find_spec("legal_rag.router.auto") is not None

    router_module = importlib.import_module("legal_rag.router.auto")
    types_module = importlib.import_module("legal_rag.types")

    AutoRouter = router_module.AutoRouter
    RetrievalResult = types_module.RetrievalResult

    router = AutoRouter(min_confidence=0.7, min_margin=0.1)
    exact = RetrievalResult(docs=[], confidence=0.0, latency_ms=0.0, reasons=[], raw_signals={})
    hybrid = RetrievalResult(
        docs=[],
        confidence=0.52,
        latency_ms=0.0,
        reasons=["hybrid_ranked"],
        raw_signals={"top1_score": 0.52, "top2_score": 0.49},
    )

    decision = router.decide(exact_result=exact, hybrid_result=hybrid)

    assert decision.selected_mode == "mini"
    assert decision.fallback_triggered is True
    assert decision.merge_policy == "mini_only"


def test_router_prefers_hybrid_when_top1_is_clear_enough() -> None:
    router_module = importlib.import_module("legal_rag.router.auto")
    types_module = importlib.import_module("legal_rag.types")

    AutoRouter = router_module.AutoRouter
    RetrievalResult = types_module.RetrievalResult

    router = AutoRouter()
    exact = RetrievalResult(docs=[], confidence=0.0, latency_ms=0.0, reasons=[], raw_signals={})
    hybrid = RetrievalResult(
        docs=[],
        confidence=0.7272727272727273,
        latency_ms=0.0,
        reasons=["hybrid_ranked"],
        raw_signals={"top1_score": 0.7272727272727273, "top2_score": 0.6363636363636364},
    )

    decision = router.decide(exact_result=exact, hybrid_result=hybrid)

    assert decision.selected_mode == "hybrid"
    assert decision.fallback_triggered is False


def test_router_prefers_hybrid_when_rrf_score_is_low_but_usable() -> None:
    router_module = importlib.import_module("legal_rag.router.auto")
    types_module = importlib.import_module("legal_rag.types")

    AutoRouter = router_module.AutoRouter
    RetrievalResult = types_module.RetrievalResult

    router = AutoRouter()
    exact = RetrievalResult(docs=[], confidence=0.0, latency_ms=0.0, reasons=[], raw_signals={})
    hybrid = RetrievalResult(
        docs=[],
        confidence=0.02,
        latency_ms=0.0,
        reasons=["hybrid_rrf"],
        raw_signals={"top1_score": 0.02, "top2_score": 0.018},
    )

    decision = router.decide(exact_result=exact, hybrid_result=hybrid)

    assert decision.selected_mode == "hybrid"
    assert decision.fallback_triggered is False


def test_router_falls_back_to_mini_when_rrf_score_is_clearly_weak() -> None:
    router_module = importlib.import_module("legal_rag.router.auto")
    types_module = importlib.import_module("legal_rag.types")

    AutoRouter = router_module.AutoRouter
    RetrievalResult = types_module.RetrievalResult

    router = AutoRouter()
    exact = RetrievalResult(docs=[], confidence=0.0, latency_ms=0.0, reasons=[], raw_signals={})
    hybrid = RetrievalResult(
        docs=[],
        confidence=0.014,
        latency_ms=0.0,
        reasons=["hybrid_rrf"],
        raw_signals={"top1_score": 0.014, "top2_score": 0.013},
    )

    decision = router.decide(exact_result=exact, hybrid_result=hybrid)

    assert decision.selected_mode == "mini"
    assert decision.fallback_triggered is True
