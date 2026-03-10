from legal_rag.types import RetrievalResult, RouteDecision


class AutoRouter:
    def __init__(self, min_confidence: float = 0.7, min_margin: float = 0.08) -> None:
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def decide(self, exact_result: RetrievalResult, hybrid_result: RetrievalResult) -> RouteDecision:
        if exact_result.docs:
            return RouteDecision(
                selected_mode="hybrid",
                fallback_triggered=False,
                confidence=exact_result.confidence,
                merge_policy="hybrid_plus_exact",
                reasons=exact_result.reasons or ["exact_match"],
            )

        top1 = float(hybrid_result.raw_signals.get("top1_score", hybrid_result.confidence))
        top2 = float(hybrid_result.raw_signals.get("top2_score", 0.0))
        if top1 >= self.min_confidence and (top1 - top2) >= self.min_margin:
            return RouteDecision(
                selected_mode="hybrid",
                fallback_triggered=False,
                confidence=hybrid_result.confidence,
                merge_policy="hybrid_plus_exact",
                reasons=hybrid_result.reasons or ["hybrid_confident"],
            )

        return RouteDecision(
            selected_mode="mini",
            fallback_triggered=True,
            confidence=hybrid_result.confidence,
            merge_policy="mini_only",
            reasons=["low_hybrid_confidence"],
        )
