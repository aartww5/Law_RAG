from legal_rag.types import AnswerContext, RetrievedDoc, RetrievalResult, RouteDecision


def dedupe_by_canonical_id(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    deduped: list[RetrievedDoc] = []
    seen: set[str] = set()
    for doc in docs:
        if doc.canonical_id in seen:
            continue
        seen.add(doc.canonical_id)
        deduped.append(doc)
    return deduped


def build_answer_context(
    question: str,
    decision: RouteDecision,
    *,
    exact_result: RetrievalResult,
    hybrid_result: RetrievalResult,
    mini_result: RetrievalResult | None,
    max_articles: int,
) -> AnswerContext:
    ordered_docs: list[RetrievedDoc] = []

    if decision.merge_policy == "hybrid_plus_exact":
        ordered_docs.extend(exact_result.docs)
        ordered_docs.extend(hybrid_result.docs)
    elif mini_result is not None:
        ordered_docs.extend(mini_result.docs)
        ordered_docs.extend(exact_result.docs)

    docs = dedupe_by_canonical_id(ordered_docs)[:max_articles]
    citations = [doc.canonical_id for doc in docs]
    return AnswerContext(
        question=question,
        docs=docs,
        route_decision=decision,
        citations=citations,
        source_summary={"doc_count": len(docs)},
    )
