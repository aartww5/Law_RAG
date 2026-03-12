from legal_rag.retrievers.exact_match import extract_article_num
from legal_rag.types import NormalizedArticle, RetrievedDoc, RetrievalResult
from legal_rag.utils.text import keyword_tokens, split_structured_query


LAW_MATCH_BONUS = 0.2
ARTICLE_MATCH_BONUS = 1.0
ARTICLE_CN_MATCH_BONUS = 0.2


class HybridRetriever:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self._fake_docs = docs or []

    @classmethod
    def fake_for_test(cls, docs: list[dict]) -> "HybridRetriever":
        return cls(docs=docs)

    @classmethod
    def from_articles(cls, articles: list[NormalizedArticle]) -> "HybridRetriever":
        docs = []
        for article in articles:
            docs.append(
                {
                    "canonical_id": article.canonical_id,
                    "content": article.content,
                    "metadata": {
                        "law_name": article.law_name,
                        "law_aliases": article.law_aliases,
                        "article_id_cn": article.article_id_cn,
                        "article_id_num": article.article_id_num,
                    },
                }
            )
        return cls(docs=docs)

    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        ranked_docs = []
        for item in self._fake_docs:
            score = item.get("score")
            if score is None:
                score = self._score_question_against_doc(question, item["content"], item.get("metadata", {}))
            ranked_docs.append({**item, "score": score})

        ranked = sorted(ranked_docs, key=lambda item: item["score"], reverse=True)
        docs = [
            RetrievedDoc(
                canonical_id=item["canonical_id"],
                content=item["content"],
                metadata=item.get("metadata", {}),
                score=item["score"],
                score_breakdown={"hybrid": item["score"]},
                retriever="hybrid",
            )
            for item in ranked
        ]
        return RetrievalResult(
            docs=docs,
            confidence=docs[0].score if docs else 0.0,
            latency_ms=0.0,
            reasons=["hybrid_ranked"] if docs else [],
            raw_signals={
                "candidate_count": len(docs),
                "top1_score": docs[0].score if docs else 0.0,
                "top2_score": docs[1].score if len(docs) > 1 else 0.0,
            },
        )

    def _score_question_against_doc(self, question: str, content: str, metadata: dict) -> float:
        background_text, current_text = split_structured_query(question)
        query_text = current_text or question
        query_tokens = set(keyword_tokens(query_text))
        doc_tokens = set(keyword_tokens(content))
        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = len(query_tokens & doc_tokens) / len(query_tokens)
        if background_text:
            background_tokens = set(keyword_tokens(background_text))
            if background_tokens:
                background_overlap = len(background_tokens & doc_tokens) / len(background_tokens)
                overlap = overlap * 0.8 + background_overlap * 0.2
        law_bonus = 0.0
        article_bonus = 0.0

        law_aliases = metadata.get("law_aliases", [])
        law_name = metadata.get("law_name")
        if law_name and law_name in question:
            law_bonus = LAW_MATCH_BONUS
        elif any(alias and alias in question for alias in law_aliases):
            law_bonus = LAW_MATCH_BONUS

        article_num = extract_article_num(question)
        if article_num and article_num == metadata.get("article_id_num"):
            article_bonus += ARTICLE_MATCH_BONUS

        article_id_cn = metadata.get("article_id_cn")
        if article_id_cn and article_id_cn in question:
            article_bonus += ARTICLE_CN_MATCH_BONUS

        return overlap + law_bonus + article_bonus
