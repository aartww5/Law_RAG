from __future__ import annotations

import logging

from legal_rag.retrievers.backends import Bm25Backend, QdrantVectorBackend, weighted_rrf
from legal_rag.retrievers.exact_match import extract_article_num
from legal_rag.types import NormalizedArticle, RetrievedDoc, RetrievalResult
from legal_rag.utils.text import keyword_tokens, split_structured_query


LOGGER = logging.getLogger(__name__)
LAW_MATCH_BONUS = 0.2
ARTICLE_MATCH_BONUS = 1.0
ARTICLE_CN_MATCH_BONUS = 0.2
CURRENT_QUERY_WEIGHT = 1.0
BACKGROUND_QUERY_WEIGHT = 0.2
BM25_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0


class HybridRetriever:
    def __init__(
        self,
        docs: list[dict] | None = None,
        *,
        bm25_backend=None,
        vector_backend=None,
    ) -> None:
        self._docs = docs or []
        self._doc_by_id = {item["canonical_id"]: item for item in self._docs}
        self._bm25_backend = bm25_backend
        self._vector_backend = vector_backend

    @classmethod
    def fake_for_test(cls, docs: list[dict]) -> "HybridRetriever":
        return cls(docs=docs)

    @classmethod
    def from_articles(
        cls,
        articles: list[NormalizedArticle],
        *,
        index_config=None,
        bm25_backend=None,
        vector_backend=None,
        enable_backends: bool = True,
    ) -> "HybridRetriever":
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

        if bm25_backend is None and enable_backends:
            try:
                bm25_backend = Bm25Backend.from_articles(articles)
            except Exception as exc:  # pragma: no cover - depends on local optional deps
                LOGGER.warning("bm25 backend unavailable: %s", exc)

        if vector_backend is None and enable_backends and index_config is not None:
            try:
                vector_backend = QdrantVectorBackend.from_articles(
                    articles,
                    storage_path=index_config.qdrant_path,
                    collection_name=index_config.qdrant_collection_name,
                    model_name=index_config.embedding_model,
                )
            except Exception as exc:  # pragma: no cover - depends on local optional deps
                LOGGER.warning("vector backend unavailable: %s", exc)

        return cls(
            docs=docs,
            bm25_backend=bm25_backend,
            vector_backend=vector_backend,
        )

    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        if self._docs and all(item.get("score") is not None for item in self._docs):
            return self._build_result_from_ranked_items(
                ranked_items=sorted(self._docs, key=lambda item: item["score"], reverse=True),
                reasons=["hybrid_ranked"],
                score_breakdown_key="hybrid",
            )

        if self._bm25_backend is not None or self._vector_backend is not None:
            backend_result = self._retrieve_with_backends(question)
            if backend_result.docs:
                return backend_result

        ranked_docs = []
        for item in self._docs:
            rule_score = self._score_question_against_doc(question, item["content"], item.get("metadata", {}))
            ranked_docs.append({**item, "score": rule_score})

        ranked = sorted(ranked_docs, key=lambda item: item["score"], reverse=True)
        return self._build_result_from_ranked_items(
            ranked_items=ranked,
            reasons=["rule_score_fallback"],
            score_breakdown_key="hybrid",
        )

    def _retrieve_with_backends(self, question: str) -> RetrievalResult:
        current_query, background_query = self._split_queries(question)
        rank_lists: list[tuple[list[tuple[str, float]], float]] = []
        reasons: list[str] = ["hybrid_rrf"]

        for query_text, query_weight in (
            (current_query, CURRENT_QUERY_WEIGHT),
            (background_query, BACKGROUND_QUERY_WEIGHT),
        ):
            if not query_text:
                continue
            if self._bm25_backend is not None:
                bm25_ranked = self._bm25_backend.retrieve(query_text)
                if bm25_ranked:
                    rank_lists.append((bm25_ranked, query_weight * BM25_WEIGHT))
                    if "bm25" not in reasons:
                        reasons.append("bm25")
            if self._vector_backend is not None:
                vector_ranked = self._vector_backend.retrieve(query_text)
                if vector_ranked:
                    rank_lists.append((vector_ranked, query_weight * VECTOR_WEIGHT))
                    if "vector" not in reasons:
                        reasons.append("vector")

        fused_scores = weighted_rrf(rank_lists)
        if not fused_scores:
            return RetrievalResult(
                docs=[],
                confidence=0.0,
                latency_ms=0.0,
                reasons=[],
                raw_signals={"candidate_count": 0, "top1_score": 0.0, "top2_score": 0.0},
            )

        ranked_items = []
        for canonical_id, rrf_score in fused_scores.items():
            item = self._doc_by_id.get(canonical_id)
            if item is None:
                continue
            law_bonus, article_bonus = self._lexical_bonus(question, item.get("metadata", {}))
            final_score = rrf_score + law_bonus + article_bonus
            ranked_items.append(
                {
                    **item,
                    "score": final_score,
                    "score_breakdown": {
                        "rrf": rrf_score,
                        "law_bonus": law_bonus,
                        "article_bonus": article_bonus,
                    },
                }
            )

        ranked = sorted(ranked_items, key=lambda item: item["score"], reverse=True)
        docs = [
            RetrievedDoc(
                canonical_id=item["canonical_id"],
                content=item["content"],
                metadata=item.get("metadata", {}),
                score=item["score"],
                score_breakdown=item.get("score_breakdown", {}),
                retriever="hybrid",
            )
            for item in ranked
        ]
        return RetrievalResult(
            docs=docs,
            confidence=docs[0].score if docs else 0.0,
            latency_ms=0.0,
            reasons=reasons,
            raw_signals={
                "candidate_count": len(docs),
                "top1_score": docs[0].score if docs else 0.0,
                "top2_score": docs[1].score if len(docs) > 1 else 0.0,
            },
        )

    def _build_result_from_ranked_items(
        self,
        *,
        ranked_items: list[dict],
        reasons: list[str],
        score_breakdown_key: str,
    ) -> RetrievalResult:
        docs = [
            RetrievedDoc(
                canonical_id=item["canonical_id"],
                content=item["content"],
                metadata=item.get("metadata", {}),
                score=item["score"],
                score_breakdown=item.get("score_breakdown", {score_breakdown_key: item["score"]}),
                retriever="hybrid",
            )
            for item in ranked_items
        ]
        return RetrievalResult(
            docs=docs,
            confidence=docs[0].score if docs else 0.0,
            latency_ms=0.0,
            reasons=reasons,
            raw_signals={
                "candidate_count": len(docs),
                "top1_score": docs[0].score if docs else 0.0,
                "top2_score": docs[1].score if len(docs) > 1 else 0.0,
            },
        )

    def _split_queries(self, question: str) -> tuple[str, str]:
        background_text, current_text = split_structured_query(question)
        query_text = current_text or question
        return query_text, background_text

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
        law_bonus, article_bonus = self._lexical_bonus(question, metadata)
        return overlap + law_bonus + article_bonus

    def _lexical_bonus(self, question: str, metadata: dict) -> tuple[float, float]:
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

        return law_bonus, article_bonus
