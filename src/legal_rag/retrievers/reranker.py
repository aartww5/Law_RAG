from __future__ import annotations

import importlib.util
import math
from typing import Protocol

from legal_rag.utils.devices import resolve_torch_device


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 8


class ModelReranker(Protocol):
    def score(self, queries: str | list[tuple[str, float]], docs: list[dict]) -> dict[str, float]:
        ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model = None

    def score(self, queries: str | list[tuple[str, float]], docs: list[dict]) -> dict[str, float]:
        weighted_queries = self._normalize_queries(queries)
        if not weighted_queries or not docs:
            return {}

        model = self._load_model()
        pairs: list[tuple[str, str]] = []
        pair_refs: list[tuple[str, float]] = []
        for query_text, weight in weighted_queries:
            doc_pairs = [(query_text, self._build_doc_text(doc)) for doc in docs]
            pairs.extend(doc_pairs)
            pair_refs.extend((doc["canonical_id"], weight) for doc in docs)
        raw_scores = model.predict(
            pairs,
            batch_size=min(self.batch_size, len(pairs)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        normalized_scores = self._normalize_scores(raw_scores)
        total_weight = sum(weight for _query, weight in weighted_queries)
        aggregated_scores: dict[str, float] = {}
        for (canonical_id, weight), score in zip(pair_refs, normalized_scores, strict=False):
            aggregated_scores[canonical_id] = aggregated_scores.get(canonical_id, 0.0) + (score * weight)
        if total_weight <= 0:
            return aggregated_scores
        return {
            canonical_id: score / total_weight
            for canonical_id, score in aggregated_scores.items()
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("sentence_transformers is not installed")

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        return self._model

    @staticmethod
    def _build_doc_text(doc: dict) -> str:
        metadata = doc.get("metadata", {})
        law_name = metadata.get("law_name", "")
        aliases = " ".join(alias for alias in metadata.get("law_aliases", []) if alias)
        article_id = metadata.get("article_id_cn") or metadata.get("article_id_num") or doc.get("canonical_id", "")
        content = doc.get("content", "")
        parts = [
            f"Law: {law_name}" if law_name else "",
            f"Aliases: {aliases}" if aliases else "",
            f"Article: {article_id}" if article_id else "",
            f"Content: {content}" if content else "",
        ]
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _normalize_scores(raw_scores) -> list[float]:
        scores = [float(score) for score in raw_scores]
        if all(0.0 <= score <= 1.0 for score in scores):
            return scores
        return [1.0 / (1.0 + math.exp(-max(min(score, 30.0), -30.0))) for score in scores]

    @staticmethod
    def _normalize_queries(queries: str | list[tuple[str, float]]) -> list[tuple[str, float]]:
        if isinstance(queries, str):
            normalized_query = queries.strip()
            return [(normalized_query, 1.0)] if normalized_query else []

        aggregated: dict[str, float] = {}
        ordered_queries: list[str] = []
        for query_text, weight in queries:
            normalized_query = str(query_text).strip()
            if not normalized_query or weight <= 0:
                continue
            if normalized_query not in aggregated:
                ordered_queries.append(normalized_query)
                aggregated[normalized_query] = 0.0
            aggregated[normalized_query] += float(weight)
        return [(query_text, aggregated[query_text]) for query_text in ordered_queries]
