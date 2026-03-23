from __future__ import annotations

import importlib.util
import math
from typing import Protocol


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 8


class ModelReranker(Protocol):
    def score(self, question: str, docs: list[dict]) -> dict[str, float]:
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
        self.device = device
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model = None

    def score(self, question: str, docs: list[dict]) -> dict[str, float]:
        if not question.strip() or not docs:
            return {}

        model = self._load_model()
        pairs = [(question, self._build_doc_text(doc)) for doc in docs]
        raw_scores = model.predict(
            pairs,
            batch_size=min(self.batch_size, len(pairs)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        normalized_scores = self._normalize_scores(raw_scores)
        return {
            doc["canonical_id"]: score
            for doc, score in zip(docs, normalized_scores, strict=False)
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
