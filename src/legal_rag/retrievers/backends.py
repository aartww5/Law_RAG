from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path
import uuid

from legal_rag.types import NormalizedArticle
from legal_rag.utils.text import normalize_question


LOGGER = logging.getLogger(__name__)
BM25_TOP_K = 20
VECTOR_TOP_K = 20
RRF_K = 60
INDEX_BATCH_SIZE = 256


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_question(text)
    if not normalized:
        return []

    jieba = _require_dependency("jieba")
    raw_tokens = jieba.lcut_for_search(normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        cleaned = token.strip().lower()
        if not cleaned or re.fullmatch(r"\W+", cleaned):
            continue
        tokens.append(cleaned)
    return tokens


def weighted_rrf(
    rank_lists: list[tuple[list[tuple[str, float]], float]],
    *,
    k: int = RRF_K,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked_items, weight in rank_lists:
        if weight <= 0:
            continue
        for rank, (canonical_id, _score) in enumerate(ranked_items):
            fused[canonical_id] = fused.get(canonical_id, 0.0) + (weight / (k + rank + 1))
    return fused


def build_search_text(article: NormalizedArticle) -> str:
    parts = [
        article.law_name,
        " ".join(article.law_aliases),
        article.article_id_cn or "",
        article.article_id_num or "",
        article.content,
    ]
    return normalize_question(" ".join(part for part in parts if part))


class Bm25Backend:
    def __init__(self, *, doc_ids: list[str], tokenized_corpus: list[list[str]], bm25) -> None:
        self.doc_ids = doc_ids
        self.tokenized_corpus = tokenized_corpus
        self._bm25 = bm25

    @classmethod
    def from_articles(cls, articles: list[NormalizedArticle]) -> "Bm25Backend":
        rank_bm25 = _require_dependency("rank_bm25")
        tokenized_corpus = [tokenize_for_bm25(build_search_text(article)) for article in articles]
        bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        return cls(
            doc_ids=[article.canonical_id for article in articles],
            tokenized_corpus=tokenized_corpus,
            bm25=bm25,
        )

    def retrieve(self, question: str, *, limit: int = BM25_TOP_K) -> list[tuple[str, float]]:
        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        results: list[tuple[str, float]] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append((self.doc_ids[index], score))
            if len(results) >= limit:
                break
        return results


class QdrantVectorBackend:
    def __init__(
        self,
        *,
        storage_path: Path,
        collection_name: str,
        model_name: str,
        articles: list[NormalizedArticle],
    ) -> None:
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.model_name = model_name
        self._articles = articles
        self._canonical_by_key = {
            (article.law_name, article.article_id_cn or ""): article.canonical_id for article in articles
        }
        self._client = None
        self._model = None
        self._is_ready = False

    @classmethod
    def from_articles(
        cls,
        articles: list[NormalizedArticle],
        *,
        storage_path: str | Path,
        collection_name: str,
        model_name: str,
    ) -> "QdrantVectorBackend":
        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)
        return cls(
            storage_path=storage,
            collection_name=collection_name,
            model_name=model_name,
            articles=articles,
        )

    def retrieve(self, question: str, *, limit: int = VECTOR_TOP_K) -> list[tuple[str, float]]:
        query = normalize_question(question)
        if not query:
            return []

        self._ensure_ready()
        vector = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[tuple[str, float]] = []
        for point in response.points:
            canonical_id = self._resolve_canonical_id(point.payload or {})
            if canonical_id is None:
                continue
            results.append((str(canonical_id), float(point.score)))
        return results

    def _ensure_ready(self) -> None:
        if self._is_ready:
            return

        qdrant_client = _require_dependency("qdrant_client")
        sentence_transformers = _require_dependency("sentence_transformers")
        self._client = qdrant_client.QdrantClient(path=str(self.storage_path))
        self._model = sentence_transformers.SentenceTransformer(self.model_name, device="cpu")
        self._ensure_collection()
        self._is_ready = True

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self.collection_name):
            return

        models = _require_dependency("qdrant_client.http.models")
        sample_vector = self._model.encode("测试", convert_to_numpy=True, normalize_embeddings=True)
        vector_size = int(len(sample_vector))
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        points = []
        for article in self._articles:
            vector = self._model.encode(
                build_search_text(article),
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, article.canonical_id)),
                    vector=vector,
                    payload={
                        "canonical_id": article.canonical_id,
                        "law_name": article.law_name,
                        "article_id_cn": article.article_id_cn,
                        "article_id_num": article.article_id_num,
                    },
                )
            )
            if len(points) >= INDEX_BATCH_SIZE:
                self._client.upsert(self.collection_name, points=points, wait=True)
                points.clear()

        if points:
            self._client.upsert(self.collection_name, points=points, wait=True)

    def _resolve_canonical_id(self, payload: dict) -> str | None:
        canonical_id = payload.get("canonical_id")
        if canonical_id:
            return str(canonical_id)

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            nested_canonical_id = metadata.get("canonical_id")
            if nested_canonical_id:
                return str(nested_canonical_id)

            law_name = metadata.get("law_name") or metadata.get("source")
            article_id_cn = metadata.get("article_id_cn") or metadata.get("article_id")
            if law_name and article_id_cn:
                return self._canonical_by_key.get((str(law_name), str(article_id_cn)))

        law_name = payload.get("law_name") or payload.get("source")
        article_id_cn = payload.get("article_id_cn") or payload.get("article_id")
        if law_name and article_id_cn:
            return self._canonical_by_key.get((str(law_name), str(article_id_cn)))
        return None


def _require_dependency(module_name: str):
    if module_name == "qdrant_client.http.models":
        if importlib.util.find_spec("qdrant_client") is None:
            raise RuntimeError("qdrant_client is not installed")
        from qdrant_client.http import models

        return models

    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"{module_name} is not installed")
    module = __import__(module_name, fromlist=["_sentinel"])
    return module
