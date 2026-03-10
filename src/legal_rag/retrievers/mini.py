import asyncio
import importlib.util
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from legal_rag.config import DEFAULT_OLLAMA_MODEL
from legal_rag.types import NormalizedArticle, RetrievedDoc, RetrievalResult


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_SKIP_CHECK_LOAD_AVAILABILITY", "1")


class MiniRetriever:
    def __init__(
        self,
        docs: list[dict] | None = None,
        *,
        is_available: bool = True,
        unavailable_reason: str | None = None,
        runtime: Any = None,
    ) -> None:
        self._fake_docs = docs or []
        self.is_available = is_available
        self.unavailable_reason = unavailable_reason
        self._runtime = runtime

    @classmethod
    def fake_for_test(cls, docs: list[dict]) -> "MiniRetriever":
        return cls(docs=docs)

    @classmethod
    def from_working_dir(
        cls,
        working_dir: str | Path,
        articles: list[NormalizedArticle],
        *,
        query_mode: str = "mini",
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        embedding_model: str = "BAAI/bge-m3",
    ) -> "MiniRetriever":
        working_path = Path(working_dir)
        if not working_path.exists():
            return cls(is_available=False, unavailable_reason="mini_working_dir_missing")

        if importlib.util.find_spec("minirag") is None:
            return cls(is_available=False, unavailable_reason="minirag_not_installed")

        runtime = _MiniRuntime(
            working_path,
            articles,
            query_mode=query_mode,
            ollama_model=ollama_model,
            embedding_model=embedding_model,
        )
        return cls(is_available=True, runtime=runtime)

    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        if not self.is_available:
            return RetrievalResult(
                docs=[],
                confidence=0.0,
                latency_ms=0.0,
                reasons=[self.unavailable_reason] if self.unavailable_reason else [],
                raw_signals={"candidate_count": 0},
            )

        if self._runtime is not None:
            started = perf_counter()
            try:
                docs = self._runtime.query(question)
            except Exception as exc:
                return RetrievalResult(
                    docs=[],
                    confidence=0.0,
                    latency_ms=(perf_counter() - started) * 1000,
                    reasons=["mini_runtime_error"],
                    raw_signals={"candidate_count": 0, "error": str(exc)},
                )
            latency_ms = (perf_counter() - started) * 1000
            return RetrievalResult(
                docs=docs,
                confidence=max((doc.score for doc in docs), default=0.0),
                latency_ms=latency_ms,
                reasons=["mini_retrieved"] if docs else [],
                raw_signals={"candidate_count": len(docs)},
            )

        docs = [
            RetrievedDoc(
                canonical_id=item["canonical_id"],
                content=item["content"],
                metadata={"chunk_id": item["chunk_id"]},
                score=item["score"],
                score_breakdown={"mini": item["score"]},
                retriever="mini",
            )
            for item in self._fake_docs
        ]
        return RetrievalResult(
            docs=docs,
            confidence=max((doc.score for doc in docs), default=0.0),
            latency_ms=0.0,
            reasons=["mini_retrieved"] if docs else [],
            raw_signals={"candidate_count": len(docs)},
        )


class _MiniRuntime:
    def __init__(
        self,
        working_dir: Path,
        articles: list[NormalizedArticle],
        *,
        query_mode: str,
        ollama_model: str,
        embedding_model: str,
    ) -> None:
        self._working_dir = working_dir
        self._articles = articles
        self._query_mode = query_mode
        self._ollama_model = ollama_model
        self._embedding_model = embedding_model
        self._rag = self._init_rag()
        self._article_by_content = {article.content: article for article in articles}

    def _init_rag(self) -> Any:
        import numpy as np
        import ollama
        from minirag import MiniRAG
        from minirag.utils import EmbeddingFunc

        embed_model = None

        def get_embed_model():
            nonlocal embed_model
            if embed_model is None:
                from sentence_transformers import SentenceTransformer

                embed_model = SentenceTransformer(self._embedding_model, device="cpu")
            return embed_model

        async def bge_m3_embed(texts: list[str]) -> np.ndarray:
            model = get_embed_model()
            embeddings = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)

        async def ollama_llm_complete(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list | None = None,
            **kwargs,
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            response = await asyncio.to_thread(
                ollama.chat,
                model=self._ollama_model,
                messages=messages,
                stream=False,
                options={"temperature": 0, "num_ctx": 32768},
            )
            content = response["message"]["content"]
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            return content

        embedding = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=bge_m3_embed,
        )

        return MiniRAG(
            working_dir=str(self._working_dir),
            llm_model_func=ollama_llm_complete,
            llm_model_name=self._ollama_model,
            llm_model_max_token_size=32768,
            embedding_func=embedding,
            chunk_token_size=512,
            chunk_overlap_token_size=50,
            entity_extract_max_gleaning=1,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            enable_llm_cache=True,
        )

    def query(self, question: str) -> list[RetrievedDoc]:
        from minirag import QueryParam

        async def do_query() -> str:
            return await self._rag.aquery(
                question,
                param=QueryParam(mode=self._query_mode, only_need_context=True, top_k=20),
            )

        context = asyncio.run(do_query())
        if not context:
            return []

        docs: list[RetrievedDoc] = []
        for index, article in enumerate(self._match_articles(context), start=1):
            score = max(0.0, 1.0 - (index - 1) * 0.05)
            docs.append(
                RetrievedDoc(
                    canonical_id=article.canonical_id,
                    content=article.content,
                    metadata={
                        "law_name": article.law_name,
                        "article_id_cn": article.article_id_cn,
                        "article_id_num": article.article_id_num,
                    },
                    score=score,
                    score_breakdown={"mini": score},
                    retriever="mini",
                )
            )
        return docs

    def _match_articles(self, context: str) -> list[NormalizedArticle]:
        matched: list[NormalizedArticle] = []
        for article in self._articles:
            if article.content and article.content in context:
                matched.append(article)
        return matched
