import logging
from collections.abc import Iterator
from textwrap import shorten

from legal_rag.config import AppConfig, RuntimeConfig
from legal_rag.generation.context_builder import build_answer_context
from legal_rag.generation.llm import SimpleGenerator
from legal_rag.indexing.corpus_builder import iter_normalized_articles
from legal_rag.rewrite import QueryRewriter
from legal_rag.retrievers.exact_match import ExactMatchRetriever
from legal_rag.retrievers.hybrid import HybridRetriever
from legal_rag.retrievers.mini import MiniRetriever
from legal_rag.router.auto import AutoRouter
from legal_rag.types import (
    ConversationState,
    ConversationTurn,
    FinalAnswer,
    NormalizedArticle,
    PreparedAnswer,
    RouteDecision,
)


LOGGER = logging.getLogger(__name__)


class LegalAssistantService:
    def __init__(
        self,
        config: AppConfig,
        *,
        exact_retriever: ExactMatchRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        mini_retriever: MiniRetriever | None = None,
        router: AutoRouter | None = None,
        generator: SimpleGenerator | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.config = config
        self.exact_retriever = exact_retriever
        self.hybrid_retriever = hybrid_retriever
        self.mini_retriever = mini_retriever
        self.router = router or AutoRouter()
        self.generator = generator or SimpleGenerator(config.runtime.ollama_model)
        self.rewriter = rewriter or QueryRewriter(config.runtime.ollama_model, enable_ollama=True)

    @classmethod
    def for_test(cls, mode: str = "auto", mini_available: bool = True) -> "LegalAssistantService":
        article = NormalizedArticle(
            canonical_id="中华人民共和国民法典:第一千一百三十八条",
            law_name="中华人民共和国民法典",
            law_aliases=["中华人民共和国民法典", "民法典"],
            article_id_cn="第一千一百三十八条",
            article_id_num="1138",
            content="《中华人民共和国民法典》第一千一百三十八条 口头遗嘱...",
            chapter=None,
            section=None,
            source="中华人民共和国民法典.txt",
            source_line=1,
        )
        secondary_article = NormalizedArticle(
            canonical_id="中华人民共和国民法典:第一千一百三十九条",
            law_name="中华人民共和国民法典",
            law_aliases=["中华人民共和国民法典", "民法典"],
            article_id_cn="第一千一百三十九条",
            article_id_num="1139",
            content="《中华人民共和国民法典》第一千一百三十九条 录音录像遗嘱...",
            chapter=None,
            section=None,
            source="中华人民共和国民法典.txt",
            source_line=2,
        )
        config = AppConfig(runtime=RuntimeConfig(mode=mode))
        exact_retriever = ExactMatchRetriever([article])
        hybrid_docs = [
            {
                "canonical_id": article.canonical_id,
                "content": article.content,
                "score": 0.95 if mode != "auto" else 0.52,
            },
            {
                "canonical_id": secondary_article.canonical_id,
                "content": secondary_article.content,
                "score": 0.49 if mode == "auto" else 0.61,
            },
        ]
        hybrid_retriever = HybridRetriever.fake_for_test(hybrid_docs)
        mini_retriever = MiniRetriever.fake_for_test(
            [
                {
                    "chunk_id": "chunk-1",
                    "canonical_id": article.canonical_id,
                    "content": article.content,
                    "score": 0.66,
                }
            ]
        )
        service = cls(
            config=config,
            exact_retriever=exact_retriever,
            hybrid_retriever=hybrid_retriever,
            mini_retriever=mini_retriever if mini_available else MiniRetriever.fake_for_test([]),
            generator=SimpleGenerator(config.runtime.ollama_model, enable_ollama=False),
            rewriter=QueryRewriter(config.runtime.ollama_model, enable_ollama=False),
        )
        service.mini_available = mini_available
        return service

    @classmethod
    def from_laws_dir(cls, laws_dir: str, mode: str = "auto") -> "LegalAssistantService":
        articles = iter_normalized_articles(laws_dir)
        config = AppConfig(runtime=RuntimeConfig(mode=mode))
        return cls.from_config(config, articles=articles)

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        articles: list[NormalizedArticle] | None = None,
    ) -> "LegalAssistantService":
        article_records = articles or iter_normalized_articles(config.index.laws_dir)
        mini_retriever = MiniRetriever.fake_for_test([])
        mini_available = False
        if config.index.mini_working_dir.exists():
            mini_retriever = MiniRetriever.from_working_dir(
                config.index.mini_working_dir,
                article_records,
                ollama_model=config.runtime.ollama_model,
            )
            mini_available = mini_retriever.is_available

        service = cls(
            config=config,
            exact_retriever=ExactMatchRetriever(article_records),
            hybrid_retriever=HybridRetriever.from_articles(article_records, index_config=config.index),
            mini_retriever=mini_retriever,
            router=AutoRouter(),
            generator=SimpleGenerator(config.runtime.ollama_model),
        )
        service.mini_available = mini_available
        return service

    def handle_message(
        self,
        question: str,
        mode: str | None = None,
        conversation_state: ConversationState | None = None,
    ) -> FinalAnswer:
        prepared = self.prepare_answer(
            question,
            mode=mode,
            conversation_state=conversation_state,
        )
        generated = self.generator.generate(prepared.context)
        return self.finalize_answer(prepared, generated.answer_text)

    def prepare_answer(
        self,
        question: str,
        mode: str | None = None,
        conversation_state: ConversationState | None = None,
    ) -> PreparedAnswer:
        active_mode = mode or self.config.runtime.mode
        state = conversation_state or ConversationState()
        rewrite_result = self.rewriter.rewrite(question, state)
        rewritten_query = rewrite_result.rewritten_query
        exact_result = self.exact_retriever.retrieve(rewritten_query) if self.exact_retriever else None
        hybrid_result = self.hybrid_retriever.retrieve(rewritten_query) if self.hybrid_retriever else None
        mini_result = None

        if active_mode == "mini":
            decision = RouteDecision("mini", False, 1.0, "mini_only", ["manual_mode"])
            mini_result = self.mini_retriever.retrieve(rewritten_query) if self.mini_retriever else None
        elif active_mode == "auto":
            decision = self.router.decide(exact_result=exact_result, hybrid_result=hybrid_result)
            if decision.selected_mode == "mini" and not getattr(self, "mini_available", True):
                decision = RouteDecision(
                    selected_mode="hybrid",
                    fallback_triggered=False,
                    confidence=hybrid_result.confidence,
                    merge_policy="hybrid_plus_exact",
                    reasons=[*decision.reasons, "mini_unavailable"],
                )
            elif decision.selected_mode == "mini" and self.mini_retriever:
                mini_result = self.mini_retriever.retrieve(rewritten_query)
                if not mini_result.docs or "mini_runtime_error" in mini_result.reasons:
                    decision = RouteDecision(
                        selected_mode="hybrid",
                        fallback_triggered=False,
                        confidence=hybrid_result.confidence,
                        merge_policy="hybrid_plus_exact",
                        reasons=[*decision.reasons, "mini_failed"],
                    )
        else:
            decision = RouteDecision("hybrid", False, hybrid_result.confidence, "hybrid_plus_exact", ["manual_mode"])

        context = build_answer_context(
            rewritten_query,
            decision,
            exact_result=exact_result,
            hybrid_result=hybrid_result,
            mini_result=mini_result,
            max_articles=self.config.runtime.max_context_articles,
        )
        LOGGER.info(
            "query_rewrite original_query=%r rewritten_query=%r rewrite_notes=%s route=%s",
            rewrite_result.original_query,
            rewrite_result.rewritten_query,
            ",".join(rewrite_result.rewrite_notes) or "none",
            decision.selected_mode,
        )
        return PreparedAnswer(
            raw_query=question,
            rewrite_result=rewrite_result,
            context=context,
        )

    def stream_answer(self, prepared: PreparedAnswer) -> Iterator[str]:
        yield from self.generator.stream_generate(prepared.context)

    def finalize_answer(self, prepared: PreparedAnswer, answer_text: str) -> FinalAnswer:
        return FinalAnswer(
            answer_text=answer_text,
            route_decision=prepared.context.route_decision,
            context=prepared.context,
            rewrite_result=prepared.rewrite_result,
        )

    def build_conversation_turn(self, answer: FinalAnswer) -> ConversationTurn:
        rewrite_result = answer.rewrite_result
        raw_query = rewrite_result.original_query if rewrite_result else answer.context.question
        rewritten_query = rewrite_result.rewritten_query if rewrite_result else answer.context.question
        summary = shorten(" ".join(answer.answer_text.splitlines()), width=160, placeholder="...")
        return ConversationTurn(
            raw_query=raw_query,
            rewritten_query=rewritten_query,
            answer_summary=summary,
            citations=answer.context.citations,
        )
