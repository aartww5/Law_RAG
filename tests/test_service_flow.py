from pathlib import Path
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import legal_rag.rewrite as rewrite_module
from legal_rag.config import AppConfig, IndexConfig, RuntimeConfig
from legal_rag.generation.llm import SimpleGenerator
from legal_rag.rewrite import QueryRewriter
from legal_rag.retrievers.exact_match import ExactMatchRetriever
from legal_rag.retrievers.hybrid import HybridRetriever
from legal_rag.retrievers.mini import MiniRetriever
from legal_rag.services import LegalAssistantService
from legal_rag.types import (
    AnswerContext,
    ConversationState,
    ConversationTurn,
    FinalAnswer,
    NormalizedArticle,
    RetrievalResult,
    RetrievedDoc,
    RewriteResult,
    RouteDecision,
)


def install_fake_ollama(monkeypatch, chat_impl) -> None:
    fake_ollama = ModuleType("ollama")
    fake_ollama.__spec__ = ModuleSpec("ollama", loader=None)
    fake_ollama.chat = chat_impl
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)


def test_service_handles_hybrid_mode_with_shared_pipeline() -> None:
    service = LegalAssistantService.for_test(mode="hybrid")
    answer = service.handle_message("民法典1138条怎么规定")

    assert answer.route_decision.selected_mode == "hybrid"
    assert answer.answer_text
    assert "相关法律条文" not in answer.answer_text


def test_service_for_test_does_not_call_external_ollama(monkeypatch) -> None:
    calls = {"count": 0}

    def fail_chat(**kwargs):
        calls["count"] += 1
        raise AssertionError("for_test should not call ollama")

    install_fake_ollama(monkeypatch, fail_chat)

    service = LegalAssistantService.for_test(mode="hybrid")
    answer = service.handle_message("口头遗嘱需要几个见证人")

    assert answer.route_decision.selected_mode == "hybrid"
    assert answer.answer_text
    assert calls["count"] == 0


def test_service_can_bootstrap_from_laws_dir(tmp_path: Path) -> None:
    laws_dir = tmp_path / "laws"
    laws_dir.mkdir()
    (laws_dir / "中华人民共和国民法典.txt").write_text(
        "第一千一百三十八条 口头遗嘱应当有两个以上见证人在场见证。",
        encoding="utf-8",
    )

    service = LegalAssistantService.from_laws_dir(laws_dir, mode="hybrid")
    answer = service.handle_message("口头遗嘱需要几个见证人")

    assert answer.route_decision.selected_mode == "hybrid"
    assert "口头遗嘱" in answer.answer_text


def test_service_can_bootstrap_from_config(tmp_path: Path) -> None:
    laws_dir = tmp_path / "RAG" / "Chinese-Laws"
    laws_dir.mkdir(parents=True)
    (laws_dir / "中华人民共和国民法典.txt").write_text(
        "第一千一百三十八条 口头遗嘱应当有两个以上见证人在场见证。",
        encoding="utf-8",
    )

    config = AppConfig.from_env(tmp_path)
    service = LegalAssistantService.from_config(config)
    answer = service.handle_message("口头遗嘱需要几个见证人")

    assert answer.route_decision.selected_mode == "hybrid"
    assert answer.context.docs


def test_service_uses_configured_ollama_model_for_generation(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_chat(*, model, messages, stream, options):
        captured["model"] = model
        return {"message": {"content": "configured model ok"}}

    install_fake_ollama(monkeypatch, fake_chat)

    article = NormalizedArticle(
        canonical_id="民法典:1138",
        law_name="中华人民共和国民法典",
        law_aliases=["中华人民共和国民法典", "民法典"],
        article_id_cn="第一千一百三十八条",
        article_id_num="1138",
        content="口头遗嘱应当有两个以上见证人在场见证。",
        chapter=None,
        section=None,
        source="民法典.txt",
        source_line=1,
    )
    config = AppConfig(
        runtime=RuntimeConfig(mode="hybrid", ollama_model="qwen35-law:q6k-stable"),
        index=IndexConfig(
            laws_dir=tmp_path / "laws",
            qdrant_path=tmp_path / "qdrant",
            bm25_cache_path=tmp_path / "bm25",
            mini_working_dir=tmp_path / "mini",
            corpus_dir=tmp_path / "corpus",
        ),
    )

    service = LegalAssistantService.from_config(config, articles=[article])
    answer = service.handle_message("口头遗嘱需要几个见证人")

    assert answer.answer_text == "configured model ok"
    assert captured["model"] == "qwen35-law:q6k-stable"


def test_service_uses_configured_ollama_model_for_rewrite(tmp_path: Path) -> None:
    article = NormalizedArticle(
        canonical_id="民法典:1138",
        law_name="中华人民共和国民法典",
        law_aliases=["中华人民共和国民法典", "民法典"],
        article_id_cn="第一千一百三十八条",
        article_id_num="1138",
        content="口头遗嘱应当有两个以上见证人在场见证。",
        chapter=None,
        section=None,
        source="民法典.txt",
        source_line=1,
    )
    config = AppConfig(
        runtime=RuntimeConfig(mode="hybrid", ollama_model="qwen35-law:q6k-stable"),
        index=IndexConfig(
            laws_dir=tmp_path / "laws",
            qdrant_path=tmp_path / "qdrant",
            bm25_cache_path=tmp_path / "bm25",
            mini_working_dir=tmp_path / "mini",
            corpus_dir=tmp_path / "corpus",
        ),
    )

    service = LegalAssistantService.from_config(config, articles=[article])

    assert service.rewriter.model_name == "qwen35-law:q6k-stable"


def test_conversation_state_keeps_recent_turns() -> None:
    state = ConversationState(max_turns=2)
    state.add_turn(
        ConversationTurn(
            raw_query="q1",
            rewritten_query="rq1",
            answer_summary="a1",
            citations=["c1"],
        )
    )
    state.add_turn(
        ConversationTurn(
            raw_query="q2",
            rewritten_query="rq2",
            answer_summary="a2",
            citations=["c2"],
        )
    )
    state.add_turn(
        ConversationTurn(
            raw_query="q3",
            rewritten_query="rq3",
            answer_summary="a3",
            citations=["c3"],
        )
    )

    assert [turn.raw_query for turn in state.turns] == ["q2", "q3"]


def test_rewrite_keeps_original_query_when_ollama_is_disabled() -> None:
    state = ConversationState(
        turns=[
            ConversationTurn(
                raw_query="个体户老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                rewritten_query="个体户老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                answer_summary="无人继承又无受遗赠的遗产，归国家用于公益事业；特定情形下归集体所有。",
                citations=["中华人民共和国民法典:第一千一百六十条"],
            )
        ]
    )

    result = QueryRewriter(enable_ollama=False).rewrite("他侄子能继承吗", state)

    assert result.original_query == "他侄子能继承吗"
    assert result.rewritten_query == "他侄子能继承吗"
    assert "llm_unavailable" in result.rewrite_notes
    assert "unchanged" in result.rewrite_notes


def test_rewrite_runs_even_when_query_stays_the_same() -> None:
    result = QueryRewriter(enable_ollama=False).rewrite("民法典第一条是什么", ConversationState())

    assert result.original_query == "民法典第一条是什么"
    assert result.rewritten_query == "民法典第一条是什么"
    assert "unchanged" in result.rewrite_notes


def test_rewrite_keeps_original_query_when_history_exists_but_ollama_is_disabled() -> None:
    state = ConversationState(
        turns=[
            ConversationTurn(
                raw_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                rewritten_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                answer_summary="无人继承又无受遗赠的遗产，归国家用于公益事业。",
                citations=["中华人民共和国民法典:第一千一百六十条"],
            )
        ]
    )

    result = QueryRewriter(enable_ollama=False).rewrite("如果没有遗嘱呢", state)

    assert result.original_query == "如果没有遗嘱呢"
    assert result.rewritten_query == "如果没有遗嘱呢"
    assert "llm_unavailable" in result.rewrite_notes
    assert "unchanged" in result.rewrite_notes


def test_rewrite_prefers_llm_with_recent_window_and_answer_summaries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(*, model, messages, stream, options):
        captured["model"] = model
        captured["messages"] = messages
        captured["stream"] = stream
        return {"message": {"content": "老王去世且没有遗嘱时，侄子能否继承遗产？"}}

    fake_importlib = type(
        "FakeImportlib",
        (),
        {"util": type("FakeUtil", (), {"find_spec": staticmethod(lambda name: object())})},
    )
    monkeypatch.setattr(rewrite_module, "importlib", fake_importlib, raising=False)
    install_fake_ollama(monkeypatch, fake_chat)

    state = ConversationState(
        turns=[
            ConversationTurn(
                raw_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                rewritten_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                answer_summary="无人继承又无受遗赠的遗产，归国家用于公益事业。",
                citations=["中华人民共和国民法典:第一千一百六十条"],
            ),
            ConversationTurn(
                raw_query="如果侄子主张继承呢？",
                rewritten_query="老王去世后如果侄子主张继承，应如何认定？",
                answer_summary="侄子通常不属于法定继承第一顺位或第二顺位。",
                citations=["中华人民共和国民法典:第一千一百二十七条"],
            ),
        ],
        max_turns=4,
    )

    result = QueryRewriter(model_name="rewrite-model").rewrite("如果没有遗嘱呢", state)

    assert result.rewritten_query == "老王去世且没有遗嘱时，侄子能否继承遗产？"
    assert "llm_rewritten" in result.rewrite_notes
    assert captured["model"] == "rewrite-model"
    assert captured["stream"] is False
    system_prompt = captured["messages"][0]["content"]
    prompt = captured["messages"][1]["content"]
    assert "不得补充历史中未出现的新事实" in system_prompt
    assert "老王去世后留下遗产" in prompt
    assert "侄子通常不属于法定继承第一顺位或第二顺位" in prompt
    assert "如果没有遗嘱呢" in prompt
    assert "第2轮检索问题" not in prompt
    assert "引用法条" not in prompt
    assert "中华人民共和国民法典:第一千一百二十七条" not in prompt


def test_rewrite_keeps_original_query_when_llm_errors(monkeypatch) -> None:
    def fail_chat(**kwargs):
        raise RuntimeError("rewrite backend failed")

    fake_importlib = type(
        "FakeImportlib",
        (),
        {"util": type("FakeUtil", (), {"find_spec": staticmethod(lambda name: object())})},
    )
    monkeypatch.setattr(rewrite_module, "importlib", fake_importlib, raising=False)
    install_fake_ollama(monkeypatch, fail_chat)

    state = ConversationState(
        turns=[
            ConversationTurn(
                raw_query="个体户老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                rewritten_query="个体户老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                answer_summary="无人继承又无受遗赠的遗产，归国家用于公益事业；特定情形下归集体所有。",
                citations=["中华人民共和国民法典:第一千一百六十条"],
            )
        ]
    )

    result = QueryRewriter(model_name="rewrite-model").rewrite("他侄子能继承吗", state)

    assert result.rewritten_query == "他侄子能继承吗"
    assert "llm_error" in result.rewrite_notes
    assert "unchanged" in result.rewrite_notes


def test_service_uses_rewritten_query_for_all_retrievers() -> None:
    rewritten_query = "在老王去世后遗产继承纠纷中，老王侄子能继承吗"
    doc = RetrievedDoc(
        canonical_id="中华人民共和国民法典:第一千一百二十七条",
        content="遗产按照下列顺序继承。",
        metadata={
            "law_name": "中华人民共和国民法典",
            "law_aliases": ["中华人民共和国民法典", "民法典"],
            "article_id_cn": "第一千一百二十七条",
            "article_id_num": "1127",
        },
        score=0.95,
        score_breakdown={"lexical": 0.95},
        retriever="mini",
    )

    class RecordingRetriever:
        def __init__(self, result: RetrievalResult) -> None:
            self.result = result
            self.queries: list[str] = []

        def retrieve(self, question: str, **kwargs) -> RetrievalResult:
            self.queries.append(question)
            return self.result

    class FixedRewriter:
        def rewrite(self, raw_query: str, state: ConversationState | None = None) -> RewriteResult:
            return RewriteResult(
                original_query=raw_query,
                rewritten_query=rewritten_query,
                rewrite_notes=["history_attached"],
            )

    class EchoGenerator:
        def __init__(self) -> None:
            self.context: AnswerContext | None = None

        def generate(self, context: AnswerContext) -> FinalAnswer:
            self.context = context
            return FinalAnswer(
                answer_text="ok",
                route_decision=context.route_decision,
                context=context,
            )

    empty_result = RetrievalResult(
        docs=[],
        confidence=0.0,
        latency_ms=0.0,
        reasons=[],
        raw_signals={},
    )
    low_confidence_hybrid = RetrievalResult(
        docs=[],
        confidence=0.3,
        latency_ms=0.0,
        reasons=["low_hybrid_confidence"],
        raw_signals={"top1_score": 0.3, "top2_score": 0.25},
    )
    mini_result = RetrievalResult(
        docs=[doc],
        confidence=0.91,
        latency_ms=0.0,
        reasons=["mini_hit"],
        raw_signals={"top1_score": 0.91},
    )

    exact = RecordingRetriever(empty_result)
    hybrid = RecordingRetriever(low_confidence_hybrid)
    mini = RecordingRetriever(mini_result)
    generator = EchoGenerator()
    service = LegalAssistantService(
        config=AppConfig(runtime=RuntimeConfig(mode="auto")),
        exact_retriever=exact,
        hybrid_retriever=hybrid,
        mini_retriever=mini,
        generator=generator,
        rewriter=FixedRewriter(),
    )
    service.mini_available = True

    answer = service.handle_message(
        "他侄子能继承吗",
        conversation_state=ConversationState(
            turns=[
                ConversationTurn(
                    raw_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                    rewritten_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                    answer_summary="无人继承又无受遗赠的遗产，归国家用于公益事业。",
                    citations=["中华人民共和国民法典:第一千一百六十条"],
                )
            ]
        ),
    )

    assert exact.queries == [rewritten_query]
    assert hybrid.queries == [rewritten_query]
    assert mini.queries == [rewritten_query]
    assert generator.context is not None
    assert generator.context.question == rewritten_query
    assert answer.context.citations == ["中华人民共和国民法典:第一千一百二十七条"]


def test_service_logs_rewrite_information(caplog) -> None:
    class FixedRewriter:
        def rewrite(self, raw_query: str, state: ConversationState | None = None) -> RewriteResult:
            return RewriteResult(
                original_query=raw_query,
                rewritten_query="日志专用改写问题",
                rewrite_notes=["history_attached"],
            )

    service = LegalAssistantService.for_test(mode="hybrid")
    service.rewriter = FixedRewriter()

    with caplog.at_level("INFO", logger="legal_rag.services"):
        service.handle_message("他侄子能继承吗")

    assert "original_query='他侄子能继承吗'" in caplog.text
    assert "rewritten_query='日志专用改写问题'" in caplog.text
    assert "route=hybrid" in caplog.text


def test_stream_generate_yields_incremental_chunks_without_ollama() -> None:
    context = AnswerContext(
        question="口头遗嘱需要几个见证人",
        docs=[
            RetrievedDoc(
                canonical_id="中华人民共和国民法典:第一千一百三十八条",
                content="口头遗嘱应当有两个以上见证人在场见证。",
                metadata={
                    "law_name": "中华人民共和国民法典",
                    "article_id_cn": "第一千一百三十八条",
                },
                score=1.0,
                score_breakdown={"exact_match": 1.0},
                retriever="exact_match",
            )
        ],
        route_decision=RouteDecision(
            selected_mode="hybrid",
            fallback_triggered=False,
            confidence=1.0,
            merge_policy="hybrid_plus_exact",
            reasons=["exact_match"],
        ),
        citations=["中华人民共和国民法典:第一千一百三十八条"],
        source_summary={"doc_count": 1},
    )

    generator = SimpleGenerator(enable_ollama=False)

    chunks = list(generator.stream_generate(context))

    assert len(chunks) > 1
    assert "".join(chunks) == generator.generate(context).answer_text


def test_stream_generate_falls_back_when_streaming_backend_errors(monkeypatch) -> None:
    context = AnswerContext(
        question="口头遗嘱需要几个见证人",
        docs=[
            RetrievedDoc(
                canonical_id="中华人民共和国民法典:第一千一百三十八条",
                content="口头遗嘱应当有两个以上见证人在场见证。",
                metadata={
                    "law_name": "中华人民共和国民法典",
                    "article_id_cn": "第一千一百三十八条",
                },
                score=1.0,
                score_breakdown={"exact_match": 1.0},
                retriever="exact_match",
            )
        ],
        route_decision=RouteDecision(
            selected_mode="hybrid",
            fallback_triggered=False,
            confidence=1.0,
            merge_policy="hybrid_plus_exact",
            reasons=["exact_match"],
        ),
        citations=["中华人民共和国民法典:第一千一百三十八条"],
        source_summary={"doc_count": 1},
    )

    import legal_rag.generation.llm as llm_module

    monkeypatch.setattr(llm_module.importlib.util, "find_spec", lambda name: object())

    def fail_chat(**kwargs):
        raise RuntimeError("stream failed")

    install_fake_ollama(monkeypatch, fail_chat)

    generator = SimpleGenerator(enable_ollama=True)
    chunks = list(generator.stream_generate(context))

    assert "".join(chunks) == generator.generate(context).answer_text


def test_stream_generate_preserves_newline_only_chunks_from_ollama(monkeypatch) -> None:
    context = AnswerContext(
        question="口头遗嘱需要几个见证人",
        docs=[
            RetrievedDoc(
                canonical_id="中华人民共和国民法典:第一千一百三十八条",
                content="口头遗嘱应当有两个以上见证人在场见证。",
                metadata={
                    "law_name": "中华人民共和国民法典",
                    "article_id_cn": "第一千一百三十八条",
                },
                score=1.0,
                score_breakdown={"exact_match": 1.0},
                retriever="exact_match",
            )
        ],
        route_decision=RouteDecision(
            selected_mode="hybrid",
            fallback_triggered=False,
            confidence=1.0,
            merge_policy="hybrid_plus_exact",
            reasons=["exact_match"],
        ),
        citations=["中华人民共和国民法典:第一千一百三十八条"],
        source_summary={"doc_count": 1},
    )

    import legal_rag.generation.llm as llm_module

    monkeypatch.setattr(llm_module.importlib.util, "find_spec", lambda name: object())

    def fake_stream_chat(**kwargs):
        return iter(
            [
                {"message": {"content": "第一段"}},
                {"message": {"content": "\n"}},
                {"message": {"content": "\n"}},
                {"message": {"content": "第二段"}},
            ]
        )

    install_fake_ollama(monkeypatch, fake_stream_chat)

    generator = SimpleGenerator(enable_ollama=True)
    chunks = list(generator.stream_generate(context))

    assert chunks == ["第一段", "\n", "\n", "第二段"]
    assert "".join(chunks) == "第一段\n\n第二段"


def test_follow_up_query_uses_previous_inheritance_context() -> None:
    articles = [
        NormalizedArticle(
            canonical_id="中华人民共和国民法典:第一千一百二十七条",
            law_name="中华人民共和国民法典",
            law_aliases=["中华人民共和国民法典", "民法典"],
            article_id_cn="第一千一百二十七条",
            article_id_num="1127",
            content="遗产按照下列顺序继承：第一顺序为配偶、子女、父母；第二顺序为兄弟姐妹、祖父母、外祖父母。",
            chapter=None,
            section=None,
            source="民法典.txt",
            source_line=1,
        ),
        NormalizedArticle(
            canonical_id="中华人民共和国合同法:第一百零七条",
            law_name="中华人民共和国合同法",
            law_aliases=["中华人民共和国合同法", "合同法"],
            article_id_cn="第一百零七条",
            article_id_num="107",
            content="当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行等违约责任。",
            chapter=None,
            section=None,
            source="合同法.txt",
            source_line=1,
        ),
    ]
    class FixedRewriter:
        def rewrite(self, raw_query: str, state: ConversationState | None = None) -> RewriteResult:
            return RewriteResult(
                original_query=raw_query,
                rewritten_query="老王去世后留下遗产且没有遗嘱时，侄子能否继承遗产？",
                rewrite_notes=["llm_rewritten"],
            )

    service = LegalAssistantService(
        config=AppConfig(runtime=RuntimeConfig(mode="hybrid")),
        exact_retriever=ExactMatchRetriever(articles),
        hybrid_retriever=HybridRetriever.from_articles(articles),
        mini_retriever=MiniRetriever.fake_for_test([]),
        generator=SimpleGenerator(enable_ollama=False),
        rewriter=FixedRewriter(),
    )
    service.mini_available = False

    state = ConversationState(
        turns=[
            ConversationTurn(
                raw_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                rewritten_query="老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？",
                answer_summary="无人继承又无受遗赠的遗产，归国家用于公益事业。",
                citations=["中华人民共和国民法典:第一千一百六十条"],
            )
        ]
    )

    answer = service.handle_message("他侄子能继承吗", conversation_state=state)

    assert answer.rewrite_result is not None
    assert answer.rewrite_result.rewritten_query == "老王去世后留下遗产且没有遗嘱时，侄子能否继承遗产？"
    assert answer.context.citations[0] == "中华人民共和国民法典:第一千一百二十七条"
