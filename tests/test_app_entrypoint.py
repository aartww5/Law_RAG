import asyncio
import subprocess
import sys
from pathlib import Path
import importlib.util
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_rag.types import AnswerContext, FinalAnswer, RetrievedDoc, RewriteResult, RouteDecision


def test_app_entrypoint_runs_as_script() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    result = subprocess.run(
        [sys.executable, str(app_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Unified legal assistant ready" in result.stdout


def test_build_service_uses_env_mode(monkeypatch) -> None:
    monkeypatch.setenv("LEGAL_RAG_MODE", "hybrid")

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    service = module.build_service()
    assert service.config.runtime.mode == "hybrid"


def test_startup_message_includes_mode_and_mini_status() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    service = module.LegalAssistantService.for_test(mode="auto", mini_available=False)
    message = module.build_startup_message(service)

    assert "mode=auto" in message
    assert "mini_available=False" in message


def test_format_answer_message_omits_rewrite_metadata() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    doc = RetrievedDoc(
        canonical_id="中华人民共和国民法典:第一千一百二十七条",
        content="遗产按照下列顺序继承。",
        metadata={"law_name": "中华人民共和国民法典", "article_id_cn": "第一千一百二十七条"},
        score=1.0,
        score_breakdown={"exact_match": 1.0},
        retriever="exact_match",
    )
    answer = FinalAnswer(
        answer_text="ok",
        route_decision=RouteDecision(
            selected_mode="hybrid",
            fallback_triggered=False,
            confidence=1.0,
            merge_policy="hybrid_plus_exact",
            reasons=["exact_match"],
        ),
        context=AnswerContext(
            question="REWRITTEN_MARKER",
            docs=[doc],
            route_decision=RouteDecision(
                selected_mode="hybrid",
                fallback_triggered=False,
                confidence=1.0,
                merge_policy="hybrid_plus_exact",
                reasons=["exact_match"],
            ),
            citations=["中华人民共和国民法典:第一千一百二十七条"],
            source_summary={"doc_count": 1},
        ),
        rewrite_result=RewriteResult(
            original_query="ORIGINAL_MARKER",
            rewritten_query="REWRITTEN_MARKER",
            rewrite_notes=["history_attached"],
        ),
    )

    message = module.format_answer_message(answer)

    assert "ORIGINAL_MARKER" not in message
    assert "REWRITTEN_MARKER" not in message
    assert "history_attached" not in message


def test_process_user_message_streams_answer_and_persists_conversation_state() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FakeSession:
        def __init__(self) -> None:
            self.data = {}

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value) -> None:
            self.data[key] = value

    class FakeAssistantMessage:
        instances = []

        def __init__(self, content="", **kwargs) -> None:
            self.content = content
            self.tokens = []
            self.sent = False
            self.updated = False
            FakeAssistantMessage.instances.append(self)

        async def send(self):
            self.sent = True
            return self

        async def stream_token(self, token: str):
            self.tokens.append(token)
            self.content += token

        async def update(self):
            self.updated = True
            return self

    class FakeService:
        def __init__(self) -> None:
            self.config = SimpleNamespace(runtime=SimpleNamespace(max_history_turns=4))
            self.received_state = None

        def prepare_answer(self, question, mode=None, conversation_state=None):
            self.received_state = conversation_state
            context = AnswerContext(
                question="基于前述情形：老王去世后留下遗产。追问：老王侄子能继承吗",
                docs=[
                    RetrievedDoc(
                        canonical_id="中华人民共和国民法典:第一千一百二十七条",
                        content="遗产按照下列顺序继承。",
                        metadata={
                            "law_name": "中华人民共和国民法典",
                            "article_id_cn": "第一千一百二十七条",
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
                citations=["中华人民共和国民法典:第一千一百二十七条"],
                source_summary={"doc_count": 1},
            )
            return SimpleNamespace(
                raw_query=question,
                rewrite_result=RewriteResult(
                    original_query=question,
                    rewritten_query=context.question,
                    rewrite_notes=["history_attached"],
                ),
                context=context,
                route_decision=context.route_decision,
            )

        def stream_answer(self, prepared):
            yield "侄子"
            yield "一般不属于法定继承人。"

        def finalize_answer(self, prepared, answer_text: str) -> FinalAnswer:
            return FinalAnswer(
                answer_text=answer_text,
                route_decision=prepared.route_decision,
                context=prepared.context,
                rewrite_result=prepared.rewrite_result,
            )

        def build_conversation_turn(self, answer: FinalAnswer):
            return module.ConversationTurn(
                raw_query=answer.rewrite_result.original_query,
                rewritten_query=answer.rewrite_result.rewritten_query,
                answer_summary=answer.answer_text,
                citations=answer.context.citations,
            )

    session = FakeSession()
    service = FakeService()
    session.set("service", service)

    asyncio.run(
        module.process_user_message(
            SimpleNamespace(content="他侄子能继承吗"),
            service=service,
            session=session,
            message_factory=FakeAssistantMessage,
        )
    )

    assistant_message = FakeAssistantMessage.instances[0]

    assert service.received_state is not None
    assert assistant_message.sent is True
    assert assistant_message.tokens == ["侄子", "一般不属于法定继承人。"]
    assert assistant_message.updated is True
    assert "[route]" in assistant_message.content
    assert "[citations]" in assistant_message.content
    assert "history_attached" not in assistant_message.content
    assert session.get("conversation_state").turns[-1].raw_query == "他侄子能继承吗"
