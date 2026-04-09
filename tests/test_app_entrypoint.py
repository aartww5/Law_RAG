import asyncio
import hashlib
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


def test_release_session_service_closes_active_service() -> None:
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

    class FakeService:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = FakeSession()
    service = FakeService()
    session.set("service", service)

    module.release_session_service(session)

    assert service.closed is True
    assert session.get("service") is None


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


def test_chat_start_initializes_session_without_sending_startup_message(monkeypatch) -> None:
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

    class FakeMessage:
        instances = []

        def __init__(self, *args, **kwargs) -> None:
            FakeMessage.instances.append((args, kwargs))

        async def send(self):
            return self

    fake_service = module.LegalAssistantService.for_test(mode="auto", mini_available=False)
    fake_session = FakeSession()

    monkeypatch.setattr(module, "build_service", lambda: fake_service)

    async def fake_bind_thread_to_current_user() -> None:
        return None

    monkeypatch.setattr(module, "bind_thread_to_current_user", fake_bind_thread_to_current_user)
    monkeypatch.setattr(module.cl, "user_session", fake_session, raising=False)
    monkeypatch.setattr(module.cl, "Message", FakeMessage, raising=False)

    asyncio.run(module.start())

    assert fake_session.get("service") is fake_service
    assert fake_session.get("conversation_state").max_turns == fake_service.config.runtime.max_history_turns
    assert FakeMessage.instances == []


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

        def __init__(self, content="", metadata=None, **kwargs) -> None:
            self.content = content
            self.metadata = metadata or {}
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
    assert assistant_message.metadata["conversation_turn"]["raw_query"] == "他侄子能继承吗"
    assert session.get("conversation_state").turns[-1].raw_query == "他侄子能继承吗"


def test_process_user_message_offloads_blocking_work_to_threads(monkeypatch) -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "has_chainlit_context", lambda: False)

    calls: list[str] = []

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(module, "asyncio", asyncio, raising=False)
    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

    class FakeSession:
        def __init__(self) -> None:
            self.data = {}

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value) -> None:
            self.data[key] = value

    class FakeAssistantMessage:
        def __init__(self, content="", **kwargs) -> None:
            self.content = content

        async def send(self):
            return self

        async def stream_token(self, token: str):
            self.content += token

        async def update(self):
            return self

    class FakeService:
        def __init__(self) -> None:
            self.config = SimpleNamespace(runtime=SimpleNamespace(max_history_turns=4))

        def prepare_answer(self, question, mode=None, conversation_state=None):
            context = AnswerContext(
                question=question,
                docs=[],
                route_decision=RouteDecision(
                    selected_mode="hybrid",
                    fallback_triggered=False,
                    confidence=1.0,
                    merge_policy="hybrid_plus_exact",
                    reasons=["exact_match"],
                ),
                citations=[],
                source_summary={"doc_count": 0},
            )
            return SimpleNamespace(
                raw_query=question,
                rewrite_result=RewriteResult(
                    original_query=question,
                    rewritten_query=question,
                    rewrite_notes=["unchanged"],
                ),
                context=context,
                route_decision=context.route_decision,
            )

        def stream_answer(self, prepared):
            yield "A"
            yield "B"

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

    asyncio.run(
        module.process_user_message(
            SimpleNamespace(content="测试问题"),
            service=FakeService(),
            session=FakeSession(),
            message_factory=FakeAssistantMessage,
        )
    )

    assert "prepare_answer" in calls
    assert "_next_stream_chunk" in calls
    assert "finalize_answer" in calls
    assert "build_conversation_turn" in calls


def test_authenticate_local_user_accepts_plain_and_hashed_passwords() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config = module.AppConfig(
        auth=module.AuthConfig(
            users=[
                module.LocalAuthUser(username="plain", display_name="Plain User", password="secret123"),
                module.LocalAuthUser(
                    username="hashed",
                    display_name="Hashed User",
                    password_hash="sha256$" + hashlib.sha256("secret456".encode("utf-8")).hexdigest(),
                ),
            ]
        )
    )

    plain_user = asyncio.run(module.authenticate_local_user("plain", "secret123", config))
    hashed_user = asyncio.run(module.authenticate_local_user("hashed", "secret456", config))

    assert plain_user is not None
    assert plain_user.identifier == "plain"
    assert hashed_user is not None
    assert hashed_user.identifier == "hashed"


def test_authenticate_local_user_rejects_invalid_credentials() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config = module.AppConfig(
        auth=module.AuthConfig(
            users=[module.LocalAuthUser(username="plain", display_name="Plain User", password="secret123")]
        )
    )

    assert asyncio.run(module.authenticate_local_user("plain", "wrong", config)) is None
    assert asyncio.run(module.authenticate_local_user("missing", "secret123", config)) is None


def test_rebuild_conversation_state_from_thread_uses_persisted_turn_metadata() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    thread = {
        "id": "thread-1",
        "steps": [
            {
                "id": "step-user-1",
                "type": "user_message",
                "metadata": {},
            },
            {
                "id": "step-assistant-1",
                "type": "assistant_message",
                "metadata": {
                    "conversation_turn": {
                        "raw_query": "他侄子能继承吗",
                        "rewritten_query": "老王去世后，其侄子是否属于法定继承人，能否继承其遗产？",
                        "answer_summary": "一般不属于法定继承人。",
                        "citations": ["中华人民共和国民法典:第一千一百二十七条"],
                    }
                },
            },
        ],
    }

    state = module.rebuild_conversation_state_from_thread(thread, max_turns=4)

    assert len(state.turns) == 1
    assert state.turns[0].raw_query == "他侄子能继承吗"
    assert state.turns[0].rewritten_query.startswith("老王去世后")


def test_ensure_chainlit_auth_secret_sets_local_default(monkeypatch) -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("unified_app_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.delenv("CHAINLIT_AUTH_SECRET", raising=False)

    module.ensure_chainlit_auth_secret()

    assert module.os.environ["CHAINLIT_AUTH_SECRET"]
