"""Unified legal RAG application entrypoint."""

import asyncio
import hashlib
import hmac
import logging
import os
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = CURRENT_DIR / "src"
LOG_DIR = CURRENT_DIR / "logs"
LOG_FILE = LOG_DIR / "unified_app.log"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def ensure_chainlit_auth_secret() -> str:
    secret = os.environ.get("CHAINLIT_AUTH_SECRET")
    if secret:
        return secret
    local_secret = hashlib.sha256(str(CURRENT_DIR).encode("utf-8")).hexdigest()
    os.environ["CHAINLIT_AUTH_SECRET"] = local_secret
    return local_secret


ensure_chainlit_auth_secret()

from legal_rag.chat_history import SQLiteChatHistoryDataLayer
from legal_rag.config import AppConfig, AuthConfig, LocalAuthUser
from legal_rag.services import LegalAssistantService
from legal_rag.types import ConversationState, ConversationTurn

try:
    import chainlit as cl
except Exception:  # pragma: no cover - optional runtime dependency
    cl = None


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    file_handler_exists = False
    stream_handler_exists = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == LOG_FILE.resolve():
            file_handler_exists = True
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            stream_handler_exists = True

    if not stream_handler_exists:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if not file_handler_exists:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return LOG_FILE


_DATA_LAYER: SQLiteChatHistoryDataLayer | None = None


def get_app_config() -> AppConfig:
    return AppConfig.from_env(PROJECT_ROOT)


def get_data_layer_instance() -> SQLiteChatHistoryDataLayer:
    global _DATA_LAYER
    if _DATA_LAYER is None:
        _DATA_LAYER = SQLiteChatHistoryDataLayer(get_app_config().storage.chat_db_path)
    return _DATA_LAYER


def build_service() -> LegalAssistantService:
    configure_logging()
    config = get_app_config()
    if config.index.laws_dir.exists():
        return LegalAssistantService.from_config(config)
    return LegalAssistantService.for_test(mode=config.runtime.mode)


def build_startup_message(service: LegalAssistantService) -> str:
    return (
        "Unified legal assistant ready "
        f"(mode={service.config.runtime.mode}, "
        f"mini_available={getattr(service, 'mini_available', False)})"
    )


def close_service_instance(service) -> None:
    close = getattr(service, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            logging.getLogger(__name__).warning("service close failed: %s", exc)


def release_session_service(session) -> None:
    service = session.get("service")
    if service is None:
        return
    close_service_instance(service)
    session.set("service", None)


def hash_password(password: str) -> str:
    digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return f"sha256${digest}"


def verify_local_password(password: str, user: LocalAuthUser) -> bool:
    if user.password_hash:
        prefix, _, digest = user.password_hash.partition("$")
        if prefix == "sha256" and digest:
            candidate = hashlib.sha256(password.encode("utf-8")).hexdigest()
            return hmac.compare_digest(candidate, digest)
        return hmac.compare_digest(hash_password(password), user.password_hash)
    if user.password is None:
        return False
    return hmac.compare_digest(user.password, password)


async def authenticate_local_user(username: str, password: str, config: AppConfig | None = None):
    if cl is None:
        return None
    active_config = config or get_app_config()
    for user in active_config.auth.users:
        if user.username != username:
            continue
        if not verify_local_password(password, user):
            return None
        return cl.User(
            identifier=user.username,
            display_name=user.display_name or user.username,
            metadata={"provider": "local_config"},
        )
    return None


def serialize_conversation_turn(turn: ConversationTurn) -> dict:
    return {
        "raw_query": turn.raw_query,
        "rewritten_query": turn.rewritten_query,
        "answer_summary": turn.answer_summary,
        "citations": list(turn.citations),
    }


def deserialize_conversation_turn(payload: dict | None) -> ConversationTurn | None:
    if not isinstance(payload, dict):
        return None
    raw_query = str(payload.get("raw_query", "")).strip()
    rewritten_query = str(payload.get("rewritten_query", "")).strip()
    answer_summary = str(payload.get("answer_summary", "")).strip()
    if not raw_query or not rewritten_query:
        return None
    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    return ConversationTurn(
        raw_query=raw_query,
        rewritten_query=rewritten_query,
        answer_summary=answer_summary,
        citations=[str(citation) for citation in citations],
    )


def rebuild_conversation_state_from_thread(thread: dict, max_turns: int) -> ConversationState:
    state = ConversationState(max_turns=max_turns)
    steps = thread.get("steps", [])
    if not isinstance(steps, list):
        return state

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type", ""))
        if "assistant_message" not in step_type:
            continue
        metadata = step.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                import json

                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        turn = deserialize_conversation_turn(metadata.get("conversation_turn"))
        if turn is not None:
            state.add_turn(turn)
    return state


async def bind_thread_to_current_user() -> None:
    if cl is None or not has_chainlit_context():
        return
    session = cl.context.session
    user = getattr(session, "user", None)
    user_id = getattr(user, "id", None)
    thread_id = getattr(session, "thread_id", None)
    if not user_id or not thread_id:
        return
    await get_data_layer_instance().update_thread(thread_id=thread_id, user_id=user_id)


def format_answer_message(answer) -> str:
    reasons = ", ".join(answer.route_decision.reasons) if answer.route_decision.reasons else "none"
    citations = "\n".join(f"- {citation}" for citation in answer.context.citations) or "- none"
    return (
        f"{answer.answer_text}\n\n"
        f"[route] mode={answer.route_decision.selected_mode}, "
        f"merge={answer.route_decision.merge_policy}, reasons={reasons}\n\n"
        f"[citations]\n{citations}"
    )


def get_or_create_conversation_state(
    session,
    service: LegalAssistantService,
) -> ConversationState:
    state = session.get("conversation_state")
    if state is None:
        state = ConversationState(max_turns=service.config.runtime.max_history_turns)
        session.set("conversation_state", state)
    return state


def has_chainlit_context() -> bool:
    if cl is None:
        return False
    try:
        _ = cl.context.session
    except Exception:
        return False
    return True


def _next_stream_chunk(stream) -> tuple[str, bool]:
    try:
        return next(stream), False
    except StopIteration:
        return "", True


async def process_user_message(
    message,
    *,
    service: LegalAssistantService,
    session,
    message_factory,
) -> None:
    raw_query = message.content.strip()
    if not raw_query:
        return

    state = get_or_create_conversation_state(session, service)
    prepared = await asyncio.to_thread(service.prepare_answer, raw_query, None, state)

    answer_chunks: list[str] = []
    stream = service.stream_answer(prepared)

    if has_chainlit_context():
        assistant_message = message_factory(content="")
        async with cl.Step(name="💭 Thinking", type="llm") as thinking_step:
            thinking_step.language = "markdown"
            is_thinking = False

            while True:
                chunk, is_done = await asyncio.to_thread(_next_stream_chunk, stream)
                if is_done:
                    break

                if "<think>" in chunk:
                    is_thinking = True
                    chunk = chunk.replace("<think>", "")
                if "</think>" in chunk:
                    is_thinking = False
                    chunk = chunk.replace("</think>", "")

                if not chunk:
                    continue

                if is_thinking:
                    await thinking_step.stream_token(chunk)
                else:
                    answer_chunks.append(chunk)
                    await assistant_message.stream_token(chunk)
                await asyncio.sleep(0)
    else:
        assistant_message = message_factory(content="")
        await assistant_message.send()
        while True:
            chunk, is_done = await asyncio.to_thread(_next_stream_chunk, stream)
            if is_done:
                break
            answer_chunks.append(chunk)
            await assistant_message.stream_token(chunk)
            await asyncio.sleep(0)

    answer = await asyncio.to_thread(service.finalize_answer, prepared, "".join(answer_chunks))
    conversation_turn = await asyncio.to_thread(service.build_conversation_turn, answer)
    state.add_turn(conversation_turn)
    session.set("conversation_state", state)

    message_metadata = getattr(assistant_message, "metadata", None)
    if not isinstance(message_metadata, dict):
        message_metadata = {}
    message_metadata["conversation_turn"] = serialize_conversation_turn(conversation_turn)
    assistant_message.metadata = message_metadata

    assistant_message.content = format_answer_message(answer)
    await assistant_message.update()


if cl is not None:
    @cl.data_layer
    def data_layer():
        return get_data_layer_instance()


    @cl.password_auth_callback
    async def password_auth_callback(username: str, password: str):
        return await authenticate_local_user(username, password)


    @cl.on_chat_start
    async def start() -> None:
        release_session_service(cl.user_session)
        service = build_service()
        cl.user_session.set("service", service)
        cl.user_session.set(
            "conversation_state",
            ConversationState(max_turns=service.config.runtime.max_history_turns),
        )
        await bind_thread_to_current_user()


    @cl.on_chat_resume
    async def on_chat_resume(thread: dict) -> None:
        release_session_service(cl.user_session)
        service = build_service()
        cl.user_session.set("service", service)
        cl.user_session.set(
            "conversation_state",
            rebuild_conversation_state_from_thread(
                thread,
                max_turns=service.config.runtime.max_history_turns,
            ),
        )
        await bind_thread_to_current_user()


    @cl.on_message
    async def on_message(message: "cl.Message") -> None:
        service = cl.user_session.get("service")
        if service is None:
            service = build_service()
            cl.user_session.set("service", service)
        if cl.user_session.get("conversation_state") is None:
            cl.user_session.set(
                "conversation_state",
                ConversationState(max_turns=service.config.runtime.max_history_turns),
            )
        await bind_thread_to_current_user()
        await process_user_message(
            message,
            service=service,
            session=cl.user_session,
            message_factory=cl.Message,
        )


    @cl.on_chat_end
    async def on_chat_end() -> None:
        release_session_service(cl.user_session)


    @cl.on_stop
    async def on_stop() -> None:
        release_session_service(cl.user_session)


def main() -> None:
    configure_logging()
    service = build_service()
    try:
        print(build_startup_message(service))
    finally:
        close_service_instance(service)


if __name__ == "__main__":
    main()
