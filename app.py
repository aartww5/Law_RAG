"""Unified legal RAG application entrypoint."""

import logging
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = CURRENT_DIR / "src"
LOG_DIR = CURRENT_DIR / "logs"
LOG_FILE = LOG_DIR / "unified_app.log"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.config import AppConfig
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


def build_service() -> LegalAssistantService:
    configure_logging()
    config = AppConfig.from_env(PROJECT_ROOT)
    if config.index.laws_dir.exists():
        return LegalAssistantService.from_config(config)
    return LegalAssistantService.for_test(mode=config.runtime.mode)


def build_startup_message(service: LegalAssistantService) -> str:
    return (
        "Unified legal assistant ready "
        f"(mode={service.config.runtime.mode}, "
        f"mini_available={getattr(service, 'mini_available', False)})"
    )


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
    prepared = service.prepare_answer(raw_query, conversation_state=state)

    answer_chunks: list[str] = []

    if cl is not None:
        assistant_message = message_factory(content="")
        async with cl.Step(name="💭 Thinking", type="llm") as thinking_step:
            thinking_step.language = "markdown"
            is_thinking = False

            for chunk in service.stream_answer(prepared):
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
    else:
        assistant_message = message_factory(content="")
        await assistant_message.send()
        for chunk in service.stream_answer(prepared):
            answer_chunks.append(chunk)
            await assistant_message.stream_token(chunk)

    answer = service.finalize_answer(prepared, "".join(answer_chunks))
    state.add_turn(service.build_conversation_turn(answer))
    session.set("conversation_state", state)

    assistant_message.content = format_answer_message(answer)
    await assistant_message.update()


if cl is not None:
    @cl.on_chat_start
    async def start() -> None:
        service = build_service()
        cl.user_session.set("service", service)
        cl.user_session.set(
            "conversation_state",
            ConversationState(max_turns=service.config.runtime.max_history_turns),
        )
        await cl.Message(content=build_startup_message(service)).send()


    @cl.on_message
    async def on_message(message: "cl.Message") -> None:
        service = cl.user_session.get("service")
        if service is None:
            service = build_service()
            cl.user_session.set("service", service)
        await process_user_message(
            message,
            service=service,
            session=cl.user_session,
            message_factory=cl.Message,
        )


def main() -> None:
    configure_logging()
    service = build_service()
    print(build_startup_message(service))


if __name__ == "__main__":
    main()
