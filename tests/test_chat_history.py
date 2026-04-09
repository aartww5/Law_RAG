import asyncio
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chainlit.types import Pagination, ThreadFilter
from chainlit.user import User

from legal_rag.chat_history import SQLiteChatHistoryDataLayer


def run(coro):
    return asyncio.run(coro)


def test_chat_history_persists_users_threads_and_steps(tmp_path: Path) -> None:
    data_layer = SQLiteChatHistoryDataLayer(tmp_path / "chat.sqlite3")

    persisted = run(
        data_layer.create_user(
            User(identifier="alice", display_name="Alice", metadata={"role": "demo"})
        )
    )
    assert persisted is not None

    run(
        data_layer.update_thread(
            thread_id="thread-1",
            name="Inheritance question",
            user_id=persisted.id,
            metadata={"topic": "inheritance"},
            tags=["law"],
        )
    )
    run(
        data_layer.create_step(
            {
                "id": "step-user-1",
                "threadId": "thread-1",
                "type": "user_message",
                "name": "user",
                "input": "",
                "output": "他侄子能继承吗",
                "createdAt": "2026-04-08T12:00:00.000000Z",
                "metadata": {},
            }
        )
    )
    run(
        data_layer.create_step(
            {
                "id": "step-assistant-1",
                "threadId": "thread-1",
                "type": "assistant_message",
                "name": "assistant",
                "input": "",
                "output": "一般不能。",
                "createdAt": "2026-04-08T12:00:01.000000Z",
                "metadata": {"conversation_turn": {"raw_query": "他侄子能继承吗"}},
            }
        )
    )

    thread = run(data_layer.get_thread("thread-1"))

    assert thread is not None
    assert thread["name"] == "Inheritance question"
    assert thread["userIdentifier"] == "alice"
    assert [step["id"] for step in thread["steps"]] == ["step-user-1", "step-assistant-1"]
    assert thread["steps"][-1]["metadata"]["conversation_turn"]["raw_query"] == "他侄子能继承吗"


def test_chat_history_lists_only_threads_for_requested_user(tmp_path: Path) -> None:
    data_layer = SQLiteChatHistoryDataLayer(tmp_path / "chat.sqlite3")
    alice = run(data_layer.create_user(User(identifier="alice")))
    bob = run(data_layer.create_user(User(identifier="bob")))
    assert alice is not None
    assert bob is not None

    run(data_layer.update_thread(thread_id="thread-a", name="A", user_id=alice.id))
    run(data_layer.update_thread(thread_id="thread-b", name="B", user_id=bob.id))

    page = run(
        data_layer.list_threads(
            Pagination(first=20),
            ThreadFilter(userId=alice.id),
        )
    )

    assert [thread["id"] for thread in page.data] == ["thread-a"]


def test_chat_history_delete_thread_removes_associated_steps(tmp_path: Path) -> None:
    data_layer = SQLiteChatHistoryDataLayer(tmp_path / "chat.sqlite3")
    persisted = run(data_layer.create_user(User(identifier="alice")))
    assert persisted is not None

    run(data_layer.update_thread(thread_id="thread-1", user_id=persisted.id))
    run(
        data_layer.create_step(
            {
                "id": "step-1",
                "threadId": "thread-1",
                "type": "assistant_message",
                "name": "assistant",
                "output": "demo",
                "createdAt": "2026-04-08T12:00:00.000000Z",
                "metadata": {},
            }
        )
    )

    run(data_layer.delete_thread("thread-1"))

    assert run(data_layer.get_thread("thread-1")) is None
    page = run(
        data_layer.list_threads(
            Pagination(first=20),
            ThreadFilter(userId=persisted.id),
        )
    )
    assert page.data == []
