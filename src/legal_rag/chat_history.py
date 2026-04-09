import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chainlit.data.base import BaseDataLayer
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.types import Feedback, FeedbackDict, PageInfo, PaginatedResponse, Pagination, ThreadDict, ThreadFilter
from chainlit.user import PersistedUser, User


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class SQLiteChatHistoryDataLayer(BaseDataLayer):
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    identifier TEXT NOT NULL UNIQUE,
                    displayName TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    createdAt TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL,
                    name TEXT,
                    userId TEXT,
                    userIdentifier TEXT,
                    tags TEXT,
                    metadata TEXT,
                    FOREIGN KEY(userId) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    threadId TEXT NOT NULL,
                    name TEXT,
                    type TEXT NOT NULL,
                    parentId TEXT,
                    streaming INTEGER,
                    waitForAnswer INTEGER,
                    isError INTEGER,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    tags TEXT,
                    input TEXT,
                    output TEXT,
                    createdAt TEXT NOT NULL,
                    start TEXT,
                    end TEXT,
                    generation TEXT,
                    showInput TEXT,
                    language TEXT,
                    FOREIGN KEY(threadId) REFERENCES threads(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedbacks (
                    id TEXT PRIMARY KEY,
                    forId TEXT NOT NULL,
                    value REAL,
                    comment TEXT,
                    FOREIGN KEY(forId) REFERENCES steps(id) ON DELETE CASCADE
                );
                """
            )

    def _get_user_sync(self, identifier: str) -> PersistedUser | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT id, identifier, displayName, metadata, createdAt FROM users WHERE identifier = ?',
                (identifier,),
            ).fetchone()
        if row is None:
            return None
        return PersistedUser(
            id=row["id"],
            identifier=row["identifier"],
            display_name=row["displayName"],
            metadata=_json_loads(row["metadata"], {}),
            createdAt=row["createdAt"],
        )

    async def get_user(self, identifier: str) -> PersistedUser | None:
        return await asyncio.to_thread(self._get_user_sync, identifier)

    def _create_user_sync(self, user: User) -> PersistedUser:
        now = _utc_now()
        with self._connect() as connection:
            existing = connection.execute(
                'SELECT id, createdAt FROM users WHERE identifier = ?',
                (user.identifier,),
            ).fetchone()
            if existing is None:
                user_id = str(uuid.uuid4())
                created_at = now
                connection.execute(
                    """
                    INSERT INTO users (id, identifier, displayName, metadata, createdAt)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        user.identifier,
                        user.display_name,
                        _json_dumps(user.metadata),
                        created_at,
                    ),
                )
            else:
                user_id = existing["id"]
                created_at = existing["createdAt"]
                connection.execute(
                    """
                    UPDATE users
                    SET displayName = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (
                        user.display_name,
                        _json_dumps(user.metadata),
                        user_id,
                    ),
                )
            connection.commit()
        return PersistedUser(
            id=user_id,
            identifier=user.identifier,
            display_name=user.display_name,
            metadata=user.metadata,
            createdAt=created_at,
        )

    async def create_user(self, user: User) -> PersistedUser | None:
        return await asyncio.to_thread(self._create_user_sync, user)

    async def delete_feedback(self, feedback_id: str) -> bool:
        def _delete() -> bool:
            with self._connect() as connection:
                connection.execute("DELETE FROM feedbacks WHERE id = ?", (feedback_id,))
                connection.commit()
            return True

        return await asyncio.to_thread(_delete)

    async def upsert_feedback(self, feedback: Feedback) -> str:
        def _upsert() -> str:
            feedback_id = feedback.id or str(uuid.uuid4())
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedbacks (id, forId, value, comment)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE
                    SET value = excluded.value, comment = excluded.comment
                    """,
                    (feedback_id, feedback.forId, float(feedback.value), feedback.comment),
                )
                connection.commit()
            return feedback_id

        return await asyncio.to_thread(_upsert)

    async def create_element(self, element: Element):
        return None

    async def get_element(self, thread_id: str, element_id: str) -> ElementDict | None:
        return None

    async def delete_element(self, element_id: str, thread_id: str | None = None):
        return None

    def _get_user_identifier_by_id(self, connection: sqlite3.Connection, user_id: str | None) -> str | None:
        if user_id is None:
            return None
        row = connection.execute(
            "SELECT identifier FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return row["identifier"]

    def _update_thread_sync(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as connection:
            existing = connection.execute(
                'SELECT createdAt, name, userId, userIdentifier, metadata, tags FROM threads WHERE id = ?',
                (thread_id,),
            ).fetchone()
            created_at = existing["createdAt"] if existing else now
            existing_metadata = _json_loads(existing["metadata"], {}) if existing else {}
            existing_tags = _json_loads(existing["tags"], None) if existing else None
            merged_metadata = existing_metadata if metadata is None else {**existing_metadata, **metadata}
            resolved_name = name if name is not None else (existing["name"] if existing else None)
            resolved_user_id = user_id if user_id is not None else (existing["userId"] if existing else None)
            resolved_user_identifier = self._get_user_identifier_by_id(connection, resolved_user_id)
            if resolved_user_identifier is None and existing:
                resolved_user_identifier = existing["userIdentifier"]
            resolved_tags = tags if tags is not None else existing_tags

            connection.execute(
                """
                INSERT INTO threads (id, createdAt, updatedAt, name, userId, userIdentifier, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE
                SET updatedAt = excluded.updatedAt,
                    name = excluded.name,
                    userId = excluded.userId,
                    userIdentifier = excluded.userIdentifier,
                    tags = excluded.tags,
                    metadata = excluded.metadata
                """,
                (
                    thread_id,
                    created_at,
                    now,
                    resolved_name,
                    resolved_user_id,
                    resolved_user_identifier,
                    _json_dumps(resolved_tags) if resolved_tags is not None else None,
                    _json_dumps(merged_metadata) if merged_metadata else None,
                ),
            )
            connection.commit()

    async def get_thread_author(self, thread_id: str) -> str:
        def _get_author() -> str:
            with self._connect() as connection:
                row = connection.execute(
                    'SELECT userIdentifier FROM threads WHERE id = ?',
                    (thread_id,),
                ).fetchone()
            if row is None or not row["userIdentifier"]:
                raise ValueError(f"Author not found for thread_id {thread_id}")
            return row["userIdentifier"]

        return await asyncio.to_thread(_get_author)

    async def delete_thread(self, thread_id: str):
        def _delete() -> None:
            with self._connect() as connection:
                connection.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
                connection.commit()

        await asyncio.to_thread(_delete)

    def _build_thread_dict(self, connection: sqlite3.Connection, thread_row: sqlite3.Row) -> ThreadDict:
        feedback_rows = connection.execute(
            "SELECT id, forId, value, comment FROM feedbacks"
        ).fetchall()
        feedback_by_step = {
            row["forId"]: FeedbackDict(
                id=row["id"],
                forId=row["forId"],
                value=row["value"],
                comment=row["comment"],
            )
            for row in feedback_rows
        }
        step_rows = connection.execute(
            """
            SELECT id, name, type, threadId, parentId, streaming, waitForAnswer, isError,
                   metadata, tags, input, output, createdAt, start, end, generation, showInput, language
            FROM steps
            WHERE threadId = ?
            ORDER BY createdAt ASC, rowid ASC
            """,
            (thread_row["id"],),
        ).fetchall()
        steps: list[StepDict] = []
        for row in step_rows:
            steps.append(
                StepDict(
                    id=row["id"],
                    name=row["name"],
                    type=row["type"],
                    threadId=row["threadId"],
                    parentId=row["parentId"],
                    streaming=bool(row["streaming"]) if row["streaming"] is not None else False,
                    waitForAnswer=bool(row["waitForAnswer"]) if row["waitForAnswer"] is not None else None,
                    isError=bool(row["isError"]) if row["isError"] is not None else None,
                    metadata=_json_loads(row["metadata"], {}),
                    tags=_json_loads(row["tags"], None),
                    input=row["input"] or "",
                    output=row["output"] or "",
                    createdAt=row["createdAt"],
                    start=row["start"],
                    end=row["end"],
                    generation=_json_loads(row["generation"], None),
                    showInput=row["showInput"],
                    language=row["language"],
                    feedback=feedback_by_step.get(row["id"]),
                )
            )
        return ThreadDict(
            id=thread_row["id"],
            createdAt=thread_row["createdAt"],
            name=thread_row["name"],
            userId=thread_row["userId"],
            userIdentifier=thread_row["userIdentifier"],
            tags=_json_loads(thread_row["tags"], None),
            metadata=_json_loads(thread_row["metadata"], {}),
            steps=steps,
            elements=[],
        )

    def _get_thread_sync(self, thread_id: str) -> ThreadDict | None:
        with self._connect() as connection:
            thread_row = connection.execute(
                """
                SELECT id, createdAt, updatedAt, name, userId, userIdentifier, tags, metadata
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            ).fetchone()
            if thread_row is None:
                return None
            return self._build_thread_dict(connection, thread_row)

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        def _list() -> PaginatedResponse[ThreadDict]:
            if not filters.userId:
                raise ValueError("userId is required")
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, createdAt, updatedAt, name, userId, userIdentifier, tags, metadata
                    FROM threads
                    WHERE userId = ?
                    ORDER BY updatedAt DESC, createdAt DESC
                    """,
                    (filters.userId,),
                ).fetchall()
                threads = [self._build_thread_dict(connection, row) for row in rows]

            if filters.search:
                keyword = filters.search.lower()
                threads = [
                    thread
                    for thread in threads
                    if any(keyword in (step.get("output") or "").lower() for step in thread["steps"])
                ]
            if filters.feedback is not None:
                feedback_value = int(filters.feedback)
                threads = [
                    thread
                    for thread in threads
                    if any(
                        step.get("feedback") and int(step["feedback"]["value"]) == feedback_value
                        for step in thread["steps"]
                    )
                ]

            start = 0
            if pagination.cursor:
                for index, thread in enumerate(threads):
                    if thread["id"] == pagination.cursor:
                        start = index + 1
                        break
            data = threads[start : start + pagination.first]
            return PaginatedResponse(
                pageInfo=PageInfo(
                    hasNextPage=len(threads) > start + pagination.first,
                    startCursor=data[0]["id"] if data else None,
                    endCursor=data[-1]["id"] if data else None,
                ),
                data=data,
            )

        return await asyncio.to_thread(_list)

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        return await asyncio.to_thread(self._get_thread_sync, thread_id)

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ):
        await asyncio.to_thread(self._update_thread_sync, thread_id, name, user_id, metadata, tags)

    async def create_step(self, step_dict: StepDict):
        def _create() -> None:
            self._update_thread_sync(step_dict["threadId"])
            created_at = step_dict.get("createdAt") or _utc_now()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO steps (
                        id, threadId, name, type, parentId, streaming, waitForAnswer, isError,
                        metadata, tags, input, output, createdAt, start, end, generation, showInput, language
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE
                    SET threadId = excluded.threadId,
                        name = excluded.name,
                        type = excluded.type,
                        parentId = excluded.parentId,
                        streaming = excluded.streaming,
                        waitForAnswer = excluded.waitForAnswer,
                        isError = excluded.isError,
                        metadata = excluded.metadata,
                        tags = excluded.tags,
                        input = excluded.input,
                        output = excluded.output,
                        createdAt = excluded.createdAt,
                        start = excluded.start,
                        end = excluded.end,
                        generation = excluded.generation,
                        showInput = excluded.showInput,
                        language = excluded.language
                    """,
                    (
                        step_dict["id"],
                        step_dict["threadId"],
                        step_dict.get("name"),
                        step_dict["type"],
                        step_dict.get("parentId"),
                        int(bool(step_dict.get("streaming"))) if step_dict.get("streaming") is not None else None,
                        int(bool(step_dict.get("waitForAnswer"))) if step_dict.get("waitForAnswer") is not None else None,
                        int(bool(step_dict.get("isError"))) if step_dict.get("isError") is not None else None,
                        _json_dumps(step_dict.get("metadata", {})),
                        _json_dumps(step_dict.get("tags")) if step_dict.get("tags") is not None else None,
                        step_dict.get("input", ""),
                        step_dict.get("output", ""),
                        created_at,
                        step_dict.get("start"),
                        step_dict.get("end"),
                        _json_dumps(step_dict.get("generation")) if step_dict.get("generation") is not None else None,
                        step_dict.get("showInput"),
                        step_dict.get("language"),
                    ),
                )
                connection.execute(
                    'UPDATE threads SET updatedAt = ? WHERE id = ?',
                    (created_at, step_dict["threadId"]),
                )
                connection.commit()

        await asyncio.to_thread(_create)

    async def update_step(self, step_dict: StepDict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        def _delete() -> None:
            with self._connect() as connection:
                connection.execute("DELETE FROM steps WHERE id = ?", (step_id,))
                connection.commit()

        await asyncio.to_thread(_delete)

    async def build_debug_url(self) -> str:
        return str(self.db_path)

    async def close(self) -> None:
        return None

    async def get_favorite_steps(self, user_id: str) -> list[StepDict]:
        def _get_favorites() -> list[StepDict]:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT s.id, s.name, s.type, s.threadId, s.parentId, s.streaming, s.waitForAnswer, s.isError,
                           s.metadata, s.tags, s.input, s.output, s.createdAt, s.start, s.end, s.generation, s.showInput, s.language
                    FROM steps s
                    INNER JOIN threads t ON t.id = s.threadId
                    WHERE t.userId = ?
                    ORDER BY s.createdAt ASC, s.rowid ASC
                    """,
                    (user_id,),
                ).fetchall()
            favorites: list[StepDict] = []
            for row in rows:
                metadata = _json_loads(row["metadata"], {})
                if metadata.get("favorite"):
                    favorites.append(
                        StepDict(
                            id=row["id"],
                            name=row["name"],
                            type=row["type"],
                            threadId=row["threadId"],
                            parentId=row["parentId"],
                            streaming=bool(row["streaming"]) if row["streaming"] is not None else False,
                            waitForAnswer=bool(row["waitForAnswer"]) if row["waitForAnswer"] is not None else None,
                            isError=bool(row["isError"]) if row["isError"] is not None else None,
                            metadata=metadata,
                            tags=_json_loads(row["tags"], None),
                            input=row["input"] or "",
                            output=row["output"] or "",
                            createdAt=row["createdAt"],
                            start=row["start"],
                            end=row["end"],
                            generation=_json_loads(row["generation"], None),
                            showInput=row["showInput"],
                            language=row["language"],
                        )
                    )
            return favorites

        return await asyncio.to_thread(_get_favorites)
