# Local Auth And Chat History Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add local password login plus persistent chat threads that restore the existing rewrite-aware conversation context.

**Architecture:** Keep the current Chainlit frontend, add auth and persistence through a local SQLite-backed data layer, and rebuild `ConversationState` from persisted assistant-step metadata during thread resume. This preserves the existing follow-up query rewrite behavior instead of treating history as UI-only text.

**Tech Stack:** Python, Chainlit 2.10, SQLite, stdlib `hashlib`/`sqlite3`, pytest

---

### Task 1: Extend Configuration For Auth And Chat Storage

**Files:**
- Modify: `src/legal_rag/config.py`
- Modify: `config.toml`
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

Add tests covering:

- parsing `storage.chat_db_path`
- parsing `auth.users`
- preferring configured auth values without breaking existing runtime/index defaults

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py`
Expected: FAIL because storage/auth config fields do not exist yet.

**Step 3: Write minimal implementation**

Add config dataclasses and parsing helpers for:

- local auth users
- chat DB path
- password/plain-text auth fields

Update the sample `config.toml` with a local-only auth/storage section.

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/legal_rag/config.py config.toml tests/test_config.py
git commit -m "feat: add local auth and chat storage config"
```

### Task 2: Add A SQLite Chainlit Data Layer

**Files:**
- Create: `src/legal_rag/chat_history.py`
- Test: `tests/test_chat_history.py`

**Step 1: Write the failing tests**

Add tests covering:

- upserting users
- creating/updating threads
- storing steps in order
- listing threads for one user only
- deleting a thread and its steps

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_chat_history.py`
Expected: FAIL because the module does not exist.

**Step 3: Write minimal implementation**

Implement a minimal `BaseDataLayer` subclass using `sqlite3` with JSON text columns and async wrappers via `asyncio.to_thread` where needed.

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_chat_history.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/legal_rag/chat_history.py tests/test_chat_history.py
git commit -m "feat: add sqlite chainlit history layer"
```

### Task 3: Add Local Login And Resume-Aware Context Restoration

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_entrypoint.py`

**Step 1: Write the failing tests**

Add tests covering:

- successful local password authentication
- failed authentication for bad credentials
- `on_chat_resume` rebuilding `ConversationState`
- `process_user_message` attaching serialized conversation-turn metadata needed for persistence

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_app_entrypoint.py`
Expected: FAIL because auth callback, resume hook, and persistence metadata do not exist.

**Step 3: Write minimal implementation**

In `app.py`:

- initialize config and SQLite data layer
- register `@cl.data_layer`
- register `@cl.password_auth_callback`
- add helpers to serialize/deserialize `ConversationTurn`
- persist turn payload on assistant messages
- rebuild `ConversationState` in `@cl.on_chat_resume`

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_app_entrypoint.py`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app_entrypoint.py
git commit -m "feat: add local auth and context-aware chat resume"
```

### Task 4: Verify Regression Coverage

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_app_entrypoint.py`
- Modify: `tests/test_chat_history.py`

**Step 1: Run focused feature tests**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_chat_history.py tests/test_app_entrypoint.py`
Expected: PASS

**Step 2: Run broader regression subset**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_service_flow.py tests/test_generation_prompt.py tests/test_query_decomposition.py`
Expected: PASS and no regression in rewrite/context behavior.

**Step 3: Refactor only if all tests are green**

Clean up helper naming or small duplication without changing behavior.

**Step 4: Re-run the same tests**

Run: `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_chat_history.py tests/test_app_entrypoint.py tests/test_service_flow.py tests/test_generation_prompt.py tests/test_query_decomposition.py`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py src/legal_rag/config.py src/legal_rag/chat_history.py tests/test_config.py tests/test_chat_history.py tests/test_app_entrypoint.py
git commit -m "test: verify local auth and persistent chat history"
```
