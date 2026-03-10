# Multi-Turn Context And Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add always-on query rewrite, short-window conversation state, backend-only rewrite logging, and streaming answer output to `unified_app`.

**Architecture:** Keep retrieval and generation layered. Introduce a lightweight conversation state plus rewrite step before retrieval, then add a streaming generation path that reuses the same prompt and fallback logic as the current synchronous generator. Chainlit remains a thin adapter that stores session state and streams answer body tokens only.

**Tech Stack:** Python, pytest, Chainlit, Ollama

---

### Task 1: Add Conversation State Types

**Files:**
- Modify: `unified_app/src/legal_rag/types.py`
- Test: `unified_app/tests/test_service_flow.py`

**Step 1: Write the failing test**

Add a test proving conversation turns can store:

- raw query
- rewritten query
- answer summary
- citations

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k conversation_state -q`

Expected: FAIL because the conversation state types do not exist yet.

**Step 3: Write minimal implementation**

Add dataclasses for:

- `ConversationTurn`
- `ConversationState`
- `RewriteResult`

Keep `ConversationState` bounded to recent turns only.

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k conversation_state -q`

Expected: PASS

### Task 2: Add Always-On Query Rewrite

**Files:**
- Create: `unified_app/src/legal_rag/rewrite.py`
- Modify: `unified_app/src/legal_rag/config.py`
- Test: `unified_app/tests/test_service_flow.py`

**Step 1: Write the failing test**

Add a test for a follow-up query:

- prior turn contains `老王去世后留下遗产...`
- current query is `他侄子能继承吗`
- rewrite result must contain `老王` or equivalent explicit inheritance context

Also add a test proving rewrite runs every turn, even when no change is needed.

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k rewrite -q`

Expected: FAIL because rewrite module does not exist yet.

**Step 3: Write minimal implementation**

Implement a small rewrite service:

- input: raw query + recent conversation state
- output: `RewriteResult`
- initial strategy can be deterministic and prompt-free if possible; if an LLM-based rewrite is needed later, keep the interface stable
- return unchanged query when no edit is needed, but still produce a `RewriteResult`

Add config for max remembered turns if needed.

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k rewrite -q`

Expected: PASS

### Task 3: Route Retrieval Through Rewritten Query

**Files:**
- Modify: `unified_app/src/legal_rag/services.py`
- Modify: `unified_app/src/legal_rag/generation/context_builder.py`
- Test: `unified_app/tests/test_service_flow.py`

**Step 1: Write the failing test**

Add a test proving:

- `handle_message(raw_query, conversation_state=...)`
- retrieval uses `rewritten_query`
- returned citations correspond to the rewritten legal issue rather than the isolated raw query

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k rewritten_query_used -q`

Expected: FAIL because `services.py` currently accepts only one raw `question`.

**Step 3: Write minimal implementation**

Update service flow to:

- accept conversation state
- run rewrite first
- pass `rewritten_query` into `exact`, `hybrid`, and `mini`
- preserve raw query for UI and summarization only

Ensure existing fallback behavior still works.

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k rewritten_query_used -q`

Expected: PASS

### Task 4: Add Backend-Only Rewrite Logging

**Files:**
- Modify: `unified_app/src/legal_rag/services.py`
- Modify: `unified_app/app.py`
- Test: `unified_app/tests/test_app_entrypoint.py`

**Step 1: Write the failing test**

Add a test proving frontend formatting does not include:

- original query / rewritten query debug fields
- rewrite notes

Add a service-level test proving rewrite information is exposed for logging hooks or logger calls.

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_app_entrypoint.py -k rewrite -q`

Expected: FAIL because rewrite metadata is not separated yet.

**Step 3: Write minimal implementation**

Add structured logging in backend only:

- original query
- rewritten query
- rewrite notes
- route decision

Do not include rewrite metadata in Chainlit message content.

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_app_entrypoint.py -k rewrite -q`

Expected: PASS

### Task 5: Add Streaming Generator Interface

**Files:**
- Modify: `unified_app/src/legal_rag/generation/llm.py`
- Test: `unified_app/tests/test_service_flow.py`

**Step 1: Write the failing test**

Add a test for `stream_generate(...)` that:

- yields incremental text chunks
- reuses the same prompt inputs as non-streaming generation
- falls back to non-streaming output if streaming backend is unavailable

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k stream_generate -q`

Expected: FAIL because no streaming generator path exists.

**Step 3: Write minimal implementation**

Add:

- `generate(...)`
- `stream_generate(...)`

with shared prompt construction and fallback logic.

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_service_flow.py -k stream_generate -q`

Expected: PASS

### Task 6: Wire Chainlit Message Streaming

**Files:**
- Modify: `unified_app/app.py`
- Test: `unified_app/tests/test_app_entrypoint.py`

**Step 1: Write the failing test**

Add an entrypoint test proving:

- Chainlit session stores conversation state
- answer body is streamed token-by-token
- route and citations are appended after stream completion

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_app_entrypoint.py -k stream -q`

Expected: FAIL because `app.py` currently sends one final message only.

**Step 3: Write minimal implementation**

Change `on_message` to:

- load conversation state from `cl.user_session`
- call service rewrite and retrieval
- create an empty assistant message
- stream answer body tokens
- update or finalize the message with route and citations after stream completion
- append the completed turn back into conversation state

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_app_entrypoint.py -k stream -q`

Expected: PASS

### Task 7: End-To-End Follow-Up Regression

**Files:**
- Modify: `unified_app/tests/test_service_flow.py`
- Modify: `unified_app/tests/test_app_entrypoint.py`

**Step 1: Write the failing test**

Add an end-to-end regression for:

1. ask about `老王` inheritance
2. ask `他侄子能继承吗`
3. confirm the second turn uses inherited context and produces inheritance-related citations rather than unrelated law hits

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_service_flow.py unified_app/tests/test_app_entrypoint.py -k follow_up -q`

Expected: FAIL before the full flow is wired.

**Step 3: Write minimal implementation**

Only the glue needed after Tasks 1-6.

**Step 4: Run test to verify it passes**

Run: `python -m pytest unified_app/tests/test_service_flow.py unified_app/tests/test_app_entrypoint.py -k follow_up -q`

Expected: PASS

### Task 8: Final Verification

**Files:**
- No code changes expected

**Step 1: Run focused regression tests**

Run:

```bash
python -m pytest unified_app/tests/test_app_entrypoint.py unified_app/tests/test_service_flow.py -q
```

Expected: PASS

**Step 2: Run full test suite**

Run:

```bash
python -m pytest unified_app/tests -q
```

Expected: PASS

**Step 3: Run a manual smoke scenario**

Use Chainlit or service smoke flow to confirm:

1. ask the first inheritance question
2. ask the pronoun follow-up
3. verify backend logs show rewrite details
4. verify frontend does not show rewrite details
5. verify answer text streams progressively
