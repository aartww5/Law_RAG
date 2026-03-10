# Multi-Turn Context And Streaming Design

**Date:** 2026-03-09
**Status:** Approved

## Goal

Extend `unified_app` so that:

- follow-up questions such as `他侄子能继承吗` are rewritten into self-contained legal queries before retrieval
- retrieval uses the rewritten query every turn, without threshold gating
- rewrite details are logged only on the backend and never shown in the Chainlit UI
- answer text is streamed to the frontend while route and citations remain grounded and stable

## Current State

The current application has two relevant limitations:

1. Each message is handled as an isolated query.
   - `app.py` stores only the `service` in Chainlit session state
   - `services.py` accepts only a single `question` string and has no conversation history input
   - retrieval therefore cannot resolve pronouns or elliptical follow-up questions

2. The frontend is non-streaming.
   - `app.py` waits for a complete answer string
   - `llm.py` exposes only a synchronous non-streaming generation path
   - Chainlit receives one final message instead of streamed tokens

## Decision Summary

The implementation will use a "rewrite before retrieval" architecture:

- every user turn always runs through query rewrite
- rewrite uses only a short recent conversation window
- all retrieval paths consume the rewritten query, not the raw query
- generation consumes both the rewritten query result set and a compact conversation summary
- rewrite metadata is logged to backend only
- streamed output is limited to answer body text; route and citations are appended once at the end

## Architecture

### New Conversation Objects

The application will add lightweight conversation state types under `src/legal_rag/`:

- `ConversationTurn`
  - `user_query`
  - `rewritten_query`
  - `answer_summary`
  - `citations`
  - `route_mode`

- `ConversationState`
  - bounded list of recent `ConversationTurn`
  - helper methods for appending new turns
  - helper method for building compact rewrite context

- `RewriteResult`
  - `original_query`
  - `rewritten_query`
  - `rewrite_notes`
  - `changed`

These objects keep state explicit and prevent `app.py` from becoming a string-concatenation dump.

### Query Rewrite Layer

Add a rewrite module that always runs before retrieval.

Inputs:

- current raw user query
- recent conversation state, capped to the last 2-4 turns

Outputs:

- rewritten self-contained query
- rewrite notes for backend logging

Rules:

- rewrite executes on every turn
- no confidence threshold and no skip heuristic
- if no substantive change is needed, return the original query unchanged with `changed=False`
- rewrite output is used as the retrieval query for `exact`, `hybrid`, and `mini`

### Retrieval Flow

The service flow becomes:

1. receive raw user query
2. run rewrite with recent conversation state
3. log rewrite result on backend only
4. run `ExactMatchRetriever` on rewritten query
5. run `HybridRetriever` on rewritten query
6. optionally run `MiniRetriever` on rewritten query
7. build grounded answer context from retrieved docs
8. generate answer using:
   - raw user query for UI-facing naturalness when needed
   - rewritten query for retrieval diagnostics
   - compact conversation summary

This keeps RAG grounded while still allowing follow-up questions to inherit context.

### Backend Logging Boundary

Rewrite information is never shown in the frontend message body.

Allowed sinks:

- console logger
- file logger if configured later

Logged fields:

- original query
- rewritten query
- rewrite notes
- selected route

Disallowed sinks:

- Chainlit message content
- citation section
- route display section shown to users

### Streaming Output Boundary

Streaming is generation-only.

The system will not stream retrieval progress or intermediate rewrite output.

Frontend flow:

1. receive user message
2. perform rewrite and retrieval synchronously in backend
3. create an empty Chainlit assistant message
4. stream answer body tokens as they are generated
5. after token stream completes, append:
   - route info
   - citations

This avoids UI flicker and prevents partially streamed route metadata from changing under the user.

### Generator Split

`SimpleGenerator` will expose two paths:

- `generate(...)` for tests and non-streaming use
- `stream_generate(...)` for Chainlit runtime

Both paths must share:

- prompt construction
- retrieval context usage
- fallback behavior

This prevents streaming and non-streaming answers from drifting semantically.

## Data Flow

### Follow-Up Query Example

Turn 1:

- raw query: `个体户老王去世后留下遗产，无人继承或受遗赠，这些遗产归谁？`
- rewritten query: same as raw or lightly normalized
- citations: include `民法典` inheritance provisions

Turn 2:

- raw query: `他侄子能继承吗`
- rewrite reads recent turn state
- rewritten query becomes something like:
  - `老王去世后，其侄子是否属于法定继承人，能否继承其遗产？`
- retrieval now operates on a legally meaningful query instead of an isolated pronoun

### Streaming Example

After retrieval completes, Chainlit sees token streaming only for the answer body. Once complete, the app appends:

- `[route] ...`
- `[citations] ...`

So the user experience remains responsive without leaking backend rewrite operations.

## Error Handling

### Rewrite Failure

If rewrite model logic fails:

- log the exception
- fall back to the raw user query
- continue retrieval normally

This keeps rewrite additive rather than becoming a hard dependency.

### Streaming Failure

If streaming generation fails after retrieval:

- fall back to non-streaming generation
- still return a complete final answer

### Mini Failure

Existing `mini_failed` degradation remains in place. Rewrite does not change the fallback contract; it only improves the quality of the query sent into retrieval.

## Testing Strategy

Add regression coverage for:

- query rewrite always executing
- follow-up pronoun question becoming self-contained
- retrieval using rewritten query rather than raw query
- rewrite metadata remaining absent from frontend message content
- streamed generation emitting answer body incrementally
- route and citations being appended only after stream completion

## Files Expected To Change

- `unified_app/app.py`
- `unified_app/src/legal_rag/types.py`
- `unified_app/src/legal_rag/services.py`
- `unified_app/src/legal_rag/config.py`
- `unified_app/src/legal_rag/generation/llm.py`
- `unified_app/src/legal_rag/generation/prompts.py`
- `unified_app/src/legal_rag/generation/context_builder.py`
- `unified_app/src/legal_rag/` new conversation or rewrite modules
- `unified_app/tests/test_app_entrypoint.py`
- `unified_app/tests/test_service_flow.py`
- new rewrite and streaming tests

## Non-Goals

- no full long-term memory store
- no user-visible rewrite panel
- no streaming retrieval visualization
- no rewrite-threshold tuning logic
- no change to citation rendering semantics beyond timing of final append
