# Local Auth And Chat History Design

**Date:** 2026-04-08
**Status:** Approved

## Goal

Extend the existing Chainlit-based frontend so that:

- users must log in with locally configured credentials
- chat threads persist across restarts
- users can view, switch, create, and delete historical conversations
- resumed threads rebuild the in-memory `ConversationState` used by the query rewrite pipeline

## Current State

- [app.py](/F:/毕设/unified_app/app.py) uses `cl.user_session` to keep `service` and `conversation_state`
- `conversation_state` exists only in memory for the current websocket session
- there is no authentication callback
- there is no persistent thread/message store backing the Chainlit UI

The context mechanism already depends on `ConversationState` and `ConversationTurn`:

- follow-up questions are rewritten using recent turns
- the rewrite result is fed into retrieval
- if old threads are only rendered in the UI and not converted back into `ConversationState`, resumed conversations will lose the retrieval context quality

## Decision Summary

The implementation will keep Chainlit as the frontend and add two local capabilities:

1. password authentication backed by accounts in `config.toml`
2. a lightweight SQLite data layer that persists Chainlit threads and steps

When a thread is resumed, the application will rebuild `ConversationState` from stored message metadata so the existing rewrite and retrieval flow keeps working.

## Architecture

### Configuration

Extend [src/legal_rag/config.py](/F:/毕设/unified_app/src/legal_rag/config.py) and [config.toml](/F:/毕设/unified_app/config.toml) with:

- `storage.chat_db_path`
- `auth.users`

Each configured user will contain:

- `username`
- `display_name`
- `password` or `password_hash`

`password_hash` will be preferred. Plain `password` support is acceptable for this local-only project and keeps configuration simple.

### Authentication

Add `@cl.password_auth_callback` in [app.py](/F:/毕设/unified_app/app.py).

Behavior:

- read configured users from `AppConfig`
- verify password against the configured record
- return `cl.User(identifier=username, display_name=display_name, metadata=...)` on success
- return `None` on failure

The callback must not leak whether the username or password was incorrect.

### SQLite Persistence

Add a new module under `src/legal_rag/` that implements a custom Chainlit `BaseDataLayer`.

SQLite tables:

- `users`
  - Chainlit user id
  - identifier
  - display name
  - metadata JSON
  - created timestamp
- `threads`
  - thread id
  - user id
  - user identifier
  - thread name
  - tags JSON
  - metadata JSON
  - created timestamp
  - updated timestamp
- `steps`
  - step id
  - thread id
  - step type
  - input/output text
  - metadata JSON
  - created/start/end timestamps

This is enough for:

- thread list rendering
- thread resume
- thread deletion
- storing user/assistant messages

Feedback and elements can remain minimal no-op implementations for now because they are not part of the requested feature set.

### Persisted Conversation Metadata

The existing answer text is not enough to rebuild `ConversationState` faithfully. Each assistant step must therefore store a compact metadata payload:

- `raw_query`
- `rewritten_query`
- `answer_summary`
- `citations`

On normal message processing:

1. run the existing rewrite/retrieval/generation flow
2. build `ConversationTurn`
3. keep it in `cl.user_session`
4. attach the same turn payload to the persisted assistant step metadata

This preserves the current context mechanism without duplicating the rewrite logic.

### Thread Resume

Add `@cl.on_chat_resume` in [app.py](/F:/毕设/unified_app/app.py).

Behavior:

1. inspect the resumed `ThreadDict`
2. walk persisted steps in creation order
3. extract assistant-step metadata containing serialized conversation turns
4. rebuild `ConversationState(max_turns=...)`
5. store the rebuilt state in `cl.user_session`
6. ensure a fresh `LegalAssistantService` is present for the resumed session

Only assistant steps carrying the turn payload should be used to reconstruct the state. This avoids relying on free-form message parsing.

### Error Handling

- invalid login returns `None`
- unreadable or malformed auth config fails fast during config loading
- SQLite initialization failure should fail startup instead of silently disabling persistence
- resume failures should log a warning and reset to an empty `ConversationState` rather than injecting partial state

## Testing Strategy

Add tests for:

- config parsing of local auth and chat storage settings
- password verification behavior
- SQLite data layer user/thread/step persistence
- thread deletion
- resumed threads rebuilding `ConversationState`
- `process_user_message` persisting the turn payload needed for follow-up rewrite context

## Non-Goals

- self-service registration
- third-party login providers
- cloud database or remote auth
- redesigning the Chainlit frontend outside small copy/config polish
