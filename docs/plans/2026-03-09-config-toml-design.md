# Config Toml Loading Design

**Date:** 2026-03-09
**Status:** Approved

## Goal

Give `unified_app` a clear project-level configuration file so users can change runtime and index settings without editing Python source, while still allowing temporary environment-variable overrides.

## Problem

The current configuration path is not obvious:

- `RuntimeConfig` looks like the source of truth
- `AppConfig.from_env()` silently overrides those defaults
- `mode` is hard-coded to `"auto"` inside `from_env()`
- there is no user-facing config file in the project directory

This makes changes in `config.py` feel like they should work, but they do not affect the runtime path used by `app.py`.

## Decision

Add `unified_app/config.toml` as the main user-editable configuration file.

Configuration precedence will be:

1. dataclass defaults in `config.py`
2. `unified_app/config.toml`
3. `LEGAL_RAG_*` environment variables

This keeps code defaults centralized, gives the project a clear config file, and preserves env vars for one-off overrides.

## Scope

The change covers:

- runtime settings such as `mode`, `ollama_model`, `streaming`
- index paths such as `laws_dir`, `mini_working_dir`, `qdrant_path`
- loading and merging config data in one place
- adding a checked-in `config.toml` with comments

The change does not add:

- multiple config profiles
- hot reload
- command-line parsing
- secret management

## Config File Shape

The file will live at:

- `unified_app/config.toml`

It will use two sections:

```toml
[runtime]
mode = "auto"
ollama_model = "qwen35-law:q6k-stable"
streaming = true
max_context_articles = 6
max_history_turns = 4

[index]
laws_dir = "RAG/Chinese-Laws"
qdrant_path = "unified_app/storage/qdrant"
bm25_cache_path = "unified_app/storage/bm25"
mini_working_dir = "test/minirag_working"
corpus_dir = "unified_app/storage/corpus"
```

Paths in the config file will be resolved relative to the repository root passed into `AppConfig.from_env(...)`, not relative to the TOML file itself. That keeps path values aligned with the rest of this workspace.

## Loading Flow

`AppConfig.from_env(root_dir)` will:

1. start from dataclass defaults
2. look for `root_dir/unified_app/config.toml`
3. apply values found in TOML
4. apply `LEGAL_RAG_*` environment variable overrides
5. return one merged `AppConfig`

MiniRAG fallback behavior stays pragmatic:

- if the caller explicitly sets `mini_working_dir` in TOML or env, use it directly
- if they do not set it, keep the current heuristic:
  - prefer `root_dir/test/minirag_working` when present
  - otherwise use `root_dir/unified_app/storage/minirag_working`

## Testing

Add focused coverage for:

- loading `mode` and path values from `config.toml`
- environment variables overriding TOML values
- preserving current defaults when no config file exists

Use test-local temporary directories created under the workspace rather than relying on `tmp_path`, because the current environment has intermittent permission issues with pytest's temp root.

## Files

- Create: `unified_app/config.toml`
- Modify: `unified_app/src/legal_rag/config.py`
- Modify: `unified_app/tests/test_config.py`

