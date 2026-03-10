# Config Toml Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a project-level `config.toml` so `unified_app` has a clear user-editable configuration source with environment-variable overrides.

**Architecture:** Keep `AppConfig` as the single merge point. Start from dataclass defaults, overlay `config.toml`, then overlay `LEGAL_RAG_*` env vars. Preserve the existing MiniRAG fallback heuristic when no explicit path is set.

**Tech Stack:** Python 3.12, `tomllib`, pytest

---

### Task 1: Add Config Loading Tests

**Files:**
- Modify: `unified_app/tests/test_config.py`

**Step 1: Write the failing test**

Add tests for:

- loading `runtime.mode` and `index.laws_dir` from `unified_app/config.toml`
- env vars overriding TOML values

Use `tempfile.TemporaryDirectory` rooted under the workspace to avoid `tmp_path` permission issues.

**Step 2: Run test to verify it fails**

Run: `python -m pytest unified_app/tests/test_config.py -k "config_toml or env_overrides_toml" -q`

Expected: FAIL because `AppConfig.from_env()` does not read TOML yet.

**Step 3: Write minimal implementation**

No implementation in this task.

**Step 4: Commit**

Skip commit unless explicitly requested.

### Task 2: Implement TOML Merge Logic

**Files:**
- Modify: `unified_app/src/legal_rag/config.py`

**Step 1: Write minimal implementation**

Add helpers to:

- locate `root_dir/unified_app/config.toml`
- load TOML data with `tomllib`
- merge runtime and index values onto defaults
- apply env var overrides after TOML
- resolve path fields relative to the repository root

Keep MiniRAG path fallback only for the unset case.

**Step 2: Run tests to verify they pass**

Run: `python -m pytest unified_app/tests/test_config.py -k "config_toml or env_overrides_toml" -q`

Expected: PASS

### Task 3: Add Checked-In Project Config File

**Files:**
- Create: `unified_app/config.toml`

**Step 1: Write the file**

Add a commented TOML file containing:

- `[runtime]`
- `[index]`

Populate it with current project defaults so users can edit the file directly.

**Step 2: Verify loading path**

Run a short Python check that imports `AppConfig` and prints the active `mode`.

Expected: the value comes from `config.toml` when env vars are absent.

### Task 4: Regression Verification

**Files:**
- No code changes expected

**Step 1: Run targeted config tests**

Run: `python -m pytest unified_app/tests/test_config.py -q`

Expected: targeted non-`tmp_path` config tests pass; existing `tmp_path` tests may still be blocked by environment temp-dir permissions.

**Step 2: Run manual smoke checks**

Verify:

- `AppConfig.from_env(repo_root)` loads `unified_app/config.toml`
- `LEGAL_RAG_MODE` overrides the TOML value
- path fields resolve relative to the repository root

**Step 3: Record any environment limitations**

If pytest `tmp_path` still fails due temp-dir permissions, report that as an environment issue rather than a config regression.
