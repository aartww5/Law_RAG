# Model Reranker Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace heuristic reranking in hybrid retrieval with a dedicated reranker model while keeping query expansion and mini fallback.

**Architecture:** Keep the first-stage retrieval path unchanged: BM25 plus vector retrieval, optional query expansion, and weighted RRF. Replace the in-process heuristic rerank bonus with a model-backed reranker that scores top-k query-document pairs and returns calibrated top1/top2 scores for routing. If the reranker cannot be loaded, retrieval should continue with pure RRF plus lexical article/law bonuses rather than rule-based rerank.

**Tech Stack:** Python, sentence-transformers CrossEncoder, local Hugging Face cache, pytest

---

### Task 1: Add config and tests for model-backed reranker selection

**Files:**
- Modify: `src/legal_rag/config.py`
- Modify: `tests/test_hybrid_retriever.py`

**Step 1: Write the failing test**

Add tests that verify:
- `IndexConfig` exposes reranker model path/name and rerank window.
- `HybridRetriever.from_articles(...)` can receive a reranker dependency and uses it when backends are enabled.

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: FAIL because config fields and reranker wiring do not exist yet.

**Step 3: Write minimal implementation**

Add reranker fields to `IndexConfig` and config loading, with defaults pointing at the local `BAAI/bge-reranker-v2-m3` cache snapshot and a default top-k window.

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: PASS for the new config/reranker tests.

**Step 5: Commit**

```bash
git add src/legal_rag/config.py tests/test_hybrid_retriever.py
git commit -m "test: cover dedicated reranker config"
```

### Task 2: Replace heuristic reranking with a dedicated reranker module

**Files:**
- Create: `src/legal_rag/retrievers/reranker.py`
- Modify: `src/legal_rag/retrievers/hybrid.py`
- Modify: `tests/test_hybrid_retriever.py`

**Step 1: Write the failing test**

Add tests that verify:
- A fake reranker can reorder RRF candidates.
- The returned score breakdown includes model reranker output, not heuristic rerank fields.
- Retrieval reasons include `model_rerank` and no longer depend on rule rerank.

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: FAIL because the dedicated reranker module is not implemented.

**Step 3: Write minimal implementation**

Create a reranker abstraction with:
- lazy model loading
- batch scoring for `(query, doc)` pairs
- score normalization
- no heuristic fallback

Wire `HybridRetriever` to:
- keep query expansion
- RRF fuse recall candidates
- rerank top-k with the dedicated model when available
- fall back to pure RRF plus lexical bonuses when unavailable

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: PASS for the reranker behavior tests.

**Step 5: Commit**

```bash
git add src/legal_rag/retrievers/reranker.py src/legal_rag/retrievers/hybrid.py tests/test_hybrid_retriever.py
git commit -m "feat: add dedicated retrieval reranker"
```

### Task 3: Verify router compatibility and real retrieval behavior

**Files:**
- Modify: `tests/test_router.py`
- Check: `src/legal_rag/router/auto.py`

**Step 1: Write the failing test**

Add a router-facing test that verifies reranked top1/top2 scores still drive mini fallback correctly.

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_router.py -q`
Expected: FAIL if score semantics changed incompatibly.

**Step 3: Write minimal implementation**

Adjust score scaling or raw signals if needed so router decisions remain coherent.

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_router.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_router.py src/legal_rag/router/auto.py
git commit -m "test: verify router with reranker scores"
```

### Task 4: Full verification and retrieval probes

**Files:**
- Check: `tests/test_service_flow.py`
- Check: `tests/test_fallback.py`
- Check: `tests/test_context_builder.py`
- Check: `tests/test_exact_match.py`
- Check: `tests/test_generation_prompt.py`

**Step 1: Run full relevant verification**

Run:
`.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py tests/test_router.py tests/test_service_flow.py tests/test_fallback.py tests/test_context_builder.py tests/test_exact_match.py tests/test_generation_prompt.py -q`

Expected: PASS.

**Step 2: Run real retrieval probes**

Use the existing probe pattern against representative questions such as:
- fake goods compensation
- parking space ownership
- lease transfer
- delivery rider liability

Expected: model rerank is active, and in-corpus hit quality is at least not worse than the heuristic baseline on covered questions.

**Step 3: Commit**

```bash
git add .
git commit -m "feat: switch hybrid retrieval to dedicated reranker"
```
