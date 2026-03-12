# BM25 And Vector Hybrid Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current rule-scored `HybridRetriever` with a true `BM25 + vector + RRF` retrieval pipeline that still fits the existing `LegalAssistantService` contract.

**Architecture:** Keep `ExactMatchRetriever`, `AutoRouter`, and answer generation unchanged. Introduce lazy-loaded BM25 and vector backends under the retriever layer, use weighted RRF fusion inside `HybridRetriever`, and preserve the current rule-score path as a soft fallback when optional dependencies are unavailable.

**Tech Stack:** `jieba`, `rank_bm25`, `sentence-transformers`, `qdrant-client`, existing dataclass-based retrieval contract, pytest.

---

### Task 1: Add Config Surface For Hybrid Indexes

**Files:**
- Modify: `src/legal_rag/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
def test_config_reads_hybrid_index_settings_from_config_toml() -> None:
    ...
    assert config.index.qdrant_collection_name == "demo_collection"
    assert config.index.embedding_model == "BAAI/bge-m3"
```

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_config_reads_hybrid_index_settings_from_config_toml -q`
Expected: FAIL because `IndexConfig` does not yet expose those fields.

**Step 3: Write minimal implementation**

Extend `IndexConfig` and TOML/env parsing with:

```python
qdrant_collection_name: str = "chinese_laws_article_based"
embedding_model: str = "BAAI/bge-m3"
```

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_config.py src/legal_rag/config.py
git commit -m "feat: add hybrid index config"
```

### Task 2: Add BM25 And Vector Backend Abstractions

**Files:**
- Create: `src/legal_rag/retrievers/backends.py`
- Test: `tests/test_hybrid_retriever.py`

**Step 1: Write the failing test**

```python
def test_weighted_rrf_fuses_rank_lists_from_multiple_backends():
    fused = weighted_rrf([...])
    assert fused[0][0] == "law:1138"
```

```python
def test_bm25_tokenizer_uses_jieba_words_and_preserves_ascii():
    tokens = tokenize_for_bm25("民法典1138条口头遗嘱")
    assert "民法典" in tokens
    assert "1138" in tokens
```

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py::test_weighted_rrf_fuses_rank_lists_from_multiple_backends tests/test_hybrid_retriever.py::test_bm25_tokenizer_uses_jieba_words_and_preserves_ascii -q`
Expected: FAIL because helpers do not exist.

**Step 3: Write minimal implementation**

Create:

- `tokenize_for_bm25(text: str) -> list[str]`
- `weighted_rrf(rank_lists: list[tuple[list[tuple[str, float]], float]], k: int = 60) -> dict[str, float]`
- `Bm25Backend`
- `QdrantVectorBackend`

Both backend constructors must import optional dependencies lazily.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: PASS for the new helper tests.

**Step 5: Commit**

```bash
git add tests/test_hybrid_retriever.py src/legal_rag/retrievers/backends.py
git commit -m "feat: add hybrid retrieval backends"
```

### Task 3: Refactor HybridRetriever To Use BM25, Vector, And RRF

**Files:**
- Modify: `src/legal_rag/retrievers/hybrid.py`
- Modify: `src/legal_rag/utils/text.py`
- Test: `tests/test_hybrid_retriever.py`

**Step 1: Write the failing test**

```python
def test_hybrid_retriever_fuses_bm25_and_vector_results_with_exact_bias():
    retriever = HybridRetriever(
        articles=articles,
        bm25_backend=FakeBm25(...),
        vector_backend=FakeVector(...),
    )
    result = retriever.retrieve("民法典第一千一百三十八条口头遗嘱")
    assert result.docs[0].canonical_id == "law:1138"
```

```python
def test_hybrid_retriever_falls_back_to_rule_scoring_when_backends_unavailable():
    retriever = HybridRetriever.from_articles(articles, enable_backends=False)
    result = retriever.retrieve("口头遗嘱需要几个见证人")
    assert result.docs[0].canonical_id == "law:1138"
```

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: FAIL because `HybridRetriever` does not accept backends or use fusion.

**Step 3: Write minimal implementation**

Implement:

- `HybridRetriever` stores article metadata and optional BM25/vector backends.
- `from_articles()` builds available backends from config.
- `retrieve()` runs current query search plus optional background query search.
- Fusion uses weighted RRF and then applies law/article bonuses.
- If no backend is available, reuse the existing rule-score path.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_hybrid_retriever.py src/legal_rag/retrievers/hybrid.py src/legal_rag/utils/text.py
git commit -m "feat: fuse bm25 and vector retrieval"
```

### Task 4: Wire Service Bootstrap To Configured Hybrid Backends

**Files:**
- Modify: `src/legal_rag/services.py`
- Test: `tests/test_service_flow.py`

**Step 1: Write the failing test**

```python
def test_service_passes_hybrid_index_config_to_retriever(monkeypatch, tmp_path):
    ...
    assert captured["collection_name"] == "demo_collection"
```

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_service_flow.py::test_service_passes_hybrid_index_config_to_retriever -q`
Expected: FAIL because config is not forwarded.

**Step 3: Write minimal implementation**

Update `LegalAssistantService.from_config()` so `HybridRetriever.from_articles()` receives the `AppConfig` index settings needed for BM25/vector backends.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_service_flow.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_service_flow.py src/legal_rag/services.py
git commit -m "feat: wire hybrid backend config"
```

### Task 5: Verify End-To-End Behavior

**Files:**
- Test: `tests/test_config.py`
- Test: `tests/test_hybrid_retriever.py`
- Test: `tests/test_service_flow.py`
- Test: `tests/test_app_entrypoint.py`

**Step 1: Run focused verification**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_hybrid_retriever.py tests/test_service_flow.py tests/test_app_entrypoint.py -q`
Expected: PASS.

**Step 2: Run full verification**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS.

**Step 3: Commit**

```bash
git add docs/plans/2026-03-12-bm25-vector-hybrid-design.md docs/plans/2026-03-12-bm25-vector-hybrid.md
git commit -m "docs: add hybrid retrieval design and plan"
```
