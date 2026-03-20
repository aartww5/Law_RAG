# Fixed-Corpus Retrieval Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `unified_app` retrieval quality under a fixed corpus by adding deterministic legal query expansion, lightweight reranking, and stricter hybrid route gating.

**Architecture:** Keep the existing retrieval pipeline shape, but upgrade `HybridRetriever` so it searches with the original query plus lower-weight legal concept expansions, reranks fused candidates with legal-intent features, and exposes stronger route signals. Then tighten `AutoRouter` so weak tied hybrid results do not automatically bypass fallback behavior.

**Tech Stack:** Python, pytest, BM25, Qdrant-backed vector retrieval, weighted RRF, deterministic feature scoring.

---

### Task 1: Lock Current Failure Cases With Tests

**Files:**
- Create: `F:\毕设\unified_app\tests\test_fixed_corpus_retrieval.py`

**Step 1: Write the failing test**

Add focused tests that encode representative in-corpus cases where the current retriever underperforms, for example:

- lease-transfer questions should rank `民法典` lease-sale provisions ahead of unrelated civil-law or ticket-law articles
- counterfeit-product compensation questions should rank `消费者权益保护法` punitive-compensation articles ahead of unrelated trademark or anti-terrorism articles
- employer-liability questions should rank `民法典` task-execution liability provisions ahead of generic traffic-accident articles when the issue is responsibility attribution

Use fake article sets and fake BM25/vector ranked lists so the tests stay deterministic and do not depend on local model state.

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: FAIL because the current retriever has no expansion or reranking layer.

**Step 3: Write minimal implementation**

Do not change production behavior yet beyond what is required to support the first failing test.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: PASS for the first locked case.

**Step 5: Commit**

```bash
git add tests/test_fixed_corpus_retrieval.py
git commit -m "test: lock fixed-corpus retrieval failure cases"
```

### Task 2: Add Deterministic Query Expansion

**Files:**
- Create: `F:\毕设\unified_app\src\legal_rag\utils\query_expansion.py`
- Modify: `F:\毕设\unified_app\src\legal_rag\retrievers\hybrid.py`
- Test: `F:\毕设\unified_app\tests\test_fixed_corpus_retrieval.py`

**Step 1: Write the failing test**

Add tests that prove:

- expansion preserves the original query
- known legal issue patterns generate stable concept expansions
- expansion does not invent new facts not implied by the query
- `HybridRetriever` sends the original query plus lower-weight expansion queries into fusion

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: FAIL because no expansion utility exists.

**Step 3: Write minimal implementation**

Implement a deterministic expander that:

- maps known issue phrases to short legal search phrases
- returns the original query plus zero to two expansions
- exposes explicit weights for the original query and expansions

Refactor `HybridRetriever` to consume multiple query variants instead of just the rewritten query.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/legal_rag/utils/query_expansion.py src/legal_rag/retrievers/hybrid.py tests/test_fixed_corpus_retrieval.py
git commit -m "feat: add deterministic legal query expansion"
```

### Task 3: Add Lightweight Legal Reranking

**Files:**
- Create: `F:\毕设\unified_app\src\legal_rag\utils\reranking.py`
- Modify: `F:\毕设\unified_app\src\legal_rag\retrievers\hybrid.py`
- Test: `F:\毕设\unified_app\tests\test_fixed_corpus_retrieval.py`

**Step 1: Write the failing test**

Add tests that prove:

- reranking promotes the correct in-corpus article when the candidate set already contains it
- generic-term matches do not outrank issue-defining legal concepts
- the reranker score breakdown is visible in returned document metadata or raw signals

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: FAIL because fusion output is currently only corrected by small law/article bonuses.

**Step 3: Write minimal implementation**

Implement a reranker that:

- scores only a bounded candidate window from fused results
- rewards concept-phrase, issue-term, and domain-alignment matches
- penalizes candidates that only overlap on generic legal words
- updates document score breakdown to include reranker contributions

Keep the reranker deterministic and cheap.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/legal_rag/utils/reranking.py src/legal_rag/retrievers/hybrid.py tests/test_fixed_corpus_retrieval.py
git commit -m "feat: rerank hybrid candidates with legal intent features"
```

### Task 4: Retune Auto Routing Against Reranked Scores

**Files:**
- Modify: `F:\毕设\unified_app\src\legal_rag\router\auto.py`
- Modify: `F:\毕设\unified_app\tests\test_fixed_corpus_retrieval.py`
- Review: `F:\毕设\unified_app\docs\plans\2026-03-13-hybrid-router-threshold-design.md`

**Step 1: Write the failing test**

Add tests that prove:

- exact-match routing still takes precedence
- tied or near-tied hybrid top scores no longer automatically route to `hybrid`
- reranked confident cases still stay on `hybrid`

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: FAIL because the current router accepts almost any non-zero RRF top1 score.

**Step 3: Write minimal implementation**

Adjust `AutoRouter` so it uses stricter default confidence and margin requirements based on the reranked score distribution. Keep the router shape simple and preserve exact-match priority.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/legal_rag/router/auto.py tests/test_fixed_corpus_retrieval.py
git commit -m "tune: tighten hybrid routing after reranking"
```

### Task 5: Add Evaluation-Level Regression Checks

**Files:**
- Create or modify: `F:\毕设\unified_app\tests\test_retrieval_regression.py`
- Read from: `F:\毕设\experiment\data\sampled_questions_30.json`
- Compare against: `F:\毕设\experiment\results\raw\rag_results.json` as baseline evidence only

**Step 1: Write the failing test**

Add lightweight regression checks that validate:

- specific in-corpus questions improve in top candidate quality
- fixed-corpus unsupported questions are not falsely asserted as solved
- route-score spread is no longer dominated by `top1 == top2`

Keep these tests deterministic by using fixture candidate sets rather than real Qdrant state where possible.

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_retrieval_regression.py -q`
Expected: FAIL until the new expansion, reranking, and routing behavior is in place.

**Step 3: Write minimal implementation**

Add only the supporting code needed for the regression checks.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py tests/test_retrieval_regression.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_retrieval_regression.py
git commit -m "test: add fixed-corpus retrieval regression coverage"
```

### Task 6: Full Verification

**Files:**
- Test: `F:\毕设\unified_app\tests\test_fixed_corpus_retrieval.py`
- Test: `F:\毕设\unified_app\tests\test_retrieval_regression.py`
- Optional smoke path: `F:\毕设\experiment\run_experiment.py`

**Step 1: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fixed_corpus_retrieval.py tests/test_retrieval_regression.py -q`
Expected: PASS.

**Step 2: Run retrieval smoke verification**

Run a limited experiment or retrieval probe using the fixed 30-question sample and confirm:

- in-corpus top1/top3 hit counts improve
- route-score margins are no longer mostly zero
- unsupported out-of-corpus laws remain explicitly unsupported

**Step 3: Commit planning docs**

```bash
git add docs/plans/2026-03-20-fixed-corpus-retrieval-improvement-design.md docs/plans/2026-03-20-fixed-corpus-retrieval-improvement.md
git commit -m "docs: add fixed-corpus retrieval improvement plan"
```
