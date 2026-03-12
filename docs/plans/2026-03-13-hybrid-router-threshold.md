# Hybrid Router Threshold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Relax `AutoRouter` defaults so `auto` mode prefers `hybrid` whenever hybrid retrieval has a basically usable result.

**Architecture:** Keep the current router decision tree and only change the default thresholds. Lock the behavior with focused router tests and one service-level regression test so future retrieval upgrades do not silently reintroduce over-eager `mini` fallback.

**Tech Stack:** Python dataclass-based routing, pytest.

---

### Task 1: Lock Router Behavior With Tests

**Files:**
- Modify: `tests/test_router.py`

**Step 1: Write the failing test**

```python
def test_router_prefers_hybrid_when_confidence_is_usable_even_without_margin():
    router = AutoRouter()
    hybrid = RetrievalResult(
        docs=[],
        confidence=0.41,
        latency_ms=0.0,
        reasons=["hybrid_rrf"],
        raw_signals={"top1_score": 0.41, "top2_score": 0.40},
    )
    decision = router.decide(exact_result=exact, hybrid_result=hybrid)
    assert decision.selected_mode == "hybrid"
```

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_router.py -q`
Expected: FAIL because current defaults still route this case to `mini`.

**Step 3: Write minimal implementation**

Change `AutoRouter` defaults to:

```python
min_confidence=0.35
min_margin=0.0
```

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_router.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_router.py src/legal_rag/router/auto.py
git commit -m "feat: relax hybrid router thresholds"
```

### Task 2: Add A Service-Level Regression Check

**Files:**
- Modify: `tests/test_service_flow.py`

**Step 1: Write the failing test**

```python
def test_auto_mode_keeps_hybrid_when_confidence_is_usable():
    service = LegalAssistantService.for_test(mode="auto")
    service.hybrid_retriever = HybridRetriever.fake_for_test([...])
    answer = service.handle_message("...")
    assert answer.route_decision.selected_mode == "hybrid"
```

**Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_service_flow.py::test_auto_mode_keeps_hybrid_when_confidence_is_usable -q`
Expected: FAIL under old thresholds.

**Step 3: Write minimal implementation**

No extra production code should be needed beyond the router threshold change. Only adjust fixtures or fake scores if the test setup needs a more precise route-shaping scenario.

**Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_service_flow.py::test_auto_mode_keeps_hybrid_when_confidence_is_usable -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_service_flow.py
git commit -m "test: cover hybrid-first auto routing"
```

### Task 3: Full Verification

**Files:**
- Test: `tests/test_router.py`
- Test: `tests/test_service_flow.py`
- Test: full suite

**Step 1: Run focused verification**

Run: `.venv\Scripts\python.exe -m pytest tests/test_router.py tests/test_service_flow.py -q`
Expected: PASS.

**Step 2: Run full verification**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS.

**Step 3: Commit planning docs**

```bash
git add docs/plans/2026-03-13-hybrid-router-threshold-design.md docs/plans/2026-03-13-hybrid-router-threshold.md
git commit -m "docs: add hybrid router threshold plan"
```
