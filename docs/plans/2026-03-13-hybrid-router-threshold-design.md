# Hybrid Router Threshold Design

**Goal:** Make `auto` routing strongly prefer the `hybrid` retriever whenever it returns a basically usable result, instead of falling back to `mini` too aggressively.

## Approved Policy

The user-approved routing policy is:

- Keep `ExactMatch` behavior unchanged.
- Keep the current `AutoRouter` structure unchanged.
- Lower the default hybrid confidence threshold from `0.7` to `0.35`.
- Lower the default hybrid margin threshold from `0.08` to `0.0`.

This means:

- If `exact_result` exists, continue using `hybrid_plus_exact`.
- If hybrid top-1 score is at least `0.35`, use `hybrid` even when top-1 and top-2 are close.
- Only fall back to `mini` when hybrid confidence is genuinely weak.

## Why This Approach

The current router was tuned for the old lightweight retriever. After upgrading hybrid retrieval to `BM25 + vector + RRF`, the old threshold now rejects many answerable cases. The score distribution changed, but the route policy did not.

Changing the thresholds is the smallest correct fix:

- No service-layer redesign.
- No new config surface.
- No new routing branches.
- Keeps `mini` as a fallback for clearly weak hybrid retrieval.

## Out Of Scope

- Dynamic thresholds by query type.
- Configurable router thresholds in `config.toml`.
- Adding document-count heuristics or coverage heuristics.
- Changing `mini` retrieval behavior.

## Verification

Tests should prove:

- Weak hybrid scores still route to `mini`.
- Moderate but usable hybrid scores now stay on `hybrid`.
- Exact-match routing remains unchanged.
