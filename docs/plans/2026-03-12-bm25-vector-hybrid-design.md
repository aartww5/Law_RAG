# BM25 And Vector Hybrid Retrieval Design

**Goal:** Upgrade `unified_app` from rule-scored lexical matching to a true hybrid retriever built from `jieba` + `rank_bm25` and `BAAI/bge-m3` + local Qdrant, while preserving the existing `ExactMatch`, multi-turn rewrite, router, and answer-generation flow.

## Chosen Approach

Three approaches were considered:

1. Directly transplant Demo retrieval code into `unified_app`.
2. Keep `unified_app`'s current service boundaries and replace the inside of `HybridRetriever` with BM25 and vector backends.
3. Stop at BM25 and postpone vectors.

Approach 2 is the right tradeoff. It lifts retrieval quality substantially without replacing the existing service contract, test structure, and `mini` fallback behavior. It also keeps the retrieval upgrade isolated from prompt, router, and UI behavior.

## Architecture

`HybridRetriever` remains the only retriever the service layer knows about. Internally it will orchestrate:

- A BM25 backend built from normalized article content using `jieba` tokenization and `rank_bm25.BM25Okapi`.
- A vector backend backed by local Qdrant and `sentence-transformers` with the `BAAI/bge-m3` model.
- Weighted RRF fusion over ranked lists from BM25 and vector retrieval.

The exact-match retriever remains separate and still takes precedence in context assembly. `AutoRouter` continues to make decisions from the merged hybrid confidence. `QueryRewriter` and the structured-background query format stay intact; when a rewritten query carries both background and current question, the hybrid retriever will run primary retrieval on the current question and lower-weight retrieval on the background text.

## Storage And Reuse

The vector store will use the configured `qdrant_path` and collection name. The default collection name will match the Demo collection (`chinese_laws_article_based`) so the app can point at the same on-disk Qdrant data when desired. If the collection does not exist, `unified_app` will build it from the local normalized corpus.

BM25 will be built in memory from the same normalized articles. This keeps the implementation smaller and avoids introducing a separate BM25 serialization format in the same change.

## Dependency Strategy

The current global Python interpreter does not have `qdrant_client`, `sentence_transformers`, `rank_bm25`, or `jieba`, while the project `.venv` does. The implementation therefore must:

- Use lazy imports inside the BM25/vector backend code so the package remains importable even outside the virtual environment.
- Use `.venv\\Scripts\\python.exe` for verification commands.

If BM25/vector dependencies are unavailable at runtime, the retriever should fail soft during backend initialization and fall back to the existing rule-score path rather than crashing the service startup.

## Scoring

Final hybrid ranking will be:

1. Gather ranked lists from BM25 and vector retrieval for the current query.
2. If structured background exists, gather additional lower-weight ranked lists for that background text.
3. Fuse ranked lists with weighted Reciprocal Rank Fusion.
4. Apply the existing law-name/article-number bonuses as a final lexical bias.

This preserves explicit article lookup quality while improving semantic recall for paraphrased questions.

## Testing

The change must be test-driven without requiring real model downloads. Unit tests will therefore rely on fake BM25/vector backends and cover:

- `jieba` tokenization behavior for legal questions.
- RRF fusion ordering.
- Hybrid retrieval ranking when BM25 and vector signals disagree.
- Graceful fallback when optional vector/BM25 dependencies are absent.
- Service bootstrap using config-supplied Qdrant settings without forcing heavy initialization in tests.

## Out Of Scope

- Cross-encoder reranker.
- Query decomposition / multi-query generation.
- Coverage补位 logic.
- LangGraph-style retrieve-grade-refine loops.
