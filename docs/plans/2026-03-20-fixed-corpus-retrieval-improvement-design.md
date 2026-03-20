# Fixed-Corpus Retrieval Improvement Design

**Goal:** Improve retrieval quality in `unified_app` without adding or changing corpus files, by upgrading query expansion, hybrid reranking, and route confidence gating.

## Constraints

- Do not modify `F:\毕设\RAG\Chinese-Laws`.
- Do not add new law files, judicial interpretations, or external knowledge sources.
- Keep the existing top-level RAG flow:
  - query rewrite
  - exact retrieval
  - hybrid retrieval
  - auto routing
  - context building
- Prefer deterministic retrieval improvements over prompt-only fixes.

## Problem Statement

The current experiment results show that the experiment side is already using the real project retrieval pipeline, but retrieval quality is still weak.

The observed failure modes split into two groups:

1. **Out-of-corpus misses**

Some golden laws in the sampled evaluation set do not exist in the current corpus, including `中华人民共和国刑法`, `中华人民共和国刑事诉讼法`, `中华人民共和国劳动合同法`, and some judicial interpretations. These cannot be recovered by retrieval tuning alone.

2. **In-corpus ranking failures**

Some golden laws do exist in the corpus but still fail to reach the top of the retrieved list. Typical examples include:

- tenancy questions that should surface `民法典` lease-transfer provisions
- counterfeit-product compensation questions that should surface `消费者权益保护法` punitive-compensation provisions
- tort questions that should surface `民法典` employer-liability provisions

The current implementation mainly relies on:

- BM25 ranking
- vector ranking
- weighted RRF fusion
- small lexical bonuses for explicit law names and article numbers

This is too weak for natural-language legal questions that mention facts and outcomes but not statute names.

## Root Causes

### 1. Exact matching is almost inactive on natural-language questions

`ExactMatchRetriever` only succeeds when the query explicitly contains both a law alias and an article number. The sampled questions are mostly plain-language questions, so exact-match routing contributes almost nothing.

### 2. Hybrid fusion lacks legal-intent expansion

The retriever searches the literal user wording. Questions such as "房东卖房让我搬走" or "假冒伪劣商品几倍赔偿" do not explicitly contain the legal concepts that best identify the right statute.

### 3. Final reranking is too shallow

After BM25 and vector retrieval are fused, the retriever only adds:

- law-name bonus
- article-number bonus

This does not distinguish between:

- the right legal concept in the right legal domain
- a merely related article with overlapping generic words such as `赔偿`, `责任`, or `处理`

### 4. Router thresholds treat weak hybrid results as confident enough

The current `AutoRouter` accepts `hybrid` whenever the top score barely clears the minimum confidence and the top-two margin is non-negative. Given the current RRF score range, this means low-information retrieval results often bypass fallback behavior.

## Options Considered

### Option A: Tighten router thresholds only

This is the smallest change, but it only prevents some bad `hybrid` selections. It does not improve retrieval when the correct in-corpus law is rankable but currently buried or missed.

### Option B: Add deterministic query expansion only

This improves recall for some paraphrased questions, but if multiple related statutes are still retrieved, there is no stronger second-stage logic to push the best match to the top.

### Option C: Deterministic query expansion + lightweight reranking + stricter routing

This is the recommended approach.

It addresses the actual failure pattern:

- expansion raises the chance that the correct in-corpus statute enters the candidate set
- reranking resolves collisions between related but wrong statutes
- tighter routing prevents obviously weak hybrid results from being treated as trustworthy

## Approved Approach

### 1. Add deterministic legal query expansion

Introduce a rule-based query expansion layer inside retrieval, not in generation.

The expansion should:

- preserve the original user question
- add one or two short legal search variants
- avoid inventing new facts
- avoid using an LLM

Examples of the intended behavior:

- "租房合同没到期，房东要卖房让我搬走" -> add concepts such as `买卖不破租赁`, `房屋租赁`, `承租人`
- "买到假冒伪劣商品，可以要求几倍赔偿" -> add concepts such as `消费者权益保护法`, `惩罚性赔偿`, `退一赔三`
- "外卖骑手送餐时撞伤行人，谁承担赔偿责任" -> add concepts such as `执行工作任务`, `用人单位`, `侵权责任`

The retriever should run BM25 and vector retrieval for:

- the original query at full weight
- expansion queries at lower weight

and then fuse all ranked lists.

### 2. Add a lightweight legal reranker over fused candidates

After multi-query fusion, rerank the top candidate window using deterministic legal features rather than only law/article string bonuses.

The reranker should reward:

- concept phrase matches between query expansions and article text
- domain-specific terms that align with the user’s legal issue
- high overlap with issue-defining phrases such as `买卖不破租赁`, `预付款`, `执行工作任务`, `惩罚性赔偿`

The reranker should penalize:

- candidates from clearly irrelevant legal domains when the query intent is narrow
- articles that only match generic terms such as `赔偿`, `责任`, `处理`

This remains a lightweight retriever-side scorer, not a cross-encoder.

### 3. Tighten route confidence after reranking

Retain the existing router shape, but revise its thresholds so that:

- tied or near-tied hybrid results are no longer automatically treated as confident
- reranked top1 and top2 separation matters
- low-information candidate sets can still fall back instead of forcing weak hybrid context into generation

The route trace exported to experiments should continue exposing:

- `top1_score`
- `top2_score`
- `score_margin`
- `candidate_count`
- `selected_mode`
- `fallback_triggered`

### 4. Keep corpus-missing questions explicitly out of the target claim

This change is not expected to solve questions whose target statute is absent from the corpus. Verification should therefore separate:

- overall sampled-set behavior
- in-corpus-only behavior

This avoids overstating retrieval gains while respecting the user’s fixed-corpus constraint.

## Proposed Code Areas

- Modify: `F:\毕设\unified_app\src\legal_rag\retrievers\hybrid.py`
- Modify: `F:\毕设\unified_app\src\legal_rag\router\auto.py`
- Create or modify helper utilities under `F:\毕设\unified_app\src\legal_rag\utils\`
- Add focused tests under `F:\毕设\unified_app\tests\`

Likely internal components:

- `LegalQueryExpander`
- `LegalHybridReranker`
- updated hybrid raw signals for route analysis

## Verification Strategy

Verification should focus on two kinds of evidence.

### Unit-level evidence

Tests should prove:

- expansion queries are generated deterministically from known issue patterns
- multi-query fusion keeps the original query dominant
- reranking promotes the correct in-corpus article in representative cases
- weak reranked results produce non-zero top-two separation only when justified

### Evaluation-level evidence

On the sampled experiment set, compare before and after:

- in-corpus `top1` hit count
- in-corpus `top3` hit count
- route distribution
- route-score spread (`top1_score`, `top2_score`, `score_margin`)

Representative regression checks should cover at least:

- tenancy transfer / sale during lease
- counterfeit or fake product compensation
- employer liability during task execution
- prepaid-card refund

## Out Of Scope

- adding or changing corpus documents
- using an LLM to hallucinate retrieval expansions
- introducing external search tools
- replacing the current retriever with a cross-encoder stack
- redesigning answer generation prompts
