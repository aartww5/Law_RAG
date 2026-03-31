# Arbitrary Query Retrieval Investigation

Date: 2026-03-31

## Trigger

User-provided arbitrary question:

`小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？`

Observed behavior:

- retrieval returned unrelated statutes such as `反家庭暴力法` / `反有组织犯罪法` / `票据法`
- the answer quality degraded even though the relevant Civil Code provisions exist in the corpus

## Scope

This investigation focused on the full retrieval path:

1. query decomposition
2. BM25 recall
3. vector recall
4. fused ranking / reranking

No further code changes were made beyond the already completed prompt/reranker cleanup in the same session.

## Expected Relevant Statutes

For this question, the likely core provisions are in the Civil Code:

- `民法典 第二十条` 不满八周岁的未成年人为无民事行为能力人
- `民法典 第一百四十四条` 无民事行为能力人实施的民事法律行为无效
- `民法典 第一百五十七条` 无效后取得的财产应当返还

Potentially relevant supporting provisions:

- `民法典 第十九条`
- `民法典 第一百四十五条`

## Findings

### 1. Benchmark-specific few-shot contamination did exist

The decomposition prompt previously contained benchmark-style example questions and outputs.

Affected file:

- [query_decomposition.py](/F:/毕设/unified_app/src/legal_rag/utils/query_decomposition.py)

That issue has already been removed in the current working tree.

### 2. BM25 is not the main blocker for this question class

Direct BM25 probes with correct legal phrasing retrieved the expected Civil Code provisions near the top:

- query `无民事行为能力人 实施的民事法律行为无效`
  - `民法典 144` ranked `1`
  - `民法典 19` ranked `3`
  - `民法典 145` ranked `6`
  - `民法典 20` ranked `9`
- query `不满八周岁 未成年人 法定代理人 代理 实施民事法律行为`
  - `民法典 20` ranked `1`
  - `民法典 19` ranked `2`
  - `民法典 145` ranked `6`
- query `民事法律行为无效 应当予以返还`
  - `民法典 157` ranked `1`
  - `民法典 145` ranked `5`
  - `民法典 144` ranked `6`

Implication:

- sparse recall can find the right statutes when given reasonable query text

### 3. The current production vector collection is stale / partial

This is the dominant finding.

The active Qdrant collection at:

- [storage/qdrant](/F:/毕设/unified_app/storage/qdrant)

contains only:

- `256` points total

and the sampled payloads show only four laws:

- `中华人民共和国反恐怖主义法`
- `中华人民共和国反有组织犯罪法`
- `中华人民共和国反洗钱法`
- `中华人民共和国反家庭暴力法`

The current full corpus parsed from:

- `F:\毕设\RAG\Chinese-Laws`

contains:

- `14415` normalized articles

and definitely includes the Civil Code target provisions.

Most importantly, direct retrieval from the active collection confirmed that these canonical IDs do **not** exist in the current vector store:

- `中华人民共和国民法典:第二十条`
- `中华人民共和国民法典:第一百四十四条`
- `中华人民共和国民法典:第一百五十七条`

Implication:

- vector recall is currently searching a stale partial subset, not the full legal corpus
- this alone explains why arbitrary questions get routed into the same small cluster of unrelated statutes

### 4. Fresh vector indexing on a tiny rebuilt control set works

A fresh temporary collection was built using only:

- target Civil Code provisions
- the frequently mis-hit distractor provisions from `反家庭暴力法` / `反有组织犯罪法` / `反恐怖主义法` / `票据法`

On that fresh tiny collection, vector retrieval behaved reasonably:

- `无民事行为能力人 实施的民事法律行为无效`
  - `民法典 144` ranked `1`
  - `民法典 145` ranked `2`
  - `民法典 20` ranked `3`
- `不满八周岁 未成年人 法定代理人 代理 实施民事法律行为`
  - `民法典 20` ranked `1`
  - `民法典 19` ranked `2`
- `民事法律行为无效 应当予以返还`
  - `民法典 157` ranked `1`
  - `民法典 144` ranked `2`

Implication:

- the embedding model is not the immediate blocker
- the production vector store content is the blocker

### 5. Decomposition still underperforms on arbitrary questions

After removing benchmark few-shot examples, the same arbitrary question decomposed into:

- original question
- `7岁儿童出售手表行为效力`
- `限制民事行为能力人实施民事法律行为效力`

This is better than before, but still flawed:

- it preserved the age fact
- but it incorrectly generalized `7岁` toward `限制民事行为能力人`, instead of `无民事行为能力人`

Implication:

- decomposition quality still matters
- but it is a secondary issue compared with the stale vector collection

## Root Cause Summary

Primary root cause:

1. The active Qdrant collection is an old partial index with only 256 points and four laws.

Secondary causes:

2. Query decomposition still makes incorrect legal abstractions on arbitrary unseen questions.
3. The code path in [backends.py](/F:/毕设/unified_app/src/legal_rag/retrievers/backends.py) only checks `collection_exists(...)` and never verifies that the collection matches the current corpus/model/text schema before reusing it.

## Why the observed behavior looked "benchmark-optimized"

The system had two overlapping problems:

1. prompt contamination from benchmark-style decomposition examples
2. a stale vector collection dominated by only a few laws that coincidentally matched recent debugging sessions

The first problem made the system look test-oriented.
The second problem made arbitrary queries collapse into the same irrelevant law families.

## Recommended Fix Order

1. Rebuild the vector collection from the full current corpus.
2. Add an index fingerprint before reusing a collection:
   - corpus article count
   - corpus checksum or manifest hash
   - embedding model name
   - search text schema version
3. Refuse to reuse the collection when the fingerprint mismatches.
4. After the vector store is corrected, reassess decomposition quality on arbitrary questions.
5. Only after that, tune router thresholds again.

## Concrete Code Hotspots

- vector collection reuse:
  - [backends.py](/F:/毕设/unified_app/src/legal_rag/retrievers/backends.py)
- service construction that always trusts the existing vector store:
  - [services.py](/F:/毕设/unified_app/src/legal_rag/services.py)
- decomposition behavior for unseen questions:
  - [query_decomposition.py](/F:/毕设/unified_app/src/legal_rag/utils/query_decomposition.py)

## Investigation Outcome

The arbitrary-question failure is real and reproducible.

The dominant cause is **not** simply benchmark-targeted prompt engineering anymore.
The dominant current cause is that the production vector retriever is backed by a stale and incomplete Qdrant collection.
