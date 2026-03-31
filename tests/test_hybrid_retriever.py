import importlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_app_config_reads_dedicated_reranker_settings_from_env() -> None:
    config_module = importlib.import_module("legal_rag.config")
    AppConfig = config_module.AppConfig

    root = Path(__file__).resolve().parents[2]
    previous_model = os.environ.get("LEGAL_RAG_RERANKER_MODEL")
    previous_reranker_device = os.environ.get("LEGAL_RAG_RERANKER_DEVICE")
    previous_top_k = os.environ.get("LEGAL_RAG_RERANK_TOP_K")
    previous_embedding_device = os.environ.get("LEGAL_RAG_EMBEDDING_DEVICE")
    os.environ["LEGAL_RAG_RERANKER_MODEL"] = "BAAI/bge-reranker-base"
    os.environ["LEGAL_RAG_RERANKER_DEVICE"] = "cuda"
    os.environ["LEGAL_RAG_RERANK_TOP_K"] = "12"
    os.environ["LEGAL_RAG_EMBEDDING_DEVICE"] = "cuda"

    try:
        config = AppConfig.from_env(root)
    finally:
        if previous_model is None:
            os.environ.pop("LEGAL_RAG_RERANKER_MODEL", None)
        else:
            os.environ["LEGAL_RAG_RERANKER_MODEL"] = previous_model
        if previous_reranker_device is None:
            os.environ.pop("LEGAL_RAG_RERANKER_DEVICE", None)
        else:
            os.environ["LEGAL_RAG_RERANKER_DEVICE"] = previous_reranker_device
        if previous_top_k is None:
            os.environ.pop("LEGAL_RAG_RERANK_TOP_K", None)
        else:
            os.environ["LEGAL_RAG_RERANK_TOP_K"] = previous_top_k
        if previous_embedding_device is None:
            os.environ.pop("LEGAL_RAG_EMBEDDING_DEVICE", None)
        else:
            os.environ["LEGAL_RAG_EMBEDDING_DEVICE"] = previous_embedding_device

    assert config.index.reranker_model == "BAAI/bge-reranker-base"
    assert config.index.reranker_device == "cuda"
    assert config.index.rerank_top_k == 12
    assert config.index.embedding_device == "cuda"


def test_hybrid_retriever_returns_ranked_docs_from_shared_contract() -> None:
    assert importlib.util.find_spec("legal_rag.retrievers.hybrid") is not None

    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    HybridRetriever = retriever_module.HybridRetriever

    retriever = HybridRetriever.fake_for_test(
        docs=[
            {
                "canonical_id": "law:1",
                "content": "\u53e3\u5934\u9057\u5631\u9700\u8981\u4e24\u4e2a\u4ee5\u4e0a\u89c1\u8bc1\u4eba\u5728\u573a",
                "score": 0.91,
            },
            {
                "canonical_id": "law:2",
                "content": "\u6253\u5370\u9057\u5631\u5e94\u5f53\u6709\u4e24\u4e2a\u4ee5\u4e0a\u89c1\u8bc1\u4eba",
                "score": 0.78,
            },
        ]
    )
    result = retriever.retrieve("\u53e3\u5934\u9057\u5631\u9700\u8981\u51e0\u4e2a\u89c1\u8bc1\u4eba")

    assert result.docs[0].canonical_id == "law:1"
    assert result.confidence > 0


def test_bm25_tokenizer_uses_jieba_words_and_preserves_ascii() -> None:
    backends_module = importlib.import_module("legal_rag.retrievers.backends")

    tokens = backends_module.tokenize_for_bm25("民法典1138条口头遗嘱")

    assert "民法典" in tokens
    assert "1138" in tokens
    assert "口头" in tokens


def test_weighted_rrf_fuses_rank_lists_from_multiple_backends() -> None:
    backends_module = importlib.import_module("legal_rag.retrievers.backends")

    fused = backends_module.weighted_rrf(
        [
            ([("law:1138", 4.0), ("law:1139", 3.0)], 1.0),
            ([("law:1139", 0.8), ("law:1138", 0.6)], 1.0),
            ([("law:1138", 0.5)], 0.5),
        ],
        k=60,
    )

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)

    assert ranked[0][0] == "law:1138"
    assert ranked[0][1] > ranked[1][1]


def test_qdrant_vector_backend_close_releases_client_state() -> None:
    backends_module = importlib.import_module("legal_rag.retrievers.backends")
    QdrantVectorBackend = backends_module.QdrantVectorBackend

    backend = QdrantVectorBackend(
        storage_path=Path("."),
        collection_name="demo",
        model_name="demo-model",
        device="cpu",
        articles=[],
    )

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = FakeClient()
    backend._client = client
    backend._model = object()
    backend._is_ready = True

    backend.close()

    assert client.closed is True
    assert backend._client is None
    assert backend._model is None
    assert backend._is_ready is False


def test_qdrant_vector_backend_reuses_shared_local_client(monkeypatch, tmp_path: Path) -> None:
    backends_module = importlib.import_module("legal_rag.retrievers.backends")
    types_module = importlib.import_module("legal_rag.types")
    QdrantVectorBackend = backends_module.QdrantVectorBackend
    NormalizedArticle = types_module.NormalizedArticle

    article = NormalizedArticle(
        canonical_id="law:1",
        law_name="中华人民共和国民法典",
        law_aliases=["民法典"],
        article_id_cn="第一百八十四条",
        article_id_num="184",
        content="因自愿实施紧急救助行为造成受助人损害的，救助人不承担民事责任。",
        chapter=None,
        section=None,
        source="civil_code.txt",
        source_line=1,
    )

    created_clients: list[object] = []
    created_models: list[object] = []

    class FakeVector(list):
        def tolist(self):
            return list(self)

    class FakeClient:
        def __init__(self, *, path: str) -> None:
            self.path = path
            self.closed = False
            self.collections: set[str] = set()
            created_clients.append(self)

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name in self.collections

        def create_collection(self, *, collection_name: str, vectors_config) -> None:
            self.collections.add(collection_name)

        def upsert(self, collection_name: str, *, points, wait: bool) -> None:
            self.collections.add(collection_name)

        def close(self) -> None:
            self.closed = True

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            self.model_name = model_name
            self.device = device
            created_models.append(self)

        def encode(self, text: str, *, convert_to_numpy: bool, normalize_embeddings: bool):
            return FakeVector([0.1, 0.2, 0.3])

    class FakeVectorParams:
        def __init__(self, *, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class FakePointStruct:
        def __init__(self, *, id: str, vector: list[float], payload: dict) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    class FakeModelsModule:
        VectorParams = FakeVectorParams
        PointStruct = FakePointStruct

        class Distance:
            COSINE = "cosine"

    class FakeQdrantModule:
        QdrantClient = FakeClient

    def fake_require_dependency(module_name: str):
        if module_name == "qdrant_client":
            return FakeQdrantModule
        if module_name == "sentence_transformers":
            return type("FakeSentenceModule", (), {"SentenceTransformer": FakeSentenceTransformer})
        if module_name == "qdrant_client.http.models":
            return FakeModelsModule
        raise AssertionError(f"unexpected dependency request: {module_name}")

    monkeypatch.setattr(backends_module, "_require_dependency", fake_require_dependency)
    monkeypatch.setattr(QdrantVectorBackend, "_shared_clients", {}, raising=False)
    monkeypatch.setattr(QdrantVectorBackend, "_shared_models", {}, raising=False)
    monkeypatch.setattr(backends_module, "resolve_torch_device", lambda preferred: preferred)

    storage_path = tmp_path / "qdrant"
    backend_one = QdrantVectorBackend(
        storage_path=storage_path,
        collection_name="laws",
        model_name="demo-embedding",
        device="cuda",
        articles=[article],
    )
    backend_two = QdrantVectorBackend(
        storage_path=storage_path,
        collection_name="laws",
        model_name="demo-embedding",
        device="cuda",
        articles=[article],
    )
    backend_one._ensure_ready()
    backend_two._ensure_ready()

    assert len(created_clients) == 1
    assert len(created_models) == 1
    assert backend_one._client is backend_two._client
    assert backend_one._model is backend_two._model
    assert created_models[0].device == "cuda"
    assert created_clients[0].closed is False

    backend_one.close()

    assert created_clients[0].closed is False

    backend_two.close()

    assert created_clients[0].closed is True


def test_hybrid_retriever_can_rank_real_articles() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="law:1138",
            law_name="\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
            law_aliases=["\u6c11\u6cd5\u5178"],
            article_id_cn="\u7b2c\u4e00\u5343\u4e00\u767e\u4e09\u5341\u516b\u6761",
            article_id_num="1138",
            content=(
                "\u300a\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178\u300b"
                "\u7b2c\u4e00\u5343\u4e00\u767e\u4e09\u5341\u516b\u6761\u89c4\u5b9a\uff0c"
                "\u53e3\u5934\u9057\u5631\u5e94\u5f53\u6709\u4e24\u4e2a\u4ee5\u4e0a\u89c1\u8bc1\u4eba\u5728\u573a\u89c1\u8bc1\u3002"
            ),
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=1,
        ),
        NormalizedArticle(
            canonical_id="law:1139",
            law_name="\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
            law_aliases=["\u6c11\u6cd5\u5178"],
            article_id_cn="\u7b2c\u4e00\u5343\u4e00\u767e\u4e09\u5341\u4e5d\u6761",
            article_id_num="1139",
            content=(
                "\u300a\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178\u300b"
                "\u7b2c\u4e00\u5343\u4e00\u767e\u4e09\u5341\u4e5d\u6761\u89c4\u5b9a\uff0c"
                "\u5f55\u97f3\u5f55\u50cf\u9057\u5631\u5e94\u5f53\u6709\u4e24\u4e2a\u4ee5\u4e0a\u89c1\u8bc1\u4eba\u5728\u573a\u89c1\u8bc1\u3002"
            ),
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=2,
        ),
    ]

    retriever = HybridRetriever.from_articles(articles)
    result = retriever.retrieve("\u53e3\u5934\u9057\u5631\u9700\u8981\u51e0\u4e2a\u89c1\u8bc1\u4eba")

    assert result.docs[0].canonical_id == "law:1138"


def test_hybrid_retriever_prioritizes_exact_article_queries() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="law:1",
            law_name="\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
            law_aliases=[
                "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
                "\u6c11\u6cd5\u5178",
            ],
            article_id_cn="\u7b2c\u4e00\u6761",
            article_id_num="1",
            content=(
                "\u300a\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178\u300b"
                "\u7b2c\u4e00\u6761\u89c4\u5b9a\uff0c\u4e3a\u4e86\u4fdd\u62a4\u6c11\u4e8b"
                "\u4e3b\u4f53\u7684\u5408\u6cd5\u6743\u76ca\uff0c\u5236\u5b9a\u672c\u6cd5\u3002"
            ),
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=1,
        ),
        NormalizedArticle(
            canonical_id="law:27",
            law_name="\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
            law_aliases=[
                "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
                "\u6c11\u6cd5\u5178",
            ],
            article_id_cn="\u7b2c\u4e8c\u5341\u4e03\u6761",
            article_id_num="27",
            content=(
                "\u300a\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178\u300b"
                "\u7b2c\u4e8c\u5341\u4e03\u6761\u89c4\u5b9a\uff0c\u6cd5\u4eba\u7684\u6cd5\u5b9a"
                "\u4ee3\u8868\u4eba\u4ee5\u6cd5\u4eba\u540d\u4e49\u4ece\u4e8b\u6c11\u4e8b\u6d3b\u52a8\u3002"
            ),
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=27,
        ),
    ]

    retriever = HybridRetriever.from_articles(articles)
    result = retriever.retrieve("\u6c11\u6cd5\u5178\u7b2c\u4e00\u6761\u662f\u4ec0\u4e48")

    assert result.docs[0].canonical_id == "law:1"
    assert result.raw_signals["top1_score"] > result.raw_signals["top2_score"]


def test_hybrid_retriever_prioritizes_article_one_in_real_corpus() -> None:
    config_module = importlib.import_module("legal_rag.config")
    corpus_module = importlib.import_module("legal_rag.indexing.corpus_builder")
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")

    AppConfig = config_module.AppConfig
    iter_normalized_articles = corpus_module.iter_normalized_articles
    HybridRetriever = retriever_module.HybridRetriever

    root = Path(__file__).resolve().parents[2]
    config = AppConfig.from_env(root)

    selected = []
    target_ids = {"1", "27", "28", "51", "102", "114"}
    for article in iter_normalized_articles(config.index.laws_dir):
        if article.law_name == "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178" and article.article_id_num in target_ids:
            selected.append(article)

    retriever = HybridRetriever.from_articles(selected)
    result = retriever.retrieve("\u6c11\u6cd5\u5178\u7b2c\u4e00\u6761\u662f\u4ec0\u4e48")

    assert result.docs[0].metadata["article_id_num"] == "1"


def test_hybrid_retriever_fuses_bm25_and_vector_results_with_exact_bias() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="law:1138",
            law_name="中华人民共和国民法典",
            law_aliases=["中华人民共和国民法典", "民法典"],
            article_id_cn="第一千一百三十八条",
            article_id_num="1138",
            content="口头遗嘱应当有两个以上见证人在场见证。",
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=1,
        ),
        NormalizedArticle(
            canonical_id="law:1139",
            law_name="中华人民共和国民法典",
            law_aliases=["中华人民共和国民法典", "民法典"],
            article_id_cn="第一千一百三十九条",
            article_id_num="1139",
            content="录音录像遗嘱应当有两个以上见证人在场见证。",
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=2,
        ),
    ]

    class FakeBm25Backend:
        def retrieve(self, question: str, *, limit: int = 20):
            return [("law:1139", 8.0), ("law:1138", 7.2)]

    class FakeVectorBackend:
        def retrieve(self, question: str, *, limit: int = 20):
            return [("law:1138", 0.92), ("law:1139", 0.86)]

    retriever = HybridRetriever.from_articles(
        articles,
        bm25_backend=FakeBm25Backend(),
        vector_backend=FakeVectorBackend(),
        enable_backends=False,
    )
    result = retriever.retrieve("民法典第一千一百三十八条口头遗嘱")

    assert result.docs[0].canonical_id == "law:1138"
    assert result.docs[0].score_breakdown["rrf"] > 0
    assert result.docs[0].score_breakdown["article_bonus"] >= 1.0


def test_hybrid_retriever_falls_back_to_rule_scoring_when_backends_unavailable() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="law:1138",
            law_name="中华人民共和国民法典",
            law_aliases=["民法典"],
            article_id_cn="第一千一百三十八条",
            article_id_num="1138",
            content="口头遗嘱应当有两个以上见证人在场见证。",
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=1,
        ),
        NormalizedArticle(
            canonical_id="law:1139",
            law_name="中华人民共和国民法典",
            law_aliases=["民法典"],
            article_id_cn="第一千一百三十九条",
            article_id_num="1139",
            content="录音录像遗嘱应当有两个以上见证人在场见证。",
            chapter=None,
            section=None,
            source="civil_code.txt",
            source_line=2,
        ),
    ]

    retriever = HybridRetriever.from_articles(articles, enable_backends=False)
    result = retriever.retrieve("口头遗嘱需要几个见证人")

    assert result.docs[0].canonical_id == "law:1138"
    assert "rule_score_fallback" in result.reasons

@pytest.mark.skip(reason="replaced by llm query decomposition")
def test_hybrid_retriever_legacy_query_expansion_path_is_retired() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="consumer:55",
            law_name="中华人民共和国消费者权益保护法",
            law_aliases=["中华人民共和国消费者权益保护法", "消费者权益保护法", "消保法"],
            article_id_cn="第五十五条",
            article_id_num="55",
            content=(
                "经营者提供商品或者服务有欺诈行为的，应当按照消费者的要求增加赔偿其受到的损失，"
                "增加赔偿的金额为消费者购买商品的价款或者接受服务的费用的三倍。"
            ),
            chapter=None,
            section=None,
            source="consumer.txt",
            source_line=55,
        ),
        NormalizedArticle(
            canonical_id="trademark:63",
            law_name="中华人民共和国商标法",
            law_aliases=["中华人民共和国商标法", "商标法"],
            article_id_cn="第六十三条",
            article_id_num="63",
            content="侵犯商标专用权的赔偿数额，按照权利人因被侵权所受到的实际损失确定。",
            chapter=None,
            section=None,
            source="trademark.txt",
            source_line=63,
        ),
        NormalizedArticle(
            canonical_id="antiterror:86",
            law_name="中华人民共和国反恐怖主义法",
            law_aliases=["中华人民共和国反恐怖主义法", "反恐怖主义法"],
            article_id_cn="第八十六条",
            article_id_num="86",
            content="有关单位未履行反恐怖主义工作职责的，依法追究责任。",
            chapter=None,
            section=None,
            source="antiterror.txt",
            source_line=86,
        ),
    ]

    class FakeBm25Backend:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, question: str, *, limit: int = 20):
            self.queries.append(question)
            if "惩罚性赔偿" in question or "退一赔三" in question:
                return [("consumer:55", 30.0)]
            return [("trademark:63", 20.0), ("antiterror:86", 19.0)]

    class FakeVectorBackend:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, question: str, *, limit: int = 20):
            self.queries.append(question)
            if "消费者权益保护法" in question:
                return [("consumer:55", 0.95)]
            return [("antiterror:86", 0.91), ("trademark:63", 0.89)]

    bm25_backend = FakeBm25Backend()
    vector_backend = FakeVectorBackend()
    retriever = HybridRetriever.from_articles(
        articles,
        bm25_backend=bm25_backend,
        vector_backend=vector_backend,
        enable_backends=False,
    )

    result = retriever.retrieve("买到假冒伪劣商品，可以要求几倍赔偿？")

    assert any(doc.canonical_id == "consumer:55" for doc in result.docs)
    assert any("惩罚性赔偿" in query for query in bm25_backend.queries)
    assert any("消费者权益保护法" in query for query in vector_backend.queries)


def test_hybrid_retriever_uses_llm_query_decomposition_and_configured_limits() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    decomposition_module = importlib.import_module("legal_rag.utils.query_decomposition")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    QueryVariant = decomposition_module.QueryVariant
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="consumer:55",
            law_name="Consumer Protection Law",
            law_aliases=["Consumer Protection Law"],
            article_id_cn="Article 55",
            article_id_num="55",
            content="Fraudulent consumer sales support triple compensation.",
            chapter=None,
            section=None,
            source="consumer.txt",
            source_line=55,
        ),
        NormalizedArticle(
            canonical_id="trademark:63",
            law_name="Trademark Law",
            law_aliases=["Trademark Law"],
            article_id_cn="Article 63",
            article_id_num="63",
            content="Trademark infringement damages.",
            chapter=None,
            section=None,
            source="trademark.txt",
            source_line=63,
        ),
    ]

    class FakeBm25Backend:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.limits: list[int] = []

        def retrieve(self, question: str, *, limit: int = 20):
            self.queries.append(question)
            self.limits.append(limit)
            if "consumer protection law" in question:
                return [("consumer:55", 20.0)]
            return [("trademark:63", 19.0)]

    class FakeVectorBackend:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.limits: list[int] = []

        def retrieve(self, question: str, *, limit: int = 20):
            self.queries.append(question)
            self.limits.append(limit)
            if "triple compensation" in question:
                return [("consumer:55", 0.95)]
            return [("trademark:63", 0.89)]

    class FakeDecomposer:
        def decompose(self, question: str, *, background: str = ""):
            assert background == ""
            return [
                QueryVariant(text=question, weight=1.0, source="original"),
                QueryVariant(
                    text="consumer protection law triple compensation",
                    weight=0.65,
                    source="legal_concept",
                ),
            ]

    bm25_backend = FakeBm25Backend()
    vector_backend = FakeVectorBackend()
    retriever = HybridRetriever.from_articles(
        articles,
        bm25_backend=bm25_backend,
        vector_backend=vector_backend,
        decomposer=FakeDecomposer(),
        bm25_top_k=50,
        vector_top_k=50,
        enable_backends=False,
    )

    result = retriever.retrieve("bought fake goods, how much compensation can be claimed")

    assert any(doc.canonical_id == "consumer:55" for doc in result.docs)
    assert result.raw_signals["query_variant_count"] == 2
    assert "query_decomposition" in result.reasons
    assert any("consumer protection law" in query for query in bm25_backend.queries)
    assert any("triple compensation" in query for query in vector_backend.queries)
    assert bm25_backend.limits == [50, 50]
    assert vector_backend.limits == [50, 50]


def test_hybrid_retriever_uses_model_reranker_to_reorder_rrf_candidates() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.hybrid")
    decomposition_module = importlib.import_module("legal_rag.utils.query_decomposition")
    types_module = importlib.import_module("legal_rag.types")

    HybridRetriever = retriever_module.HybridRetriever
    QueryVariant = decomposition_module.QueryVariant
    NormalizedArticle = types_module.NormalizedArticle

    articles = [
        NormalizedArticle(
            canonical_id="civil:1191",
            law_name="Civil Code",
            law_aliases=["Civil Code"],
            article_id_cn="Article 1191",
            article_id_num="1191",
            content="Employer is liable when a worker causes harm while performing assigned work.",
            chapter=None,
            section=None,
            source="civil.txt",
            source_line=1191,
        ),
        NormalizedArticle(
            canonical_id="traffic:76",
            law_name="Road Traffic Safety Law",
            law_aliases=["Road Traffic Safety Law"],
            article_id_cn="Article 76",
            article_id_num="76",
            content="Traffic accident losses are handled first through insurance and then fault-based compensation.",
            chapter=None,
            section=None,
            source="traffic.txt",
            source_line=76,
        ),
    ]

    class FakeBm25Backend:
        def retrieve(self, question: str, *, limit: int = 20):
            return [("traffic:76", 20.0), ("civil:1191", 18.0)]

    class FakeVectorBackend:
        def retrieve(self, question: str, *, limit: int = 20):
            return [("traffic:76", 0.92), ("civil:1191", 0.88)]

    class FakeDecomposer:
        def decompose(self, question: str, *, background: str = ""):
            assert background == ""
            return [
                QueryVariant(text=question, weight=1.0, source="original"),
                QueryVariant(
                    text="assigned work employer liability",
                    weight=0.65,
                    source="legal_concept",
                ),
            ]

    class FakeReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[list[tuple[str, float]], list[str]]] = []

        def score(self, queries: list[tuple[str, float]], docs: list[dict]) -> dict[str, float]:
            self.calls.append((queries, [doc["canonical_id"] for doc in docs]))
            return {"traffic:76": 0.05, "civil:1191": 0.95}

    reranker = FakeReranker()
    retriever = HybridRetriever.from_articles(
        articles,
        bm25_backend=FakeBm25Backend(),
        vector_backend=FakeVectorBackend(),
        reranker=reranker,
        decomposer=FakeDecomposer(),
        enable_backends=False,
    )

    result = retriever.retrieve("delivery rider injures pedestrian liability")

    assert result.docs[0].canonical_id == "civil:1191"
    assert result.docs[0].score_breakdown["model_rerank"] == 0.95
    assert "rerank" not in result.docs[0].score_breakdown
    assert "generic_penalty" not in result.docs[0].score_breakdown
    assert "model_rerank" in result.reasons
    assert result.raw_signals["rerank_query_count"] == 2
    assert reranker.calls[0][0] == [
        ("delivery rider injures pedestrian liability", 1.0),
        ("assigned work employer liability", 0.65),
    ]
    assert reranker.calls[0][1] == ["traffic:76", "civil:1191"]
