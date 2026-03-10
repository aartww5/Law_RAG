import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
