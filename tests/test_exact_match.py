import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_exact_match_returns_standard_retrieval_result() -> None:
    assert importlib.util.find_spec("legal_rag.retrievers") is not None
    assert importlib.util.find_spec("legal_rag.retrievers.exact_match") is not None

    retriever_module = importlib.import_module("legal_rag.retrievers.exact_match")
    types_module = importlib.import_module("legal_rag.types")

    ExactMatchRetriever = retriever_module.ExactMatchRetriever
    NormalizedArticle = types_module.NormalizedArticle

    article = NormalizedArticle(
        canonical_id="中华人民共和国民法典:第一千一百三十八条",
        law_name="中华人民共和国民法典",
        law_aliases=["民法典"],
        article_id_cn="第一千一百三十八条",
        article_id_num="1138",
        content="《中华人民共和国民法典》第一千一百三十八条 口头遗嘱...",
        chapter=None,
        section=None,
        source="中华人民共和国民法典.txt",
        source_line=1,
    )
    retriever = ExactMatchRetriever([article])
    result = retriever.retrieve("民法典1138条怎么规定")

    assert result.docs[0].canonical_id == article.canonical_id
    assert "exact_article_match" in result.reasons


def test_exact_match_supports_chinese_article_number_queries() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.exact_match")
    types_module = importlib.import_module("legal_rag.types")

    ExactMatchRetriever = retriever_module.ExactMatchRetriever
    NormalizedArticle = types_module.NormalizedArticle

    article = NormalizedArticle(
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
    )

    retriever = ExactMatchRetriever([article])
    result = retriever.retrieve("\u6c11\u6cd5\u5178\u7b2c\u4e00\u6761\u662f\u4ec0\u4e48")

    assert result.docs[0].canonical_id == "law:1"
    assert result.raw_signals["matched_article_num"] == "1"
