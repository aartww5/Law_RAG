from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_rag.config import AppConfig, RuntimeConfig
from legal_rag.generation.llm import SimpleGenerator
from legal_rag.retrievers.exact_match import ExactMatchRetriever
from legal_rag.retrievers.hybrid import HybridRetriever
from legal_rag.retrievers.mini import MiniRetriever
from legal_rag.services import LegalAssistantService
from legal_rag.types import NormalizedArticle


def test_auto_disables_mini_fallback_when_mini_is_unavailable() -> None:
    service = LegalAssistantService.for_test(mode="auto", mini_available=False)
    answer = service.handle_message("\u4e34\u7ec8\u53e3\u5934\u9057\u5631\u6709\u6548\u5417")

    assert answer.route_decision.selected_mode == "hybrid"
    assert "mini_unavailable" in answer.route_decision.reasons


def test_auto_degrades_to_hybrid_when_mini_returns_runtime_error() -> None:
    article = NormalizedArticle(
        canonical_id="law:1138",
        law_name="\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178",
        law_aliases=["\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u6c11\u6cd5\u5178", "\u6c11\u6cd5\u5178"],
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
    )

    class BrokenRuntime:
        def query(self, question: str):
            raise AttributeError("'str' object has no attribute 'get'")

    service = LegalAssistantService(
        config=AppConfig(runtime=RuntimeConfig(mode="auto")),
        exact_retriever=ExactMatchRetriever([]),
        hybrid_retriever=HybridRetriever.fake_for_test(
            [
                {
                    "canonical_id": article.canonical_id,
                    "content": article.content,
                    "metadata": {
                        "law_name": article.law_name,
                        "law_aliases": article.law_aliases,
                        "article_id_cn": article.article_id_cn,
                        "article_id_num": article.article_id_num,
                    },
                    "score": 0.72,
                },
                {
                    "canonical_id": "other:1",
                    "content": "\u5176\u4ed6\u6cd5\u6761",
                    "metadata": {"law_name": "\u5176\u4ed6\u6cd5\u5f8b", "article_id_num": "1"},
                    "score": 0.71,
                },
            ]
        ),
        mini_retriever=MiniRetriever(is_available=True, runtime=BrokenRuntime()),
        generator=SimpleGenerator(enable_ollama=False),
    )
    service.mini_available = True

    answer = service.handle_message("\u53e3\u5934\u9057\u5631\u9700\u8981\u51e0\u4e2a\u89c1\u8bc1\u4eba")

    assert answer.route_decision.selected_mode == "hybrid"
    assert "mini_failed" in answer.route_decision.reasons
    assert answer.context.citations
