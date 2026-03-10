import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_mini_retriever_maps_graph_hits_back_to_canonical_articles() -> None:
    assert importlib.util.find_spec("legal_rag.retrievers.mini") is not None

    retriever_module = importlib.import_module("legal_rag.retrievers.mini")
    MiniRetriever = retriever_module.MiniRetriever

    retriever = MiniRetriever.fake_for_test(
        docs=[
            {
                "chunk_id": "chunk-1",
                "canonical_id": "law:1138",
                "content": "口头遗嘱...",
                "score": 0.66,
            }
        ]
    )
    result = retriever.retrieve("临终口头遗嘱有效吗")

    assert result.docs[0].canonical_id == "law:1138"
    assert result.docs[0].retriever == "mini"


def test_mini_retriever_from_working_dir_degrades_when_backend_is_missing(tmp_path: Path, monkeypatch) -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.mini")
    types_module = importlib.import_module("legal_rag.types")

    MiniRetriever = retriever_module.MiniRetriever
    NormalizedArticle = types_module.NormalizedArticle

    working_dir = tmp_path / "minirag_working"
    working_dir.mkdir()

    article = NormalizedArticle(
        canonical_id="law:1138",
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

    original_find_spec = retriever_module.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "minirag":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(retriever_module.importlib.util, "find_spec", fake_find_spec)

    retriever = MiniRetriever.from_working_dir(working_dir, [article])

    assert retriever.is_available is False
    result = retriever.retrieve("临终口头遗嘱有效吗")
    assert result.docs == []


def test_mini_retriever_degrades_when_runtime_query_fails() -> None:
    retriever_module = importlib.import_module("legal_rag.retrievers.mini")

    MiniRetriever = retriever_module.MiniRetriever

    class BrokenRuntime:
        def query(self, question: str):
            raise RuntimeError("backend unavailable")

    retriever = MiniRetriever(is_available=True, runtime=BrokenRuntime())
    result = retriever.retrieve("临终口头遗嘱有效吗")

    assert result.docs == []
    assert "mini_runtime_error" in result.reasons
