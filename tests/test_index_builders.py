import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_index_builders_use_shared_corpus_locations(tmp_path: Path) -> None:
    assert importlib.util.find_spec("legal_rag.indexing.build_hybrid_index") is not None
    assert importlib.util.find_spec("legal_rag.indexing.build_mini_index") is not None

    hybrid_module = importlib.import_module("legal_rag.indexing.build_hybrid_index")
    mini_module = importlib.import_module("legal_rag.indexing.build_mini_index")

    storage = tmp_path / "storage"
    hybrid = hybrid_module.hybrid_index_paths(storage)
    mini = mini_module.mini_index_paths(storage)

    assert hybrid.corpus_jsonl == storage / "corpus" / "normalized_articles.jsonl"
    assert mini.working_dir == storage / "minirag_working"
