from dataclasses import dataclass
from pathlib import Path


@dataclass
class HybridIndexPaths:
    corpus_jsonl: Path
    qdrant_path: Path
    bm25_cache_path: Path


def hybrid_index_paths(storage_dir: str | Path) -> HybridIndexPaths:
    storage = Path(storage_dir)
    return HybridIndexPaths(
        corpus_jsonl=storage / "corpus" / "normalized_articles.jsonl",
        qdrant_path=storage / "qdrant",
        bm25_cache_path=storage / "bm25",
    )

