from dataclasses import dataclass
from pathlib import Path


@dataclass
class MiniIndexPaths:
    corpus_jsonl: Path
    working_dir: Path


def mini_index_paths(storage_dir: str | Path) -> MiniIndexPaths:
    storage = Path(storage_dir)
    return MiniIndexPaths(
        corpus_jsonl=storage / "corpus" / "normalized_articles.jsonl",
        working_dir=storage / "minirag_working",
    )
