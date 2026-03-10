import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_corpus_builder_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    assert importlib.util.find_spec("legal_rag.indexing") is not None
    assert importlib.util.find_spec("legal_rag.indexing.corpus_builder") is not None

    corpus_builder = importlib.import_module("legal_rag.indexing.corpus_builder")
    build_canonical_corpus = corpus_builder.build_canonical_corpus

    laws_dir = tmp_path / "laws"
    laws_dir.mkdir()
    (laws_dir / "中华人民共和国民法典.txt").write_text(
        "第一千一百三十八条 口头遗嘱...",
        encoding="utf-8",
    )
    output_dir = tmp_path / "corpus"

    result = build_canonical_corpus(laws_dir=laws_dir, output_dir=output_dir)

    assert (output_dir / "normalized_articles.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert result.article_count == 1
