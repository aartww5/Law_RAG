import json
from dataclasses import asdict, dataclass
from pathlib import Path

from legal_rag.documents.article_parser import parse_article_record
from legal_rag.documents.loader import iter_raw_law_lines
from legal_rag.documents.normalize import normalize_law_line
from legal_rag.types import NormalizedArticle


@dataclass
class BuildCorpusResult:
    article_count: int
    output_dir: Path


def iter_normalized_articles(laws_dir: str | Path) -> list[NormalizedArticle]:
    articles: list[NormalizedArticle] = []
    for source_path, law_name, line_no, raw_line in iter_raw_law_lines(laws_dir):
        normalized_line, _ = normalize_law_line(raw_line, law_name)
        if normalized_line is None:
            continue
        articles.append(
            parse_article_record(
                normalized_line,
                source=Path(source_path).name,
                source_line=line_no,
            )
        )
    return articles


def build_canonical_corpus(laws_dir: str | Path, output_dir: str | Path) -> BuildCorpusResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    articles = iter_normalized_articles(laws_dir)
    jsonl_path = output_path / "normalized_articles.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"article_count": len(articles)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return BuildCorpusResult(article_count=len(articles), output_dir=output_path)
