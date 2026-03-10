import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_article_line_is_normalized_and_parsed() -> None:
    assert importlib.util.find_spec("legal_rag.documents") is not None
    assert importlib.util.find_spec("legal_rag.documents.aliases") is not None
    assert importlib.util.find_spec("legal_rag.documents.article_parser") is not None
    assert importlib.util.find_spec("legal_rag.documents.normalize") is not None

    aliases_module = importlib.import_module("legal_rag.documents.aliases")
    parser_module = importlib.import_module("legal_rag.documents.article_parser")
    normalize_module = importlib.import_module("legal_rag.documents.normalize")

    build_law_aliases = aliases_module.build_law_aliases
    parse_article_record = parser_module.parse_article_record
    normalize_law_line = normalize_module.normalize_law_line

    line, action = normalize_law_line(
        "第一千一百三十八条 口头遗嘱...",
        "中华人民共和国民法典",
    )
    record = parse_article_record(line, source="民法典.txt", source_line=1)
    aliases = build_law_aliases("中华人民共和国民法典")

    assert action == "prefix_article"
    assert record.article_id_num == "1138"
    assert "民法典" in aliases
