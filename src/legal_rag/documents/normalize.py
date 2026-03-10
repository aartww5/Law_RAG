import re


LAW_PREFIX_RE = re.compile(r"^《[^》]+》")
ARTICLE_RE = re.compile(r"^第[0-9一二三四五六七八九十百千万零两]+条")


def normalize_law_line(line: str, law_name: str) -> tuple[str | None, str]:
    text = " ".join(line.strip().split())
    if not text:
        return None, "skip_empty"
    if ARTICLE_RE.match(text):
        return f"《{law_name}》{text}", "prefix_article"
    if LAW_PREFIX_RE.match(text):
        return text, "keep_prefixed"
    return None, "skip_non_article"

