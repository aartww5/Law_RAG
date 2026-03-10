import re

from legal_rag.documents.aliases import build_law_aliases
from legal_rag.types import NormalizedArticle


LAW_RE = re.compile(r"^《([^》]+)》")
ARTICLE_RE = re.compile(r"(第[0-9一二三四五六七八九十百千万零两]+条)")

_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "两": 2,
}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _chinese_number_to_int(text: str) -> int:
    if text.isdigit():
        return int(text)

    total = 0
    section = 0
    number = 0

    for char in text:
        if char in _DIGITS:
            number = _DIGITS[char]
            continue

        unit = _UNITS.get(char)
        if unit is None:
            continue
        if unit == 10000:
            section = (section + number) * unit
            total += section
            section = 0
        else:
            section += (number or 1) * unit
        number = 0

    return total + section + number


def parse_article_record(line: str, source: str, source_line: int) -> NormalizedArticle:
    law_match = LAW_RE.search(line)
    article_match = ARTICLE_RE.search(line)

    law_name = law_match.group(1) if law_match else source.removesuffix(".txt")
    article_id_cn = article_match.group(1) if article_match else None
    article_id_num = None
    if article_id_cn:
        article_id_num = str(_chinese_number_to_int(article_id_cn[1:-1]))

    canonical_suffix = article_id_cn or f"line-{source_line}"
    return NormalizedArticle(
        canonical_id=f"{law_name}:{canonical_suffix}",
        law_name=law_name,
        law_aliases=build_law_aliases(law_name),
        article_id_cn=article_id_cn,
        article_id_num=article_id_num,
        content=line,
        chapter=None,
        section=None,
        source=source,
        source_line=source_line,
    )
