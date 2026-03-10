import re

from legal_rag.types import NormalizedArticle, RetrievedDoc, RetrievalResult


ARTICLE_NUM_RE = re.compile(r"\u7b2c?\s*([0-9]{1,6})\s*\u6761")
ARTICLE_CN_RE = re.compile(r"\u7b2c([0-9\u4e00-\u9fff]{1,12})\u6761")

_DIGITS = {
    "\u96f6": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u4e24": 2,
}
_UNITS = {"\u5341": 10, "\u767e": 100, "\u5343": 1000, "\u4e07": 10000}


def to_retrieved_doc(article: NormalizedArticle, score: float, retriever: str) -> RetrievedDoc:
    return RetrievedDoc(
        canonical_id=article.canonical_id,
        content=article.content,
        metadata={
            "law_name": article.law_name,
            "law_aliases": article.law_aliases,
            "article_id_cn": article.article_id_cn,
            "article_id_num": article.article_id_num,
            "source": article.source,
            "source_line": article.source_line,
        },
        score=score,
        score_breakdown={"exact_match": score},
        retriever=retriever,
    )


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


def extract_article_num(question: str) -> str | None:
    article_num_match = ARTICLE_NUM_RE.search(question)
    if article_num_match:
        return article_num_match.group(1)

    article_cn_match = ARTICLE_CN_RE.search(question)
    if article_cn_match:
        return str(_chinese_number_to_int(article_cn_match.group(1)))
    return None


class ExactMatchRetriever:
    def __init__(self, articles: list[NormalizedArticle]) -> None:
        self._articles = articles

    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        article_num = extract_article_num(question)

        matches: list[NormalizedArticle] = []
        reasons: list[str] = []
        for article in self._articles:
            alias_hit = any(alias and alias in question for alias in article.law_aliases)
            article_num_hit = article_num is not None and article.article_id_num == article_num
            if alias_hit and article_num_hit:
                matches.append(article)
                reasons = ["exact_law_match", "exact_article_match"]

        docs = [to_retrieved_doc(article, score=1.0, retriever="exact_match") for article in matches]
        confidence = 1.0 if docs else 0.0
        return RetrievalResult(
            docs=docs,
            confidence=confidence,
            latency_ms=0.0,
            reasons=reasons,
            raw_signals={"match_count": len(docs), "matched_article_num": article_num},
        )
