from __future__ import annotations

from legal_rag.utils.query_expansion import QueryVariant
from legal_rag.utils.text import normalize_question


GENERIC_MATCH_TERMS: tuple[str, ...] = ("赔偿", "责任", "损失", "处理")
PHRASE_WEIGHT_OVERRIDES = {
    "买卖不破租赁": 0.45,
    "执行工作任务": 0.45,
    "惩罚性赔偿": 0.35,
    "退一赔三": 0.35,
    "用人单位": 0.25,
    "侵权责任": 0.2,
    "承租人": 0.18,
    "房屋租赁": 0.22,
}


def build_doc_search_text(content: str, metadata: dict) -> str:
    parts = [
        metadata.get("law_name", ""),
        " ".join(metadata.get("law_aliases", []) or []),
        metadata.get("article_id_cn", ""),
        metadata.get("article_id_num", ""),
        content,
    ]
    return normalize_question(" ".join(str(part) for part in parts if part))


def compute_rerank_bonus(
    question: str,
    *,
    content: str,
    metadata: dict,
    query_variants: list[QueryVariant],
) -> tuple[float, dict[str, float]]:
    search_text = build_doc_search_text(content, metadata)
    if not search_text:
        return 0.0, {"rerank": 0.0, "generic_penalty": 0.0}

    matched_phrases: set[str] = set()
    for variant in query_variants[1:]:
        for phrase in variant.concepts:
            normalized_phrase = normalize_question(phrase)
            if normalized_phrase and normalized_phrase in search_text:
                matched_phrases.add(normalized_phrase)

    rerank_bonus = min(sum(_phrase_weight(phrase) for phrase in matched_phrases), 1.2)

    generic_penalty = 0.0
    if rerank_bonus == 0.0:
        generic_hits = sum(1 for term in GENERIC_MATCH_TERMS if term in question and term in search_text)
        generic_penalty = min(0.08 * generic_hits, 0.2)

    total_bonus = rerank_bonus - generic_penalty
    return total_bonus, {
        "rerank": rerank_bonus,
        "generic_penalty": generic_penalty,
    }


def _phrase_weight(phrase: str) -> float:
    if phrase in PHRASE_WEIGHT_OVERRIDES:
        return PHRASE_WEIGHT_OVERRIDES[phrase]
    if phrase.endswith("法") or phrase.endswith("法典"):
        return 0.18
    if len(phrase) >= 6:
        return 0.24
    if len(phrase) >= 4:
        return 0.18
    return 0.12
