from __future__ import annotations

from dataclasses import dataclass

from legal_rag.utils.text import normalize_question


DEFAULT_EXPANSION_WEIGHT = 0.45
MAX_EXPANSIONS = 2


@dataclass(frozen=True)
class QueryVariant:
    text: str
    weight: float
    source: str
    concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpansionRule:
    source: str
    triggers: tuple[str, ...]
    concepts: tuple[str, ...]


_EXPANSION_RULES: tuple[ExpansionRule, ...] = (
    ExpansionRule(
        source="consumer_compensation",
        triggers=("假冒伪劣", "假货", "欺诈", "几倍赔偿", "三倍赔偿", "退一赔三"),
        concepts=("消费者权益保护法", "惩罚性赔偿", "退一赔三", "三倍赔偿", "欺诈"),
    ),
    ExpansionRule(
        source="prepaid_refund",
        triggers=("预付费", "预付款", "充值卡", "预付费卡", "跑路"),
        concepts=("消费者权益保护法", "预付款", "退回预付款", "行政投诉"),
    ),
    ExpansionRule(
        source="lease_transfer",
        triggers=("租房", "租赁", "房东", "卖房", "搬走", "租客"),
        concepts=("民法典", "买卖不破租赁", "房屋租赁", "承租人"),
    ),
    ExpansionRule(
        source="task_execution_liability",
        triggers=("外卖骑手", "送餐", "骑手", "配送", "撞伤", "撞人"),
        concepts=("民法典", "执行工作任务", "用人单位", "侵权责任"),
    ),
    ExpansionRule(
        source="parking_space",
        triggers=("停车位", "车位", "开发商", "业主"),
        concepts=("民法典", "建筑物区分所有权", "车位", "车库", "业主"),
    ),
    ExpansionRule(
        source="lost_property",
        triggers=("遗失物", "丢失", "拾得", "捡到"),
        concepts=("民法典", "拾得遗失物", "返还", "保管"),
    ),
)


def expand_query_variants(
    question: str,
    *,
    original_weight: float = 1.0,
    expansion_weight: float = DEFAULT_EXPANSION_WEIGHT,
    max_expansions: int = MAX_EXPANSIONS,
) -> list[QueryVariant]:
    normalized = normalize_question(question)
    if not normalized:
        return []

    variants = [QueryVariant(text=normalized, weight=original_weight, source="original")]
    seen = {normalized}
    expansion_count = 0

    for rule in _EXPANSION_RULES:
        if expansion_count >= max_expansions:
            break
        if not any(trigger in normalized for trigger in rule.triggers):
            continue

        concepts = tuple(concept for concept in rule.concepts if concept)
        expansion_text = normalize_question(" ".join((normalized, *concepts)))
        if not expansion_text or expansion_text in seen:
            continue

        variants.append(
            QueryVariant(
                text=expansion_text,
                weight=expansion_weight,
                source=rule.source,
                concepts=concepts,
            )
        )
        seen.add(expansion_text)
        expansion_count += 1

    return variants
