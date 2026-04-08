from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import logging
import re
from typing import Protocol

from legal_rag.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_NUM_CTX
from legal_rag.utils.text import normalize_question


LOGGER = logging.getLogger(__name__)
DEFAULT_DECOMPOSITION_NUM_PREDICT = 1024
SOURCE_WEIGHTS = {
    "issue": 0.8,
    "fact_focus": 0.7,
    "legal_concept": 0.65,
    "law_alias": 0.55,
    "statutory_phrase": 0.9,
    "decomposition": 0.6,
}
DECOMPOSITION_SYSTEM_PROMPT = """你是法律检索争点提取器。你的任务是从用户问题中提取法律检索争点和规范概念，供程序后续构造检索查询。

规则：
1. 保持与用户相同的语言。中文问题必须输出中文争点和中文规范概念。
2. 不得虚构新事实、日期、主体、结果或条号。
3. issues 只写检索争点，不要写完整答案、分析结论或长句。
4. doctrines 只写适合法条检索的规范概念或规范短语，优先提炼可能直接出现在法条原文中的表述，例如构成要件、责任后果、行为效力、返还义务、法定阈值表述。
5. 如果原问题包含年龄、金额、数量、时间、身份关系或行为对象，提炼 doctrines 时不得丢失这些关键事实锚点。
6. 除非问题明确涉及，否则不要引入相邻但不直接对应争点的概念或无关法域。
7. 只输出 JSON，格式如下：
{"issues":["..."],"doctrines":["..."]}
8. 非“法名+条号”类问题必须至少输出 1 个 issue 和 1 个 doctrine；总提取项少于 2 视为失败。
9. 最多输出 2 个 issues 和 3 个 doctrines，要求简短、可检索、去掉解释性文字。

输出要求：
- 不要输出 markdown、解释、代码块、<think> 标签或其他额外文本。
"""
REPAIR_SYSTEM_PROMPT = """你是法律检索查询拆解修复器。

任务：把给定法律问题修复成合法、可解析的 JSON 争点提取输出。
要求：
1. 必须输出 JSON，格式为 {"issues":[...],"doctrines":[...]}。
2. 必须保留中文问题的中文争点和中文规范概念。
3. 非“法名+条号”类问题至少给出 1 个 issue 和 1 个 doctrine；总提取项少于 2 视为失败。
4. doctrines 优先给出规范短语、责任后果和法定阈值表述。
5. 如果原问题包含年龄、金额、数量、时间或身份关系，doctrines 中至少一条要保留这些关键事实。
6. 禁止解释、分析、markdown、代码块、<think> 标签。
"""


@dataclass(frozen=True)
class QueryVariant:
    text: str
    weight: float
    source: str


class QueryDecomposer(Protocol):
    def decompose(self, question: str, *, background: str = "") -> list[QueryVariant]:
        ...


class OllamaQueryDecomposer:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        *,
        enable_ollama: bool = True,
        max_queries: int = 4,
        num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
        num_predict: int = DEFAULT_DECOMPOSITION_NUM_PREDICT,
    ) -> None:
        self.model_name = model_name
        self.enable_ollama = enable_ollama
        self.max_queries = max(1, max_queries)
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def decompose(self, question: str, *, background: str = "") -> list[QueryVariant]:
        normalized = normalize_question(question)
        if not normalized:
            return []

        variants = [QueryVariant(text=normalized, weight=1.0, source="original")]
        if _looks_like_direct_statute_lookup(normalized):
            return variants
        expects_chinese = _contains_chinese(normalized)
        if not self.enable_ollama or self.max_queries <= 1:
            return variants
        if importlib.util.find_spec("ollama") is None:
            return variants

        raw_output = ""
        try:
            raw_output = self._chat(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=self._build_user_prompt(normalized, background=background),
            )
            candidates = self._parse_variants(raw_output)
            if not self._has_sufficient_candidates(candidates):
                repair_output = self._chat(
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    user_prompt=self._build_repair_prompt(
                        normalized,
                        background=background,
                        previous_output=raw_output,
                    ),
                )
                candidates = self._parse_variants(repair_output)
                if repair_output:
                    raw_output = repair_output
        except Exception as exc:  # pragma: no cover - depends on local ollama runtime
            LOGGER.warning("query decomposition llm failed: %s", exc)
            return variants

        seen = {normalized}
        added = 0
        for candidate in candidates:
            if len(variants) >= self.max_queries:
                break
            candidate_text = normalize_question(candidate.text)
            if not candidate_text or candidate_text in seen:
                continue
            if expects_chinese and not _contains_chinese(candidate_text):
                continue
            variants.append(QueryVariant(text=candidate_text, weight=candidate.weight, source=candidate.source))
            seen.add(candidate_text)
            added += 1

        LOGGER.info(
            "query_decomposition question=%r variant_count=%s variants=%s raw_output=%r",
            normalized,
            len(variants),
            [variant.text for variant in variants],
            _truncate_for_log(raw_output),
        )
        return variants

    def _chat(self, *, system_prompt: str, user_prompt: str) -> str:
        import ollama

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            think=False,
            format="json",
            options={
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        )
        return response["message"]["content"].strip()

    def _build_user_prompt(self, question: str, *, background: str = "") -> str:
        lines = [
            f"原问题：{question}",
        ]
        if background:
            lines.append(f"背景：{background}")
        lines.extend(
            [
                "请提取 1 到 2 个 issues，以及 1 到 3 个 doctrines。",
                '请直接填充 JSON 模板：{"issues":["..."],"doctrines":["..."]}',
                "issues 写检索争点，doctrines 写法条原文可能出现的规范短语或法律概念。",
                "如果原问题包含年龄、金额、数量、时间或身份关系，至少一条 doctrine 要保留这些关键事实。",
                "只输出 JSON，不要解释。",
            ]
        )
        return "\n".join(lines)

    def _build_repair_prompt(self, question: str, *, background: str = "", previous_output: str = "") -> str:
        lines = [
            f"原问题：{question}",
        ]
        if background:
            lines.append(f"背景：{background}")
        if previous_output:
            lines.append(f"上一次输出：{previous_output}")
        lines.extend(
            [
                "上一次输出无效或没有可用争点提取。",
                "请重新输出合法 JSON。",
                "至少给出 1 个中文 issue 和 1 个中文 doctrine。",
                '请直接填充 JSON 模板：{"issues":["..."],"doctrines":["..."]}',
                "如果原问题包含年龄、金额、数量、时间或身份关系，至少一条 doctrine 要保留这些关键事实。",
                "只输出 JSON，不要解释。",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _has_sufficient_candidates(candidates: list[QueryVariant]) -> bool:
        return len(candidates) >= 2

    def _parse_variants(self, text: str) -> list[QueryVariant]:
        payload = self._parse_json_payload(text)
        variants = self._build_variants_from_issue_payload(payload)
        if not variants:
            raw_queries = payload.get("queries", [])
            if isinstance(raw_queries, dict):
                raw_queries = list(raw_queries.values())
            if isinstance(raw_queries, list):
                variants = self._build_variants_from_query_items(raw_queries)
        if variants:
            return variants
        return _recover_variants_from_reasoning(text)

    def _build_variants_from_issue_payload(self, payload: dict) -> list[QueryVariant]:
        doctrines = _normalize_string_items(payload.get("doctrines", []))
        issues = _normalize_string_items(payload.get("issues", []))
        variants: list[QueryVariant] = []
        seen: set[str] = set()

        for index, candidate_text in enumerate(doctrines):
            source = "statutory_phrase" if index == 0 else "legal_concept"
            if candidate_text in seen:
                continue
            seen.add(candidate_text)
            variants.append(
                QueryVariant(
                    text=candidate_text,
                    weight=SOURCE_WEIGHTS[source],
                    source=source,
                )
            )

        for candidate_text in issues:
            if candidate_text in seen:
                continue
            seen.add(candidate_text)
            variants.append(
                QueryVariant(
                    text=candidate_text,
                    weight=SOURCE_WEIGHTS["issue"],
                    source="issue",
                )
            )
        return variants

    def _build_variants_from_query_items(self, raw_queries: list) -> list[QueryVariant]:
        variants: list[QueryVariant] = []
        for item in raw_queries:
            if isinstance(item, str):
                candidate_text = normalize_question(item.strip())
                if candidate_text:
                    variants.append(
                        QueryVariant(
                            text=candidate_text,
                            weight=SOURCE_WEIGHTS["decomposition"],
                            source="decomposition",
                        )
                    )
                continue
            if not isinstance(item, dict):
                continue
            candidate_text = normalize_question(str(item.get("text", "")).strip())
            if not candidate_text:
                continue
            source = str(item.get("source", "decomposition")).strip().lower() or "decomposition"
            weight = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["decomposition"])
            variants.append(QueryVariant(text=candidate_text, weight=weight, source=source))
        return variants

    @staticmethod
    def _parse_json_payload(text: str) -> dict:
        cleaned = _strip_reasoning_and_fences(text)

        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


def _strip_reasoning_and_fences(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _truncate_for_log(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _normalize_string_items(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        normalized = normalize_question(str(item).strip())
        if normalized:
            items.append(normalized)
    return items


def _looks_like_direct_statute_lookup(text: str) -> bool:
    normalized = normalize_question(text)
    if not normalized:
        return False
    has_law_name = bool(re.search(r"(法|法典|条例|规定|解释)", normalized))
    has_article = bool(re.search(r"(第[\d一二三四五六七八九十百千万零两]+条|article\s*\d+)", normalized, flags=re.IGNORECASE))
    return has_law_name and has_article


def _recover_variants_from_reasoning(text: str) -> list[QueryVariant]:
    candidates: list[QueryVariant] = []
    seen: set[str] = set()
    patterns = [
        re.compile(
            r"查询\s*\d+\s*[（(]?(issue|fact[_ ]?focus|legal[_ ]?concept|law[_ ]?alias|statutory[_ ]?phrase)?[）)]?[^“\"\n]*[“\"]([^”\"\n]+)[”\"]",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"提炼为[“\"]([^”\"\n]+)[”\"]",
            flags=re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(text):
            if pattern.groups == 2:
                source = (match.group(1) or "decomposition").replace(" ", "_").lower()
                candidate_text = match.group(2)
            else:
                source = "decomposition"
                candidate_text = match.group(1)
            candidate_text = normalize_question(candidate_text.strip())
            if not candidate_text or candidate_text in seen:
                continue
            seen.add(candidate_text)
            candidates.append(
                QueryVariant(
                    text=candidate_text,
                    weight=SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["decomposition"]),
                    source=source,
                )
            )
    return candidates
