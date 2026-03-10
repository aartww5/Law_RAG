from dataclasses import dataclass, field


@dataclass
class NormalizedArticle:
    canonical_id: str
    law_name: str
    law_aliases: list[str]
    article_id_cn: str | None
    article_id_num: str | None
    content: str
    chapter: str | None
    section: str | None
    source: str
    source_line: int


@dataclass
class RetrievedDoc:
    canonical_id: str
    content: str
    metadata: dict
    score: float
    score_breakdown: dict[str, float]
    retriever: str


@dataclass
class RetrievalResult:
    docs: list[RetrievedDoc]
    confidence: float
    latency_ms: float
    reasons: list[str]
    raw_signals: dict


@dataclass
class RouteDecision:
    selected_mode: str
    fallback_triggered: bool
    confidence: float
    merge_policy: str
    reasons: list[str]


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    rewrite_notes: list[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    raw_query: str
    rewritten_query: str
    answer_summary: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 4

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]


@dataclass
class AnswerContext:
    question: str
    docs: list[RetrievedDoc]
    route_decision: RouteDecision
    citations: list[str]
    source_summary: dict


@dataclass
class PreparedAnswer:
    raw_query: str
    rewrite_result: RewriteResult
    context: AnswerContext


@dataclass
class FinalAnswer:
    answer_text: str
    route_decision: RouteDecision
    context: AnswerContext
    rewrite_result: RewriteResult | None = None
