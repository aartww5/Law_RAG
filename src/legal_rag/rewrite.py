import re

from legal_rag.types import ConversationState, ConversationTurn, RewriteResult


SUBJECT_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{1,12})(?:去世|死亡|身故)")
FOLLOW_UP_PATTERN = re.compile(r"(?:^他|^她|^其|^该|上述|前述|此|这|那|侄子|侄女|继承|受遗赠)")


class QueryRewriter:
    def rewrite(self, raw_query: str, state: ConversationState | None = None) -> RewriteResult:
        query = raw_query.strip()
        notes: list[str] = []
        rewritten = query
        last_turn = state.turns[-1] if state and state.turns else None

        if last_turn is not None:
            subject = self._extract_subject(last_turn)
            if subject:
                resolved = self._replace_subject_reference(rewritten, subject)
                if resolved != rewritten:
                    rewritten = resolved
                    notes.append("subject_resolved")

            if FOLLOW_UP_PATTERN.search(query):
                prior_context = last_turn.rewritten_query or last_turn.raw_query
                if prior_context and prior_context not in rewritten:
                    rewritten = f"基于前述情形：{prior_context}。追问：{rewritten}"
                    notes.append("history_attached")

        if rewritten == query:
            notes.append("unchanged")

        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            rewrite_notes=notes,
        )

    def _extract_subject(self, turn: ConversationTurn) -> str | None:
        for candidate in (turn.rewritten_query, turn.raw_query, turn.answer_summary):
            match = SUBJECT_PATTERN.search(candidate)
            if match:
                return match.group(1)
        return None

    def _replace_subject_reference(self, query: str, subject: str) -> str:
        if query.startswith("他"):
            return subject + query[1:]
        if query.startswith("她"):
            return subject + query[1:]
        if query.startswith("其"):
            return subject + "的" + query[1:]
        if query.startswith("该"):
            return subject + query[1:]
        return query
