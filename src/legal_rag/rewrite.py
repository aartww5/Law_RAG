import importlib.util
import logging

from legal_rag.config import DEFAULT_OLLAMA_MODEL
from legal_rag.types import ConversationState, ConversationTurn, RewriteResult


LOGGER = logging.getLogger(__name__)
LLM_REWRITE_SYSTEM_PROMPT = """你是法律检索问题改写器。
你的唯一任务是：根据最近几轮对话和当前用户问题，输出一个适合检索的、自包含的问题。

严格规则：
1. 只能重组和改写历史中已经出现的信息。
2. 不得补充历史中未出现的新事实。
3. 不得回答问题，不得给出法律结论，不得解释原因。
4. 如果当前问题已经足够完整，只做最小改写或原样输出。
5. 只输出最终改写后的一个问题，不要输出分析、标签、项目符号或多余文字。"""


class QueryRewriter:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        *,
        enable_ollama: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enable_ollama = enable_ollama

    def rewrite(self, raw_query: str, state: ConversationState | None = None) -> RewriteResult:
        query = raw_query.strip()
        if not query:
            return RewriteResult(original_query="", rewritten_query="", rewrite_notes=["unchanged"])

        window_turns = self._get_window_turns(state)
        if not window_turns:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_notes=["unchanged"],
            )

        llm_rewritten = None
        llm_notes: list[str] = []
        if self.enable_ollama and importlib.util.find_spec("ollama") is not None:
            try:
                llm_rewritten = self._rewrite_with_llm(query, window_turns)
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                LOGGER.warning("query rewrite llm failed: %s", exc)
                llm_notes.append("llm_error")
        else:
            llm_notes.append("llm_unavailable")

        if llm_rewritten:
            rewritten = self._clean_llm_output(llm_rewritten, query)
            notes = [*llm_notes, "llm_rewritten" if rewritten != query else "llm_unchanged"]
            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                rewrite_notes=notes,
            )

        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            rewrite_notes=[*llm_notes, "unchanged"],
        )

    def _rewrite_with_llm(self, query: str, turns: list[ConversationTurn]) -> str:
        import ollama

        history = self._format_history_for_llm(turns)
        prompt = (
            "请将当前问题改写为一个适合法律检索的自包含问题。\n\n"
            f"最近对话窗口：\n{history}\n\n"
            f"当前用户问题：{query}\n"
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": LLM_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            options={"temperature": 0},
        )
        return response["message"]["content"].strip()

    def _format_history_for_llm(self, turns: list[ConversationTurn]) -> str:
        lines: list[str] = []
        for index, turn in enumerate(turns, start=1):
            lines.append(f"第{index}轮用户问题：{(turn.raw_query or '').strip()}")
            if turn.answer_summary:
                lines.append(f"第{index}轮系统回答摘要：{turn.answer_summary.strip()}")
        return "\n".join(lines)

    def _clean_llm_output(self, text: str, fallback: str) -> str:
        content = text.strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return fallback
        candidate = lines[-1].strip().strip("`").strip("\"' ")

        for prefix in ("改写后的问题：", "改写后问题：", "重写后的问题：", "问题："):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                break
        return candidate or fallback

    def _get_window_turns(self, state: ConversationState | None) -> list[ConversationTurn]:
        if state is None or not state.turns:
            return []
        window_size = max(1, state.max_turns)
        return state.turns[-window_size:]
