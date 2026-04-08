import importlib.util
import logging
from textwrap import shorten

from legal_rag.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_NUM_CTX, DEFAULT_OLLAMA_NUM_PREDICT
from legal_rag.generation.prompts import build_legal_prompt
from legal_rag.generation.stream import iter_text_chunks
from legal_rag.types import AnswerContext, FinalAnswer


LOGGER = logging.getLogger(__name__)


class SimpleGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        *,
        enable_ollama: bool = True,
        num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
        num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT,
    ) -> None:
        self.model_name = model_name
        self.enable_ollama = enable_ollama
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(self, context: AnswerContext) -> FinalAnswer:
        answer_text = self._generate_answer_text(context)
        return FinalAnswer(
            answer_text=answer_text,
            route_decision=context.route_decision,
            context=context,
        )

    def stream_generate(self, context: AnswerContext):
        if self.enable_ollama and importlib.util.find_spec("ollama") is not None:
            try:
                import ollama

                prompt = build_legal_prompt(context)
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    options=self._build_ollama_options(),
                )
                chunk_count = 0
                char_count = 0
                done_reason = "not_reported"
                for chunk in response:
                    message = chunk.get("message", {})
                    text = message.get("content", "")
                    if text:
                        chunk_count += 1
                        char_count += len(text)
                        yield text
                    if chunk.get("done"):
                        done_reason = str(chunk.get("done_reason") or "unknown")
                LOGGER.info(
                    "generation_stream_complete model=%s done_reason=%s chunk_count=%s char_count=%s",
                    self.model_name,
                    done_reason,
                    chunk_count,
                    char_count,
                )
                return
            except Exception:
                pass

        yield from iter_text_chunks(self._generate_answer_text(context))

    def _generate_answer_text(self, context: AnswerContext) -> str:
        if self.enable_ollama and importlib.util.find_spec("ollama") is not None:
            try:
                import ollama

                prompt = build_legal_prompt(context)
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options=self._build_ollama_options(),
                )
                LOGGER.info(
                    "generation_complete model=%s done_reason=%s",
                    self.model_name,
                    response.get("done_reason", "not_reported"),
                )
                return response["message"]["content"].strip()
            except Exception:
                pass

        if not context.docs:
            return "No sufficient legal basis was retrieved. Please provide a more specific law name, article number, or factual description."

        first_doc = context.docs[0]
        law_name = first_doc.metadata.get("law_name", "Relevant law")
        article_id = first_doc.metadata.get("article_id_cn") or first_doc.canonical_id
        summary = shorten(first_doc.content, width=120, placeholder="...")
        lines = [
            "The most relevant retrieved legal basis is:",
            f"1. {law_name} {article_id}: {summary}",
        ]
        if len(context.docs) > 1:
            lines.append(f"{len(context.docs) - 1} more supporting articles were also retrieved.")
        lines.append("A final conclusion still depends on the concrete facts, timing, and dispute focus.")
        return "\n".join(lines)

    def _build_ollama_options(self) -> dict[str, int]:
        return {
            "temperature": 0,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
