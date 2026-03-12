import importlib.util
from textwrap import shorten

from legal_rag.config import DEFAULT_OLLAMA_MODEL
from legal_rag.generation.prompts import build_legal_prompt
from legal_rag.generation.stream import iter_text_chunks
from legal_rag.types import AnswerContext, FinalAnswer


class SimpleGenerator:
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, *, enable_ollama: bool = True) -> None:
        self.model_name = model_name
        self.enable_ollama = enable_ollama

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
                    options={"temperature": 0},
                )
                for chunk in response:
                    text = chunk["message"]["content"]
                    if text:
                        yield text
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
                    options={"temperature": 0},
                )
                return response["message"]["content"].strip()
            except Exception:
                pass

        if not context.docs:
            return "未检索到足够的法律依据，请尝试补充更具体的法名、条号或事实描述。"

        first_doc = context.docs[0]
        law_name = first_doc.metadata.get("law_name", "相关法律")
        article_id = first_doc.metadata.get("article_id_cn") or first_doc.canonical_id
        summary = shorten(first_doc.content, width=120, placeholder="...")
        lines = [
            "根据检索到的法律条文，当前最相关的依据如下：",
            f"1. 《{law_name}》{article_id}: {summary}",
        ]
        if len(context.docs) > 1:
            lines.append(f"另检索到 {len(context.docs) - 1} 条可辅助参考的相关条文。")
        lines.append("如需更精确结论，请结合具体事实、时间节点和争议焦点进一步核对。")
        return "\n".join(lines)
