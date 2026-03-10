from legal_rag.types import AnswerContext


def build_legal_prompt(context: AnswerContext) -> str:
    evidence = "\n\n".join(doc.content for doc in context.docs)
    return f"""你是一位专业的法律顾问。

相关法律条文：
{evidence}

用户问题：{context.question}
"""

