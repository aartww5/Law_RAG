import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_prompt_includes_question_and_quoted_articles() -> None:
    assert importlib.util.find_spec("legal_rag.generation.prompts") is not None

    prompts_module = importlib.import_module("legal_rag.generation.prompts")
    types_module = importlib.import_module("legal_rag.types")

    build_legal_prompt = prompts_module.build_legal_prompt
    AnswerContext = types_module.AnswerContext
    RetrievedDoc = types_module.RetrievedDoc
    RouteDecision = types_module.RouteDecision

    context = AnswerContext(
        question="口头遗嘱有效吗",
        docs=[
            RetrievedDoc(
                "law:1138",
                "《中华人民共和国民法典》第一千一百三十八条 口头遗嘱...",
                {},
                1.0,
                {},
                "hybrid",
            )
        ],
        route_decision=RouteDecision("hybrid", False, 0.9, "hybrid_plus_exact", ["exact_article_match"]),
        citations=[],
        source_summary={},
    )
    prompt = build_legal_prompt(context)

    assert "用户问题：口头遗嘱有效吗" in prompt
    assert "第一千一百三十八条" in prompt
