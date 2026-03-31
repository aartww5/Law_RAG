from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import legal_rag.utils.query_decomposition as decomposition_module


def install_fake_ollama(monkeypatch, chat_impl) -> None:
    fake_ollama = ModuleType("ollama")
    fake_ollama.__spec__ = ModuleSpec("ollama", loader=None)
    fake_ollama.chat = chat_impl
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)


def test_decomposer_parses_chinese_queries_after_think_wrapper(monkeypatch) -> None:
    def fake_chat(*, model, messages, stream, options, **kwargs):
        return {
            "message": {
                "content": (
                    "<think>analysis</think>\n"
                    '{"queries":['
                    '{"source":"fact_focus","text":"7岁 手表 卖给 二手商店"},'
                    '{"source":"statutory_phrase","text":"无民事行为能力人 实施的民事法律行为无效"}'
                    "]}"
                )
            }
        }

    fake_importlib = type(
        "FakeImportlib",
        (),
        {"util": type("FakeUtil", (), {"find_spec": staticmethod(lambda name: object())})},
    )
    monkeypatch.setattr(decomposition_module, "importlib", fake_importlib, raising=False)
    install_fake_ollama(monkeypatch, fake_chat)

    decomposer = decomposition_module.OllamaQueryDecomposer(
        model_name="demo-model",
        enable_ollama=True,
        max_queries=4,
    )

    variants = decomposer.decompose("小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？")

    assert len(variants) == 3
    assert variants[0].source == "original"
    assert variants[1].text == "7岁 手表 卖给 二手商店"
    assert variants[2].text == "无民事行为能力人 实施的民事法律行为无效"


def test_decomposer_retries_with_llm_repair_when_first_response_has_too_few_queries(monkeypatch) -> None:
    calls: list[str] = []

    def fake_chat(*, model, messages, stream, options, **kwargs):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return {"message": {"content": '{"queries":[{"source":"fact_focus","text":"7岁"}]}'}}
        return {
            "message": {
                "content": (
                    '{"queries":['
                    '{"source":"fact_focus","text":"7岁 手表 买卖"},'
                    '{"source":"statutory_phrase","text":"民事法律行为无效 返还财产"}'
                    "]}"
                )
            }
        }

    fake_importlib = type(
        "FakeImportlib",
        (),
        {"util": type("FakeUtil", (), {"find_spec": staticmethod(lambda name: object())})},
    )
    monkeypatch.setattr(decomposition_module, "importlib", fake_importlib, raising=False)
    install_fake_ollama(monkeypatch, fake_chat)

    decomposer = decomposition_module.OllamaQueryDecomposer(
        model_name="demo-model",
        enable_ollama=True,
        max_queries=4,
    )

    variants = decomposer.decompose("小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？")

    assert len(calls) == 2
    assert len(variants) == 3
    assert any(variant.text == "7岁 手表 买卖" for variant in variants)
    assert any(variant.text == "民事法律行为无效 返还财产" for variant in variants)


def test_decomposer_recovers_queries_from_truncated_reasoning_output(monkeypatch) -> None:
    def fake_chat(*, model, messages, stream, options, **kwargs):
        return {
            "message": {
                "content": (
                    "<think>\n"
                    "查询 1（Fact Focus）：提炼为“7岁 手表 卖给 二手商店”。\n"
                    "查询 2（Statutory Phrase）：提炼为“无民事行为能力人 实施的民事法律行为无效”。\n"
                    "查询 3（Statutory Phrase）：提炼为“民事法律行为无效 返还财产”。\n"
                    "</think>\n"
                    '{"queries":[{"source":"fact_focus","text":"7岁 手表 卖给 二手商店'
                )
            }
        }

    fake_importlib = type(
        "FakeImportlib",
        (),
        {"util": type("FakeUtil", (), {"find_spec": staticmethod(lambda name: object())})},
    )
    monkeypatch.setattr(decomposition_module, "importlib", fake_importlib, raising=False)
    install_fake_ollama(monkeypatch, fake_chat)

    decomposer = decomposition_module.OllamaQueryDecomposer(
        model_name="demo-model",
        enable_ollama=True,
        max_queries=4,
    )

    variants = decomposer.decompose("小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？")

    assert len(variants) == 4
    assert variants[1].text == "7岁 手表 卖给 二手商店"
    assert variants[2].text == "无民事行为能力人 实施的民事法律行为无效"
    assert variants[3].text == "民事法律行为无效 返还财产"


def test_decomposition_prompt_avoids_benchmark_specific_few_shot_examples() -> None:
    prompt = decomposition_module.DECOMPOSITION_SYSTEM_PROMPT

    assert "买到假冒伪劣商品" not in prompt
    assert "小马路遇一儿童落水" not in prompt
    assert "至少一条额外查询要保留这些事实" in prompt
