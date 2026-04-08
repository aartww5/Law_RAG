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


def _install_ollama_available(monkeypatch) -> None:
    fake_importlib = type(
        "FakeImportlib",
        (),
        {"util": type("FakeUtil", (), {"find_spec": staticmethod(lambda name: object())})},
    )
    monkeypatch.setattr(decomposition_module, "importlib", fake_importlib, raising=False)


def test_decomposer_builds_query_variants_from_issue_extraction_payload(monkeypatch) -> None:
    def fake_chat(*, model, messages, stream, options, **kwargs):
        return {
            "message": {
                "content": (
                    "<think>analysis</think>\n"
                    '{"issues":["未成年人行为能力","民事法律行为效力"],'
                    '"doctrines":["不满八周岁的未成年人","无民事行为能力人实施的民事法律行为无效",'
                    '"民事法律行为无效后返还财产"]}'
                )
            }
        }

    _install_ollama_available(monkeypatch)
    install_fake_ollama(monkeypatch, fake_chat)

    decomposer = decomposition_module.OllamaQueryDecomposer(
        model_name="demo-model",
        enable_ollama=True,
        max_queries=6,
    )

    variants = decomposer.decompose(
        "小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？"
    )

    assert [variant.text for variant in variants] == [
        "小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？",
        "不满八周岁的未成年人",
        "无民事行为能力人实施的民事法律行为无效",
        "民事法律行为无效后返还财产",
        "未成年人行为能力",
        "民事法律行为效力",
    ]
    assert [variant.source for variant in variants] == [
        "original",
        "statutory_phrase",
        "legal_concept",
        "legal_concept",
        "issue",
        "issue",
    ]


def test_decomposer_retries_with_repair_when_issue_extraction_is_incomplete(monkeypatch) -> None:
    calls: list[str] = []

    def fake_chat(*, model, messages, stream, options, **kwargs):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return {"message": {"content": '{"issues":["未成年人行为能力"],"doctrines":[]}'}}  # too thin
        return {
            "message": {
                "content": (
                    '{"issues":["未成年人行为能力"],'
                    '"doctrines":["不满八周岁的未成年人","无民事行为能力人实施的民事法律行为无效"]}'
                )
            }
        }

    _install_ollama_available(monkeypatch)
    install_fake_ollama(monkeypatch, fake_chat)

    decomposer = decomposition_module.OllamaQueryDecomposer(
        model_name="demo-model",
        enable_ollama=True,
        max_queries=4,
    )

    variants = decomposer.decompose(
        "小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？"
    )

    assert len(calls) == 2
    assert [variant.text for variant in variants] == [
        "小刘7岁时，将父亲送给他的一块手表卖给了二手商店，其父母能要求退回吗？",
        "不满八周岁的未成年人",
        "无民事行为能力人实施的民事法律行为无效",
        "未成年人行为能力",
    ]


def test_decomposer_bypasses_issue_extraction_for_direct_statute_lookup(monkeypatch) -> None:
    calls: list[str] = []

    def fake_chat(*, model, messages, stream, options, **kwargs):
        calls.append(messages[-1]["content"])
        return {"message": {"content": '{"issues":[],"doctrines":["不满十八周岁"]}' }}

    _install_ollama_available(monkeypatch)
    install_fake_ollama(monkeypatch, fake_chat)

    decomposer = decomposition_module.OllamaQueryDecomposer(
        model_name="demo-model",
        enable_ollama=True,
        max_queries=4,
    )

    variants = decomposer.decompose("民法典第二十条是什么？")

    assert calls == []
    assert len(variants) == 1
    assert variants[0].source == "original"
    assert variants[0].text == "民法典第二十条是什么？"


def test_decomposition_prompt_requests_issue_and_doctrine_json() -> None:
    prompt = decomposition_module.DECOMPOSITION_SYSTEM_PROMPT

    assert '"issues"' in prompt
    assert '"doctrines"' in prompt
    assert '"queries"' not in prompt
    assert "买到假冒伪劣商品" not in prompt
    assert "小马路遇一儿童落水" not in prompt
