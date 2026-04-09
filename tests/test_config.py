from pathlib import Path
import shutil
import sys
from uuid import uuid4


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_rag.config import AppConfig


def make_test_root() -> Path:
    root = Path(__file__).resolve().parents[2] / ".config_test_tmp" / uuid4().hex
    root.mkdir(parents=True)
    return root


def test_config_splits_runtime_and_index_settings() -> None:
    config = AppConfig()
    assert config.runtime.mode == "auto"
    assert config.runtime.max_context_articles == 6
    assert config.index.mini_working_dir.name == "minirag_working"


def test_config_can_bootstrap_from_project_root(tmp_path: Path, monkeypatch) -> None:
    laws_dir = tmp_path / "RAG" / "Chinese-Laws"
    laws_dir.mkdir(parents=True)
    mini_dir = tmp_path / "test" / "minirag_working"
    mini_dir.mkdir(parents=True)

    monkeypatch.setenv("LEGAL_RAG_MODE", "mini")

    config = AppConfig.from_env(tmp_path)

    assert config.runtime.mode == "mini"
    assert config.index.laws_dir == laws_dir
    assert config.index.mini_working_dir == mini_dir


def test_config_toml_sets_runtime_and_index_defaults() -> None:
    root = make_test_root()
    try:
        config_dir = root / "unified_app"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            "\n".join(
                [
                    "[runtime]",
                    'mode = "hybrid"',
                    "",
                    "[index]",
                    'laws_dir = "custom-laws"',
                ]
            ),
            encoding="utf-8",
        )

        config = AppConfig.from_env(root)

        assert config.runtime.mode == "hybrid"
        assert config.index.laws_dir == root / "custom-laws"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_env_overrides_config_toml(monkeypatch) -> None:
    root = make_test_root()
    try:
        config_dir = root / "unified_app"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            "\n".join(
                [
                    "[runtime]",
                    'mode = "hybrid"',
                    "",
                    "[index]",
                    'laws_dir = "custom-laws"',
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("LEGAL_RAG_MODE", "mini")

        config = AppConfig.from_env(root)

        assert config.runtime.mode == "mini"
        assert config.index.laws_dir == root / "custom-laws"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_config_reads_hybrid_index_settings_from_config_toml() -> None:
    root = make_test_root()
    try:
        config_dir = root / "unified_app"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            "\n".join(
                [
                    "[index]",
                    'qdrant_collection_name = "demo_collection"',
                    'embedding_model = "BAAI/bge-m3"',
                    'embedding_device = "cpu"',
                    'embedding_build_device = "cuda"',
                    'reranker_device = "cuda"',
                ]
            ),
            encoding="utf-8",
        )

        config = AppConfig.from_env(root)

        assert config.index.qdrant_collection_name == "demo_collection"
        assert config.index.embedding_model == "BAAI/bge-m3"
        assert config.index.embedding_device == "cpu"
        assert config.index.embedding_build_device == "cuda"
        assert config.index.reranker_device == "cuda"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_config_reads_retrieval_and_generation_limits_from_config_toml() -> None:
    root = make_test_root()
    try:
        config_dir = root / "unified_app"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            "\n".join(
                [
                    "[runtime]",
                    "ollama_num_ctx = 24576",
                    "ollama_num_predict = 1536",
                    "",
                    "[index]",
                    "bm25_top_k = 50",
                    "vector_top_k = 60",
                    "decomposition_max_queries = 4",
                ]
            ),
            encoding="utf-8",
        )

        config = AppConfig.from_env(root)

        assert config.runtime.ollama_num_ctx == 24576
        assert config.runtime.ollama_num_predict == 1536
        assert config.index.bm25_top_k == 50
        assert config.index.vector_top_k == 60
        assert config.index.decomposition_max_queries == 4
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_config_reads_local_auth_and_chat_storage_settings() -> None:
    root = make_test_root()
    try:
        config_dir = root / "unified_app"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            "\n".join(
                [
                    "[storage]",
                    'chat_db_path = "storage/chat.sqlite3"',
                    "",
                    "[[auth.users]]",
                    'username = "demo"',
                    'display_name = "Demo User"',
                    'password = "secret123"',
                    "",
                    "[[auth.users]]",
                    'username = "hashed"',
                    'display_name = "Hashed User"',
                    'password_hash = "sha256$abc123"',
                ]
            ),
            encoding="utf-8",
        )

        config = AppConfig.from_env(root)

        assert config.storage.chat_db_path == root / "storage" / "chat.sqlite3"
        assert [user.username for user in config.auth.users] == ["demo", "hashed"]
        assert config.auth.users[0].password == "secret123"
        assert config.auth.users[1].password_hash == "sha256$abc123"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_config_defaults_keep_auth_and_storage_empty() -> None:
    config = AppConfig()

    assert config.auth.users == []
    assert config.storage.chat_db_path.name == "chat_history.sqlite3"
