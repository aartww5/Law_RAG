from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib


DEFAULT_OLLAMA_MODEL = "qwen35-law:q6k-stable"
CONFIG_FILE_NAME = "config.toml"


@dataclass
class RuntimeConfig:
    mode: str = "auto"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    streaming: bool = True
    max_context_articles: int = 6
    max_history_turns: int = 4


@dataclass
class IndexConfig:
    laws_dir: Path = Path("RAG/Chinese-Laws")
    qdrant_path: Path = Path("unified_app/storage/qdrant")
    bm25_cache_path: Path = Path("unified_app/storage/bm25")
    mini_working_dir: Path = Path("unified_app/storage/minirag_working")
    corpus_dir: Path = Path("unified_app/storage/corpus")


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    @classmethod
    def from_env(cls, root_dir: str | Path) -> "AppConfig":
        root = Path(root_dir)
        config_data = _load_config_toml(root)
        runtime = _build_runtime_config(config_data)
        index = _build_index_config(root, config_data)
        return cls(runtime=runtime, index=index)


def _load_config_toml(root: Path) -> dict:
    config_path = root / "unified_app" / CONFIG_FILE_NAME
    if not config_path.exists():
        return {}
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _build_runtime_config(config_data: dict) -> RuntimeConfig:
    defaults = RuntimeConfig()
    runtime_data = config_data.get("runtime", {})

    runtime = RuntimeConfig(
        mode=_normalize_mode(runtime_data.get("mode", defaults.mode), defaults.mode),
        ollama_model=str(runtime_data.get("ollama_model", defaults.ollama_model)).strip() or defaults.ollama_model,
        streaming=_parse_bool(runtime_data.get("streaming", defaults.streaming), defaults.streaming),
        max_context_articles=_parse_int(
            runtime_data.get("max_context_articles", defaults.max_context_articles),
            defaults.max_context_articles,
        ),
        max_history_turns=_parse_int(
            runtime_data.get("max_history_turns", defaults.max_history_turns),
            defaults.max_history_turns,
        ),
    )

    if "LEGAL_RAG_MODE" in os.environ:
        runtime.mode = _normalize_mode(os.environ["LEGAL_RAG_MODE"], runtime.mode)
    if "LEGAL_RAG_OLLAMA_MODEL" in os.environ:
        runtime.ollama_model = os.environ["LEGAL_RAG_OLLAMA_MODEL"].strip() or runtime.ollama_model
    if "LEGAL_RAG_STREAMING" in os.environ:
        runtime.streaming = _parse_bool(os.environ["LEGAL_RAG_STREAMING"], runtime.streaming)
    if "LEGAL_RAG_MAX_CONTEXT_ARTICLES" in os.environ:
        runtime.max_context_articles = _parse_int(
            os.environ["LEGAL_RAG_MAX_CONTEXT_ARTICLES"],
            runtime.max_context_articles,
        )
    if "LEGAL_RAG_MAX_HISTORY_TURNS" in os.environ:
        runtime.max_history_turns = _parse_int(
            os.environ["LEGAL_RAG_MAX_HISTORY_TURNS"],
            runtime.max_history_turns,
        )
    return runtime


def _build_index_config(root: Path, config_data: dict) -> IndexConfig:
    defaults = IndexConfig()
    index_data = config_data.get("index", {})

    laws_dir = _resolve_path(index_data.get("laws_dir"), root, defaults.laws_dir)
    qdrant_path = _resolve_path(index_data.get("qdrant_path"), root, defaults.qdrant_path)
    bm25_cache_path = _resolve_path(index_data.get("bm25_cache_path"), root, defaults.bm25_cache_path)
    corpus_dir = _resolve_path(index_data.get("corpus_dir"), root, defaults.corpus_dir)

    mini_working_dir_value = index_data.get("mini_working_dir")
    if mini_working_dir_value is None:
        mini_working_dir = _default_mini_working_dir(root)
    else:
        mini_working_dir = _resolve_path(mini_working_dir_value, root, defaults.mini_working_dir)

    if "LEGAL_RAG_LAWS_DIR" in os.environ:
        laws_dir = _resolve_path(os.environ["LEGAL_RAG_LAWS_DIR"], root, laws_dir)
    if "LEGAL_RAG_QDRANT_PATH" in os.environ:
        qdrant_path = _resolve_path(os.environ["LEGAL_RAG_QDRANT_PATH"], root, qdrant_path)
    if "LEGAL_RAG_BM25_CACHE_PATH" in os.environ:
        bm25_cache_path = _resolve_path(os.environ["LEGAL_RAG_BM25_CACHE_PATH"], root, bm25_cache_path)
    if "LEGAL_RAG_CORPUS_DIR" in os.environ:
        corpus_dir = _resolve_path(os.environ["LEGAL_RAG_CORPUS_DIR"], root, corpus_dir)
    if "LEGAL_RAG_MINI_WORKING_DIR" in os.environ:
        mini_working_dir = _resolve_path(os.environ["LEGAL_RAG_MINI_WORKING_DIR"], root, mini_working_dir)

    return IndexConfig(
        laws_dir=laws_dir,
        qdrant_path=qdrant_path,
        bm25_cache_path=bm25_cache_path,
        mini_working_dir=mini_working_dir,
        corpus_dir=corpus_dir,
    )


def _resolve_path(value: str | Path | None, root: Path, fallback: Path) -> Path:
    if value is None:
        path = fallback
    else:
        path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _default_mini_working_dir(root: Path) -> Path:
    preferred = root / "test" / "minirag_working"
    if preferred.exists():
        return preferred
    return root / "unified_app" / "storage" / "minirag_working"


def _normalize_mode(value: object, fallback: str) -> str:
    text = str(value).strip().lower()
    return text or fallback


def _parse_bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _parse_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
