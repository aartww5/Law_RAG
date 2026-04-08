from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib


DEFAULT_OLLAMA_MODEL = "qwen35-law:q6k-stable"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_DEVICE = "auto"
DEFAULT_RERANK_TOP_K = 20
DEFAULT_BM25_TOP_K = 50
DEFAULT_VECTOR_TOP_K = 50
DEFAULT_DECOMPOSITION_MAX_QUERIES = 4
DEFAULT_OLLAMA_NUM_CTX = 32768
DEFAULT_OLLAMA_NUM_PREDICT = 2048
CONFIG_FILE_NAME = "config.toml"


@dataclass
class RuntimeConfig:
    mode: str = "auto"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    ollama_num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT
    streaming: bool = True
    max_context_articles: int = 6
    max_history_turns: int = 4


@dataclass
class IndexConfig:
    laws_dir: Path = Path("RAG/Chinese-Laws")
    qdrant_path: Path = Path("unified_app/storage/qdrant")
    qdrant_collection_name: str = "chinese_laws_article_based"
    bm25_cache_path: Path = Path("unified_app/storage/bm25")
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = DEFAULT_MODEL_DEVICE
    embedding_build_device: str = DEFAULT_MODEL_DEVICE
    bm25_top_k: int = DEFAULT_BM25_TOP_K
    vector_top_k: int = DEFAULT_VECTOR_TOP_K
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_device: str = DEFAULT_MODEL_DEVICE
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    decomposition_max_queries: int = DEFAULT_DECOMPOSITION_MAX_QUERIES
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
        ollama_num_ctx=_parse_int(
            runtime_data.get("ollama_num_ctx", defaults.ollama_num_ctx),
            defaults.ollama_num_ctx,
        ),
        ollama_num_predict=_parse_int(
            runtime_data.get("ollama_num_predict", defaults.ollama_num_predict),
            defaults.ollama_num_predict,
        ),
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
    if "LEGAL_RAG_OLLAMA_NUM_CTX" in os.environ:
        runtime.ollama_num_ctx = _parse_int(os.environ["LEGAL_RAG_OLLAMA_NUM_CTX"], runtime.ollama_num_ctx)
    if "LEGAL_RAG_OLLAMA_NUM_PREDICT" in os.environ:
        runtime.ollama_num_predict = _parse_int(
            os.environ["LEGAL_RAG_OLLAMA_NUM_PREDICT"],
            runtime.ollama_num_predict,
        )
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
    qdrant_collection_name = (
        str(index_data.get("qdrant_collection_name", defaults.qdrant_collection_name)).strip()
        or defaults.qdrant_collection_name
    )
    bm25_cache_path = _resolve_path(index_data.get("bm25_cache_path"), root, defaults.bm25_cache_path)
    embedding_model = str(index_data.get("embedding_model", defaults.embedding_model)).strip() or defaults.embedding_model
    embedding_device = str(index_data.get("embedding_device", defaults.embedding_device)).strip() or defaults.embedding_device
    embedding_build_device = (
        str(index_data.get("embedding_build_device", defaults.embedding_build_device)).strip()
        or defaults.embedding_build_device
    )
    bm25_top_k = _parse_int(index_data.get("bm25_top_k", defaults.bm25_top_k), defaults.bm25_top_k)
    vector_top_k = _parse_int(index_data.get("vector_top_k", defaults.vector_top_k), defaults.vector_top_k)
    reranker_model = str(index_data.get("reranker_model", defaults.reranker_model)).strip() or defaults.reranker_model
    reranker_device = str(index_data.get("reranker_device", defaults.reranker_device)).strip() or defaults.reranker_device
    rerank_top_k = _parse_int(index_data.get("rerank_top_k", defaults.rerank_top_k), defaults.rerank_top_k)
    decomposition_max_queries = _parse_int(
        index_data.get("decomposition_max_queries", defaults.decomposition_max_queries),
        defaults.decomposition_max_queries,
    )
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
    if "LEGAL_RAG_QDRANT_COLLECTION_NAME" in os.environ:
        qdrant_collection_name = os.environ["LEGAL_RAG_QDRANT_COLLECTION_NAME"].strip() or qdrant_collection_name
    if "LEGAL_RAG_BM25_CACHE_PATH" in os.environ:
        bm25_cache_path = _resolve_path(os.environ["LEGAL_RAG_BM25_CACHE_PATH"], root, bm25_cache_path)
    if "LEGAL_RAG_EMBEDDING_MODEL" in os.environ:
        embedding_model = os.environ["LEGAL_RAG_EMBEDDING_MODEL"].strip() or embedding_model
    if "LEGAL_RAG_EMBEDDING_DEVICE" in os.environ:
        embedding_device = os.environ["LEGAL_RAG_EMBEDDING_DEVICE"].strip() or embedding_device
    if "LEGAL_RAG_EMBEDDING_BUILD_DEVICE" in os.environ:
        embedding_build_device = os.environ["LEGAL_RAG_EMBEDDING_BUILD_DEVICE"].strip() or embedding_build_device
    if "LEGAL_RAG_BM25_TOP_K" in os.environ:
        bm25_top_k = _parse_int(os.environ["LEGAL_RAG_BM25_TOP_K"], bm25_top_k)
    if "LEGAL_RAG_VECTOR_TOP_K" in os.environ:
        vector_top_k = _parse_int(os.environ["LEGAL_RAG_VECTOR_TOP_K"], vector_top_k)
    if "LEGAL_RAG_RERANKER_MODEL" in os.environ:
        reranker_model = os.environ["LEGAL_RAG_RERANKER_MODEL"].strip() or reranker_model
    if "LEGAL_RAG_RERANKER_DEVICE" in os.environ:
        reranker_device = os.environ["LEGAL_RAG_RERANKER_DEVICE"].strip() or reranker_device
    if "LEGAL_RAG_RERANK_TOP_K" in os.environ:
        rerank_top_k = _parse_int(os.environ["LEGAL_RAG_RERANK_TOP_K"], rerank_top_k)
    if "LEGAL_RAG_DECOMPOSITION_MAX_QUERIES" in os.environ:
        decomposition_max_queries = _parse_int(
            os.environ["LEGAL_RAG_DECOMPOSITION_MAX_QUERIES"],
            decomposition_max_queries,
        )
    if "LEGAL_RAG_CORPUS_DIR" in os.environ:
        corpus_dir = _resolve_path(os.environ["LEGAL_RAG_CORPUS_DIR"], root, corpus_dir)
    if "LEGAL_RAG_MINI_WORKING_DIR" in os.environ:
        mini_working_dir = _resolve_path(os.environ["LEGAL_RAG_MINI_WORKING_DIR"], root, mini_working_dir)

    return IndexConfig(
        laws_dir=laws_dir,
        qdrant_path=qdrant_path,
        qdrant_collection_name=qdrant_collection_name,
        bm25_cache_path=bm25_cache_path,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        embedding_build_device=embedding_build_device,
        bm25_top_k=bm25_top_k,
        vector_top_k=vector_top_k,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        rerank_top_k=rerank_top_k,
        decomposition_max_queries=decomposition_max_queries,
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
