from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from `.env` into os.environ (existing vars win).

    Dependency-free on purpose: secrets stay in `.env` (never in Git), and both
    scripts and the API get them without extra packages.
    """
    env_path = path or (BASE_DIR / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


class Config:
    # Paths (env-overridable)
    DATA_DIR = _env_path("FINANCE_DATA_DIR", BASE_DIR / "data")
    MODEL_DIR = _env_path("FINANCE_MODEL_DIR", BASE_DIR / "models")
    LOG_DIR = _env_path("FINANCE_LOG_DIR", BASE_DIR / "logs")
    DB_PATH = _env_path("FINANCE_DB_PATH", DATA_DIR / "stock_system.db")

    # Model / prediction semantics (identical to legacy B version)
    MODEL_VERSION = os.environ.get("FINANCE_MODEL_VERSION", "xgb-b-20260625-fulla")
    MODEL_REGRESSOR_PATH = MODEL_DIR / "xgb_regressor.pkl"
    MODEL_CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.pkl"
    FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
    MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.json"

    START_DATE = "2020-01-01"
    PREDICTION_HORIZON_DAYS = 5
    TOP_N = 20

    # Pool rules (identical to legacy B version)
    INCLUDE_MARKETS_FOR_TRAINING = ["sh", "sz"]
    EXCLUDE_FROM_RECOMMENDATION_PREFIXES = ["300", "301", "688", "689"]
    EXCLUDE_STOCK_NAME_KEYWORDS = ["ST", "*ST", "退"]
    MIN_LISTING_DAYS = 120
    MIN_AMOUNT = 50_000_000
    MIN_TRAINING_OBSERVATIONS_PER_STOCK = 60
    TOP_N_OPTIONS = [10, 20, 50]

    # Scoring weights (B version)
    SCORE_W_PRED_RETURN = 0.75
    SCORE_W_STRONG_PROB = 15.0
    SCORE_W_RISK = 10.0

    # Allowed board values for ranking API
    BOARD_VALUES = ["main", "growth", "star", "bj"]

    # RAG: embedding + vector store (Phase 2)
    # EMBEDDING_VERSION must change whenever the model/weights change, otherwise
    # old and new vectors would silently mix in one collection.
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda")
    EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
    EMBEDDING_VERSION = os.environ.get("EMBEDDING_VERSION", "bge-m3-v1")
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
    # auto = try the local HF cache first, then the network; 1 = offline only;
    # 0 = network first. A cached model still triggers remote HEAD checks, which
    # hang on networks where huggingface.co is unreachable.
    EMBEDDING_LOCAL_FILES_ONLY = os.environ.get("EMBEDDING_LOCAL_FILES_ONLY", "auto")

    # Qdrant local (embedded) mode by default: same API as the server, so
    # switching to a Docker service later only changes the connection target.
    QDRANT_PATH = _env_path("QDRANT_PATH", DATA_DIR / "qdrant")
    QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "financial_chunks")
    QDRANT_URL = os.environ.get("QDRANT_URL", "")  # set to use a Qdrant server

    # Phase 3: hybrid retrieval (dense + BM25 -> RRF -> reranker)
    BM25_PATH = _env_path("BM25_PATH", DATA_DIR / "bm25")
    # BM25 candidate window. 0 (default) = score the whole corpus and apply the
    # metadata filters afterwards, so a filter can never hide a document that
    # ranks outside an arbitrary window (see docs/bm25_latency_comparison.md).
    # >0 = truncate to that many top-scoring candidates (explicit opt-in; a
    # WARNING is logged whenever truncation actually happens).
    BM25_WINDOW = int(os.environ.get("BM25_WINDOW", "0"))
    # Consistency guard: when the BM25 manifest disagrees with the chunk corpus we
    # fail loudly (IndexOutOfSync). Set BM25_ALLOW_STALE=1 for offline demos/evals
    # where a warning is preferable to an exception.
    BM25_ALLOW_STALE = os.environ.get("BM25_ALLOW_STALE", "0").lower() in ("1", "true", "yes")
    RRF_K = int(os.environ.get("RRF_K", "60"))
    DENSE_TOP_K = int(os.environ.get("DENSE_TOP_K", "20"))
    BM25_TOP_K = int(os.environ.get("BM25_TOP_K", "20"))
    FUSION_TOP_K = int(os.environ.get("FUSION_TOP_K", "15"))
    RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "5"))
    RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", EMBEDDING_DEVICE)
    RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "1").lower() not in ("0", "false", "no")

    # Phase 7: Agent (LangGraph + LLM tool calling)
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
    LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
    AGENT_MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "6"))
    AGENT_TOOL_TIMEOUT = float(os.environ.get("AGENT_TOOL_TIMEOUT", "60"))
