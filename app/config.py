from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


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

    # Qdrant local (embedded) mode by default: same API as the server, so
    # switching to a Docker service later only changes the connection target.
    QDRANT_PATH = _env_path("QDRANT_PATH", DATA_DIR / "qdrant")
    QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "financial_chunks")
    QDRANT_URL = os.environ.get("QDRANT_URL", "")  # set to use a Qdrant server
