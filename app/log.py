from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from app.config import Config


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring a basic handler once if needed.

    The project previously had no logging configuration at all (only the legacy
    file-based `write_log`), which meant INFO messages were dropped by the root
    logger's WARNING default. This keeps that untouched and adds just enough for
    the BM25 diagnostics to be visible in CLI runs and in uvicorn.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logging.getLogger(name)


def ensure_dirs() -> None:
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(action: str, level: str = "INFO") -> None:
    ensure_dirs()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now} [{level}] {action}\n"
    log_path = Path(Config.LOG_DIR) / "system.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def read_recent_logs(limit: int = 80) -> list[str]:
    log_path = Path(Config.LOG_DIR) / "system.log"
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]
