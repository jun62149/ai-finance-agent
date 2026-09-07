from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.config import Config


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
