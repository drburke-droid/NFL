"""Structured logging for the pipeline."""

from __future__ import annotations
import logging
import os
import sys
from datetime import datetime
from src.utils.config import load_config


def setup_logger(name: str, cfg: dict | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to console and a timestamped log file."""
    cfg = cfg or load_config()
    log_dir = cfg["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
