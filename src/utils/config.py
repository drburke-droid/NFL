"""Config loader — reads YAML and provides typed access."""

from __future__ import annotations
import os
import yaml
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "config", "default.yaml")


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML config, falling back to default.yaml."""
    path = path or _DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against project root
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            if isinstance(val, str) and not os.path.isabs(val):
                cfg["paths"][key] = os.path.join(_PROJECT_ROOT, val)
    return cfg


def get_db_path(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["paths"]["db"]


def project_root() -> str:
    return _PROJECT_ROOT
