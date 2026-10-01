"""Load paths from a local JSON file that is never committed with real values."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(
        os.environ.get("MWFMRI_CONFIG", REPOSITORY_ROOT / "config.example.json")
    ).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path)
    return config


def task_config(task: str) -> dict:
    config = load_config()
    if task not in config["tasks"]:
        raise KeyError(f"Task {task!r} is missing from {config['_config_path']}")
    return config["tasks"][task]


def resolve_path(value: str) -> Path:
    """Resolve relative config paths against the selected config file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(load_config()["_config_path"]).parent / path
    return path.resolve()
