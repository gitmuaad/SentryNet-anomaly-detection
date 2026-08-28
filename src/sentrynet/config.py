"""Loads settings from config/config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (the directory containing ``config/config.yaml``)."""
    env = os.environ.get("SENTRYNET_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Fallback: <root>/src/sentrynet/config.py -> <root>
    return here.parents[2]


@dataclass(frozen=True)
class Config:
    """Thin, dictionary-backed view over ``config/config.yaml``."""

    data: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.data["project"]["seed"])

    def path(self, key: str) -> Path:
        """Resolve a key from the ``paths`` block into an absolute path."""
        return (self.root / self.data["paths"][key]).resolve()

    def ensure_dirs(self) -> None:
        for key in (
            "processed_dir",
            "splits_dir",
            "artifacts_dir",
            "figures_dir",
            "tables_dir",
            "metrics_dir",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration.

    Parameters
    ----------
    path:
        Optional explicit path. Defaults to ``<repo_root>/config/config.yaml``.
    """
    root = repo_root()
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(data=data, root=root)
