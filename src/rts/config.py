"""Run configuration: a TOML file with environment overrides for the two filesystem roots."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("data")
    out_dir: Path = Path("runs")
    in_channels: int = 4
    n_folds: int = 3
    seed: int = 0
    epochs: int = 2
    batch_size: int = 4
    lr: float = 1e-3
    threshold: float = 0.5
    device: str = "cpu"

    @property
    def annotations(self) -> Path:
        return self.data_root / "annotations.json"


def load_config(path: str | Path | None = None) -> Config:
    raw = tomllib.loads(Path(path).read_text()) if path else {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**raw)
    # Env vars win so the same TOML runs on a laptop and on a GPU box with different mounts.
    cfg.data_root = Path(os.environ.get("RTS_DATA", cfg.data_root))
    cfg.out_dir = Path(os.environ.get("RTS_OUT", cfg.out_dir))
    return cfg
