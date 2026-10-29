"""Entry points: `rts synth | folds | train | smoke`."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from .config import Config, load_config
from .data import read_coco
from .folds import site_folds
from .synthetic import make_synthetic
from .train import run_fold


def _folds(cfg: Config):
    images = read_coco(cfg.annotations)["images"]
    return site_folds([im["id"] for im in images], [im["site"] for im in images], cfg.n_folds, cfg.seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rts")
    ap.add_argument("--config", default=None, help="TOML config path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="write synthetic chips + annotations under data_root")
    sub.add_parser("folds", help="print the site-grouped folds")
    tr = sub.add_parser("train", help="train and score one fold")
    tr.add_argument("--fold", type=int, default=0)
    sub.add_parser("smoke", help="synthetic data -> folds -> train fold 0 -> COCO AP")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "synth":
        print(make_synthetic(cfg.data_root, channels=cfg.in_channels, seed=cfg.seed))
    elif args.cmd == "folds":
        for k, (tr_ids, val_ids) in enumerate(_folds(cfg)):
            print(json.dumps({"fold": k, "n_train": len(tr_ids), "n_val": len(val_ids)}))
    elif args.cmd == "train":
        tr_ids, val_ids = _folds(cfg)[args.fold]
        print(json.dumps(run_fold(cfg, tr_ids, val_ids)))
    elif args.cmd == "smoke":
        with tempfile.TemporaryDirectory() as tmp:
            cfg = replace(cfg, data_root=Path(tmp))
            make_synthetic(cfg.data_root, channels=cfg.in_channels, seed=cfg.seed)
            tr_ids, val_ids = _folds(cfg)[0]
            print(json.dumps(run_fold(cfg, tr_ids, val_ids)))
