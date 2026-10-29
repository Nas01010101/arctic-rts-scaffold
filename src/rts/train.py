"""Train one fold, predict its validation chips, score them."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import ChipDataset, channel_stats, read_coco
from .evaluate import evaluate
from .model import build_model
from .postprocess import probs_to_instances


def run_fold(cfg: Config, train_ids: list[int], val_ids: list[int], model_name: str = "stub") -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    coco = read_coco(cfg.annotations)
    by_id = {im["id"]: im for im in coco["images"]}
    mean, std = channel_stats(cfg.data_root, [by_id[i] for i in train_ids])

    train = DataLoader(ChipDataset(cfg.data_root, coco, train_ids, mean, std),
                       batch_size=cfg.batch_size, shuffle=True)
    val = DataLoader(ChipDataset(cfg.data_root, coco, val_ids, mean, std), batch_size=cfg.batch_size)

    model = build_model(model_name, cfg.in_channels).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(cfg.epochs):
        model.train()
        for x, y, _ in train:
            opt.zero_grad()
            loss_fn(model(x.to(cfg.device)), y.to(cfg.device)).backward()
            opt.step()

    model.eval()
    preds: list[dict] = []
    with torch.no_grad():
        for x, _, ids in val:
            probs = torch.sigmoid(model(x.to(cfg.device)))[:, 0].cpu().numpy()
            for p, iid in zip(probs, ids.tolist()):
                preds += probs_to_instances(p, iid, cfg.threshold)
    return evaluate(coco, preds, val_ids)
