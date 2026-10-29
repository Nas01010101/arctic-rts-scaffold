"""Chip loading from a COCO-format release.

Layout under `data_root`:
    annotations.json        COCO images / annotations (RLE segmentation); each image may carry "site"
    <images[].file_name>    .npz with an `image` array of shape (H, W, C)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


def read_coco(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def semantic_mask(coco_anns: list[dict], height: int, width: int) -> np.ndarray:
    """Union of the instance masks of one image, uint8 (H, W)."""
    out = np.zeros((height, width), np.uint8)
    for a in coco_anns:
        out |= mask_utils.decode(a["segmentation"]).astype(np.uint8)
    return out


def channel_stats(root: Path, images: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the given (training) chips only, so validation chips never
    contribute to normalisation."""
    chips = []
    for im in images:
        with np.load(root / im["file_name"]) as z:
            chips.append(z["image"].reshape(-1, z["image"].shape[-1]))
    x = np.concatenate(chips).astype(np.float64)
    return x.mean(0).astype(np.float32), (x.std(0) + 1e-6).astype(np.float32)


class ChipDataset(Dataset):
    """Yields (image CHW float32, semantic mask 1HW float32, image_id)."""

    def __init__(self, root: Path, coco: dict, image_ids: list[int], mean: np.ndarray, std: np.ndarray):
        self.root, self.mean, self.std = Path(root), mean, std
        by_id = {im["id"]: im for im in coco["images"]}
        self.images = [by_id[i] for i in image_ids]
        self.anns: dict[int, list[dict]] = {i: [] for i in image_ids}
        for a in coco["annotations"]:
            if a["image_id"] in self.anns:
                self.anns[a["image_id"]].append(a)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        im = self.images[idx]
        with np.load(self.root / im["file_name"]) as z:
            x = z["image"]
        x = np.nan_to_num((x - self.mean) / self.std).astype(np.float32)
        y = semantic_mask(self.anns[im["id"]], im["height"], im["width"]).astype(np.float32)
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y)[None], im["id"]
