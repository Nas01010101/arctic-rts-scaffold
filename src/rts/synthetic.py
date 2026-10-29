"""Random chips in the release layout, so every stage runs without the challenge data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage


def make_synthetic(root: Path, n_images: int = 24, size: int = 64, channels: int = 4,
                   n_sites: int = 6, seed: int = 0) -> Path:
    """Write `n_images` chips plus `annotations.json` under `root`; return the annotation path."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "chips").mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    images, annotations = [], []
    for i in range(n_images):
        img = rng.normal(size=(size, size, channels)).astype(np.float32)
        union = np.zeros((size, size), bool)
        for _ in range(rng.integers(0, 4)):
            cy, cx = rng.integers(8, size - 8, 2)
            ry, rx = rng.integers(3, 8, 2)
            blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            union |= blob
        img[union] += 1.5  # gives the stub model a signal to fit
        name = f"chips/{i:04d}.npz"
        np.savez_compressed(root / name, image=img)
        images.append({"id": i, "file_name": name, "height": size, "width": size, "site": f"site_{i % n_sites}"})
        labels, n = ndimage.label(union)
        for k in range(1, n + 1):
            m = np.asfortranarray(labels == k, dtype=np.uint8)
            rle = mask_utils.encode(m)
            rle["counts"] = rle["counts"].decode()
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": 1,
                                "segmentation": rle, "area": float(m.sum()), "iscrowd": 0,
                                "bbox": mask_utils.toBbox(rle).tolist()})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "retrogressive_thaw_slump"}]}
    out = root / "annotations.json"
    out.write_text(json.dumps(coco))
    return out
