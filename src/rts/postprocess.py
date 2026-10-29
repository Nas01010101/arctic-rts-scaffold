"""Probability map -> COCO instance results."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage


def probs_to_instances(prob: np.ndarray, image_id: int, threshold: float) -> list[dict]:
    """Binarise, split into connected components, score each by its mean probability."""
    labels, n = ndimage.label(prob >= threshold)
    out = []
    for k in range(1, n + 1):
        comp = labels == k
        rle = mask_utils.encode(np.asfortranarray(comp, dtype=np.uint8))
        rle["counts"] = rle["counts"].decode()
        out.append({"image_id": int(image_id), "category_id": 1, "segmentation": rle,
                    "score": float(prob[comp].mean())})
    return out
