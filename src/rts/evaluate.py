"""COCO segmentation AP over a subset of images."""
from __future__ import annotations

import contextlib
import copy
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def evaluate(coco_gt: dict, preds: list[dict], image_ids: list[int]) -> dict[str, float]:
    ids = set(image_ids)
    gt = copy.deepcopy(coco_gt)
    gt["images"] = [i for i in gt["images"] if i["id"] in ids]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in ids]
    if not preds:
        return {k: 0.0 for k in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        ev = COCOeval(cg, cg.loadRes(preds), "segm")
        ev.params.imgIds = sorted(ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(v) for k, v in zip(STAT_NAMES, ev.stats[:6])}
