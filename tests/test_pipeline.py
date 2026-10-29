import numpy as np

from rts.config import Config, load_config
from rts.data import ChipDataset, channel_stats, read_coco
from rts.evaluate import evaluate
from rts.postprocess import probs_to_instances
from rts.synthetic import make_synthetic
from rts.train import run_fold


def test_dataset_shapes(tmp_path):
    ann = make_synthetic(tmp_path, n_images=4, size=32, channels=4)
    coco = read_coco(ann)
    ids = [im["id"] for im in coco["images"]]
    mean, std = channel_stats(tmp_path, coco["images"])
    x, y, iid = ChipDataset(tmp_path, coco, ids, mean, std)[0]
    assert x.shape == (4, 32, 32) and y.shape == (1, 32, 32) and iid == ids[0]


def test_ground_truth_components_score_perfect_ap(tmp_path):
    coco = read_coco(make_synthetic(tmp_path, n_images=6, size=48))
    preds = [{"image_id": a["image_id"], "category_id": 1, "score": 1.0,
              "segmentation": a["segmentation"]} for a in coco["annotations"]]
    assert evaluate(coco, preds, [im["id"] for im in coco["images"]])["AP"] == 1.0


def test_probs_to_instances_splits_components():
    p = np.zeros((16, 16)); p[1:4, 1:4] = 0.9; p[10:14, 10:14] = 0.8
    out = probs_to_instances(p, 7, 0.5)
    assert len(out) == 2 and all(o["image_id"] == 7 for o in out)


def test_run_fold_returns_ap_keys(tmp_path):
    make_synthetic(tmp_path, n_images=12, size=32)
    cfg = Config(data_root=tmp_path, epochs=1)
    ids = list(range(12))
    out = run_fold(cfg, ids[:8], ids[8:])
    assert {"AP", "AP50", "APl"} <= set(out)


def test_config_rejects_unknown_keys(tmp_path):
    import pytest
    p = tmp_path / "c.toml"; p.write_text("bogus = 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_env_overrides_data_root(monkeypatch):
    monkeypatch.setenv("RTS_DATA", "/x")
    assert str(load_config().data_root) == "/x"
