"""Site-grouped cross-validation.

Chips from one site share terrain, season and sensor conditions, so a random chip split leaks
site-specific appearance into validation. `site_folds` keeps every site entirely inside one fold.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def site_folds(image_ids: Sequence[int], sites: Sequence[str | int], n_folds: int, seed: int = 0
               ) -> list[tuple[list[int], list[int]]]:
    """Return `n_folds` (train_ids, val_ids) pairs.

    Sites are shuffled with `seed`, then assigned largest-first to the fold with the fewest chips,
    which balances fold sizes while keeping assignment deterministic for a given seed.
    """
    if len(image_ids) != len(sites):
        raise ValueError("image_ids and sites must have the same length")
    by_site: dict = defaultdict(list)
    for iid, site in zip(image_ids, sites):
        by_site[site].append(int(iid))
    if len(by_site) < n_folds:
        raise ValueError(f"{len(by_site)} sites cannot fill {n_folds} folds")

    keys = sorted(by_site, key=str)
    np.random.default_rng(seed).shuffle(keys)
    keys.sort(key=lambda k: -len(by_site[k]))  # stable: ties keep the shuffled order
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for k in keys:
        min(folds, key=len).extend(by_site[k])

    everything = {i for f in folds for i in f}
    return [([i for i in sorted(everything) if i not in set(val)], sorted(val)) for val in folds]
