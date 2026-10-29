from rts.folds import site_folds


def test_sites_never_split_across_train_and_val():
    ids = list(range(30))
    sites = [f"s{i % 6}" for i in ids]
    folds = site_folds(ids, sites, 3, seed=1)
    site_of = dict(zip(ids, sites))
    for train, val in folds:
        assert not {site_of[i] for i in train} & {site_of[i] for i in val}
        assert sorted(train + val) == ids


def test_val_sets_partition_all_images():
    ids = list(range(20))
    folds = site_folds(ids, [i % 5 for i in ids], 5)
    assert sorted(i for _, val in folds for i in val) == ids


def test_deterministic_for_seed():
    ids = list(range(20))
    sites = [i % 5 for i in ids]
    assert site_folds(ids, sites, 5, seed=3) == site_folds(ids, sites, 5, seed=3)


def test_too_few_sites_raises():
    import pytest
    with pytest.raises(ValueError):
        site_folds([0, 1], ["a", "a"], 2)
