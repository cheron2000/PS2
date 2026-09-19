"""Executable tests for the T6 SEN2Vénus loader. Run: python3 test_sen2venus.py"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.datasets.sen2naip import upsample_nearest
from src.datasets.sen2venus import SEN2VenusDataset, _HAS_TORCH


def make_scene(h=24, w=24, c=4, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(h, w))
    for _ in range(3):
        base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1)) / 3
    return np.stack([base + rng.normal(0, .03, size=(h, w)) for _ in range(c)]) * 1000 + 2000


def test_scale_two_end_to_end():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr"))
        os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(seed=10)
        hr = upsample_nearest(lr, 2)
        np.save(os.path.join(root, "lr", "scene0.npy"), lr)
        np.save(os.path.join(root, "hr", "scene0.npy"), hr)
        ds = SEN2VenusDataset(root, max_shift=4, min_ncc_score=.3)
        assert len(ds) == 1, ds.dropped_pairs
        sample = ds[0]
        assert sample["scale"] == 2
        assert sample["hr"].shape == (4, 48, 48), sample["hr"].shape
        assert sample["lr"].shape == (4, 24, 24), sample["lr"].shape
        assert 0 <= float(np.asarray(sample["lr"]).min()) <= 1
        print("[PASS] test_scale_two_end_to_end")
    finally:
        shutil.rmtree(root)


def test_bad_pair_is_dropped():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr"))
        os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(seed=1)
        hr_bad = make_scene(h=48, w=48, seed=999)
        np.save(os.path.join(root, "lr", "bad.npy"), lr)
        np.save(os.path.join(root, "hr", "bad.npy"), hr_bad)
        ds = SEN2VenusDataset(root, min_ncc_score=.3)
        assert len(ds) == 0
        assert ds.dropped_pairs and ds.dropped_pairs[0][0] == "bad"
        print("[PASS] test_bad_pair_is_dropped")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_scale_two_end_to_end()
    test_bad_pair_is_dropped()
    print(f"All SEN2Vénus tests passed. (torch available: {_HAS_TORCH})")
