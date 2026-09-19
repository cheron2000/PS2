"""Synthetic T10 tests. Run: python3 test_cartosat_pairing.py"""
import json
import os
import shutil
import tempfile

import numpy as np

from src.datasets.sen2naip import upsample_nearest
from src.datasets.cartosat_pairing import (
    CartosatPairingDataset,
    PairRejected,
    prepare_pair,
)


def scene(h=16, w=16, c=4, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w), dtype=np.float32)
    for _ in range(2):
        base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1)) / 3
    return np.stack([np.clip(base + rng.normal(0, .01, (h, w)), 0, 1) for _ in range(c)])


def shifted(arr, dy, dx):
    _, h, w = arr.shape
    pad = max(abs(dy), abs(dx)) + 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    return padded[:, pad + dy:pad + dy + h, pad + dx:pad + dx + w]


def test_prepare_pair():
    lr = scene(seed=1)
    hr = shifted(upsample_nearest(lr, 4), 3, -2) * 0.8 + 0.05
    pair = prepare_pair(lr, hr, scale=4, max_shift=6, min_ncc_score=.5,
                        metadata={"scene_id": "cartosat-demo"})
    assert pair.lr.shape == (4, 15, 15), pair.lr.shape
    assert pair.hr.shape == (4, 60, 60), pair.hr.shape
    assert pair.valid_mask.shape == pair.hr.shape[1:]
    assert pair.alignment_score > .9
    assert pair.metadata["scene_id"] == "cartosat-demo"
    assert np.isfinite(pair.harmonization_gain).all()
    print("[PASS] test_prepare_pair")


def test_dataset_keeps_good_and_drops_bad():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr"))
        os.makedirs(os.path.join(root, "hr"))
        os.makedirs(os.path.join(root, "metadata"))
        lr = scene(seed=2)
        np.save(os.path.join(root, "lr", "good.npy"), lr)
        np.save(os.path.join(root, "hr", "good.npy"), upsample_nearest(lr, 4))
        with open(os.path.join(root, "metadata", "good.json"), "w", encoding="utf-8") as f:
            json.dump({"acquisition": "2026-01-01", "crs": "EPSG:32643"}, f)
        np.save(os.path.join(root, "lr", "bad.npy"), scene(seed=3))
        np.save(os.path.join(root, "hr", "bad.npy"), scene(h=64, w=64, seed=999))
        ds = CartosatPairingDataset(root, min_ncc_score=.4)
        assert len(ds) == 1, ds.dropped_pairs
        sample = ds[0]
        assert sample["lr"].shape[0] == 4
        assert tuple(sample["hr"].shape[-2:]) == (sample["lr"].shape[-2] * 4, sample["lr"].shape[-1] * 4)
        assert sample["metadata"]["crs"] == "EPSG:32643"
        assert ds.dropped_pairs[0][0] == "bad"
        print("[PASS] test_dataset_keeps_good_and_drops_bad")
    finally:
        shutil.rmtree(root)


def test_low_score_rejected():
    try:
        prepare_pair(scene(seed=4), scene(h=64, w=64, seed=5), min_ncc_score=.5)
    except PairRejected:
        print("[PASS] test_low_score_rejected")
        return
    raise AssertionError("unrelated Cartosat pair should be rejected")


if __name__ == "__main__":
    test_prepare_pair()
    test_dataset_keeps_good_and_drops_bad()
    test_low_score_rejected()
    print("All Cartosat pairing tests passed.")
