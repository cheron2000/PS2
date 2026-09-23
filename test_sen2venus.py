"""Executable tests for the T6 SEN2Vénus loader. Run: python3 test_sen2venus.py"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.datasets.sen2naip import upsample_nearest
from src.datasets.sen2venus import SEN2VenusDataset, _HAS_TORCH
from src.datasets.provenance import build_pair_manifest, write_manifest


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



def _grid_pair(width, height, scale, origin=(500000, 300000), crs="EPSG:32643"):
    ox, oy = origin
    return (
        {"crs": crs, "transform": [10, 0, ox, 0, -10, oy], "width": width, "height": height},
        {"crs": crs, "transform": [10 / scale, 0, ox, 0, -10 / scale, oy],
         "width": width * scale, "height": height * scale},
    )


def _write_manifest_pair(root, pair_id, lr_grid=None, hr_grid=None):
    metadata = {}
    if lr_grid is not None:
        metadata["lr_grid"] = lr_grid
    if hr_grid is not None:
        metadata["hr_grid"] = hr_grid
    path = os.path.join(root, "provenance.json")
    write_manifest(path, build_pair_manifest(
        root, "SEN2Vénus-fixture", "Sentinel-2+Vénus",
        [{"id": pair_id, "lr_path": f"lr/{pair_id}.npy",
          "hr_path": f"hr/{pair_id}.npy", "metadata": metadata}],
    ))
    return path


def test_grid_aware_pair_accepted_and_returned():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr")); os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(h=12, w=12, seed=20); hr = upsample_nearest(lr, 2)
        np.save(os.path.join(root, "lr", "grid.npy"), lr)
        np.save(os.path.join(root, "hr", "grid.npy"), hr)
        lg, hg = _grid_pair(12, 12, 2)
        mp = _write_manifest_pair(root, "grid", lg, hg)
        ds = SEN2VenusDataset(root, manifest_path=mp, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 1, ds.dropped_pairs
        sample = ds[0]
        assert sample["lr_grid"]["crs"] == "EPSG:32643"
        assert sample["hr_grid"]["width"] == 24
        print("[PASS] test_grid_aware_pair_accepted_and_returned")
    finally:
        shutil.rmtree(root)


def test_grid_aware_pair_rejects_crs_mismatch():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr")); os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(h=12, w=12, seed=21); hr = upsample_nearest(lr, 2)
        np.save(os.path.join(root, "lr", "crs.npy"), lr)
        np.save(os.path.join(root, "hr", "crs.npy"), hr)
        lg, hg = _grid_pair(12, 12, 2); hg["crs"] = "EPSG:4326"
        mp = _write_manifest_pair(root, "crs", lg, hg)
        ds = SEN2VenusDataset(root, manifest_path=mp, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 0 and "grid contract violation" in ds.dropped_pairs[0][1]
        print("[PASS] test_grid_aware_pair_rejects_crs_mismatch")
    finally:
        shutil.rmtree(root)


def test_grid_aware_pair_rejects_non_scale_aligned_shift():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr")); os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(h=16, w=16, seed=22); full = upsample_nearest(lr, 2)
        pad = 4; dy0, dx0 = 1, -1
        padded = np.pad(full, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        hr = padded[:, pad + dy0:pad + dy0 + full.shape[1], pad + dx0:pad + dx0 + full.shape[2]]
        np.save(os.path.join(root, "lr", "phase.npy"), lr)
        np.save(os.path.join(root, "hr", "phase.npy"), hr)
        lg, hg = _grid_pair(16, 16, 2)
        mp = _write_manifest_pair(root, "phase", lg, hg)
        ds = SEN2VenusDataset(root, manifest_path=mp, max_shift=4, min_ncc_score=0.3)
        assert len(ds) == 0 and "grid contract violation" in ds.dropped_pairs[0][1]
        print("[PASS] test_grid_aware_pair_rejects_non_scale_aligned_shift")
    finally:
        shutil.rmtree(root)


def test_grid_aware_pair_rejects_dimension_mismatch():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr")); os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(h=12, w=12, seed=23); hr = upsample_nearest(lr, 2)
        np.save(os.path.join(root, "lr", "dim.npy"), lr)
        np.save(os.path.join(root, "hr", "dim.npy"), hr)
        lg, hg = _grid_pair(999, 999, 2)
        mp = _write_manifest_pair(root, "dim", lg, hg)
        ds = SEN2VenusDataset(root, manifest_path=mp, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 0 and "grid contract violation" in ds.dropped_pairs[0][1]
        print("[PASS] test_grid_aware_pair_rejects_dimension_mismatch")
    finally:
        shutil.rmtree(root)


def test_grid_aware_pair_requires_both_grids():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "lr")); os.makedirs(os.path.join(root, "hr"))
        lr = make_scene(h=12, w=12, seed=24); hr = upsample_nearest(lr, 2)
        np.save(os.path.join(root, "lr", "half.npy"), lr)
        np.save(os.path.join(root, "hr", "half.npy"), hr)
        lg, _ = _grid_pair(12, 12, 2)
        mp = _write_manifest_pair(root, "half", lg, None)
        ds = SEN2VenusDataset(root, manifest_path=mp, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 0 and "both lr_grid and hr_grid" in ds.dropped_pairs[0][1]
        print("[PASS] test_grid_aware_pair_requires_both_grids")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_scale_two_end_to_end()
    test_bad_pair_is_dropped()
    print(f"All SEN2Vénus tests passed. (torch available: {_HAS_TORCH})")
    test_grid_aware_pair_accepted_and_returned()
    test_grid_aware_pair_rejects_crs_mismatch()
    test_grid_aware_pair_rejects_non_scale_aligned_shift()
    test_grid_aware_pair_rejects_dimension_mismatch()
    test_grid_aware_pair_requires_both_grids()
