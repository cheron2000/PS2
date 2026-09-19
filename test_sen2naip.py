"""
test_sen2naip.py — executable end-to-end test for src/datasets/sen2naip.py (T5).

Run: python3 test_sen2naip.py
Requires: numpy (torch optional — test covers both paths).
"""
import shutil
import sys
import tempfile
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.datasets.sen2naip import (
    estimate_pair_shift, apply_shift_and_crop, normalized_cross_correlation,
    upsample_nearest, SEN2NAIPDataset, _HAS_TORCH,
)


def make_synthetic_scene(H=40, W=40, C=4, seed=0):
    rng = np.random.default_rng(seed)
    # smooth-ish synthetic "scene": low-frequency structure + noise, so NCC has
    # real signal to lock onto (pure noise would have no correlation structure
    # to recover a shift from)
    base = rng.normal(0, 1, size=(H, W))
    for _ in range(3):
        base = (base + np.roll(base, 1, axis=0) + np.roll(base, 1, axis=1)) / 3
    scene = np.stack([base + rng.normal(0, 0.05, size=(H, W)) for _ in range(C)])
    return (scene * 1000 + 2000).clip(0, 10000)  # plausible S2-like DN range


def test_ncc_self_score_is_one():
    a = np.random.default_rng(1).normal(size=(20, 20))
    score = normalized_cross_correlation(a, a)
    assert abs(score - 1.0) < 1e-9, f"NCC of a signal with itself should be 1.0, got {score}"
    print("[PASS] test_ncc_self_score_is_one")


def test_upsample_nearest_shape():
    x = np.zeros((4, 10, 10))
    up = upsample_nearest(x, 4)
    assert up.shape == (4, 40, 40), up.shape
    print("[PASS] test_upsample_nearest_shape")


def test_estimate_pair_shift_recovers_known_shift():
    scale = 4
    lr = make_synthetic_scene(H=20, W=20, seed=2)
    hr_full = upsample_nearest(lr, scale)  # perfectly aligned "HR" derived from LR

    true_dy, true_dx = 3, -2
    # simulate a misaligned HR by shifting hr_full and cropping to a common region
    C, H, W = hr_full.shape
    pad = 8
    hr_padded = np.pad(hr_full, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    hr_shifted = hr_padded[:, pad + true_dy: pad + true_dy + H, pad + true_dx: pad + true_dx + W]

    dy, dx, score = estimate_pair_shift(lr, hr_shifted, scale=scale, max_shift=6)
    # estimate_pair_shift searches for the shift applied *to HR* that best aligns
    # it back with LR, which is -true_dy, -true_dx relative to how we shifted it above
    assert (dy, dx) == (-true_dy, -true_dx), f"expected recovered shift {(-true_dy, -true_dx)}, got {(dy, dx)}"
    assert score > 0.9, f"expected high NCC after recovering exact shift, got {score}"
    print(f"[PASS] test_estimate_pair_shift_recovers_known_shift (recovered shift={(dy, dx)}, score={score:.3f})")


def test_apply_shift_and_crop_shapes_consistent():
    scale = 4
    lr = make_synthetic_scene(H=16, W=16, seed=3)
    hr = upsample_nearest(lr, scale)
    lr_c, hr_c = apply_shift_and_crop(lr, hr, dy=2, dx=-3, scale=scale)
    assert hr_c.shape[1] == lr_c.shape[1] * scale, (hr_c.shape, lr_c.shape)
    assert hr_c.shape[2] == lr_c.shape[2] * scale, (hr_c.shape, lr_c.shape)
    print(f"[PASS] test_apply_shift_and_crop_shapes_consistent (lr={lr_c.shape}, hr={hr_c.shape})")


def test_dataset_end_to_end_with_synthetic_files():
    scale = 4
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))

        # pair 0: well-aligned (small shift, should be corrected and kept)
        lr0 = make_synthetic_scene(H=24, W=24, seed=10)
        hr0 = upsample_nearest(lr0, scale)
        pad = 6
        hr0_padded = np.pad(hr0, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        hr0_shifted = hr0_padded[:, pad + 2:pad + 2 + hr0.shape[1], pad - 1:pad - 1 + hr0.shape[2]]
        np.save(os.path.join(tmpdir, "lr", "pair0.npy"), lr0)
        np.save(os.path.join(tmpdir, "hr", "pair0.npy"), hr0_shifted)

        # pair 1: unrelated HR (simulates a badly mismatched/corrupt pair) -> should be dropped
        lr1 = make_synthetic_scene(H=24, W=24, seed=11)
        hr1_unrelated = make_synthetic_scene(H=24 * scale, W=24 * scale, C=4, seed=999)
        np.save(os.path.join(tmpdir, "lr", "pair1.npy"), lr1)
        np.save(os.path.join(tmpdir, "hr", "pair1.npy"), hr1_unrelated)

        # pair 2: no matching HR file at all -> should be dropped
        lr2 = make_synthetic_scene(H=24, W=24, seed=12)
        np.save(os.path.join(tmpdir, "lr", "pair2.npy"), lr2)

        ds = SEN2NAIPDataset(tmpdir, scale=scale, max_shift=6, min_ncc_score=0.3)

        assert len(ds) == 1, f"expected exactly 1 kept pair, got {len(ds)}; dropped={ds.dropped_pairs}"
        kept_id = os.path.splitext(os.path.basename(ds.pairs[0][0]))[0]
        assert kept_id == "pair0", f"expected pair0 to be kept, got {kept_id}"

        dropped_ids = {d[0] for d in ds.dropped_pairs}
        assert dropped_ids == {"pair1", "pair2"}, f"unexpected dropped set: {ds.dropped_pairs}"

        sample = ds[0]
        assert sample["lr"].shape[0] == 4
        assert sample["hr"].shape[1] == sample["lr"].shape[1] * scale
        assert sample["hr"].shape[2] == sample["lr"].shape[2] * scale
        assert 0.0 <= float(np.asarray(sample["lr"]).max()) <= 1.0, "reflectance normalization should be in [0,1]"

        kind = "torch.Tensor" if _HAS_TORCH else "numpy.ndarray"
        print(f"[PASS] test_dataset_end_to_end_with_synthetic_files "
              f"(kept=1/3 pairs as expected, sample tensors are {kind}, "
              f"lr={tuple(sample['lr'].shape)}, hr={tuple(sample['hr'].shape)})")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_ncc_self_score_is_one()
    test_upsample_nearest_shape()
    test_estimate_pair_shift_recovers_known_shift()
    test_apply_shift_and_crop_shapes_consistent()
    test_dataset_end_to_end_with_synthetic_files()
    print(f"\nAll tests passed. (torch available in this run: {_HAS_TORCH})")
\n\ndef test_dataset_rejects_malformed_lr_shape():\n    tmpdir = tempfile.mkdtemp()\n    try:\n        os.makedirs(os.path.join(tmpdir, "lr"))\n        os.makedirs(os.path.join(tmpdir, "hr"))\n        np.save(os.path.join(tmpdir, "lr", "bad.npy"), np.ones((4, 4)))\n        np.save(os.path.join(tmpdir, "hr", "bad.npy"), np.ones((4, 16, 16)))\n        ds = SEN2NAIPDataset(tmpdir)\n        assert len(ds) == 0\n        assert ds.dropped_pairs[0][0] == "bad"\n        assert "shape (C,H,W)" in ds.dropped_pairs[0][1]\n    finally:\n        shutil.rmtree(tmpdir)\n\n\ndef test_dataset_rejects_invalid_scale():\n    tmpdir = tempfile.mkdtemp()\n    try:\n        os.makedirs(os.path.join(tmpdir, "lr"))\n        os.makedirs(os.path.join(tmpdir, "hr"))\n        np.save(os.path.join(tmpdir, "lr", "x.npy"), np.ones((4, 4, 4)))\n        np.save(os.path.join(tmpdir, "hr", "x.npy"), np.ones((4, 16, 16)))\n        for scale in (0, -1, 1.5, True):\n            try:\n                SEN2NAIPDataset(tmpdir, scale=scale)\n            except ValueError:\n                pass\n            else:\n                raise AssertionError(f"scale {scale!r} should be rejected")\n    finally:\n        shutil.rmtree(tmpdir)\n\n