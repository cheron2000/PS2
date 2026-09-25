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
from src.datasets.provenance import build_pair_manifest, write_manifest


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
        kept_id = os.path.splitext(os.path.basename(ds.pairs[0]["lr_path"]))[0]
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

def test_dataset_rejects_invalid_scale():
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        np.save(os.path.join(tmpdir, "lr", "x.npy"), np.ones((4, 4, 4)))
        np.save(os.path.join(tmpdir, "hr", "x.npy"), np.ones((4, 16, 16)))
        for scale in (0, -1, 1.5, True):
            try:
                SEN2NAIPDataset(tmpdir, scale=scale)
            except ValueError:
                pass
            else:
                raise AssertionError(f"scale {scale!r} should be rejected")
    finally:
        shutil.rmtree(tmpdir)

def test_dataset_rejects_malformed_lr_shape():
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        np.save(os.path.join(tmpdir, "lr", "bad.npy"), np.ones((4, 4)))
        np.save(os.path.join(tmpdir, "hr", "bad.npy"), np.ones((4, 16, 16)))
        ds = SEN2NAIPDataset(tmpdir)
        assert len(ds) == 0
        assert ds.dropped_pairs[0][0] == "bad"
        assert "shape (C,H,W)" in ds.dropped_pairs[0][1]
    finally:
        shutil.rmtree(tmpdir)

def test_dataset_normalizes_sentinel2_hr_by_10000():
    """Regression test confirming that when the downloaded data contains
    Sentinel-2-scale reflectance values (0-10000) for BOTH lr and hr (as in
    the HuggingFace SEN2NAIPv2 dataset), both sides are normalized by /10000.

    This replaces the previous test_dataset_normalizes_naip_hr_by_255_not_10000
    which tested against 8-bit NAIP data (0-255). The actual downloaded dataset
    uses matching Sentinel-2 reflectance scale on both sides.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        rng = np.random.default_rng(0)
        # Both LR and HR use Sentinel-2 L2A reflectance DN range (roughly 0-10000)
        lr_data = rng.integers(0, 10000, size=(4, 8, 8)).astype(np.float32)
        hr_data = rng.integers(0, 10000, size=(4, 32, 32)).astype(np.float32)
        hr_data[0, 0, 0] = 10000.0  # guarantee the true max is exercised
        np.save(os.path.join(tmpdir, "lr", "tile.npy"), lr_data)
        np.save(os.path.join(tmpdir, "hr", "tile.npy"), hr_data)

        ds = SEN2NAIPDataset(tmpdir, scale=4, min_ncc_score=-1.0)  # accept any alignment, only checking normalization
        assert len(ds) == 1, f"expected the synthetic pair to be kept, dropped_pairs={ds.dropped_pairs}"
        sample = ds[0]
        hr_norm = sample["hr"]

        hr_max = float(hr_norm.max())
        # Both sides should be normalized to roughly [0, 1] when divided by 10000
        assert hr_max <= 1.0 + 1e-6, f"HR normalized values must stay within [0, 1], got max {hr_max}"
        assert hr_max > 0.5, (
            f"HR (Sentinel-2 scale) normalized max is {hr_max:.4f} -- should be near 1.0 "
            f"when the input contains values near 10000"
        )

        # Both LR and HR should use the same divisor (10000) for Sentinel-2-scale data
        assert sample["hr_stats"]["divisor"] == 10000.0, (
            f"expected HR stats to record divisor=10000.0 (Sentinel-2 scale), got {sample['hr_stats']['divisor']}"
        )
        assert sample["lr_stats"]["divisor"] == 10000.0, (
            f"expected LR stats to record divisor=10000.0 (Sentinel-2), got {sample['lr_stats']['divisor']}"
        )
    finally:
        shutil.rmtree(tmpdir)


def test_dataset_accepts_sealed_provenance_manifest():
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        lr = make_synthetic_scene(H=12, W=12, seed=33)
        hr = upsample_nearest(lr, 4)
        np.save(os.path.join(tmpdir, "lr", "manifest-scene.npy"), lr)
        np.save(os.path.join(tmpdir, "hr", "manifest-scene.npy"), hr)
        manifest_path = os.path.join(tmpdir, "provenance.json")
        write_manifest(
            manifest_path,
            build_pair_manifest(
                tmpdir,
                "SEN2NAIP-converted",
                "Sentinel-2+NAIP",
                [{
                    "id": "manifest-scene",
                    "lr_path": "lr/manifest-scene.npy",
                    "hr_path": "hr/manifest-scene.npy",
                }],
                metadata={"converter": "fixture"},
            ),
        )
        ds = SEN2NAIPDataset(
            tmpdir,
            manifest_path=manifest_path,
            max_shift=0,
            min_ncc_score=0.9,
        )
        assert len(ds) == 1
        assert os.path.basename(ds.pairs[0]["lr_path"]) == "manifest-scene.npy"
    finally:
        shutil.rmtree(tmpdir)


def _grid_pair(width, height, scale, origin=(500000, 300000), crs="EPSG:32643"):
    """Matching T16's own test convention (test_t16_pairing.py's grid_pair()):
    LR at 10m, HR at 10/scale m, same CRS and origin, scale-aligned pixel size."""
    ox, oy = origin
    lr_grid = {"crs": crs, "transform": [10, 0, ox, 0, -10, oy], "width": width, "height": height}
    hr_grid = {"crs": crs, "transform": [10 / scale, 0, ox, 0, -10 / scale, oy],
               "width": width * scale, "height": height * scale}
    return lr_grid, hr_grid


def test_grid_aware_pair_accepted_with_valid_scale_aligned_grids():
    """T25: a perfectly-aligned pair with valid CRS/grid metadata should be
    accepted via the grid-aware path, and the returned sample should carry
    lr_grid/hr_grid (cropped/translated, not just echoed back verbatim)."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        lr = make_synthetic_scene(H=12, W=12, seed=40)
        hr = upsample_nearest(lr, 4)
        np.save(os.path.join(tmpdir, "lr", "grid-scene.npy"), lr)
        np.save(os.path.join(tmpdir, "hr", "grid-scene.npy"), hr)

        lr_grid, hr_grid = _grid_pair(12, 12, scale=4)
        manifest_path = os.path.join(tmpdir, "provenance.json")
        write_manifest(
            manifest_path,
            build_pair_manifest(
                tmpdir, "SEN2NAIP-converted", "Sentinel-2+NAIP",
                [{
                    "id": "grid-scene", "lr_path": "lr/grid-scene.npy", "hr_path": "hr/grid-scene.npy",
                    "metadata": {"lr_grid": lr_grid, "hr_grid": hr_grid},
                }],
            ),
        )

        ds = SEN2NAIPDataset(tmpdir, manifest_path=manifest_path, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 1, f"expected the grid-valid pair to be kept, dropped={ds.dropped_pairs}"
        sample = ds[0]
        assert "lr_grid" in sample and "hr_grid" in sample, "grid-aware pairs should return grid metadata in the sample"
        assert sample["lr_grid"]["crs"] == "EPSG:32643"
        assert sample["hr_grid"]["width"] == sample["lr_grid"]["width"] * 4
    finally:
        shutil.rmtree(tmpdir)


def test_grid_aware_pair_rejects_crs_mismatch():
    """T25 acceptance criterion: differing CRS must be rejected, not silently
    paired as if the grids matched."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        lr = make_synthetic_scene(H=12, W=12, seed=41)
        hr = upsample_nearest(lr, 4)
        np.save(os.path.join(tmpdir, "lr", "crs-mismatch.npy"), lr)
        np.save(os.path.join(tmpdir, "hr", "crs-mismatch.npy"), hr)

        lr_grid, hr_grid = _grid_pair(12, 12, scale=4)
        hr_grid["crs"] = "EPSG:4326"  # deliberately different from lr_grid's EPSG:32643
        manifest_path = os.path.join(tmpdir, "provenance.json")
        write_manifest(
            manifest_path,
            build_pair_manifest(
                tmpdir, "SEN2NAIP-converted", "Sentinel-2+NAIP",
                [{
                    "id": "crs-mismatch", "lr_path": "lr/crs-mismatch.npy", "hr_path": "hr/crs-mismatch.npy",
                    "metadata": {"lr_grid": lr_grid, "hr_grid": hr_grid},
                }],
            ),
        )

        ds = SEN2NAIPDataset(tmpdir, manifest_path=manifest_path, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 0, f"expected the CRS-mismatched pair to be dropped, kept={len(ds)}"
        assert ds.dropped_pairs[0][0] == "crs-mismatch"
        assert "grid contract violation" in ds.dropped_pairs[0][1]
    finally:
        shutil.rmtree(tmpdir)


def test_grid_aware_pair_rejects_non_scale_aligned_shift():
    """T25's actual point: this is the specific bug the Wide Research report
    named for this file — a registration shift that isn't an exact multiple
    of the scale factor used to be silently floor-divided (a real phase
    error), and must now be REJECTED instead when grid metadata declares
    the pairing should be exact."""
    scale = 4
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        lr = make_synthetic_scene(H=20, W=20, seed=2)  # same fixture as
        hr_full = upsample_nearest(lr, scale)           # test_estimate_pair_shift_recovers_known_shift,
        true_dy, true_dx = 3, -2                          # which proves this recovers shift (-3, 2) —
        C, H, W = hr_full.shape                            # neither -3 nor 2 is a multiple of scale=4.
        pad = 8
        hr_padded = np.pad(hr_full, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        hr_shifted = hr_padded[:, pad + true_dy: pad + true_dy + H, pad + true_dx: pad + true_dx + W]

        np.save(os.path.join(tmpdir, "lr", "phase-mismatch.npy"), lr)
        np.save(os.path.join(tmpdir, "hr", "phase-mismatch.npy"), hr_shifted)

        lr_grid, hr_grid = _grid_pair(20, 20, scale=scale)
        manifest_path = os.path.join(tmpdir, "provenance.json")
        write_manifest(
            manifest_path,
            build_pair_manifest(
                tmpdir, "SEN2NAIP-converted", "Sentinel-2+NAIP",
                [{
                    "id": "phase-mismatch", "lr_path": "lr/phase-mismatch.npy", "hr_path": "hr/phase-mismatch.npy",
                    "metadata": {"lr_grid": lr_grid, "hr_grid": hr_grid},
                }],
            ),
        )

        ds = SEN2NAIPDataset(tmpdir, manifest_path=manifest_path, max_shift=6, min_ncc_score=0.3)
        assert len(ds) == 0, f"expected the non-scale-aligned-shift pair to be rejected, kept={len(ds)}"
        assert ds.dropped_pairs[0][0] == "phase-mismatch"
        assert "grid contract violation" in ds.dropped_pairs[0][1]

        # Confirm the SAME pair, WITHOUT grid metadata, is accepted as before —
        # proving this is genuinely opt-in stricter behavior, not a regression
        # in the shape-only fallback path.
        manifest_path_no_grid = os.path.join(tmpdir, "provenance-no-grid.json")
        write_manifest(
            manifest_path_no_grid,
            build_pair_manifest(
                tmpdir, "SEN2NAIP-converted", "Sentinel-2+NAIP",
                [{"id": "phase-mismatch", "lr_path": "lr/phase-mismatch.npy", "hr_path": "hr/phase-mismatch.npy"}],
            ),
        )
        ds_no_grid = SEN2NAIPDataset(tmpdir, manifest_path=manifest_path_no_grid, max_shift=6, min_ncc_score=0.3)
        assert len(ds_no_grid) == 1, "the same pair without grid metadata should still be accepted via the shape-only fallback"
    finally:
        shutil.rmtree(tmpdir)


def test_grid_aware_pair_rejects_dimension_mismatch():
    """Grid metadata whose declared width/height don't match the actual
    array shape must be rejected, not silently trusted."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        lr = make_synthetic_scene(H=12, W=12, seed=42)
        hr = upsample_nearest(lr, 4)
        np.save(os.path.join(tmpdir, "lr", "dim-mismatch.npy"), lr)
        np.save(os.path.join(tmpdir, "hr", "dim-mismatch.npy"), hr)

        lr_grid, hr_grid = _grid_pair(999, 999, scale=4)  # deliberately wrong dims
        manifest_path = os.path.join(tmpdir, "provenance.json")
        write_manifest(
            manifest_path,
            build_pair_manifest(
                tmpdir, "SEN2NAIP-converted", "Sentinel-2+NAIP",
                [{
                    "id": "dim-mismatch", "lr_path": "lr/dim-mismatch.npy", "hr_path": "hr/dim-mismatch.npy",
                    "metadata": {"lr_grid": lr_grid, "hr_grid": hr_grid},
                }],
            ),
        )

        ds = SEN2NAIPDataset(tmpdir, manifest_path=manifest_path, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 0, f"expected the dimension-mismatched pair to be dropped, kept={len(ds)}"
        assert "grid contract violation" in ds.dropped_pairs[0][1]
    finally:
        shutil.rmtree(tmpdir)


def test_grid_aware_pair_requires_both_grids_present():
    """Only one of lr_grid/hr_grid present (not both) should be rejected as
    an incomplete/inconsistent contract, not silently treated as 'no grid info'."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "lr"))
        os.makedirs(os.path.join(tmpdir, "hr"))
        lr = make_synthetic_scene(H=12, W=12, seed=43)
        hr = upsample_nearest(lr, 4)
        np.save(os.path.join(tmpdir, "lr", "half-grid.npy"), lr)
        np.save(os.path.join(tmpdir, "hr", "half-grid.npy"), hr)

        lr_grid, _hr_grid = _grid_pair(12, 12, scale=4)
        manifest_path = os.path.join(tmpdir, "provenance.json")
        write_manifest(
            manifest_path,
            build_pair_manifest(
                tmpdir, "SEN2NAIP-converted", "Sentinel-2+NAIP",
                [{
                    "id": "half-grid", "lr_path": "lr/half-grid.npy", "hr_path": "hr/half-grid.npy",
                    "metadata": {"lr_grid": lr_grid},  # hr_grid deliberately omitted
                }],
            ),
        )

        ds = SEN2NAIPDataset(tmpdir, manifest_path=manifest_path, max_shift=0, min_ncc_score=0.9)
        assert len(ds) == 0, f"expected the half-declared-grid pair to be dropped, kept={len(ds)}"
        assert "both lr_grid and hr_grid" in ds.dropped_pairs[0][1]
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_ncc_self_score_is_one()
    test_upsample_nearest_shape()
    test_estimate_pair_shift_recovers_known_shift()
    test_apply_shift_and_crop_shapes_consistent()
    test_dataset_end_to_end_with_synthetic_files()
    test_dataset_rejects_invalid_scale()
    test_dataset_rejects_malformed_lr_shape()
    test_dataset_normalizes_sentinel2_hr_by_10000()
    test_dataset_accepts_sealed_provenance_manifest()
    test_grid_aware_pair_accepted_with_valid_scale_aligned_grids()
    test_grid_aware_pair_rejects_crs_mismatch()
    test_grid_aware_pair_rejects_non_scale_aligned_shift()
    test_grid_aware_pair_rejects_dimension_mismatch()
    test_grid_aware_pair_requires_both_grids_present()
    print(f"\nAll tests passed. (torch available in this run: {_HAS_TORCH})")
