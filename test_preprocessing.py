"""test_preprocessing.py — synthetic-data tests for src/preprocessing.py (T4).

Run: python3 test_preprocessing.py
Pure numpy, no torch/network needed — runs anywhere, including this sandbox.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from preprocessing import (
    build_validity_mask,
    normalize_bands,
    denormalize_bands,
    tile_into_patches,
    SCL_CLOUD_HIGH_PROB,
    SCL_VEGETATION,
    SCL_SNOW,
    DEFAULT_INVALID_CLASSES,
)


def test_validity_mask_basic():
    scl = np.array([
        [SCL_VEGETATION, SCL_CLOUD_HIGH_PROB],
        [SCL_SNOW, SCL_VEGETATION],
    ])
    mask = build_validity_mask(scl)
    expected = np.array([[True, False], [True, True]])  # snow valid by default, cloud not
    assert np.array_equal(mask, expected), f"got {mask}"


def test_validity_mask_custom_invalid_classes():
    scl = np.full((4, 4), SCL_SNOW)
    mask = build_validity_mask(scl, invalid_classes=DEFAULT_INVALID_CLASSES | {SCL_SNOW})
    assert not mask.any(), "snow should be invalid when explicitly added to the invalid set"


def test_validity_mask_rejects_non_2d():
    try:
        build_validity_mask(np.zeros((2, 4, 4)))
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_normalize_reflectance_roundtrip():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 10000, size=(4, 16, 16)).astype(np.float64)
    normalized, stats = normalize_bands(data, method="reflectance")
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    recovered = denormalize_bands(normalized, stats)
    assert np.allclose(recovered, data, atol=1e-6)


def test_normalize_reflectance_clips_out_of_range():
    data = np.array([[[-500.0, 15000.0]]])  # out of the [0, 10000] DN range
    normalized, _ = normalize_bands(data, method="reflectance")
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0


def test_normalize_percentile_roundtrip():
    rng = np.random.default_rng(1)
    data = rng.normal(500, 100, size=(4, 20, 20))
    normalized, stats = normalize_bands(data, method="percentile")
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    recovered = denormalize_bands(normalized, stats)
    # percentile stretch clips outliers, so roundtrip isn't exact for the
    # full range — but should be exact within the clipped [2nd, 98th] band
    mid_mask = (data >= stats["lo"]) & (data <= stats["hi"])
    assert np.allclose(recovered[mid_mask], data[mid_mask], atol=1e-6)


def test_normalize_zscore_roundtrip():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, size=(4, 16, 16))
    normalized, stats = normalize_bands(data, method="zscore")
    recovered = denormalize_bands(normalized, stats)
    assert np.allclose(recovered, data, atol=1e-6)


def test_normalize_constant_band_no_div_by_zero():
    data = np.full((2, 8, 8), 500.0)  # zero variance band
    normalized, stats = normalize_bands(data, method="zscore")
    assert np.all(np.isfinite(normalized)), "should not produce NaN/inf on a constant band"


def test_normalize_unknown_method_raises():
    try:
        normalize_bands(np.zeros((2, 4, 4)), method="not_a_real_method")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_tile_non_overlapping_count():
    data = np.zeros((4, 64, 64))
    patches = tile_into_patches(data, patch_size=16)
    assert len(patches) == 16, f"expected 16 non-overlapping 16x16 tiles in a 64x64 image, got {len(patches)}"


def test_tile_shapes_correct():
    data = np.random.rand(4, 32, 32)
    patches = tile_into_patches(data, patch_size=16)
    for p in patches:
        assert p["patch"].shape == (4, 16, 16), p["patch"].shape


def test_tile_drops_low_validity_patches():
    data = np.zeros((1, 32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16, :16] = True  # only the top-left 16x16 tile is fully valid
    patches = tile_into_patches(data, patch_size=16, validity_mask=mask, min_valid_fraction=0.9)
    assert len(patches) == 1, f"expected exactly 1 valid tile, got {len(patches)}"
    assert patches[0]["row"] == 0 and patches[0]["col"] == 0


def test_tile_overlapping_stride():
    data = np.zeros((1, 32, 32))
    non_overlap = tile_into_patches(data, patch_size=16, stride=16)
    overlap = tile_into_patches(data, patch_size=16, stride=8)
    assert len(overlap) > len(non_overlap), "smaller stride should produce more (overlapping) tiles"


def test_tile_rejects_bad_dims():
    try:
        tile_into_patches(np.zeros((32, 32)), patch_size=16)  # missing channel dim
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def main():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
