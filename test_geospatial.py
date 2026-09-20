"""Executable T16 tests; run with: python3 test_geospatial.py"""
import numpy as np
from src.datasets.geospatial import (
    GridContractError, GridSpec, combine_valid_masks, crop_pair_with_grids,
    mask_from_nodata, nearest_expand_mask, require_scale_aligned_shift,
    validate_grid_pair,
)


def grids(crs="EPSG:32643", hr_origin=(500000.0, 300000.0)):
    lr = GridSpec(crs, (10.0, 0.0, 500000.0, 0.0, -10.0, 300000.0), 4, 4)
    hr = GridSpec(crs, (2.5, 0.0, hr_origin[0], 0.0, -2.5, hr_origin[1]), 16, 16)
    return lr, hr


def test_valid_pair_and_transform_preservation():
    lr_grid, hr_grid = grids()
    validate_grid_pair(lr_grid, hr_grid, 4)
    lr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    hr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    lr2, hr2, lg2, hg2 = crop_pair_with_grids(lr, hr, 4, -4, 4, lr_grid, hr_grid)
    assert lr2.shape == (1, 3, 3) and hr2.shape == (1, 12, 12)
    assert lg2.transform == (10.0, 0.0, 500010.0, 0.0, -10.0, 300000.0)
    assert hg2.transform == (2.5, 0.0, 500000.0, 0.0, -2.5, 299990.0)
    assert (lg2.width, lg2.height) == (3, 3)
    assert (hg2.width, hg2.height) == (12, 12)
    print("PASS test_valid_pair_and_transform_preservation")


def test_reject_crs_and_pixel_phase_mismatch():
    lr, hr = grids()
    try:
        validate_grid_pair(lr, GridSpec("EPSG:4326", hr.transform, 16, 16), 4)
    except GridContractError:
        pass
    else:
        raise AssertionError("CRS mismatch must be rejected")
    _, bad_hr = grids(hr_origin=(500001.25, 300000.0))
    try:
        validate_grid_pair(lr, bad_hr, 4)
    except GridContractError:
        pass
    else:
        raise AssertionError("fractional HR origin must be rejected")
    print("PASS test_reject_crs_and_pixel_phase_mismatch")


def test_reject_non_grid_aligned_shift():
    try:
        require_scale_aligned_shift(1, 0, 4)
    except GridContractError:
        pass
    else:
        raise AssertionError("non-multiple HR shift must not use floor division")
    assert require_scale_aligned_shift(4, -8, 4) == (1, -2)
    print("PASS test_reject_non_grid_aligned_shift")


def test_mask_propagation_and_nodata():
    mask = np.array([[True, False], [True, True]])
    expanded = nearest_expand_mask(mask, 2)
    assert expanded.shape == (4, 4) and not expanded[0, 2] and expanded[3, 3]
    combined = combine_valid_masks(expanded, np.ones((4, 4), dtype=bool))
    assert np.array_equal(combined, expanded)
    data = np.array([[[1, 0], [np.nan, 3]], [[2, 0], [4, 3]]], dtype=np.float32)
    assert np.array_equal(mask_from_nodata(data, 0), np.array([[True, False], [False, True]]))
    print("PASS test_mask_propagation_and_nodata")


if __name__ == "__main__":
    test_valid_pair_and_transform_preservation()
    test_reject_crs_and_pixel_phase_mismatch()
    test_reject_non_grid_aligned_shift()
    test_mask_propagation_and_nodata()
    print("All T16 geospatial tests passed.")
