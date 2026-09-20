"""T16 Cartosat integration tests; run with: python3 test_t16_pairing.py"""
import numpy as np
from src.datasets.cartosat_pairing import PairRejected, prepare_pair
from src.datasets.sen2naip import upsample_nearest


def grid_pair(h=4, w=4):
    lr = {"crs": "EPSG:32643", "transform": [10, 0, 500000, 0, -10, 300000], "width": w, "height": h}
    hr = {"crs": "EPSG:32643", "transform": [2.5, 0, 500000, 0, -2.5, 300000], "width": w * 4, "height": h * 4}
    return lr, hr


def test_grid_metadata_and_masks_round_trip():
    rng = np.random.default_rng(7)
    lr = rng.uniform(0.1, 0.9, size=(4, 4, 4)).astype(np.float32)
    hr = upsample_nearest(lr, 4)
    lr_mask = np.ones((4, 4), dtype=bool)
    hr_mask = np.ones((16, 16), dtype=bool)
    lr_mask[1, 2] = False
    pair = prepare_pair(lr, hr, metadata={"lr_grid": grid_pair()[0], "hr_grid": grid_pair()[1]}, lr_valid_mask=lr_mask, hr_valid_mask=hr_mask)
    assert pair.valid_mask.shape == (16, 16)
    assert not pair.valid_mask[4, 8]
    assert pair.metadata["lr_grid"]["width"] == 4
    assert pair.metadata["hr_grid"]["width"] == 16
    print("PASS test_grid_metadata_and_masks_round_trip")


def test_grid_mismatch_is_rejected():
    rng = np.random.default_rng(8)
    lr = rng.uniform(0.1, 0.9, size=(4, 4, 4)).astype(np.float32)
    hr = upsample_nearest(lr, 4)
    lr_grid, hr_grid = grid_pair()
    hr_grid["crs"] = "EPSG:4326"
    try:
        prepare_pair(lr, hr, metadata={"lr_grid": lr_grid, "hr_grid": hr_grid})
    except PairRejected as exc:
        assert "CRS mismatch" in str(exc)
    else:
        raise AssertionError("CRS mismatch must be rejected")
    print("PASS test_grid_mismatch_is_rejected")


if __name__ == "__main__":
    test_grid_metadata_and_masks_round_trip()
    test_grid_mismatch_is_rejected()
    print("All T16 pairing integration tests passed.")
