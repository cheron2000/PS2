"""T24 mask-aware processing regression tests."""

import sys
from pathlib import Path
import tempfile

import numpy as np

# Consistency fix (agent4, T25 turn) — same issue as tests/test_scene_protocol.py
# hit last turn: every test file in this repo runs directly via
# `python3 test_whatever.py` from the repo root, but a script inside tests/
# needs the repo root explicitly added to sys.path for `from src....` imports
# to resolve when invoked that way (python -m tests.test_mask_aware already
# worked; direct execution didn't).
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.provenance import build_pair_manifest, write_manifest
from src.datasets.sen2naip import SEN2NAIPDataset
from src.infer import _normalise_input
from src.metrics import metric_report
from src.preprocessing import normalize_bands


def _pair_arrays():
    rng = np.random.default_rng(4)
    lr = rng.random((4, 8, 8), dtype=np.float32)
    hr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    # Make the four bands structurally distinct for the band-schema checks.
    hr[1] *= 0.8
    hr[2] *= 0.6
    hr[3] *= 1.2
    return lr, hr


def test_normalization_uses_only_valid_pixels():
    data = np.ones((2, 4, 4), dtype=np.float32) * 10000
    data[:, 0, 0] = 999999
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    normalized, stats = normalize_bands(data, valid_mask=mask)
    assert np.isclose(normalized[:, 1, 1], 1.0).all()
    assert np.isclose(normalized[:, 0, 0], 0.0).all()
    assert stats["valid_mask_applied"] is True


def test_sen2naip_propagates_manifest_masks():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "lr").mkdir()
        (root / "hr").mkdir()
        lr, hr = _pair_arrays()
        lr_path, hr_path = root / "lr" / "x.npy", root / "hr" / "x.npy"
        np.save(lr_path, lr)
        np.save(hr_path, hr)
        lr_mask = np.ones(lr.shape[1:], dtype=bool)
        hr_mask = np.ones(hr.shape[1:], dtype=bool)
        lr_mask[0, 0] = False
        hr_mask[:4, :4] = False
        np.save(root / "lr_mask.npy", lr_mask)
        np.save(root / "hr_mask.npy", hr_mask)
        manifest = build_pair_manifest(
            root, "SEN2NAIP-test", "Sentinel-2/NAIP",
            [{
                "id": "x",
                "lr_path": "lr/x.npy",
                "hr_path": "hr/x.npy",
                "lr_mask_path": "lr_mask.npy",
                "hr_mask_path": "hr_mask.npy",
            }],
        )
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, manifest)
        ds = SEN2NAIPDataset(str(root), manifest_path=str(manifest_path), min_ncc_score=0.0)
        sample = ds[0]
        assert "valid_mask" in sample
        assert sample["valid_mask"].shape == sample["hr"].shape[-2:]
        assert not bool(sample["valid_mask"][0, 0])


def test_inference_normalization_masks_invalid_pixels():
    data = np.ones((4, 3, 3), dtype=np.float32) * 10000
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False
    normalized, stats = _normalise_input(data, "reflectance", valid_mask=mask)
    assert np.isclose(normalized[:, 1, 1], 1.0).all()
    assert np.isclose(normalized[:, 0, 0], 0.0).all()
    assert stats["valid_mask_applied"] is True


def test_metrics_respect_valid_mask():
    rng = np.random.default_rng(9)
    target = rng.random((2, 16, 16))
    prediction = target.copy()
    prediction[:, 0, 0] += 1000.0
    mask = np.ones((16, 16), dtype=bool)
    mask[0, 0] = False
    report = metric_report(prediction, target, valid_mask=mask)
    assert report["psnr_db"] == float("inf")
    assert report["alignment_ncc"] > 0.99


if __name__ == "__main__":
    test_normalization_uses_only_valid_pixels()
    test_sen2naip_propagates_manifest_masks()
    test_inference_normalization_masks_invalid_pixels()
    test_metrics_respect_valid_mask()
    print("mask-aware tests passed")
