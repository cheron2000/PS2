"""test_infer.py — end-to-end integration test for src/infer.py (T11).

astrasr's T11 turn included a shape-only smoke_test() but flagged that no
execution verification happened against the real repo file, and the
georeferenced export path (_write_geotiff, the fiddliest part - affine
transform scaling, CRS propagation) wasn't exercised by smoke_test() at
all. This test covers that gap: a real synthetic checkpoint, a real
synthetic input GeoTIFF (not a bare numpy array), run through the actual
load_checkpoint -> predict -> write_output path, checking the output
GeoTIFF's transform is correctly scaled by the SR factor.

Run: python3 test_infer.py (from repo root, matching infer.py's own
`from src.model import SRModel` absolute-import convention)
"""
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # for `from src.model import ...` inside infer.py

import numpy as np
import torch
from rasterio.transform import Affine
from rasterio.crs import CRS
import rasterio

from src.model import SRModel
from src.infer import load_checkpoint, predict, write_output, _load_input, smoke_test


def _make_checkpoint(path, **model_kwargs):
    model = SRModel(**model_kwargs)
    torch.save({"model_state": model.state_dict(), "model_config": model_kwargs}, path)
    return model


def test_smoke_test_still_passes():
    """Regression check on astrasr's original test before adding new coverage."""
    smoke_test()


def test_load_checkpoint_recovers_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        kwargs = dict(in_channels=4, out_channels=4, base_channels=8,
                      num_attn_blocks=1, window_size=4, num_heads=2, scale=2)
        _make_checkpoint(ckpt_path, **kwargs)

        model, config = load_checkpoint(ckpt_path, torch.device("cpu"))
        assert config["scale"] == 2
        assert config["base_channels"] == 8


def test_load_checkpoint_accepts_raw_state_dict():
    """load_checkpoint's docstring says it accepts a bare state_dict as a
    convenience path, not just the {model_state, model_config} wrapper —
    worth actually checking, since that's a branch smoke_test() never hits.

    Must use SRModel()'s DEFAULT args here: with no model_config saved,
    load_checkpoint() reconstructs SRModel() with its defaults, so the
    saved state_dict has to match those exactly, not an arbitrary small
    test-sized model — a custom-sized model here would fail to load for
    a reason that has nothing to do with load_checkpoint's own logic
    (caught this the first time this test was written)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "raw_ckpt.pt")
        model = SRModel()  # all defaults: base_channels=64, num_attn_blocks=6, scale=4, ...
        torch.save(model.state_dict(), ckpt_path)  # raw state_dict, no wrapper

        loaded_model, config = load_checkpoint(ckpt_path, torch.device("cpu"))
        assert config == {}, "raw state_dict path should report empty config, not fabricate one"
        # Confirm it's actually usable, not just loaded without error:
        x = torch.rand(1, 4, 8, 8)
        with torch.no_grad():
            mean, log_var = loaded_model(x)
        assert mean.shape == (1, 4, 32, 32)  # default scale=4


def test_end_to_end_geotiff_pipeline():
    """The actual gap astrasr flagged: full pipeline through real
    georeferenced I/O, not just tensor shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Real synthetic input GeoTIFF (4-band, 16x16, realistic-ish
        #    Sentinel-2 DN values so reflectance normalization is exercised meaningfully)
        rng = np.random.default_rng(0)
        input_data = rng.integers(0, 4000, size=(4, 16, 16)).astype(np.float32)
        input_transform = Affine.translation(500000, 4500000) * Affine.scale(10, -10)  # 10m Sentinel-2 pixels
        input_crs = CRS.from_epsg(32643)  # UTM 43N, plausible for India

        input_path = os.path.join(tmpdir, "input.tif")
        with rasterio.open(
            input_path, "w", driver="GTiff", height=16, width=16, count=4,
            dtype="float32", crs=input_crs, transform=input_transform,
        ) as dst:
            dst.write(input_data)

        # 2. Real synthetic checkpoint, scale=4 (matching the project's
        #    primary 10m -> 2.5m route)
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        model_kwargs = dict(in_channels=4, out_channels=4, base_channels=8,
                             num_attn_blocks=1, window_size=4, num_heads=2, scale=4)
        _make_checkpoint(ckpt_path, **model_kwargs)

        # 3. Run the actual load -> predict -> write pipeline
        device = torch.device("cpu")
        model, config = load_checkpoint(ckpt_path, device)
        data, metadata = _load_input(input_path)
        assert metadata["crs"] is not None and metadata["transform"] is not None

        normalized = np.clip(data / 10000.0, 0.0, 1.0).astype(np.float32)
        mean, variance = predict(model, normalized, device)
        assert mean.shape == (4, 64, 64), f"expected 4x upsampling to 64x64, got {mean.shape}"

        output_path = os.path.join(tmpdir, "output.tif")
        write_output(output_path, mean, metadata, scale=4)

        # 4. Verify the OUTPUT georeferencing is correct: same CRS, and the
        #    transform's pixel size should be 1/4 of the input's (finer
        #    resolution over the same ground extent) — this is exactly the
        #    "scaled affine transform" logic astrasr's turn implemented but
        #    couldn't execution-test.
        with rasterio.open(output_path) as src:
            assert src.crs == input_crs
            assert src.width == 64 and src.height == 64
            input_pixel_size = input_transform.a   # 10
            output_pixel_size = src.transform.a
            assert abs(output_pixel_size - input_pixel_size / 4) < 1e-9, (
                f"expected output pixel size {input_pixel_size / 4}, got {output_pixel_size}"
            )
            # Same top-left origin — only resolution should change, not extent placement.
            assert abs(src.transform.c - input_transform.c) < 1e-9
            assert abs(src.transform.f - input_transform.f) < 1e-9


def test_uncertainty_output_is_written_separately():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_kwargs = dict(in_channels=4, out_channels=4, base_channels=8,
                             num_attn_blocks=1, window_size=4, num_heads=2, scale=2)
        model = SRModel(**model_kwargs)
        data = np.random.default_rng(0).random((4, 8, 8), dtype=np.float32)
        mean, variance = predict(model, data, torch.device("cpu"))

        mean_path = os.path.join(tmpdir, "mean.npy")
        var_path = os.path.join(tmpdir, "var.npy")
        write_output(mean_path, mean, {}, scale=2)
        write_output(var_path, variance, {}, scale=2)

        assert np.array_equal(np.load(mean_path), mean)
        assert np.array_equal(np.load(var_path), variance)
        assert np.all(np.load(var_path) >= 0), "variance must be non-negative"


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
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
