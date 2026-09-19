"""test_worldcover.py — tests for src/datasets/worldcover.py (T13).

Unlike T5/T6/T10's synthetic numpy-array tests, this exercises REAL
rasterio GeoTIFF I/O and REAL CRS/reprojection logic — writes a
synthetic-but-genuinely-valid WorldCover-shaped tile to a temp file,
reads it back, reprojects it, and checks the result. No network access
needed (rasterio itself works offline; only fetching a real WorldCover
tile from S3 needs network, which this sandbox doesn't have).

Run: python3 test_worldcover.py
"""
import sys
import tempfile
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS
from datasets.worldcover import (
    remap_worldcover_classes,
    load_worldcover_tile,
    align_to_reference,
    write_geotiff,
    WORLDCOVER_CLASSES,
    NUM_WORLDCOVER_CLASSES,
    NODATA_LABEL,
)


def test_remap_known_classes():
    raw = np.array([[10, 40], [80, 100]])
    remapped = remap_worldcover_classes(raw)
    expected = np.array([[0, 3], [7, 10]])  # per WORLDCOVER_CLASSES mapping
    assert np.array_equal(remapped, expected), f"got {remapped}"


def test_remap_nodata_and_unknown_map_to_negative_one():
    raw = np.array([[NODATA_LABEL, 999], [10, 20]])
    remapped = remap_worldcover_classes(raw)
    assert remapped[0, 0] == -1 and remapped[0, 1] == -1
    assert remapped[1, 0] == 0 and remapped[1, 1] == 1


def test_remap_covers_all_documented_classes():
    all_raw_codes = np.array(list(WORLDCOVER_CLASSES.keys())).reshape(1, -1)
    remapped = remap_worldcover_classes(all_raw_codes)
    assert set(remapped.flatten().tolist()) == set(range(NUM_WORLDCOVER_CLASSES))


def test_real_geotiff_roundtrip():
    """Write a real GeoTIFF, read it back with load_worldcover_tile(),
    confirm data/transform/crs all survive the round trip."""
    rng = np.random.default_rng(0)
    raw_codes = list(WORLDCOVER_CLASSES.keys())
    data = rng.choice(raw_codes, size=(32, 32)).astype(np.uint8)
    transform = Affine.translation(500000, 4500000) * Affine.scale(10, -10)  # 10m pixels, UTM-like origin
    crs = CRS.from_epsg(32643)  # UTM zone 43N - plausible zone for India

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "synthetic_worldcover.tif")
        write_geotiff(path, data, transform, crs, dtype="uint8")

        read_data, read_transform, read_crs = load_worldcover_tile(path)

        assert np.array_equal(read_data, data), "data didn't survive GeoTIFF round trip"
        assert read_transform == transform, f"transform mismatch: {read_transform} vs {transform}"
        assert read_crs == crs, f"CRS mismatch: {read_crs} vs {crs}"


def test_align_to_reference_identity_transform():
    """When the reference grid IS the WorldCover grid, alignment should
    be a no-op (every pixel maps to itself)."""
    rng = np.random.default_rng(1)
    raw_codes = list(WORLDCOVER_CLASSES.keys())
    data = rng.choice(raw_codes, size=(16, 16)).astype(np.uint8)
    transform = Affine.translation(500000, 4500000) * Affine.scale(10, -10)
    crs = CRS.from_epsg(32643)

    aligned = align_to_reference(
        data, transform, crs,
        reference_transform=transform, reference_crs=crs, reference_shape=(16, 16),
    )
    assert np.array_equal(aligned, data), "identity reprojection should reproduce the source exactly"


def test_align_to_reference_different_grid():
    """Reprojecting onto a shifted+different-resolution grid should still
    produce a valid-shaped, class-code-only output (no interpolated
    fractional values — nearest-neighbor resampling for categorical data)."""
    data = np.zeros((10, 10), dtype=np.uint8)
    data[:5, :] = 10   # Tree cover
    data[5:, :] = 80   # Water
    src_transform = Affine.translation(0, 100) * Affine.scale(10, -10)
    crs = CRS.from_epsg(32643)

    # Reference grid: finer resolution (5m instead of 10m), same origin/extent roughly
    ref_transform = Affine.translation(0, 100) * Affine.scale(5, -5)
    aligned = align_to_reference(
        data, src_transform, crs,
        reference_transform=ref_transform, reference_crs=crs, reference_shape=(20, 20),
    )
    assert aligned.shape == (20, 20)
    present_values = set(np.unique(aligned).tolist())
    # Only real class codes (plus possibly 0 as reproject's fill value)
    # should appear — never an interpolated in-between value like 45.
    assert present_values <= {0, 10, 80}, f"unexpected values after nearest-neighbor reprojection: {present_values}"


def test_remap_output_compatible_with_eval_downstream():
    """Confirm the remapped output actually satisfies src/eval_downstream.py's
    confusion_matrix() precondition (values in [0, num_classes)) once nodata
    is filtered — this is the actual point of T13: feeding real labels into
    T8's harness without T8 needing to change."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from eval_downstream import confusion_matrix

    raw = np.array([[10, 40, NODATA_LABEL], [80, 100, 20]])
    remapped = remap_worldcover_classes(raw)
    valid_mask = remapped >= 0
    valid_labels = remapped[valid_mask]

    # Using the same array as both "prediction" and "truth" here just to
    # confirm confusion_matrix() accepts the value range without raising —
    # the actual SR-vs-bicubic comparison is exercised in test_eval_downstream.py.
    cm = confusion_matrix(valid_labels, valid_labels, num_classes=NUM_WORLDCOVER_CLASSES)
    assert cm.shape == (NUM_WORLDCOVER_CLASSES, NUM_WORLDCOVER_CLASSES)


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
