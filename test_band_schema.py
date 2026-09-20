"""test_band_schema.py — tests for src/datasets/band_schema.py (T20).

Run: python3 test_band_schema.py
Pure numpy, no torch/network needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from datasets.band_schema import (
    BandSchema,
    Band,
    BandSchemaError,
    validate_bands,
    compute_checksum,
    build_manifest,
    SENTINEL2_L2A_4BAND,
    NAIP_4BAND,
    CARTOSAT3_MX_4BAND,
    SEN2VENUS_4BAND,
)


def _random_4band(seed=0, size=16):
    rng = np.random.default_rng(seed)
    return rng.random((4, size, size))


def test_valid_4band_passes():
    data = _random_4band()
    validate_bands(data, SENTINEL2_L2A_4BAND)  # should not raise


def test_wrong_band_count_rejected():
    data = np.random.default_rng(0).random((3, 16, 16))
    try:
        validate_bands(data, SENTINEL2_L2A_4BAND)
        assert False, "should have raised on band count mismatch"
    except BandSchemaError as e:
        assert "expected exactly 4 bands" in str(e)


def test_extra_bands_rejected_by_default():
    data = np.random.default_rng(0).random((5, 16, 16))
    try:
        validate_bands(data, SENTINEL2_L2A_4BAND)
        assert False, "should have raised — allow_extra_bands is False by default"
    except BandSchemaError:
        pass


def test_extra_bands_allowed_when_flagged():
    schema = BandSchema(
        product_name="test", bands=SENTINEL2_L2A_4BAND.bands, allow_extra_bands=True
    )
    data = np.random.default_rng(0).random((5, 16, 16))
    validate_bands(data, schema)  # should not raise


def test_duplicated_band_rejected():
    """The audit's specific named concern: a channel accidentally
    repeated (e.g. band-order bug truncating/copying a channel)."""
    data = _random_4band()
    data[3] = data[0]  # NIR is an exact copy of blue — a real bug signature
    try:
        validate_bands(data, SENTINEL2_L2A_4BAND)
        assert False, "should have raised on duplicated band"
    except BandSchemaError as e:
        assert "near-identical" in str(e)


def test_near_duplicate_band_rejected():
    data = _random_4band()
    data[3] = data[0] + np.random.default_rng(1).normal(0, 1e-6, data[0].shape)  # tiny noise, still near-identical
    try:
        validate_bands(data, SENTINEL2_L2A_4BAND)
        assert False, "should have raised on near-duplicate band"
    except BandSchemaError:
        pass


def test_genuinely_correlated_but_distinct_bands_pass():
    """Real spectral bands (e.g. red/green over vegetation) can be
    fairly correlated without being a bug — shouldn't false-positive."""
    rng = np.random.default_rng(2)
    base = rng.random((16, 16))
    data = np.stack([
        base + rng.normal(0, 0.15, (16, 16)),
        base + rng.normal(0, 0.15, (16, 16)),
        base + rng.normal(0, 0.15, (16, 16)),
        rng.random((16, 16)),  # NIR, genuinely independent
    ])
    validate_bands(data, SENTINEL2_L2A_4BAND)  # should not raise despite correlated visible bands


def test_constant_band_rejected():
    data = _random_4band()
    data[1] = 0.5  # a perfectly flat "green" band — looks like missing data, not signal
    try:
        validate_bands(data, SENTINEL2_L2A_4BAND)
        assert False, "should have raised on constant band"
    except BandSchemaError as e:
        assert "constant" in str(e)


def test_wrong_ndim_rejected():
    try:
        validate_bands(np.zeros((16, 16)), SENTINEL2_L2A_4BAND)
        assert False, "should have raised on 2D input"
    except BandSchemaError:
        pass


def test_all_schemas_are_internally_valid():
    """Sanity check every predefined schema is well-formed (4 bands each,
    matching this project's R/G/B/NIR convention throughout)."""
    for schema in [SENTINEL2_L2A_4BAND, NAIP_4BAND, CARTOSAT3_MX_4BAND, SEN2VENUS_4BAND]:
        assert schema.num_bands == 4, f"{schema.product_name}: expected 4 bands, got {schema.num_bands}"
        names = [b.name for b in schema.bands]
        assert len(set(names)) == 4, f"{schema.product_name}: band names should be unique, got {names}"


def test_checksum_deterministic():
    data = _random_4band(seed=5)
    assert compute_checksum(data) == compute_checksum(data.copy())


def test_checksum_differs_for_different_data():
    a = _random_4band(seed=5)
    b = _random_4band(seed=6)
    assert compute_checksum(a) != compute_checksum(b)


def test_checksum_differs_for_different_shape_same_bytes():
    """Same underlying bytes, different shape, should NOT collide —
    this is why the header includes shape/dtype, not just raw bytes."""
    flat = np.arange(16, dtype=np.float64)
    a = flat.reshape(4, 4)
    b = flat.reshape(2, 8)
    assert compute_checksum(a) != compute_checksum(b)


def test_build_manifest_round_trip():
    data = _random_4band(seed=7)
    manifest = build_manifest(
        data, source_id="test_001", product="SEN2NAIP", sensor="Sentinel-2",
        acquisition_date="2026-01-15", processing_level="L2A", region="test-AOI",
    )
    d = manifest.to_dict()
    assert d["source_id"] == "test_001"
    assert d["checksum"] == compute_checksum(data)
    assert d["extra"]["region"] == "test-AOI"


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
