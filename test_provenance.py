"""Tests for the T23 provenance-manifest bridge.

Run: python3 test_provenance.py
Pure Python plus NumPy; no remote dataset or archive access is required.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets.provenance import (  # noqa: E402
    ProvenanceError,
    build_pair_manifest,
    load_numpy_pair,
    read_manifest,
    write_manifest,
)


def _make_pair(root: Path):
    (root / "lr").mkdir()
    (root / "hr").mkdir()
    lr = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    hr = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    np.save(root / "lr" / "scene-001.npy", lr)
    np.save(root / "hr" / "scene-001.npy", hr)
    return lr, hr


def test_build_write_read_and_load_pair():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        lr, hr = _make_pair(root)
        manifest = build_pair_manifest(
            root,
            product="SEN2NAIP-converted",
            sensor="Sentinel-2+NAIP",
            pairs=[
                {
                    "id": "scene-001",
                    "lr_path": "lr/scene-001.npy",
                    "hr_path": "hr/scene-001.npy",
                    "metadata": {"aoi": "test-AOI", "scale": 4},
                }
            ],
            metadata={"converter": "fixture-converter", "source_format": "TACO"},
        )
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, manifest)
        loaded = read_manifest(manifest_path)
        assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
        got_lr, got_hr, pair_metadata = load_numpy_pair(manifest_path, "scene-001")
        assert np.array_equal(got_lr, lr)
        assert np.array_equal(got_hr, hr)
        assert pair_metadata["aoi"] == "test-AOI"


def test_source_tampering_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_pair(root)
        manifest = build_pair_manifest(
            root,
            "fixture",
            "fixture-sensor",
            [{"id": "scene-001", "lr_path": "lr/scene-001.npy", "hr_path": "hr/scene-001.npy"}],
        )
        path = root / "manifest.json"
        write_manifest(path, manifest)
        with (root / "lr" / "scene-001.npy").open("ab") as handle:
            handle.write(b"tampered")
        try:
            read_manifest(path)
            raise AssertionError("tampered source should be rejected")
        except ProvenanceError as exc:
            assert "checksum mismatch" in str(exc)


def test_manifest_tampering_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_pair(root)
        path = root / "manifest.json"
        write_manifest(
            path,
            build_pair_manifest(
                root,
                "fixture",
                "fixture-sensor",
                [{"id": "scene-001", "lr_path": "lr/scene-001.npy", "hr_path": "hr/scene-001.npy"}],
            ),
        )
        payload = json.loads(path.read_text())
        payload["product"] = "edited-after-sealing"
        path.write_text(json.dumps(payload))
        try:
            read_manifest(path, verify_files=False)
            raise AssertionError("edited manifest should be rejected")
        except ProvenanceError as exc:
            assert "seal mismatch" in str(exc)


def test_source_outside_root_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "root"
        root.mkdir()
        outside = Path(tmpdir) / "outside.npy"
        np.save(outside, np.zeros((1, 2, 2), dtype=np.float32))
        try:
            build_pair_manifest(
                root,
                "fixture",
                "fixture-sensor",
                [{"id": "bad", "lr_path": outside, "hr_path": outside}],
            )
            raise AssertionError("outside-root source should be rejected")
        except ProvenanceError as exc:
            assert "inside manifest root" in str(exc)


def test_duplicate_pair_ids_are_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_pair(root)
        pair = {"id": "same", "lr_path": "lr/scene-001.npy", "hr_path": "hr/scene-001.npy"}
        try:
            build_pair_manifest(root, "fixture", "fixture-sensor", [pair, pair])
            raise AssertionError("duplicate pair IDs should be rejected")
        except ProvenanceError as exc:
            assert "duplicate pair id" in str(exc)


def main():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
