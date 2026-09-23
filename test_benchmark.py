"""Tests for the T26 controlled benchmark harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmark import BenchmarkError, run_benchmark  # noqa: E402
from datasets.sen2naip import upsample_nearest  # noqa: E402


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    lr = rng.random((4, 8, 8)).astype(np.float32)
    target = upsample_nearest(lr, 2)
    mask = np.ones(target.shape[-2:], dtype=bool)
    mask[:2, :2] = False
    return {"scene_id": f"scene-{seed}", "lr": lr, "target": target, "valid_mask": mask}


def test_fixture_benchmark_runs_all_required_baselines():
    result = run_benchmark(
        [_scene(1), _scene(2)],
        scale=2,
        sr_fn=lambda lr: upsample_nearest(lr, 2),
        metadata={"data_status": "fixture"},
    )
    assert result["evidence"] == "development-fixture"
    assert set(("nearest", "bicubic", "sr")).issubset(result["methods"])
    assert result["scene_count"] == 2
    # This fixture uses nearest-neighbor SR, so it is not evidence that SR
    # beats bicubic. The harness must report the delta, not manufacture a pass;
    # exact reconstructions may correctly yield NaN for inf-minus-inf PSNR.
    assert "psnr_db" in result["sr_minus_bicubic"]
    assert set(result["per_scene"]["sr"]) == {"scene-1", "scene-2"}


def test_real_claim_requires_verified_manifests():
    try:
        run_benchmark([_scene()], scale=2, claim_real=True, metadata={"data_status": "fixture"})
        raise AssertionError("unverified fixture must not be labelled real")
    except BenchmarkError as exc:
        assert "verified_real" in str(exc)


def test_real_claim_requires_both_manifest_references():
    try:
        run_benchmark(
            [_scene()],
            scale=2,
            claim_real=True,
            metadata={"data_status": "verified_real", "provenance_manifest": "source.json"},
        )
        raise AssertionError("real claim without split manifest must be rejected")
    except BenchmarkError as exc:
        assert "scene_split_manifest" in str(exc)


def test_verified_real_label_is_explicit():
    result = run_benchmark(
        [_scene()],
        scale=2,
        claim_real=True,
        metadata={
            "data_status": "verified_real",
            "provenance_manifest": "source.json",
            "scene_split_manifest": "split.json",
        },
    )
    assert result["evidence"] == "verified-real"


def test_prediction_shape_mismatch_is_rejected():
    try:
        run_benchmark([_scene()], scale=2, sr_fn=lambda lr: lr)
        raise AssertionError("wrong-shaped SR output should be rejected")
    except BenchmarkError as exc:
        assert "prediction" in str(exc)


def test_invalid_mask_is_rejected():
    scene = _scene()
    scene["valid_mask"] = np.zeros((16, 16), dtype=bool)
    try:
        run_benchmark([scene], scale=2)
        raise AssertionError("empty mask should be rejected")
    except BenchmarkError as exc:
        assert "no valid pixels" in str(exc)


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
