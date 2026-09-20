"""T17 scene protocol regression tests."""
import tempfile
from pathlib import Path
import numpy as np

from src.scene_protocol import (
    SceneProtocolError,
    build_scene_splits,
    validate_scene_split_manifest,
    save_scene_split_manifest,
    load_scene_split_manifest,
    summarize_scene_metric,
    validate_worldcover_schema,
    evaluate_scene_downstream,
    uncertainty_calibration,
)


def test_scene_split_is_disjoint_and_persistent():
    records = [{"scene_id": f"s{i}", "sample_id": f"s{i}_p{j}"} for i in range(6) for j in range(2)]
    manifest = build_scene_splits(records, seed=17)
    validate_scene_split_manifest(manifest, expected_sample_ids=[r["sample_id"] for r in records])
    all_scenes = [scene for group in manifest["scenes"].values() for scene in group]
    assert len(all_scenes) == 6
    assert len(set(all_scenes)) == 6
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits.json"
        save_scene_split_manifest(manifest, path)
        assert load_scene_split_manifest(path) == manifest


def test_scene_split_rejects_leakage():
    records = [{"scene_id": f"s{i}", "sample_id": f"p{i}"} for i in range(3)]
    manifest = build_scene_splits(records)
    manifest["scenes"]["test"].append(manifest["scenes"]["train"][0])
    try:
        validate_scene_split_manifest(manifest)
    except SceneProtocolError:
        pass
    else:
        raise AssertionError("scene leakage must be rejected")


def test_worldcover_schema_and_scene_macro_iou():
    labels = np.array([[0, 1], [1, -1]], dtype=np.int64)
    assert validate_worldcover_schema(labels)["valid_fraction"] == 0.75
    records = [
        {
            "scene_id": "a",
            "true_labels": labels,
            "sr_image": np.zeros((1, 2, 2), dtype=np.float32),
            "bicubic_image": np.zeros((1, 2, 2), dtype=np.float32),
            "hr_image": np.zeros((1, 2, 2), dtype=np.float32),
        },
        {
            "scene_id": "b",
            "true_labels": np.array([[0, 1], [0, 1]], dtype=np.int64),
            "sr_image": np.zeros((1, 2, 2), dtype=np.float32),
            "bicubic_image": np.zeros((1, 2, 2), dtype=np.float32),
        },
    ]
    def classifier(image):
        return np.array([[0, 1], [0, 1]], dtype=np.int64)
    result = evaluate_scene_downstream(records, classifier_fn=classifier, num_classes=2)
    assert set(result["summary"]) == {"sr", "bicubic", "hr"}
    assert 0.0 <= result["summary"]["sr"]["ci_low"] <= result["summary"]["sr"]["mean"] <= result["summary"]["sr"]["ci_high"] <= 1.0


def test_uncertainty_calibration_is_finite_and_covers():
    mean = np.zeros((100,), dtype=float)
    target = np.zeros((100,), dtype=float)
    variance = np.ones((100,), dtype=float)
    result = uncertainty_calibration(mean, variance, target)
    assert np.isfinite(result["gaussian_nll"])
    assert result["mean_standardized_squared_error"] == 0.0
    assert all(value == 1.0 for value in result["coverage"].values())


if __name__ == "__main__":
    test_scene_split_is_disjoint_and_persistent()
    test_scene_split_rejects_leakage()
    test_worldcover_schema_and_scene_macro_iou()
    test_uncertainty_calibration_is_finite_and_covers()
    print("scene protocol tests passed")
