"""Scene-level evaluation protocol for T17."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


class SceneProtocolError(ValueError):
    pass


def _stable_key(scene_id: str) -> str:
    return hashlib.sha256(str(scene_id).encode("utf-8")).hexdigest()


def build_scene_splits(records: Sequence[Mapping[str, Any]], *,
                       train_fraction: float = 0.70,
                       val_fraction: float = 0.15,
                       test_fraction: float = 0.15,
                       seed: int = 0) -> dict[str, Any]:
    """Deterministic scene-disjoint split. Fractions apply to scenes, not patches."""
    fractions = (train_fraction, val_fraction, test_fraction)
    if not np.isfinite(fractions).all() or min(fractions) <= 0 or not np.isclose(sum(fractions), 1.0):
        raise SceneProtocolError("train/val/test fractions must be positive, finite, and sum to 1")
    if not records:
        raise SceneProtocolError("cannot split an empty record set")
    scene_to_samples: dict[str, list[str]] = {}
    for record in records:
        if "scene_id" not in record or "sample_id" not in record:
            raise SceneProtocolError("each record requires scene_id and sample_id")
        scene_to_samples.setdefault(str(record["scene_id"]), []).append(str(record["sample_id"]))
    scenes = sorted(scene_to_samples, key=_stable_key)
    if len(scenes) < 3:
        raise SceneProtocolError("at least 3 scenes are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    split_scenes = {
        "train": scenes[:n_train],
        "val": scenes[n_train:n_train + n_val],
        "test": scenes[n_train + n_val:],
    }
    split_samples = {
        split: [sample for scene in scene_ids for sample in scene_to_samples[scene]]
        for split, scene_ids in split_scenes.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": {"train": float(train_fraction), "val": float(val_fraction), "test": float(test_fraction)},
        "scenes": split_scenes,
        "samples": split_samples,
    }


def validate_scene_split_manifest(manifest: Mapping[str, Any],
                                  expected_sample_ids: Sequence[str] | None = None) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SceneProtocolError("unsupported scene manifest schema_version")
    scenes = manifest.get("scenes")
    samples = manifest.get("samples")
    if not isinstance(scenes, Mapping) or not isinstance(samples, Mapping):
        raise SceneProtocolError("manifest requires scenes and samples")
    names = ("train", "val", "test")
    scene_sets = {n: set(map(str, scenes.get(n, []))) for n in names}
    sample_sets = {n: set(map(str, samples.get(n, []))) for n in names}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if scene_sets[left] & scene_sets[right]:
                raise SceneProtocolError("scene leakage detected")
            if sample_sets[left] & sample_sets[right]:
                raise SceneProtocolError("sample leakage detected")
    if expected_sample_ids is not None:
        expected = set(map(str, expected_sample_ids))
        actual = set().union(*sample_sets.values())
        if actual != expected:
            raise SceneProtocolError(
                f"sample coverage mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
            )


def save_scene_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    validate_scene_split_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def load_scene_split_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_scene_split_manifest(manifest)
    return manifest


def _bootstrap_ci(values: Sequence[float], confidence: float, seed: int) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise SceneProtocolError("confidence must be between 0 and 1")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(4000, values.size))]
    means = sample.mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize_scene_metric(per_scene: Mapping[str, float], *,
                           confidence: float = 0.95, seed: int = 0) -> dict[str, Any]:
    names = sorted(per_scene)
    values = [float(per_scene[n]) for n in names]
    finite = [v for v in values if np.isfinite(v)]
    low, high = _bootstrap_ci(finite, confidence, seed)
    return {
        "scene_count": len(names),
        "finite_scene_count": len(finite),
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "confidence": float(confidence),
        "ci_low": low,
        "ci_high": high,
        "per_scene": dict(per_scene),
    }


def validate_worldcover_schema(labels: np.ndarray, *, num_classes: int = 11,
                               nodata_label: int = -1) -> dict[str, Any]:
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise SceneProtocolError("WorldCover labels must be a 2-D integer array")
    valid = labels != nodata_label
    if np.any(valid & ((labels < 0) | (labels >= num_classes))):
        raise SceneProtocolError("WorldCover contains labels outside the declared schema")
    return {
        "num_classes": int(num_classes),
        "nodata_label": int(nodata_label),
        "valid_fraction": float(np.mean(valid)),
        "compatible": True,
    }


def evaluate_scene_downstream(scene_records: Sequence[Mapping[str, Any]], *,
                              classifier_fn: Callable, num_classes: int,
                              classifier_kwargs: Mapping[str, Any] | None = None,
                              confidence: float = 0.95, seed: int = 0) -> dict[str, Any]:
    """Run one classifier and one label protocol independently per scene."""
    if not scene_records:
        raise SceneProtocolError("scene_records cannot be empty")
    from src.eval_downstream import confusion_matrix, per_class_iou
    kwargs = dict(classifier_kwargs or {})
    methods = ("sr", "bicubic", "hr")
    scores = {m: {} for m in methods}
    for record in scene_records:
        scene_id = str(record["scene_id"])
        truth = np.asarray(record["true_labels"])
        validate_worldcover_schema(truth, num_classes=num_classes)
        valid = truth >= 0
        for method in methods:
            image = record.get(method + "_image")
            if image is None:
                continue
            pred = np.asarray(classifier_fn(np.asarray(image), **kwargs))
            if pred.shape != truth.shape:
                raise SceneProtocolError(f"{scene_id}/{method}: prediction shape mismatch")
            if not np.issubdtype(pred.dtype, np.integer):
                raise SceneProtocolError(f"{scene_id}/{method}: predictions must be integer")
            pred_valid = pred[valid]
            truth_valid = truth[valid]
            if pred_valid.size == 0:
                scores[method][scene_id] = float("nan")
                continue
            if np.any((pred_valid < 0) | (pred_valid >= num_classes)):
                raise SceneProtocolError(f"{scene_id}/{method}: prediction outside label schema")
            cm = confusion_matrix(pred_valid, truth_valid, num_classes)
            scores[method][scene_id] = float(np.nanmean(per_class_iou(cm)))
    result = {"per_scene": {}, "summary": {}}
    for index, method in enumerate(methods):
        if scores[method]:
            result["per_scene"][method] = scores[method]
            result["summary"][method] = summarize_scene_metric(
                scores[method], confidence=confidence, seed=seed + index
            )
    if "sr" in result["summary"] and "bicubic" in result["summary"]:
        result["sr_minus_bicubic_mean_iou"] = (
            result["summary"]["sr"]["mean"] - result["summary"]["bicubic"]["mean"]
        )
    return result


def uncertainty_calibration(mean: np.ndarray, variance: np.ndarray, target: np.ndarray, *,
                            valid_mask: np.ndarray | None = None,
                            interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS) -> dict[str, Any]:
    """Gaussian NLL, standardized squared error, and empirical interval coverage."""
    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    target = np.asarray(target, dtype=float)
    if mean.shape != variance.shape or mean.shape != target.shape:
        raise SceneProtocolError("mean, variance and target must have identical shapes")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(target).all():
        raise SceneProtocolError("uncertainty inputs must be finite")
    if np.any(variance <= 0):
        raise SceneProtocolError("predictive variance must be strictly positive")
    mask = np.ones(mean.shape, dtype=bool) if valid_mask is None else np.broadcast_to(
        np.asarray(valid_mask, dtype=bool), mean.shape
    )
    if not np.any(mask):
        raise SceneProtocolError("valid_mask contains no valid elements")
    error = target[mask] - mean[mask]
    var = variance[mask]
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + error * error / var)
    standardized = error * error / var
    coverage = {}
    for level in interval_levels:
        if not 0.0 < level < 1.0:
            raise SceneProtocolError("interval levels must be between 0 and 1")
        z = _normal_quantile(0.5 + float(level) / 2.0)
        coverage[f"{level:.2f}"] = float(np.mean(np.abs(error) <= z * np.sqrt(var)))
    return {
        "gaussian_nll": float(nll),
        "mean_standardized_squared_error": float(np.mean(standardized)),
        "coverage": coverage,
        "sample_count": int(error.size),
    }


def _normal_quantile(p: float) -> float:
    """Acklam-free approximation via bisection of the standard normal CDF."""
    if not 0.0 < p < 1.0:
        raise SceneProtocolError("probability must be between 0 and 1")
    lo, hi = -9.0, 9.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + np.math.erf(mid / np.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
