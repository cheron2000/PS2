"""
src/eval_downstream.py — Downstream-task utility comparison (T8, narrowed).

Per solution-draft.md v11's round-5 finding: image-fidelity metrics
(PSNR/SSIM/SAM, src/metrics.py) don't reliably predict whether SR output
actually helps a downstream task — the correlation can be weak or even
negative (GeoSR-Bench, arXiv 2605.00310). This module answers the actual
question the PS asks for ("interpretability and analytical utility"):
given SR output, a bicubic baseline, and (optionally) real HR, which one
produces a more USEFUL land-cover classification against ground truth?

SCOPE NOTE — T8 was tagged L with a suggestion to split. Splitting it:
  - THIS file: the comparison/metrics logic (confusion matrix, per-class
    IoU, overall accuracy) plus a real-but-simple baseline classifier
    (NDVI thresholding), so the harness is genuinely testable end-to-end
    on synthetic data without needing a trained deep classifier.
  - Spun out as a new task (see tasks.md T13): loading REAL ESA WorldCover
    labels. esa-worldcover.org is outside this sandbox's network
    allowlist (same limitation every prior data task hit — HuggingFace
    for T5, the official SEN2Vénus release for T6) — so real-label
    loading has to happen in an environment with that access.

EXECUTION STATUS: written and unit-tested by sonnet5 on synthetic label
arrays — see test_eval_downstream.py. Pure numpy, no torch/network
dependency, runs anywhere.
"""
import numpy as np


def confusion_matrix(pred_labels: np.ndarray, true_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """pred_labels, true_labels: integer arrays of any matching shape, values in [0, num_classes).
    Returns (num_classes, num_classes) array, rows=true, cols=predicted."""
    if pred_labels.shape != true_labels.shape:
        raise ValueError(f"shape mismatch: {pred_labels.shape} vs {true_labels.shape}")
    pred_flat = pred_labels.flatten()
    true_flat = true_labels.flatten()
    if pred_flat.min() < 0 or pred_flat.max() >= num_classes or true_flat.min() < 0 or true_flat.max() >= num_classes:
        raise ValueError(f"label values must be in [0, {num_classes})")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true_flat, pred_flat), 1)
    return matrix


def per_class_iou(confusion: np.ndarray) -> np.ndarray:
    """IoU per class from a confusion matrix. Classes with no true or
    predicted pixels (0/0) return NaN rather than 0 — a class that never
    appears shouldn't silently look like total failure."""
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denom = true_positive + false_positive + false_negative
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, true_positive / np.where(denom == 0, 1, denom), np.nan)
    return iou


def overall_accuracy(confusion: np.ndarray) -> float:
    total = confusion.sum()
    if total == 0:
        return float("nan")
    return float(np.trace(confusion) / total)


def ndvi_threshold_classifier(image: np.ndarray, nir_band: int, red_band: int,
                               water_threshold: float = -0.1, vegetation_threshold: float = 0.3) -> np.ndarray:
    """A real, standard, if simple, land-cover proxy: NDVI = (NIR-Red)/(NIR+Red),
    thresholded into 3 classes. This is a legitimate, well-established remote
    sensing technique (not a strawman baseline) — appropriate as the
    hackathon-scale downstream-task stand-in this harness needs to be
    end-to-end runnable without a trained deep classifier.

    image: (C, H, W) reflectance-scaled array (0-1 range expected, see
    preprocessing.normalize_bands with method="reflectance").
    Returns (H, W) integer label array: 0=water, 1=non-vegetated, 2=vegetation.
    """
    if image.ndim != 3:
        raise ValueError(f"expected (C, H, W) array, got shape {image.shape}")
    nir = image[nir_band].astype(np.float64)
    red = image[red_band].astype(np.float64)
    denom = nir + red
    ndvi = np.divide(nir - red, denom, out=np.zeros_like(denom), where=denom != 0)

    labels = np.ones(ndvi.shape, dtype=np.int64)  # default: non-vegetated (1)
    labels[ndvi < water_threshold] = 0             # water
    labels[ndvi >= vegetation_threshold] = 2        # vegetation
    return labels


def compare_downstream_utility(
    true_labels: np.ndarray,
    num_classes: int,
    classifier_fn,
    sr_image: np.ndarray,
    bicubic_image: np.ndarray,
    hr_image: np.ndarray = None,
    classifier_kwargs: dict = None,
) -> dict:
    """The actual comparison solution-draft.md's Evaluation Protocol asks
    for: run the same classifier on SR output, the bicubic baseline, and
    (if available) real HR, score each against true_labels, and report
    them side by side — a judge-facing "+X% IoU" number, not just a
    fidelity metric.

    classifier_fn: callable(image, **classifier_kwargs) -> (H, W) label array.
    Returns {"sr": {...}, "bicubic": {...}, "hr": {...} or omitted}, each
    with "overall_accuracy", "per_class_iou", "mean_iou".
    """
    classifier_kwargs = classifier_kwargs or {}

    def _score(image, name):
        pred_labels = classifier_fn(image, **classifier_kwargs)
        if pred_labels.shape != true_labels.shape:
            raise ValueError(
                f"{name}: classifier output shape {pred_labels.shape} != true_labels shape {true_labels.shape}"
            )
        matrix = confusion_matrix(pred_labels, true_labels, num_classes)
        iou = per_class_iou(matrix)
        return {
            "overall_accuracy": overall_accuracy(matrix),
            "per_class_iou": iou.tolist(),
            "mean_iou": float(np.nanmean(iou)),
        }

    results = {
        "sr": _score(sr_image, "sr"),
        "bicubic": _score(bicubic_image, "bicubic"),
    }
    if hr_image is not None:
        results["hr"] = _score(hr_image, "hr")

    results["sr_improvement_over_bicubic_mean_iou"] = (
        results["sr"]["mean_iou"] - results["bicubic"]["mean_iou"]
    )
    return results


# T17 scene-level protocol exports. Imported here so callers can keep using the
# downstream-evaluation module as the public entry point without duplicating
# protocol code.
from src.scene_protocol import (
    SceneProtocolError,
    build_scene_splits,
    validate_scene_split_manifest,
    save_scene_split_manifest,
    load_scene_split_manifest,
    summarize_scene_metric,
    evaluate_scene_downstream,
    validate_worldcover_schema,
    uncertainty_calibration,
)
