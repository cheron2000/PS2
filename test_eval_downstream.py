"""test_eval_downstream.py — synthetic-data tests for src/eval_downstream.py (T8).

Run: python3 test_eval_downstream.py
Pure numpy, no torch/network needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from eval_downstream import (
    confusion_matrix,
    per_class_iou,
    overall_accuracy,
    ndvi_threshold_classifier,
    compare_downstream_utility,
)


def test_confusion_matrix_perfect_prediction():
    true = np.array([[0, 1], [1, 2]])
    pred = true.copy()
    cm = confusion_matrix(pred, true, num_classes=3)
    assert np.array_equal(cm, np.diag([1, 2, 1])), f"got {cm}"


def test_confusion_matrix_shape_mismatch_raises():
    try:
        confusion_matrix(np.zeros((2, 2)), np.zeros((3, 3)), num_classes=2)
        assert False, "should have raised"
    except ValueError:
        pass


def test_confusion_matrix_out_of_range_raises():
    try:
        confusion_matrix(np.array([[0, 5]]), np.array([[0, 1]]), num_classes=2)
        assert False, "should have raised on label >= num_classes"
    except ValueError:
        pass


def test_per_class_iou_perfect_is_one():
    true = np.array([0, 0, 1, 1, 2, 2])
    cm = confusion_matrix(true, true, num_classes=3)
    iou = per_class_iou(cm)
    assert np.allclose(iou, [1.0, 1.0, 1.0]), f"got {iou}"


def test_per_class_iou_missing_class_is_nan():
    true = np.array([0, 0, 1, 1])  # class 2 never appears
    pred = true.copy()
    cm = confusion_matrix(pred, true, num_classes=3)
    iou = per_class_iou(cm)
    assert np.isnan(iou[2]), f"expected NaN for absent class, got {iou[2]}"
    assert np.allclose(iou[:2], [1.0, 1.0])


def test_overall_accuracy_half_correct():
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])  # 3/4 correct
    cm = confusion_matrix(pred, true, num_classes=2)
    acc = overall_accuracy(cm)
    assert abs(acc - 0.75) < 1e-9, f"got {acc}"


def test_ndvi_classifier_detects_vegetation():
    # 4-band image: bands 0,1,2,3 = R,G,B,NIR. High NIR, low Red -> vegetation.
    image = np.zeros((4, 4, 4))
    image[3, :, :] = 0.8   # NIR high everywhere
    image[0, :, :] = 0.1   # Red low everywhere -> NDVI = (0.8-0.1)/0.9 = 0.78 -> vegetation
    labels = ndvi_threshold_classifier(image, nir_band=3, red_band=0)
    assert np.all(labels == 2), f"expected all-vegetation, got unique values {np.unique(labels)}"


def test_ndvi_classifier_detects_water():
    image = np.zeros((4, 4, 4))
    image[3, :, :] = 0.05  # NIR low
    image[0, :, :] = 0.1   # Red slightly higher -> negative NDVI -> water
    labels = ndvi_threshold_classifier(image, nir_band=3, red_band=0)
    assert np.all(labels == 0), f"expected all-water, got unique values {np.unique(labels)}"


def test_ndvi_classifier_handles_zero_denominator():
    image = np.zeros((4, 4, 4))  # NIR=Red=0 everywhere -> denom=0
    labels = ndvi_threshold_classifier(image, nir_band=3, red_band=0)
    assert labels.shape == (4, 4)
    assert np.all(np.isfinite(labels))


def test_compare_downstream_utility_sr_better_than_bicubic():
    rng = np.random.default_rng(0)
    true_labels = rng.integers(0, 3, size=(8, 8))

    def perfect_classifier(image):
        return image[0].astype(np.int64)  # "classifier" that just reads labels back out

    def noisy_classifier(image):
        labels = image[0].astype(np.int64).copy()
        labels[0, 0] = (labels[0, 0] + 1) % 3  # introduce one error
        return labels

    # Encode true_labels into band 0 of the "sr" and "bicubic" stand-in images
    sr_image = np.zeros((1, 8, 8))
    sr_image[0] = true_labels
    bicubic_image = sr_image.copy()

    results = compare_downstream_utility(
        true_labels=true_labels,
        num_classes=3,
        classifier_fn=lambda img: perfect_classifier(img) if img is sr_image else noisy_classifier(img),
        sr_image=sr_image,
        bicubic_image=bicubic_image,
    )
    assert results["sr"]["overall_accuracy"] == 1.0
    assert results["bicubic"]["overall_accuracy"] < 1.0
    assert results["sr_improvement_over_bicubic_mean_iou"] > 0


def test_compare_downstream_utility_includes_hr_when_given():
    true_labels = np.zeros((4, 4), dtype=np.int64)
    dummy_image = np.zeros((1, 4, 4))

    def zero_classifier(image):
        return np.zeros((4, 4), dtype=np.int64)

    results = compare_downstream_utility(
        true_labels=true_labels,
        num_classes=2,
        classifier_fn=zero_classifier,
        sr_image=dummy_image,
        bicubic_image=dummy_image,
        hr_image=dummy_image,
    )
    assert "hr" in results
    assert results["hr"]["overall_accuracy"] == 1.0


def test_compare_downstream_utility_shape_mismatch_raises():
    true_labels = np.zeros((4, 4), dtype=np.int64)
    dummy_image = np.zeros((1, 4, 4))

    def wrong_shape_classifier(image):
        return np.zeros((2, 2), dtype=np.int64)

    try:
        compare_downstream_utility(
            true_labels=true_labels,
            num_classes=2,
            classifier_fn=wrong_shape_classifier,
            sr_image=dummy_image,
            bicubic_image=dummy_image,
        )
        assert False, "should have raised on classifier output shape mismatch"
    except ValueError:
        pass


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
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
