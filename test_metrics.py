"""Executable tests for core evaluation metrics. Run: python3 test_metrics.py."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.metrics import ergas, bandwise_metrics


def test_ergas_identity():
    x = np.ones((4, 8, 8), dtype=float)
    assert abs(ergas(x, x, scale=4)) < 1e-12
    print("[PASS] ERGAS identity")


def test_ergas_known_error_and_scale():
    target = np.ones((2, 4, 4), dtype=float)
    pred = target * 1.1
    # RMSE/mean = 0.1 for each band; ERGAS = 100/4 * 0.1.
    assert abs(ergas(pred, target, scale=4) - 2.5) < 1e-12
    assert ergas(pred, target, scale=2) > ergas(pred, target, scale=4)
    print("[PASS] ERGAS scale/error")


def test_ergas_mask():
    target = np.ones((1, 2, 2), dtype=float)
    pred = target.copy()
    pred[0, 0, 0] = 100.0
    mask = np.array([[False, True], [True, True]])
    assert ergas(pred, target, scale=4, valid_mask=mask) < 1e-12
    print("[PASS] ERGAS mask")


def test_bandwise_metrics():
    target = np.ones((2, 4, 4), dtype=float)
    pred = target.copy()
    pred[0] *= 2
    report = bandwise_metrics(pred, target, data_range=2.0)
    assert len(report["bands"]) == 2
    assert report["bands"][0]["mse"] == 1.0
    assert report["bands"][1]["mse"] == 0.0
    assert report["bands"][0]["target_mean"] == 1.0
    print("[PASS] bandwise metrics")


if __name__ == "__main__":
    test_ergas_identity()
    test_ergas_known_error_and_scale()
    test_ergas_mask()
    test_bandwise_metrics()
    print("All metric extension tests passed.")
