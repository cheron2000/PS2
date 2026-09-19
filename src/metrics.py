"""
Evaluation metrics for Sentinel-2 super-resolution experiments.

NumPy-only so they can run independently of the PyTorch model.

Conventions:
  * Images may be (H, W) or channel-first (C, H, W).
  * Values must be finite. PSNR/SSIM use the supplied data_range.
  * Spatial alignment uses an integer-pixel normalized cross-correlation search.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple
import numpy as np

def _validate_pair(prediction: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {true.shape}")
    if pred.ndim not in (2, 3):
        raise ValueError(f"expected (H,W) or (C,H,W), got {pred.shape}")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
        raise ValueError("prediction and target must contain only finite values")
    if pred.shape[-2] < 2 or pred.shape[-1] < 2:
        raise ValueError("spatial dimensions must be at least 2x2")
    return pred, true

def _masked_values(pred, true, valid_mask=None):
    if valid_mask is None:
        return pred.reshape(-1), true.reshape(-1)
    mask = np.asarray(valid_mask, dtype=bool)
    spatial = pred.shape[-2:]
    if mask.shape == spatial:
        mask = np.broadcast_to(mask, pred.shape)
    elif mask.shape != pred.shape:
        raise ValueError(f"valid_mask must have shape {spatial} or {pred.shape}, got {mask.shape}")
    if not np.any(mask):
        raise ValueError("valid_mask contains no valid pixels")
    return pred[mask], true[mask]

def mse(prediction, target, valid_mask=None) -> float:
    pred, true = _validate_pair(prediction, target)
    p, t = _masked_values(pred, true, valid_mask)
    return float(np.mean((p - t) ** 2))

def psnr(prediction, target, data_range=1.0, valid_mask=None, zero_mse=math.inf) -> float:
    """Peak signal-to-noise ratio in dB."""
    if data_range <= 0 or not np.isfinite(data_range):
        raise ValueError(f"data_range must be finite and positive, got {data_range}")
    error = mse(prediction, target, valid_mask)
    if error == 0:
        return float(zero_mse)
    return float(10.0 * np.log10((data_range ** 2) / error))

def _uniform_filter_2d(image, window_size):
    """H x W box filter using an integral image."""
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    w = window_size
    return (integral[w:, w:] - integral[:-w, w:] -
            integral[w:, :-w] + integral[:-w, :-w]) / float(w * w)

def _ssim_map(pred, true, data_range, window_size):
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    mu_p = _uniform_filter_2d(pred, window_size)
    mu_t = _uniform_filter_2d(true, window_size)
    mu_p2 = _uniform_filter_2d(pred * pred, window_size)
    mu_t2 = _uniform_filter_2d(true * true, window_size)
    mu_pt = _uniform_filter_2d(pred * true, window_size)
    var_p = np.maximum(mu_p2 - mu_p * mu_p, 0.0)
    var_t = np.maximum(mu_t2 - mu_t * mu_t, 0.0)
    cov_pt = mu_pt - mu_p * mu_t
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_p * mu_t + c1) * (2 * cov_pt + c2) /
            ((mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)))

def ssim(prediction, target, data_range=1.0, window_size=11, valid_mask=None) -> float:
    """Mean structural similarity over channels, using a uniform local window."""
    if data_range <= 0 or not np.isfinite(data_range):
        raise ValueError(f"data_range must be finite and positive, got {data_range}")
    pred, true = _validate_pair(prediction, target)
    channels_p = pred[None, ...] if pred.ndim == 2 else pred
    channels_t = true[None, ...] if true.ndim == 2 else true

    center_mask = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape == pred.shape[-2:]:
            center_mask = mask
        elif mask.shape == pred.shape:
            center_mask = np.all(mask, axis=0) if pred.ndim == 3 else mask
        else:
            raise ValueError(f"valid_mask shape {mask.shape} does not match image")
        full_valid = _uniform_filter_2d(center_mask.astype(np.float64), window_size) >= 1.0 - 1e-12
        if not np.any(full_valid):
            raise ValueError("valid_mask has no fully-valid SSIM windows")
    else:
        full_valid = None

    scores = []
    for p, t in zip(channels_p, channels_t):
        values = _ssim_map(p, t, data_range, window_size)
        scores.append(float(np.mean(values) if full_valid is None else np.mean(values[full_valid])))
    return float(np.mean(scores))

def spectral_angle_error(prediction, target, valid_mask=None, degrees=True) -> float:
    """Mean spectral angle error (SAM) per pixel."""
    pred, true = _validate_pair(prediction, target)
    if pred.ndim == 2:
        pred, true = pred[None, ...], true[None, ...]
    p = pred.reshape(pred.shape[0], -1).T
    t = true.reshape(true.shape[0], -1).T
    mask = np.ones(p.shape[0], dtype=bool)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape == pred.shape[-2:]:
            mask &= vm.reshape(-1)
        elif vm.shape == pred.shape:
            mask &= np.all(vm, axis=0).reshape(-1)
        else:
            raise ValueError(f"valid_mask shape {vm.shape} does not match image")
    p_norm = np.linalg.norm(p, axis=1)
    t_norm = np.linalg.norm(t, axis=1)
    mask &= (p_norm > 0) & (t_norm > 0)
    if not np.any(mask):
        raise ValueError("no non-zero spectral vectors remain after masking")
    cosine = np.sum(p[mask] * t[mask], axis=1) / (p_norm[mask] * t_norm[mask])
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    value = float(np.mean(angles))
    return float(np.degrees(value) if degrees else value)

def _overlap_slices(height, width, dy, dx):
    """Return prediction/target slices for target shifted by (dy, dx)."""
    if abs(dy) >= height or abs(dx) >= width:
        return None
    if dy >= 0:
        y_t, y_p = slice(dy, height), slice(0, height - dy)
    else:
        y_t, y_p = slice(0, height + dy), slice(-dy, height)
    if dx >= 0:
        x_t, x_p = slice(dx, width), slice(0, width - dx)
    else:
        x_t, x_p = slice(0, width + dx), slice(-dx, width)
    return y_p, x_p, y_t, x_t

def _ncc(a, b):
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return 0.0 if denom == 0 else float(np.dot(aa, bb) / denom)

def spatial_alignment_check(prediction, target, max_shift=4, min_ncc=0.90) -> Dict[str, object]:
    """Search integer shifts and report best NCC, overlap, and pass/fail."""
    pred, true = _validate_pair(prediction, target)
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if not 0.0 <= min_ncc <= 1.0:
        raise ValueError("min_ncc must be in [0, 1]")
    h, w = pred.shape[-2:]
    best_score, best_dy, best_dx, best_overlap = -1.0, 0, 0, 0.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            slices = _overlap_slices(h, w, dy, dx)
            if slices is None:
                continue
            y_p, x_p, y_t, x_t = slices
            p_overlap = pred[..., y_p, x_p]
            t_overlap = true[..., y_t, x_t]
            score = _ncc(p_overlap, t_overlap)
            overlap = p_overlap.shape[-2] * p_overlap.shape[-1] / float(h * w)
            if score > best_score or (np.isclose(score, best_score) and overlap > best_overlap):
                best_score, best_dy, best_dx, best_overlap = score, dy, dx, overlap
    return {
        "best_shift": (int(best_dy), int(best_dx)),
        "ncc": float(best_score),
        "overlap_fraction": float(best_overlap),
        "aligned": bool(best_score >= min_ncc),
        "threshold": float(min_ncc),
    }

def metric_report(prediction, target, data_range=1.0, ssim_window=11, valid_mask=None) -> Dict[str, float]:
    """Return PSNR, SSIM, SAM and spatial-alignment diagnostics."""
    alignment = spatial_alignment_check(prediction, target)
    return {
        "psnr_db": psnr(prediction, target, data_range, valid_mask),
        "ssim": ssim(prediction, target, data_range, ssim_window, valid_mask),
        "sam_degrees": spectral_angle_error(prediction, target, valid_mask),
        "alignment_ncc": alignment["ncc"],
        "alignment_dy": float(alignment["best_shift"][0]),
        "alignment_dx": float(alignment["best_shift"][1]),
    }

def smoke_test():
    rng = np.random.default_rng(7)
    target = rng.random((4, 32, 32))
    assert math.isinf(psnr(target, target))
    assert abs(ssim(target, target) - 1.0) < 1e-10
    # arccos's derivative is very steep near cos=1, so float64 rounding in
    # the norm/dot-product computation shows up amplified in the angle
    # even for genuinely identical vectors (~1e-7 degrees here) — 1e-4 deg
    # is still a meaningful "basically zero" check without chasing float
    # noise. Found and fixed by sonnet5, T8 turn, while verifying T7 (see
    # build-status.md) — same class of issue as eval_downstream.py's own
    # SAM-adjacent metric hit during its own testing.
    assert abs(spectral_angle_error(target, target)) < 1e-4
    shifted = np.roll(target, shift=(2, -1), axis=(-2, -1))
    report = spatial_alignment_check(shifted, target, max_shift=4)
    assert report["best_shift"] == (-2, 1), report
    assert report["ncc"] > 0.95, report
    noisy = target + 0.01 * rng.standard_normal(target.shape)
    assert psnr(noisy, target) > 35.0
    assert 0.0 < ssim(noisy, target) < 1.0
    print("metrics smoke test passed")

if __name__ == "__main__":
    smoke_test()
