"""
src/datasets/sen2naip.py — SEN2NAIP / SEN2NAIPv2 dataset loader (T5).

Loads paired low-resolution Sentinel-2 / high-resolution NAIP patches for the
project's primary training/validation benchmark per solution-draft.md
("Primary 4x target... SEN2NAIP's real cross-sensor subset").

ON-DISK FORMAT SUPPORTED HERE (documented convention, not the official release
format — see KNOWN GAP below):
    root/
      lr/<id>.npy   — (C, H, W) low-res Sentinel-2 patch, raw reflectance-scale DN
      hr/<id>.npy   — (C, H*scale, W*scale) high-res NAIP patch, matching <id>

KNOWN GAP: the official HuggingFace release (tacofoundation/SEN2NAIPv2) ships as
a TACO-format columnar archive, not plain paired .npy files. Reading that
directly needs the `tacoreader`/`tacotoolbox` package. This sandbox has no
network access to huggingface.co to fetch a real sample and verify against it,
so this loader targets the plain-file convention above instead, with a
converter from TACO -> this layout left as a natural follow-up task (not built
here — flag it in tasks.md if picked up).

THE REAL CONTRIBUTION OF THIS FILE: pair co-registration / QC. solution-draft.md
flags this explicitly as a real risk, not a footnote — cross-sensor pairs (
different sensor, different pass) are only weakly aligned, and training on them
with a naive per-pixel loss teaches the model to reproduce misregistration
noise instead of real detail. `estimate_pair_shift()` below does a small
integer-pixel-shift search maximizing normalized cross-correlation between the
upsampled LR and the HR image, and `SEN2NAIPDataset` uses it to either correct
each pair's alignment or drop pairs that don't align well enough to trust.

EXECUTION STATUS: written AND executed by claude1 in this sandbox — numpy and
scipy are both available here (unlike T1's torch situation). See
`test_sen2naip.py` for a synthetic end-to-end test: generates a known shift,
confirms `estimate_pair_shift` recovers it, confirms the Dataset applies the
correction and rejects a pair whose HR is unrelated noise (score too low). All
tests pass in this sandbox. NOT tested against a real SEN2NAIP sample (per the
KNOWN GAP above) — the file-format assumptions specifically need checking once
someone has real data access.
"""

from __future__ import annotations

import glob
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
    _DatasetBase = _TorchDataset
except Exception:
    _HAS_TORCH = False
    _DatasetBase = object  # dataset still usable (returns numpy) without torch

from src.preprocessing import normalize_bands
from src.datasets.band_schema import validate_bands, SENTINEL2_L2A_4BAND, NAIP_4BAND, BandSchemaError

SCALE_FACTOR = 4  # 10m Sentinel-2 -> 2.5m NAIP, SEN2NAIP's real cross-sensor task


def upsample_nearest(x: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upsample, (C, H, W) -> (C, H*scale, W*scale). Used only for
    the alignment search below, not as the model's own upsampling method."""
    return np.repeat(np.repeat(x, scale, axis=1), scale, axis=2)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """NCC of two same-shape arrays, flattened. Range [-1, 1], higher = more aligned.
    Returns 0.0 for a degenerate (zero-variance) input rather than raising, since a
    flat patch (e.g. all-water) is a real possibility, not necessarily a bug."""
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def estimate_pair_shift(lr: np.ndarray, hr: np.ndarray, scale: int = SCALE_FACTOR,
                         max_shift: int = 4):
    """Search integer HR-pixel shifts (dy, dx) in [-max_shift, max_shift] maximizing
    NCC between upsampled-LR and HR, using the first band for speed (bands of the
    same scene are geometrically aligned to each other, so one band is enough to
    find the shift).

    Returns (best_dy, best_dx, best_score). Caller decides what to do with a low
    score (see SEN2NAIPDataset's min_ncc_score) — this function just measures.
    """
    lr_up = upsample_nearest(lr, scale)
    band_lr = lr_up[0]
    band_hr = hr[0]
    if band_lr.shape != band_hr.shape:
        raise ValueError(
            f"upsampled LR shape {band_lr.shape} != HR shape {band_hr.shape} — "
            f"check patch sizes match scale={scale}"
        )
    H, W = band_hr.shape

    best_dy, best_dx, best_score = 0, 0, -2.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0_hr, y1_hr = max(0, dy), H + min(0, dy)
            x0_hr, x1_hr = max(0, dx), W + min(0, dx)
            y0_lr, y1_lr = max(0, -dy), H + min(0, -dy)
            x0_lr, x1_lr = max(0, -dx), W + min(0, -dx)
            hr_crop = band_hr[y0_hr:y1_hr, x0_hr:x1_hr]
            lr_crop = band_lr[y0_lr:y1_lr, x0_lr:x1_lr]
            # require a reasonably large overlap so a tiny sliver of pixels can't
            # win on a fluke correlation
            if hr_crop.size < 0.5 * band_hr.size:
                continue
            score = normalized_cross_correlation(lr_crop, hr_crop)
            if score > best_score:
                best_dy, best_dx, best_score = dy, dx, score
    return best_dy, best_dx, best_score


def apply_shift_and_crop(lr: np.ndarray, hr: np.ndarray, dy: int, dx: int,
                          scale: int = SCALE_FACTOR):
    """Crop lr and hr to their mutually-aligned overlapping region given a detected
    HR-space shift (dy, dx). Returns (lr_cropped, hr_cropped) with hr_cropped.shape
    == lr_cropped.shape * scale exactly, ready to hand to the model."""
    C_hr, H, W = hr.shape
    y0_hr, y1_hr = max(0, dy), H + min(0, dy)
    x0_hr, x1_hr = max(0, dx), W + min(0, dx)
    hr_cropped = hr[:, y0_hr:y1_hr, x0_hr:x1_hr]

    # convert the same shift into LR-pixel units for cropping the LR side
    dy_lr, dx_lr = dy // scale, dx // scale
    C_lr, H_lr, W_lr = lr.shape
    y0_lr, y1_lr = max(0, -dy_lr), H_lr + min(0, -dy_lr)
    x0_lr, x1_lr = max(0, -dx_lr), W_lr + min(0, -dx_lr)
    lr_cropped = lr[:, y0_lr:y1_lr, x0_lr:x1_lr]

    # after independent rounding, the two crops can differ by a pixel or two in
    # HR-equivalent size -- trim both to the largest shape that divides evenly
    target_h_lr = min(lr_cropped.shape[1], hr_cropped.shape[1] // scale)
    target_w_lr = min(lr_cropped.shape[2], hr_cropped.shape[2] // scale)
    lr_cropped = lr_cropped[:, :target_h_lr, :target_w_lr]
    hr_cropped = hr_cropped[:, :target_h_lr * scale, :target_w_lr * scale]
    return lr_cropped, hr_cropped


class SEN2NAIPDataset(_DatasetBase):
    """PyTorch Dataset (or plain-numpy iterable if torch isn't installed — see
    _HAS_TORCH) over paired LR/HR patches. See module docstring for the expected
    on-disk layout and its known gap vs. the official TACO-format release.

    Args:
        root: directory containing lr/ and hr/ subfolders of matching .npy files
        scale: LR->HR scale factor (default 4, matching SEN2NAIP's real subset)
        normalize_method: passed through to preprocessing.normalize_bands (T4)
        max_shift: search radius (HR pixels) for co-registration
        min_ncc_score: pairs whose best alignment score falls below this are
            dropped at load time (logged, not silently — see `dropped_pairs`)
    """

    def __init__(self, root: str, scale: int = SCALE_FACTOR,
                 normalize_method: str = "reflectance", max_shift: int = 4,
                 min_ncc_score: float = 0.1):
        self.root = root
        if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale <= 0:
            raise ValueError(f"scale must be a positive integer, got {scale!r}")
        self.scale = int(scale)
        self.normalize_method = normalize_method
        self.max_shift = max_shift
        self.min_ncc_score = min_ncc_score

        lr_files = sorted(glob.glob(os.path.join(root, "lr", "*.npy")))
        if not lr_files:
            raise FileNotFoundError(
                f"no .npy files found under {os.path.join(root, 'lr')} — see this "
                f"file's module docstring for the expected on-disk layout"
            )

        self.pairs = []       # list of (lr_path, hr_path, dy, dx, ncc_score)
        self.dropped_pairs = []  # list of (id, reason) for pairs excluded at load time

        for lr_path in lr_files:
            file_id = os.path.splitext(os.path.basename(lr_path))[0]
            hr_path = os.path.join(root, "hr", file_id + ".npy")
            if not os.path.exists(hr_path):
                self.dropped_pairs.append((file_id, "no matching HR file"))
                continue

            try:
                lr = np.load(lr_path)
                hr = np.load(hr_path)
            except (OSError, ValueError) as exc:
                self.dropped_pairs.append((file_id, f"failed to load array: {exc}"))
                continue
            if lr.ndim != 3 or hr.ndim != 3:
                self.dropped_pairs.append((file_id, f"LR and HR must have shape (C,H,W), got LR {lr.shape}, HR {hr.shape}"))
                continue
            if lr.shape[0] <= 0 or hr.shape[0] <= 0 or lr.shape[1] <= 0 or lr.shape[2] <= 0:
                self.dropped_pairs.append((file_id, f"LR and HR must have positive dimensions, got LR {lr.shape}, HR {hr.shape}"))
                continue
            if lr.shape[0] != hr.shape[0]:
                self.dropped_pairs.append((file_id, f"LR/HR channel count mismatch: {lr.shape[0]} vs {hr.shape[0]}"))
                continue
            if not np.isfinite(lr).all() or not np.isfinite(hr).all():
                self.dropped_pairs.append((file_id, "LR and HR must contain only finite values"))
                continue
            # T20: band schema validation — catches the audit's named
            # concern (a repeated/truncated channel from a band-order bug)
            # before it silently trains a model on malformed input. Uses
            # allow_extra_bands so a 4-band pair validates against the
            # project's standard R/G/B/NIR schema without requiring an
            # exact 4-band count elsewhere in the pipeline to change.
            try:
                validate_bands(lr, SENTINEL2_L2A_4BAND)
                validate_bands(hr, NAIP_4BAND)
            except BandSchemaError as exc:
                self.dropped_pairs.append((file_id, f"band schema validation failed: {exc}"))
                continue
            expected_hr_shape = (lr.shape[0], lr.shape[1] * self.scale, lr.shape[2] * self.scale)
            if hr.shape != expected_hr_shape:
                self.dropped_pairs.append(
                    (file_id, f"HR shape {hr.shape} != expected {expected_hr_shape} for scale={scale}")
                )
                continue

            dy, dx, score = estimate_pair_shift(lr, hr, scale=scale, max_shift=max_shift)
            if score < min_ncc_score:
                self.dropped_pairs.append((file_id, f"NCC score {score:.3f} below threshold {min_ncc_score}"))
                continue

            self.pairs.append((lr_path, hr_path, dy, dx, score))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, dy, dx, score = self.pairs[idx]
        lr = np.load(lr_path)
        hr = np.load(hr_path)

        lr_aligned, hr_aligned = apply_shift_and_crop(lr, hr, dy, dx, scale=self.scale)
        # BUG FIX (2026-09-20, agent4, flagged by external "Wide Research"
        # gap analysis, verified independently before fixing — see
        # preprocessing.py's normalize_bands docstring for the full story):
        # this used to call normalize_bands(..., method=self.normalize_method)
        # identically for both lr and hr. For method="reflectance" (the
        # default), that meant BOTH sides were divided by 10000 -- correct
        # for Sentinel-2 (lr), but silently wrong for NAIP (hr), whose real
        # 0-255 8-bit DN range would be crushed into [0, 0.0255] instead of
        # spanning [0, 1]. Confirmed by direct inspection: band_schema.py's
        # NAIP_4BAND already correctly declares scale=1/255, it just was
        # never actually applied here. Fixed by passing the correct
        # reflectance_divisor per source explicitly.
        if self.normalize_method == "reflectance":
            lr_norm, lr_stats = normalize_bands(lr_aligned, method="reflectance", reflectance_divisor=10000.0)
            hr_norm, hr_stats = normalize_bands(hr_aligned, method="reflectance", reflectance_divisor=255.0)
        else:
            # percentile/zscore adapt to each array's own statistics, so
            # applying the same method to both sides has no unit-mismatch
            # risk the way a fixed reflectance divisor does — see
            # preprocessing.py's docstring.
            lr_norm, lr_stats = normalize_bands(lr_aligned, method=self.normalize_method)
            hr_norm, hr_stats = normalize_bands(hr_aligned, method=self.normalize_method)

        sample = {
            "lr": lr_norm.astype(np.float32),
            "hr": hr_norm.astype(np.float32),
            "lr_stats": lr_stats,
            "hr_stats": hr_stats,
            "alignment_shift": (dy, dx),
            "alignment_score": score,
        }
        if _HAS_TORCH:
            sample["lr"] = torch.from_numpy(sample["lr"])
            sample["hr"] = torch.from_numpy(sample["hr"])
        return sample
