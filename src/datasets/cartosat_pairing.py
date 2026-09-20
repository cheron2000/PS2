"""
Cartosat↔Sentinel-2 pairing utilities for India-specific validation (T10).

This module is intentionally array-first. It accepts pre-extracted, coarsely
resampled Sentinel-2 and Cartosat patches as ``(C,H,W)`` NumPy arrays, then:

1. validates the expected integer prototype scale;
2. estimates a small HR-space integer shift with NCC;
3. crops both arrays to a common aligned footprint;
4. applies per-band affine reflectance harmonization to the Cartosat patch;
5. returns a validity mask and provenance metadata for downstream evaluation.

The Cartosat portal download, orthorectification, band mapping, and CRS reprojection
remain acquisition-side responsibilities. This module does not silently invent those
steps; callers must provide arrays in a common projected grid and record the source
metadata. For the prototype, Cartosat can be resampled to a 2.5 m reference grid so
the default integer scale is 4 (10 m Sentinel-2 -> 2.5 m reference).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
    _DatasetBase = _TorchDataset
except Exception:
    _HAS_TORCH = False
    _DatasetBase = object

from src.datasets.sen2naip import (
    apply_shift_and_crop,
    estimate_pair_shift,
    upsample_nearest,
)
from src.datasets.geospatial import (
    GridContractError,
    GridSpec,
    combine_valid_masks,
    crop_pair_with_grids,
    mask_from_nodata,
    propagate_valid_mask,
)

SCALE_FACTOR = 4


class PairRejected(ValueError):
    """Raised when a Cartosat pair fails structural or alignment quality checks."""


def _validate_chw(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise PairRejected(f"{name} must have shape (C,H,W), got {array.shape}")
    if min(array.shape) <= 0:
        raise PairRejected(f"{name} has an empty dimension: {array.shape}")
    if not np.isfinite(array).all():
        raise PairRejected(f"{name} contains non-finite values")
    return array.astype(np.float32, copy=False)


def _match_channels(lr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """Return HR with the LR band count using explicit repeat/truncate semantics."""
    if hr.shape[0] == lr.shape[0]:
        return hr
    if hr.shape[0] == 1:
        return np.repeat(hr, lr.shape[0], axis=0)
    if hr.shape[0] > lr.shape[0]:
        return hr[: lr.shape[0]]
    raise PairRejected(
        f"Cartosat has {hr.shape[0]} bands but Sentinel-2 has {lr.shape[0]}; "
        "provide a band-mapping step before pairing"
    )


def harmonize_reflectance(
    lr: np.ndarray,
    hr: np.ndarray,
    scale: int = SCALE_FACTOR,
    clip_range: tuple[float, float] = (0.0, 1.0),
    min_std: float = 1e-6,
):
    """Affine-match HR band mean/std to nearest-neighbor LR on the overlap.

    This is a conservative prototype harmonizer, not a substitute for a calibrated
    sensor-response model. It returns the transformed HR array plus per-band gains
    and offsets so the operation is auditable.
    """
    lr = _validate_chw(lr, "lr")
    hr = _validate_chw(hr, "hr")
    if scale < 1 or int(scale) != scale:
        raise PairRejected(f"scale must be a positive integer, got {scale}")
    scale = int(scale)
    if hr.shape[1:] != (lr.shape[1] * scale, lr.shape[2] * scale):
        raise PairRejected(f"HR shape {hr.shape} does not match scale={scale} for LR {lr.shape}")
    hr = _match_channels(lr, hr)
    lr_up = upsample_nearest(lr, scale)
    gains = np.ones(lr.shape[0], dtype=np.float32)
    offsets = np.zeros(lr.shape[0], dtype=np.float32)
    result = np.empty_like(hr, dtype=np.float32)
    for band in range(lr.shape[0]):
        lr_band = lr_up[band]
        hr_band = hr[band]
        lr_mean, hr_mean = float(lr_band.mean()), float(hr_band.mean())
        lr_std, hr_std = float(lr_band.std()), float(hr_band.std())
        gain = lr_std / max(hr_std, min_std)
        offset = lr_mean - gain * hr_mean
        gains[band] = gain
        offsets[band] = offset
        result[band] = np.clip(gain * hr_band + offset, *clip_range)
    return result, gains, offsets


def _harmonize_with_scale(lr, hr, scale, clip_range, min_std):
    """Scale-generic internal version used by prepare_pair."""
    lr_up = upsample_nearest(lr, scale)
    hr = _match_channels(lr, hr)
    gains = np.ones(lr.shape[0], dtype=np.float32)
    offsets = np.zeros(lr.shape[0], dtype=np.float32)
    result = np.empty_like(hr, dtype=np.float32)
    for band in range(lr.shape[0]):
        lm, hm = float(lr_up[band].mean()), float(hr[band].mean())
        ls, hs = float(lr_up[band].std()), float(hr[band].std())
        gains[band] = ls / max(hs, min_std)
        offsets[band] = lm - gains[band] * hm
        result[band] = np.clip(gains[band] * hr[band] + offsets[band], *clip_range)
    return result, gains, offsets


@dataclass
class CartosatPair:
    lr: np.ndarray
    hr: np.ndarray
    valid_mask: np.ndarray
    alignment_shift: tuple[int, int]
    alignment_score: float
    harmonization_gain: np.ndarray
    harmonization_offset: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def prepare_pair(
    lr: np.ndarray,
    cartosat: np.ndarray,
    *,
    scale: int = SCALE_FACTOR,
    max_shift: int = 8,
    min_ncc_score: float = 0.1,
    clip_range: tuple[float, float] = (0.0, 1.0),
    metadata: Optional[Mapping[str, Any]] = None,
    lr_valid_mask: Optional[np.ndarray] = None,
    hr_valid_mask: Optional[np.ndarray] = None,
) -> CartosatPair:
    """Align and harmonize one pair, optionally enforcing a CRS/grid contract.

    If metadata contains ``lr_grid`` and ``hr_grid`` objects, both are parsed as
    :class:`GridSpec` and shape-only pairing is rejected unless CRS, pixel size,
    affine phase, and the registration shift are all representable on the LR
    grid. Optional masks (or ``lr_nodata``/``hr_nodata`` metadata) are cropped
    with the data and intersected into the returned HR-resolution validity mask.
    """
    lr = _validate_chw(lr, "lr")
    cartosat = _validate_chw(cartosat, "cartosat")
    if scale < 1 or int(scale) != scale:
        raise PairRejected(f"scale must be a positive integer, got {scale}")
    scale = int(scale)
    expected = (cartosat.shape[0], lr.shape[1] * scale, lr.shape[2] * scale)
    if cartosat.shape != expected:
        raise PairRejected(f"Cartosat shape {cartosat.shape} != expected {expected}")

    record = dict(metadata or {})
    lr_grid = hr_grid = None
    if "lr_grid" in record or "hr_grid" in record:
        if "lr_grid" not in record or "hr_grid" not in record:
            raise PairRejected("both lr_grid and hr_grid metadata are required for CRS-aware pairing")
        try:
            lr_grid = GridSpec.from_mapping(record["lr_grid"], name="lr_grid")
            hr_grid = GridSpec.from_mapping(record["hr_grid"], name="hr_grid")
        except GridContractError as exc:
            raise PairRejected(str(exc)) from exc
        if (lr_grid.height, lr_grid.width) != lr.shape[1:] or (hr_grid.height, hr_grid.width) != cartosat.shape[1:]:
            raise PairRejected("grid dimensions do not match the supplied arrays")

    dy, dx, score = estimate_pair_shift(lr, cartosat, scale=scale, max_shift=max_shift)
    if score < min_ncc_score:
        raise PairRejected(f"NCC score {score:.3f} below threshold {min_ncc_score}")
    try:
        if lr_grid is not None:
            lr_aligned, hr_aligned, lr_grid_out, hr_grid_out = crop_pair_with_grids(
                lr, cartosat, dy, dx, scale, lr_grid, hr_grid
            )
        else:
            lr_aligned, hr_aligned = apply_shift_and_crop(lr, cartosat, dy, dx, scale=scale)
            lr_grid_out = hr_grid_out = None
    except GridContractError as exc:
        raise PairRejected(str(exc)) from exc

    lr_mask = mask_from_nodata(lr, record.get("lr_nodata")) if lr_valid_mask is None else propagate_valid_mask(lr_valid_mask, expected_shape=lr.shape[1:], name="lr_valid_mask")
    hr_mask = mask_from_nodata(cartosat, record.get("hr_nodata")) if hr_valid_mask is None else propagate_valid_mask(hr_valid_mask, expected_shape=cartosat.shape[1:], name="hr_valid_mask")
    if lr_grid is not None:
        lr_mask, hr_mask, _, _ = crop_pair_with_grids(
            lr_mask[None].astype(np.float32), hr_mask[None].astype(np.float32), dy, dx, scale, lr_grid, hr_grid
        )
        lr_mask, hr_mask = lr_mask[0].astype(bool), hr_mask[0].astype(bool)
    else:
        lr_mask, hr_mask = apply_shift_and_crop(lr_mask[None].astype(np.float32), hr_mask[None].astype(np.float32), dy, dx, scale=scale)
        lr_mask, hr_mask = lr_mask[0].astype(bool), hr_mask[0].astype(bool)
    hr_harmonized, gains, offsets = _harmonize_with_scale(
        lr_aligned, hr_aligned, scale, clip_range, min_std=1e-6
    )
    lr_valid_hr = np.repeat(np.repeat(lr_mask, scale, axis=0), scale, axis=1)
    valid_mask = combine_valid_masks(lr_valid_hr, hr_mask, np.isfinite(hr_harmonized).all(axis=0))
    if lr_grid_out is not None:
        record["lr_grid"] = lr_grid_out.to_dict()
        record["hr_grid"] = hr_grid_out.to_dict()
    record.update({"scale": scale, "source": "Cartosat", "reference_gsd_m": 10.0 / scale})
    return CartosatPair(
        lr=lr_aligned.astype(np.float32),
        hr=hr_harmonized.astype(np.float32),
        valid_mask=valid_mask.astype(np.float32),
        alignment_shift=(int(dy), int(dx)),
        alignment_score=float(score),
        harmonization_gain=gains,
        harmonization_offset=offsets,
        metadata=record,
    )


class CartosatPairingDataset(_DatasetBase):
    """Load ``lr/*.npy`` + ``hr/*.npy`` pairs and apply ``prepare_pair`` at init."""

    def __init__(self, root: str, *, scale: int = SCALE_FACTOR, max_shift: int = 8,
                 min_ncc_score: float = 0.1):
        self.root = root
        self.scale = scale
        self.pairs = []
        self.dropped_pairs = []
        for lr_path in sorted(glob.glob(os.path.join(root, "lr", "*.npy"))):
            file_id = os.path.splitext(os.path.basename(lr_path))[0]
            hr_path = os.path.join(root, "hr", file_id + ".npy")
            if not os.path.exists(hr_path):
                self.dropped_pairs.append((file_id, "no Cartosat file"))
                continue
            metadata = {}
            metadata_path = os.path.join(root, "metadata", file_id + ".json")
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
            try:
                pair = prepare_pair(
                    np.load(lr_path), np.load(hr_path), scale=scale,
                    max_shift=max_shift, min_ncc_score=min_ncc_score,
                    metadata=metadata,
                )
            except PairRejected as exc:
                self.dropped_pairs.append((file_id, str(exc)))
                continue
            self.pairs.append(pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample = {
            "lr": pair.lr,
            "hr": pair.hr,
            "valid_mask": pair.valid_mask,
            "alignment_shift": pair.alignment_shift,
            "alignment_score": pair.alignment_score,
            "harmonization_gain": pair.harmonization_gain,
            "harmonization_offset": pair.harmonization_offset,
            "metadata": pair.metadata,
        }
        if _HAS_TORCH:
            for key in ("lr", "hr", "valid_mask"):
                sample[key] = torch.from_numpy(sample[key])
        return sample
