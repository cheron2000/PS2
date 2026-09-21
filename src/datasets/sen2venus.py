"""
src/datasets/sen2venus.py — SEN2Vénus secondary validation loader (T6).

Supported on-disk convention:
    root/
      lr/<id>.npy  -- Sentinel-2 low-resolution patch, shape (C, H, W)
      hr/<id>.npy  -- Vénus high-resolution patch, shape (C, 2H, 2W)

The public SEN2Vénus release layout is not verified in this sandbox. This loader
therefore deliberately uses the same plain paired-array interchange convention as
T5, so a small dataset-specific converter can be added later without changing the
validation interface.

SEN2Vénus is a secondary 5 m validation route: 10 m Sentinel-2 input to 5 m
reference imagery is a 2x spatial scale. Pair alignment uses the verified NCC
search and crop utilities from sen2naip.py, while the default normalization is
reflectance-compatible with the rest of the pipeline.
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
    _DatasetBase = object

from src.preprocessing import normalize_bands
from src.datasets.sen2naip import (
    estimate_pair_shift,
    apply_shift_and_crop,
)
from src.datasets.provenance import read_manifest

SCALE_FACTOR = 2  # 10 m Sentinel-2 -> 5 m SEN2Vénus validation target


class SEN2VenusDataset(_DatasetBase):
    """Paired Sentinel-2/SEN2Vénus validation patches.

    Args:
        root: directory containing matching ``lr/*.npy`` and ``hr/*.npy`` files.
        scale: LR-to-HR factor; defaults to 2 for the 5 m validation route.
        normalize_method: method passed to ``normalize_bands``.
        max_shift: HR-pixel radius for integer NCC co-registration search.
        min_ncc_score: pairs below this score are excluded from the dataset.
        manifest_path: optional sealed provenance manifest whose declared pairs
            replace glob discovery and are checksum-verified before loading.
    """

    def __init__(self, root: str, scale: int = SCALE_FACTOR,
                 normalize_method: str = "reflectance", max_shift: int = 4,
                 min_ncc_score: float = 0.1, manifest_path: str | None = None):
        if scale < 1 or int(scale) != scale:
            raise ValueError(f"scale must be a positive integer, got {scale}")
        self.root = root
        self.scale = int(scale)
        self.normalize_method = normalize_method
        self.max_shift = max_shift
        self.min_ncc_score = min_ncc_score

        if manifest_path is not None:
            manifest = read_manifest(manifest_path, verify_files=True)
            root = os.path.dirname(os.path.abspath(manifest_path))
            manifest_pairs = [
                (
                    str(pair["id"]),
                    os.path.join(root, pair["lr_path"]),
                    os.path.join(root, pair["hr_path"]),
                    os.path.join(root, pair["lr_mask_path"]) if pair.get("lr_mask_path") else None,
                    os.path.join(root, pair["hr_mask_path"]) if pair.get("hr_mask_path") else None,
                )
                for pair in manifest["pairs"]
            ]
        else:
            lr_files = sorted(glob.glob(os.path.join(root, "lr", "*.npy")))
            manifest_pairs = [
                (os.path.splitext(os.path.basename(path))[0], path,
                 os.path.join(root, "hr", os.path.splitext(os.path.basename(path))[0] + ".npy"), None, None)
                for path in lr_files
            ]
        if not manifest_pairs:
            raise FileNotFoundError(
                f"no .npy files found under {os.path.join(root, 'lr')} — "
                "expected the documented lr/<id>.npy, hr/<id>.npy layout"
            )

        self.pairs = []
        self.dropped_pairs = []
        for file_id, lr_path, hr_path, lr_mask_path, hr_mask_path in manifest_pairs:
            if not os.path.exists(hr_path):
                self.dropped_pairs.append((file_id, "no matching HR file"))
                continue

            lr = np.load(lr_path)
            hr = np.load(hr_path)
            if lr.ndim != 3 or hr.ndim != 3:
                self.dropped_pairs.append((file_id, "LR and HR must have shape (C,H,W)"))
                continue
            expected = (lr.shape[0], lr.shape[1] * self.scale, lr.shape[2] * self.scale)
            if hr.shape != expected:
                self.dropped_pairs.append(
                    (file_id, f"HR shape {hr.shape} != expected {expected} for scale={self.scale}")
                )
                continue

            dy, dx, score = estimate_pair_shift(
                lr, hr, scale=self.scale, max_shift=self.max_shift
            )
            if score < self.min_ncc_score:
                self.dropped_pairs.append(
                    (file_id, f"NCC score {score:.3f} below threshold {self.min_ncc_score}")
                )
                continue
            self.pairs.append((lr_path, hr_path, dy, dx, score, lr_mask_path, hr_mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, dy, dx, score, lr_mask_path, hr_mask_path = self.pairs[idx]
        lr = np.load(lr_path)
        hr = np.load(hr_path)
        lr_aligned, hr_aligned = apply_shift_and_crop(
            lr, hr, dy, dx, scale=self.scale
        )
        lr_mask_aligned = None
        valid_mask = None
        if lr_mask_path or hr_mask_path:
            lr_mask = np.ones(lr.shape[1:], dtype=bool) if lr_mask_path is None else np.asarray(np.load(lr_mask_path), dtype=bool)
            hr_mask = np.ones(hr.shape[1:], dtype=bool) if hr_mask_path is None else np.asarray(np.load(hr_mask_path), dtype=bool)
            if lr_mask.shape != lr.shape[1:] or hr_mask.shape != hr.shape[1:]:
                raise ValueError("LR/HR validity masks must match source spatial shapes")
            lr_mask_aligned, hr_mask_aligned = apply_shift_and_crop(
                lr_mask[None, ...].astype(np.float32),
                hr_mask[None, ...].astype(np.float32),
                dy, dx, scale=self.scale,
            )
            lr_mask_aligned = lr_mask_aligned[0].astype(bool)
            hr_mask_aligned = hr_mask_aligned[0].astype(bool)
            valid_mask = hr_mask_aligned & np.repeat(
                np.repeat(lr_mask_aligned, self.scale, axis=0),
                self.scale, axis=1,
            )
        lr_norm, lr_stats = normalize_bands(
            lr_aligned, method=self.normalize_method, valid_mask=lr_mask_aligned
        )
        hr_norm, hr_stats = normalize_bands(
            hr_aligned, method=self.normalize_method, valid_mask=valid_mask
        )
        sample = {
            "lr": lr_norm.astype(np.float32),
            "hr": hr_norm.astype(np.float32),
            "lr_stats": lr_stats,
            "hr_stats": hr_stats,
            "alignment_shift": (dy, dx),
            "alignment_score": score,
            "scale": self.scale,
            **({"valid_mask": valid_mask} if valid_mask is not None else {}),
        }
        if _HAS_TORCH:
            sample["lr"] = torch.from_numpy(sample["lr"])
            sample["hr"] = torch.from_numpy(sample["hr"])
        return sample


# British/diacritic-friendly aliases for callers using the dataset's published name.
SEN2VENUSDataset = SEN2VenusDataset
SEN2VénusDataset = SEN2VenusDataset


if __name__ == "__main__":
    print("SEN2VenusDataset ready; expected root layout is lr/<id>.npy + hr/<id>.npy")
