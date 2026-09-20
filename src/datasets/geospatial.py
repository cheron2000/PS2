"""Grid-aware pairing primitives for T16.

The prototype loaders historically paired arrays by shape alone.  This module
adds a dependency-light contract that can be used by raster adapters without
requiring rasterio at import time: validated CRS/affine metadata, explicit
scale-aligned registration policy, transform-preserving crops, and validity
mask propagation.

Transforms use the common six-value affine convention
(a, b, c, d, e, f): x=a*col+b*row+c; y=d*col+e*row+f.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


class GridContractError(ValueError):
    """Raised when two source grids cannot be safely paired."""


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    nodata: float | int | None = None

    def __post_init__(self):
        if not isinstance(self.crs, str) or not self.crs.strip():
            raise GridContractError("crs must be a non-empty string")
        if len(self.transform) != 6 or not all(isfinite(float(v)) for v in self.transform):
            raise GridContractError("transform must contain six finite numbers")
        a, b, _c, d, e, _f = [float(v) for v in self.transform]
        if abs(a * e - b * d) <= 1e-15:
            raise GridContractError("transform must be invertible")
        if isinstance(self.width, bool) or isinstance(self.height, bool) or self.width <= 0 or self.height <= 0:
            raise GridContractError("width and height must be positive integers")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str = "grid") -> "GridSpec":
        if not isinstance(value, Mapping):
            raise GridContractError(f"{name} must be an object")
        missing = [key for key in ("crs", "transform", "width", "height") if key not in value]
        if missing:
            raise GridContractError(f"{name} missing required fields: {', '.join(missing)}")
        return cls(
            crs=str(value["crs"]),
            transform=tuple(float(v) for v in value["transform"]),
            width=int(value["width"]),
            height=int(value["height"]),
            nodata=value.get("nodata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"crs": self.crs, "transform": list(self.transform), "width": self.width, "height": self.height, "nodata": self.nodata}

    def translated(self, col: int, row: int, *, width: int | None = None, height: int | None = None) -> "GridSpec":
        """Return the grid after cropping from the top-left pixel offset."""
        a, b, c, d, e, f = self.transform
        out_width = self.width - col if width is None else int(width)
        out_height = self.height - row if height is None else int(height)
        return GridSpec(self.crs, (a, b, a * col + b * row + c, d, e, d * col + e * row + f), out_width, out_height, self.nodata)


def validate_grid_pair(lr: GridSpec, hr: GridSpec, scale: int, *, tolerance: float = 1e-6) -> None:
    """Validate that HR is a scale-refined grid of LR, not merely same-shaped arrays."""
    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or int(scale) <= 0:
        raise GridContractError(f"scale must be a positive integer, got {scale!r}")
    scale = int(scale)
    if lr.crs != hr.crs:
        raise GridContractError(f"CRS mismatch: LR={lr.crs!r}, HR={hr.crs!r}")
    la, lb, lc, ld, le, lf = lr.transform
    ha, hb, hc, hd, he, hf = hr.transform
    if any(abs(v) > tolerance for v in (lb, ld, hb, hd)):
        raise GridContractError("rotated/sheared grids are not supported by the array cropper; reproject first")
    if not np.isclose(abs(ha), abs(la) / scale, rtol=0, atol=tolerance) or not np.isclose(abs(he), abs(le) / scale, rtol=0, atol=tolerance):
        raise GridContractError("HR pixel size is not LR pixel size divided by scale")
    # The origins may differ, but only by an integral HR pixel; otherwise a
    # shape-preserving crop cannot represent the true geographic phase.
    col_offset = (lc - hc) / ha
    row_offset = (lf - hf) / he
    if not np.isclose(col_offset, round(col_offset), atol=tolerance) or not np.isclose(row_offset, round(row_offset), atol=tolerance):
        raise GridContractError("LR/HR origins are not aligned to an integer HR pixel phase")


def require_scale_aligned_shift(dy: int, dx: int, scale: int) -> tuple[int, int]:
    """Convert an HR registration shift to LR crop units without floor-division drift."""
    if int(dy) != dy or int(dx) != dx or int(scale) != scale or int(scale) <= 0:
        raise GridContractError("dy, dx, and scale must be integers with scale > 0")
    dy, dx, scale = int(dy), int(dx), int(scale)
    if dy % scale or dx % scale:
        raise GridContractError(
            f"registration shift ({dy}, {dx}) is not aligned to the {scale}x LR grid; "
            "reject it or use a subpixel/geospatial resampler"
        )
    return dy // scale, dx // scale


def crop_pair_with_grids(lr: np.ndarray, hr: np.ndarray, dy: int, dx: int, scale: int, lr_grid: GridSpec, hr_grid: GridSpec):
    """Crop an aligned pair and return LR/HR arrays plus updated GridSpecs."""
    validate_grid_pair(lr_grid, hr_grid, scale)
    dy_lr, dx_lr = require_scale_aligned_shift(dy, dx, scale)
    if lr.ndim != 3 or hr.ndim != 3 or hr.shape[1:] != (lr.shape[1] * scale, lr.shape[2] * scale):
        raise GridContractError("arrays must be CHW and HR spatial shape must equal LR shape times scale")
    y0h, y1h = max(0, dy), hr.shape[1] + min(0, dy)
    x0h, x1h = max(0, dx), hr.shape[2] + min(0, dx)
    y0l, y1l = max(0, -dy_lr), lr.shape[1] + min(0, -dy_lr)
    x0l, x1l = max(0, -dx_lr), lr.shape[2] + min(0, -dx_lr)
    lr_out = lr[:, y0l:y1l, x0l:x1l]
    hr_out = hr[:, y0h:y1h, x0h:x1h]
    h = min(lr_out.shape[1], hr_out.shape[1] // scale)
    w = min(lr_out.shape[2], hr_out.shape[2] // scale)
    lr_out, hr_out = lr_out[:, :h, :w], hr_out[:, :h * scale, :w * scale]
    return lr_out, hr_out, lr_grid.translated(x0l, y0l, width=w, height=h), hr_grid.translated(x0h, y0h, width=w * scale, height=h * scale)


def propagate_valid_mask(mask: np.ndarray, *, expected_shape: tuple[int, int], name: str = "mask") -> np.ndarray:
    """Validate and normalize a 2-D mask to boolean True=valid semantics."""
    mask = np.asarray(mask)
    if mask.shape != expected_shape:
        raise GridContractError(f"{name} shape {mask.shape} != expected {expected_shape}")
    if mask.dtype.kind == "f" and not np.isfinite(mask).all():
        raise GridContractError(f"{name} contains non-finite values")
    return mask.astype(bool, copy=False)


def nearest_expand_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """Expand a boolean LR mask to the HR grid using categorical nearest semantics."""
    mask = propagate_valid_mask(mask, expected_shape=mask.shape)
    if isinstance(scale, bool) or int(scale) != scale or int(scale) <= 0:
        raise GridContractError(f"scale must be a positive integer, got {scale!r}")
    scale = int(scale)
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def combine_valid_masks(*masks: np.ndarray) -> np.ndarray:
    """Return the intersection of same-shaped boolean validity masks."""
    if not masks:
        raise GridContractError("at least one mask is required")
    normalized = [np.asarray(mask, dtype=bool) for mask in masks]
    if any(mask.shape != normalized[0].shape for mask in normalized[1:]):
        raise GridContractError("all masks must have the same shape")
    return np.logical_and.reduce(normalized)


def mask_from_nodata(data: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """Build a valid mask from finite values and an optional finite nodata code."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise GridContractError(f"data must be CHW, got {data.shape}")
    mask = np.isfinite(data).all(axis=0)
    if nodata is not None:
        mask &= np.all(data != nodata, axis=0)
    return mask


if __name__ == "__main__":
    grid = GridSpec("EPSG:32643", (10, 0, 500000, 0, -10, 300000), 32, 32)
    hr = GridSpec("EPSG:32643", (2.5, 0, 500000, 0, -2.5, 300000), 128, 128)
    validate_grid_pair(grid, hr, 4)
    print("geospatial grid smoke test passed")
