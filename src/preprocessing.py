"""
src/preprocessing.py — Sentinel-2 L2A preprocessing for SIH26142 (T4).

Implements the preprocessing step from solution-draft.md v11's Technical
Architecture (step 1, "Ingest: L2A + cloud mask, normalize/resample"):
  1. Cloud/invalid-pixel masking via the Scene Classification Layer (SCL)
  2. Band normalization (reflectance scaling, or percentile/z-score stretch)
  3. Tiling into fixed-size patches, with a validity filter so cloud-heavy
     tiles don't end up in training

EXECUTION STATUS: written and unit-tested by sonnet5 (Claude Sonnet 5,
onboarding this turn — see BUILD_AGENTS.md Participants) on synthetic
numpy arrays — see test_preprocessing.py. All tests pass in this sandbox.
No real Sentinel-2 L2A scene was available here to test against real
cloud patterns; the SCL class-value logic follows ESA's documented
Sentinel-2 L2A SCL band spec (stable since launch), not empirical data —
worth a sanity check against one real .SAFE product before relying on it
for real training data selection.

Framework-agnostic (pure numpy) by design, matching T1's claude1's choice
to keep structural logic testable without a full torch install — T5/T6's
dataset loaders can wrap this with torch tensors as needed.
"""
import numpy as np

# Sentinel-2 L2A Scene Classification Layer (SCL) band values, per ESA's
# documented spec (stable since the L2A processing baseline was introduced).
SCL_NO_DATA = 0
SCL_SATURATED_DEFECTIVE = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM_PROB = 8
SCL_CLOUD_HIGH_PROB = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW = 11

DEFAULT_INVALID_CLASSES = frozenset({
    SCL_NO_DATA, SCL_SATURATED_DEFECTIVE, SCL_CLOUD_SHADOW,
    SCL_CLOUD_MEDIUM_PROB, SCL_CLOUD_HIGH_PROB, SCL_THIN_CIRRUS,
})
# Snow (11) is deliberately NOT in the default invalid set: it's real
# ground signal, not a sensor artifact. Dark area (2) and water (6) and
# unclassified (7) are also left valid by default — they're usable
# imagery, just not vegetation. Override invalid_classes per use case
# rather than assuming this default fits every task.


def build_validity_mask(scl: np.ndarray, invalid_classes=DEFAULT_INVALID_CLASSES) -> np.ndarray:
    """scl: (H, W) integer array of SCL class values.
    Returns a boolean (H, W) mask, True = valid/usable pixel."""
    if scl.ndim != 2:
        raise ValueError(f"expected 2D SCL array, got shape {scl.shape}")
    invalid = np.isin(scl, list(invalid_classes))
    return ~invalid


def normalize_bands(data: np.ndarray, method: str = "reflectance", stats: dict = None, reflectance_divisor: float = 10000.0, valid_mask: np.ndarray = None):
    """data: (C, H, W) raw digital-number or reflectance array.

    method:
      "reflectance" — fixed-divisor scaling to [0, 1]: DN / reflectance_divisor,
        clipped. Defaults to 10000.0 (Sentinel-2 L2A convention). **This
        default is only correct for the source it was written for
        (Sentinel-2-family reflectance products). Applying it unchanged to
        a source with a different native DN range — e.g. NAIP's 0-255
        8-bit imagery — silently crushes that source's real range into a
        tiny sliver of [0, 1] instead of spanning it.** Confirmed as a
        real bug in this exact form (2026-09-20, external "Wide Research"
        audit, verified independently by agent4 via direct code
        inspection before fixing): src/datasets/sen2naip.py called this
        with the same default divisor for both Sentinel-2 LR and NAIP HR.
        Fixed there by passing reflectance_divisor=255.0 for the NAIP
        side — see that file. If you add a new paired dataset with
        differently-scaled sources, pass the correct divisor per side
        explicitly; don't assume the default fits both.
      "percentile"  — per-band 2nd/98th percentile stretch to [0, 1]. Safe
        to apply identically to differently-scaled sources, since it's
        computed from each array's own statistics, not a fixed divisor.
      "zscore"      — per-band (x - mean) / std. Same safety property as
        percentile — adapts to each array's own scale automatically.

    Returns (normalized_data, stats_used) — stats_used lets you invert the
    normalization later (needed to turn model output back into physical
    reflectance units for the georeferenced export step, per
    solution-draft.md's "Export: CRS + affine, band metadata kept").
    """
    if data.ndim != 3:
        raise ValueError(f"expected (C, H, W) array, got shape {data.shape}")
    mask = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != data.shape[-2:]:
            raise ValueError(f"valid_mask shape {mask.shape} doesn't match data spatial shape {data.shape[-2:]}")
        if not np.any(mask):
            raise ValueError("valid_mask contains no valid pixels")
        if not np.isfinite(data[:, mask]).all():
            raise ValueError("valid pixels must contain only finite values")
        work_data = np.where(mask[None, ...], data, 0.0)
    else:
        if not np.isfinite(data).all():
            raise ValueError("data must contain only finite values; provide a validity mask for nodata/cloud pixels")
        work_data = data

    if method == "reflectance":
        normalized = np.clip(work_data.astype(np.float64) / reflectance_divisor, 0.0, 1.0)
        if mask is not None:
            normalized[:, ~mask] = 0.0
        return normalized, {"method": "reflectance", "divisor": reflectance_divisor, "valid_mask_applied": mask is not None}

    if method == "percentile":
        if stats is None:
            if mask is None:
                lo = np.percentile(work_data, 2, axis=(1, 2), keepdims=True)
                hi = np.percentile(work_data, 98, axis=(1, 2), keepdims=True)
            else:
                lo = np.array([np.percentile(work_data[c][mask], 2) for c in range(data.shape[0])])[:, None, None]
                hi = np.array([np.percentile(work_data[c][mask], 98) for c in range(data.shape[0])])[:, None, None]
        else:
            lo, hi = stats["lo"], stats["hi"]
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        normalized = np.clip((work_data.astype(np.float64) - lo) / denom, 0.0, 1.0)
        if mask is not None:
            normalized[:, ~mask] = 0.0
        return normalized, {"method": "percentile", "lo": lo, "hi": hi, "valid_mask_applied": mask is not None}

    if method == "zscore":
        if stats is None:
            if mask is None:
                mean = data.mean(axis=(1, 2), keepdims=True)
                std = data.std(axis=(1, 2), keepdims=True)
            else:
                mean = np.array([data[c][mask].mean() for c in range(data.shape[0])])[:, None, None]
                std = np.array([data[c][mask].std() for c in range(data.shape[0])])[:, None, None]
        else:
            mean, std = stats["mean"], stats["std"]
        std_safe = np.where(std == 0, 1.0, std)
        normalized = (work_data.astype(np.float64) - mean) / std_safe
        if mask is not None:
            normalized[:, ~mask] = 0.0
        return normalized, {"method": "zscore", "mean": mean, "std": std, "valid_mask_applied": mask is not None}

    raise ValueError(f"unknown normalization method: {method!r}")


def denormalize_bands(normalized: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of normalize_bands, using the stats dict it returned."""
    method = stats["method"]
    if method == "reflectance":
        return normalized * stats["divisor"]
    if method == "percentile":
        return normalized * (stats["hi"] - stats["lo"]) + stats["lo"]
    if method == "zscore":
        return normalized * stats["std"] + stats["mean"]
    raise ValueError(f"unknown normalization method: {method!r}")


def tile_into_patches(data: np.ndarray, patch_size: int, stride: int = None,
                       validity_mask: np.ndarray = None, min_valid_fraction: float = 0.8):
    """data: (C, H, W). Splits into patch_size x patch_size tiles.

    stride defaults to patch_size (non-overlapping tiles); pass a smaller
    stride for overlapping tiles (more training samples, more redundancy).

    If validity_mask (H, W boolean, True=valid) is given, tiles with less
    than min_valid_fraction valid pixels are dropped — keeps cloud-heavy
    tiles out of training instead of silently teaching the model on
    masked-out garbage.

    Returns a list of dicts: {"patch": (C, patch_size, patch_size) array,
    "row": int, "col": int, "valid_fraction": float}. Edge remainder
    (data not evenly divisible by patch_size/stride) is dropped, not
    padded — a partial tile would need its own masking logic downstream,
    not worth the complexity for the volume of data this project has.
    """
    if data.ndim != 3:
        raise ValueError(f"expected (C, H, W) array, got shape {data.shape}")
    C, H, W = data.shape
    if stride is None:
        stride = patch_size
    if patch_size <= 0 or stride <= 0:
        raise ValueError(f"patch_size and stride must be positive, got {patch_size}, {stride}")
    if validity_mask is not None and validity_mask.shape != (H, W):
        raise ValueError(f"validity_mask shape {validity_mask.shape} doesn't match data spatial shape {(H, W)}")
    if not np.isfinite(min_valid_fraction) or not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError(f"min_valid_fraction must be finite and in [0, 1], got {min_valid_fraction}")

    patches = []
    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            patch = data[:, row:row + patch_size, col:col + patch_size]
            if validity_mask is not None:
                mask_patch = validity_mask[row:row + patch_size, col:col + patch_size]
                valid_fraction = float(mask_patch.mean())
                if valid_fraction < min_valid_fraction:
                    continue
            else:
                valid_fraction = 1.0
            patches.append({"patch": patch, "row": row, "col": col, "valid_fraction": valid_fraction})
    return patches
