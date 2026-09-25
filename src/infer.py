"""
Inference CLI for the Sentinel-2 super-resolution prototype (T11).

Loads a T9 checkpoint, runs one Sentinel-2 tile, and writes:
  * super-resolved mean image
  * per-band variance (exp(log-variance))

Inputs can be NumPy CHW arrays (.npy) or geospatial rasters readable by
rasterio. For geospatial input, CRS/transform are propagated to GeoTIFF
outputs. For .npy input, pass --metadata-json containing optional "crs",
"transform", "width", "height", and "nodata" fields.

Examples:
    python -m src.infer --checkpoint runs/demo/best.pt \
        --input tile.tif --output sr.tif --uncertainty-output uncertainty.tif

    python -m src.infer --checkpoint runs/demo/best.pt \
        --input tile.npy --metadata-json tile.json --output sr.tif

The script deliberately keeps rasterio optional. NumPy input/output remains
available in environments that only have PyTorch; georeferenced GeoTIFF
export requires rasterio.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.model import SRModel
from src.telemetry import EventLogger


def _load_json(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("metadata JSON must contain an object")
    return value


def _load_input(path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load CHW input and raster metadata. GeoTIFF support is optional."""
    path = Path(path)
    metadata = dict(metadata or {})
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim != 3:
            raise ValueError(f"expected CHW or HW input, got {data.shape}")
        return data.astype(np.float32, copy=False), metadata

    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required for non-.npy input. Install rasterio or provide a .npy tile."
        ) from exc

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        metadata.update(
            {
                "crs": src.crs.to_string() if src.crs else None,
                "transform": tuple(src.transform),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "driver": src.driver,
            }
        )
    return data, metadata


def _normalise_input(data: np.ndarray, mode: str, valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    mask = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != data.shape[-2:]:
            raise ValueError(f"valid_mask shape {mask.shape} does not match input spatial shape {data.shape[-2:]}")
        if not np.any(mask):
            raise ValueError("valid_mask contains no valid pixels")
        if not np.isfinite(data[:, mask]).all():
            raise ValueError("valid input pixels must be finite")
        data = np.where(mask[None, ...], data, 0.0)
    if mode == "none":
        return data.astype(np.float32, copy=False), {"method": "none", "valid_mask_applied": mask is not None}
    if mode == "reflectance":
        normalized = np.clip(data / 10000.0, 0.0, 1.0).astype(np.float32)
        if mask is not None:
            normalized[:, ~mask] = 0.0
        return normalized, {"method": "reflectance", "divisor": 10000.0, "valid_mask_applied": mask is not None}
    raise ValueError("input normalization must be 'none' or 'reflectance'")


def _denormalise_output(data: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    if stats.get("method") == "reflectance":
        return data * float(stats["divisor"])
    return data


def load_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
) -> Tuple[SRModel, Dict[str, Any]]:
    """Load a T9 checkpoint and reconstruct the saved model configuration.

    SECURITY (fix for audit finding #6, 2026-09-19): torch.load() with its
    historical default (weights_only=False) uses pickle under the hood and
    will execute arbitrary code embedded in a malicious checkpoint file --
    a real remote-code-execution risk for any file loaded from outside your
    own training run, not a hypothetical one. weights_only=True restricts
    deserialization to tensors and a small set of safe builtin containers
    (dict, OrderedDict, int, float, str, list), which is exactly what
    train.py's save_checkpoint() produces -- so this is not just safer, it
    matches what this codebase actually writes. If you hit an
    UnpicklingError here on a checkpoint from OUTSIDE this codebase, that's
    the check doing its job -- inspect the file before considering a
    narrower, explicit allowlist rather than disabling this.
    """
    try:
        # Allowlist numpy types that appear in checkpoints (RNG state uses
        # np.ndarray with uint32 dtype). This keeps weights_only=True safe
        # while supporting our own checkpoint format.
        import torch.serialization
        _safe_types = [np.ndarray, np.dtype, np.empty]
        try:
            _safe_types.append(type(np.dtype(np.uint32)))
            _safe_types.append(type(np.dtype(np.float32)))
            _safe_types.append(type(np.dtype(np.float64)))
        except Exception:
            pass
        try:
            from numpy._core.multiarray import _reconstruct
            _safe_types.append(_reconstruct)
        except ImportError:
            pass
        with torch.serialization.safe_globals(_safe_types):
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"failed to load checkpoint safely (weights_only=True): {exc}. "
            f"If this checkpoint is from a trusted source and uses types outside "
            f"torch's safe-loading allowlist, inspect it before deciding whether "
            f"to load it any other way -- do not silently fall back to "
            f"weights_only=False."
        ) from exc
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        config = dict(payload.get("model_config") or {})
    elif isinstance(payload, dict):
        # Accept a raw state_dict as a convenience for manually saved models.
        state = payload
        config = {}
    else:
        raise ValueError("checkpoint must be a torch checkpoint/state_dict dictionary")

    allowed = {
        "in_channels",
        "out_channels",
        "base_channels",
        "num_attn_blocks",
        "window_size",
        "num_heads",
        "scale",
    }
    config = {key: value for key, value in config.items() if key in allowed}
    model = SRModel(**config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


def validate_input_preflight(
    data_chw: np.ndarray,
    expected_channels: int,
    max_pixels: Optional[int] = 64_000_000,
    max_bytes: Optional[int] = 4_000_000_000,
) -> None:
    """T18 (external audit's T17, narrowed — see tasks.md): reject
    malformed/oversized/non-finite input BEFORE any tensor is allocated,
    not after. Called from predict() before torch.from_numpy()/model(...),
    not as an afterthought once the array has already been converted and
    (on GPU) copied to device memory.

    Defaults: max_pixels=64M (~8000x8000, generous for a single Sentinel-2
    tile crop -- a full untiled Sentinel-2 scene is roughly 11000x11000 per
    10m band, so this is deliberately smaller than "a real full scene" to
    force genuinely huge inputs through tiled inference rather than a single
    unbounded allocation) and max_bytes=4GB
    (a rough guard against a technically-small-pixel-count but absurdly
    high band-count array). Both are overridable per call site, not fixed
    constants, since "reasonable" depends on available hardware.
    """
    if data_chw.ndim != 3:
        raise ValueError(f"expected CHW input, got array with {data_chw.ndim} dimensions, shape {data_chw.shape}")

    channels, height, width = data_chw.shape
    if channels != expected_channels:
        raise ValueError(f"checkpoint expects {expected_channels} input bands, got {channels}")
    if height <= 0 or width <= 0:
        raise ValueError(f"input has non-positive spatial dimensions: {height}x{width}")

    num_pixels = height * width
    if max_pixels is not None and num_pixels > max_pixels:
        raise ValueError(
            f"input is {height}x{width} ({num_pixels:,} pixels), exceeding max_pixels={max_pixels:,}. "
            f"This is a preflight rejection specifically so a malicious or accidentally-huge input "
            f"can't force an unbounded allocation -- tile the input or "
            f"raise max_pixels explicitly if you know the hardware can handle it."
        )

    if max_bytes is not None and data_chw.nbytes > max_bytes:
        raise ValueError(
            f"input is {data_chw.nbytes:,} bytes, exceeding max_bytes={max_bytes:,} "
            f"(shape {data_chw.shape}, dtype {data_chw.dtype})"
        )

    if not np.isfinite(data_chw).all():
        raise ValueError(
            "input contains non-finite values (NaN/Inf). Rejected before model allocation -- "
            "a NaN/Inf silently propagates through every downstream layer rather than failing loudly."
        )


@torch.no_grad()
def predict(
    model: SRModel,
    data_chw: np.ndarray,
    device: torch.device,
    max_pixels: Optional[int] = 64_000_000,
    max_bytes: Optional[int] = 4_000_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and variance arrays in CHW format."""
    expected_channels = int(model.stem.in_channels)
    # Preflight BEFORE allocation -- this replaces the narrower ndim/channel-only
    # checks the previous version had here, since validate_input_preflight()
    # covers those plus size and finiteness, all before torch.from_numpy().
    validate_input_preflight(data_chw, expected_channels, max_pixels=max_pixels, max_bytes=max_bytes)
    tensor = torch.from_numpy(np.ascontiguousarray(data_chw)).unsqueeze(0).to(device=device, dtype=torch.float32)
    mean, log_var = model(tensor)
    # Clamp only for conversion to variance. This prevents a pathological checkpoint
    # from producing inf while retaining the trained log-variance tensor otherwise.
    variance = torch.exp(log_var.clamp(min=-20.0, max=10.0))
    return mean.squeeze(0).cpu().numpy(), variance.squeeze(0).cpu().numpy()


def _tile_starts(length: int, core_size: int) -> list[int]:
    """Return core-window starts that cover an axis without a tiny tail tile."""
    if length <= 0 or core_size <= 0:
        raise ValueError("length and core_size must be positive")
    last_start = max(0, length - core_size)
    starts = list(range(0, last_start + 1, core_size))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.no_grad()
def predict_tiled(
    model: SRModel,
    data_chw: np.ndarray,
    device: torch.device,
    tile_size: int = 1024,
    overlap: int = 64,
    max_pixels: Optional[int] = 64_000_000,
    max_bytes: Optional[int] = 4_000_000_000,
    telemetry: Optional[EventLogger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run bounded overlapping inference and stitch non-overlapping core tiles.

    ``tile_size`` is the size of each core region in input pixels. ``overlap``
    adds context on every side before the model forward pass; only the core is
    copied into the output. Crop-and-place avoids seams caused by averaging
    predictions made with different amounts of context, while the context
    reduces boundary effects for convolutional and windowed-attention layers.
    The full input and output arrays remain in memory, but model activations
    are bounded by one context-expanded tile.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < tile_size, got {overlap}")

    expected_channels = int(model.stem.in_channels)
    # Large scenes are intentionally exempt from whole-scene pixel/byte caps;
    # each context-expanded tile is still checked by predict() below.
    validate_input_preflight(
        data_chw,
        expected_channels,
        max_pixels=None,
        max_bytes=None,
    )
    _, height, width = data_chw.shape

    y_starts = _tile_starts(height, tile_size)
    x_starts = _tile_starts(width, tile_size)
    mean_out = None
    variance_out = None

    for core_y0 in y_starts:
        core_y1 = min(core_y0 + tile_size, height)
        tile_y0 = max(0, core_y0 - overlap)
        tile_y1 = min(height, core_y1 + overlap)
        for core_x0 in x_starts:
            core_x1 = min(core_x0 + tile_size, width)
            tile_x0 = max(0, core_x0 - overlap)
            tile_x1 = min(width, core_x1 + overlap)
            tile = data_chw[:, tile_y0:tile_y1, tile_x0:tile_x1]
            tile_mean, tile_variance = predict(
                model,
                tile,
                device,
                max_pixels=max_pixels,
                max_bytes=max_bytes,
            )
            if telemetry is not None:
                telemetry.emit("tile_completed", core_y0=core_y0, core_x0=core_x0, core_height=core_y1-core_y0, core_width=core_x1-core_x0)

            if mean_out is None:
                scale = tile_mean.shape[-1] // tile.shape[-1]
                if scale <= 0 or tile_mean.shape[-2] != tile.shape[-2] * scale:
                    raise ValueError(
                        f"model output shape {tile_mean.shape} is not an integer upscale of tile {tile.shape}"
                    )
                mean_out = np.empty(
                    (tile_mean.shape[0], height * scale, width * scale),
                    dtype=tile_mean.dtype,
                )
                variance_out = np.empty_like(mean_out)

            local_y0 = (core_y0 - tile_y0) * scale
            local_y1 = local_y0 + (core_y1 - core_y0) * scale
            local_x0 = (core_x0 - tile_x0) * scale
            local_x1 = local_x0 + (core_x1 - core_x0) * scale
            out_y0, out_y1 = core_y0 * scale, core_y1 * scale
            out_x0, out_x1 = core_x0 * scale, core_x1 * scale
            mean_out[:, out_y0:out_y1, out_x0:out_x1] = tile_mean[:, local_y0:local_y1, local_x0:local_x1]
            variance_out[:, out_y0:out_y1, out_x0:out_x1] = tile_variance[:, local_y0:local_y1, local_x0:local_x1]

    assert mean_out is not None and variance_out is not None
    return mean_out, variance_out


def _write_npy(path: Path, data: np.ndarray) -> None:
    """Atomic write (T18): write to a temp file in the same directory, then
    os.replace() to the final path. os.replace() is atomic on POSIX and
    Windows when source/destination are on the same filesystem, which same-
    directory guarantees -- so a crash or kill mid-write can never leave a
    truncated/corrupt file sitting at the real output path; worst case, an
    orphaned .tmp file is left, and the destination path either has the
    complete previous version or doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp_path, data.astype(np.float32, copy=False))
    # np.save appends .npy to the filename if it's not already the suffix,
    # so tmp_path becomes e.g. "out.npy.tmp.npy" on disk -- reconcile that
    # explicitly rather than relying on np.save's suffix-guessing behavior.
    actual_tmp_path = tmp_path if tmp_path.suffix == ".npy" else tmp_path.with_suffix(tmp_path.suffix + ".npy")
    os.replace(actual_tmp_path, path)


def _write_geotiff(path: Path, data: np.ndarray, metadata: Dict[str, Any], scale: int) -> None:
    """Write CHW output using the input georeferencing, scaled by SR factor."""
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required for GeoTIFF export. Install rasterio or use .npy outputs."
        ) from exc

    # BUG-005 fix (astrasr, audit remediation) checked key *presence*, not
    # whether the value was actually usable — metadata.get("crs") is None
    # is a real, easy-to-hit case (e.g. an input raster with no CRS set,
    # or a hand-built metadata dict), and it slipped straight through the
    # old `"crs" not in metadata` check since the key existed with value
    # None. Caught by agent4 via test_infer.py's
    # test_geotiff_export_rejects_missing_crs, which astrasr's own
    # remediation turn added but which never actually ran due to a
    # separate file-corruption bug (see build-status.md) — so this was
    # never execution-verified until now.
    if metadata.get("crs") is None or metadata.get("transform") is None:
        raise ValueError(
            "GeoTIFF export needs georeferencing. Supply a raster input or metadata JSON "
            "with both 'crs' and 'transform'."
        )

    transform = Affine(*metadata["transform"])
    transform = transform * Affine.scale(1.0 / scale, 1.0 / scale)
    height, width = data.shape[-2:]
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": data.shape[0],
        "dtype": "float32",
        "crs": metadata["crs"],
        "transform": transform,
        "compress": "deflate",
        "predictor": 3,
    }
    if metadata.get("nodata") is not None:
        profile["nodata"] = metadata["nodata"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32, copy=False))
    os.replace(tmp_path, path)  # atomic on the same filesystem, see _write_npy's comment


def write_output(
    path: str | Path,
    data: np.ndarray,
    metadata: Dict[str, Any],
    scale: int,
) -> None:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        _write_npy(path, data)
    else:
        _write_geotiff(path, data, metadata, scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="CHW .npy or rasterio-readable raster")
    parser.add_argument("--output", required=True, help="SR output .npy or georeferenced .tif")
    parser.add_argument(
        "--uncertainty-output",
        help="Optional per-band variance output (.npy or georeferenced .tif)",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional metadata sidecar for .npy input; used for GeoTIFF georeferencing",
    )
    parser.add_argument(
        "--input-normalization",
        choices=("none", "reflectance"),
        default="none",
        help="Use reflectance to convert Sentinel-2 L2A DN values with DN/10000",
    )
    parser.add_argument(
        "--valid-mask",
        help="Optional .npy boolean HxW mask (True=valid). Invalid input pixels are zeroed and excluded from outputs.",
    )
    parser.add_argument(
        "--output-normalization",
        choices=("same", "reflectance", "none"),
        default="same",
        help="same keeps model units; reflectance multiplies normalized output by 10000",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=64_000_000,
        help="T18: reject input larger than this before allocation (default ~8000x8000)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=4_000_000_000,
        help="T18: reject input using more raw memory than this before allocation",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="T21: input-pixel core size for bounded tiled inference; 0 runs one full-scene pass",
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=64,
        help="T21: context pixels added around each tile core (default: 64)",
    )
    parser.add_argument("--telemetry", default=None, help="optional JSONL telemetry path")
    args = parser.parse_args()
    telemetry = EventLogger(args.telemetry)
    telemetry.start(mode="inference", input=args.input, output=args.output, device=args.device, tiled=bool(args.tile_size))

    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    metadata = _load_json(args.metadata_json)
    data, metadata = _load_input(args.input, metadata)
    valid_mask = None
    if args.valid_mask:
        valid_mask = np.asarray(np.load(args.valid_mask), dtype=bool)
        if valid_mask.shape != data.shape[-2:]:
            raise ValueError(f"valid mask shape {valid_mask.shape} does not match input spatial shape {data.shape[-2:]}")
    data, input_stats = _normalise_input(data, args.input_normalization, valid_mask=valid_mask)
    if args.tile_size:
        mean, variance = predict_tiled(
            model,
            data,
            device,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            max_pixels=args.max_pixels,
            max_bytes=args.max_bytes,
            telemetry=telemetry,
        )
    else:
        mean, variance = predict(model, data, device, max_pixels=args.max_pixels, max_bytes=args.max_bytes)

    if args.output_normalization == "same":
        output = mean
        output_variance = variance
    elif args.output_normalization == "reflectance":
        # UNIT FIX (audit finding #5, 2026-09-19): mean was being denormalised
        # by the reflectance divisor (x10000) while variance was written out
        # completely unscaled -- inconsistent units between the two outputs.
        # Var(a*X) = a^2 * Var(X), so if the mean is scaled by `divisor`, the
        # variance must be scaled by `divisor**2` to describe the *same*
        # rescaled quantity. Writing raw model-space variance next to
        # reflectance-space mean made the uncertainty output silently wrong
        # by a factor of 10000**2 = 1e8 whenever --output-normalization
        # reflectance was used -- not a rounding error, a unit-system bug.
        divisor = 10000.0
        output = _denormalise_output(mean, {"method": "reflectance", "divisor": divisor})
        output_variance = variance * (divisor ** 2)
    else:
        output = mean
        output_variance = variance

    scale = int(config.get("scale", 4))
    if valid_mask is not None:
        output_mask = np.repeat(np.repeat(valid_mask, scale, axis=0), scale, axis=1)
        output = np.where(output_mask[None, ...], output, 0.0)
        output_variance = np.where(output_mask[None, ...], output_variance, 0.0)
    write_output(args.output, output, metadata, scale)
    if args.uncertainty_output:
        write_output(args.uncertainty_output, output_variance, metadata, scale)

    telemetry.finish(mode="inference", input_shape=list(data.shape), output_shape=list(output.shape), uncertainty=bool(args.uncertainty_output))
    print(
        f"inference complete: input={tuple(data.shape)}, output={tuple(output.shape)}, "
        f"tiled={'yes' if args.tile_size else 'no'}, "
        f"uncertainty={'yes' if args.uncertainty_output else 'no'}, device={device}"
    )


def smoke_test() -> None:
    """Minimal shape test for the inference helper; requires PyTorch."""
    torch.manual_seed(0)
    model = SRModel(
        in_channels=4,
        out_channels=4,
        base_channels=8,
        num_attn_blocks=1,
        window_size=4,
        num_heads=2,
        scale=2,
    )
    model.eval()
    x = np.random.default_rng(0).random((4, 8, 8), dtype=np.float32)
    mean, variance = predict(model, x, torch.device("cpu"))
    assert mean.shape == (4, 16, 16)
    assert variance.shape == (4, 16, 16)
    assert np.isfinite(mean).all() and np.isfinite(variance).all()
    print("[smoke_test OK] inference shapes and finite uncertainty verified")


if __name__ == "__main__":
    main()
