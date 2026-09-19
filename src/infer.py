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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.model import SRModel


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


def _normalise_input(data: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if mode == "none":
        return data.astype(np.float32, copy=False), {"method": "none"}
    if mode == "reflectance":
        return np.clip(data / 10000.0, 0.0, 1.0).astype(np.float32), {
            "method": "reflectance",
            "divisor": 10000.0,
        }
    raise ValueError("input normalization must be 'none' or 'reflectance'")


def _denormalise_output(data: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    if stats.get("method") == "reflectance":
        return data * float(stats["divisor"])
    return data


def load_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
) -> Tuple[SRModel, Dict[str, Any]]:
    """Load a T9 checkpoint and reconstruct the saved model configuration."""
    payload = torch.load(checkpoint, map_location=device)
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


@torch.no_grad()
def predict(model: SRModel, data_chw: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and variance arrays in CHW format."""
    if data_chw.ndim != 3:
        raise ValueError(f"expected CHW input, got {data_chw.shape}")
    expected_channels = int(model.stem.in_channels)
    if data_chw.shape[0] != expected_channels:
        raise ValueError(
            f"checkpoint expects {expected_channels} input bands, got {data_chw.shape[0]}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(data_chw)).unsqueeze(0).to(device=device, dtype=torch.float32)
    mean, log_var = model(tensor)
    # Clamp only for conversion to variance. This prevents a pathological checkpoint
    # from producing inf while retaining the trained log-variance tensor otherwise.
    variance = torch.exp(log_var.clamp(min=-20.0, max=10.0))
    return mean.squeeze(0).cpu().numpy(), variance.squeeze(0).cpu().numpy()


def _write_npy(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data.astype(np.float32, copy=False))


def _write_geotiff(path: Path, data: np.ndarray, metadata: Dict[str, Any], scale: int) -> None:
    """Write CHW output using the input georeferencing, scaled by SR factor."""
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required for GeoTIFF export. Install rasterio or use .npy outputs."
        ) from exc

    if "crs" not in metadata or "transform" not in metadata:
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32, copy=False))


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
        "--output-normalization",
        choices=("same", "reflectance", "none"),
        default="same",
        help="same keeps model units; reflectance multiplies normalized output by 10000",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    metadata = _load_json(args.metadata_json)
    data, metadata = _load_input(args.input, metadata)

    data, input_stats = _normalise_input(data, args.input_normalization)
    mean, variance = predict(model, data, device)

    if args.output_normalization == "same":
        output = mean
    elif args.output_normalization == "reflectance":
        output = _denormalise_output(mean, {"method": "reflectance", "divisor": 10000.0})
    else:
        output = mean

    scale = int(config.get("scale", 4))
    write_output(args.output, output, metadata, scale)
    if args.uncertainty_output:
        write_output(args.uncertainty_output, variance, metadata, scale)

    print(
        f"inference complete: input={tuple(data.shape)}, output={tuple(output.shape)}, "
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
