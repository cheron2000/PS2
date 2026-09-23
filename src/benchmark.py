"""Controlled, data-gated benchmark runner for SR methods.

The runner intentionally separates *protocol execution* from *data access*.
It can test the protocol on fixtures, but refuses to label a run as real-world
validation unless the caller supplies verified-real status and both sealed
scene/source manifests.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.metrics import metric_report
from src.scene_protocol import summarize_scene_metric


class BenchmarkError(ValueError):
    """Raised when benchmark inputs or evidence labels violate the contract."""


def nearest_upsample(image: np.ndarray, scale: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or isinstance(scale, bool) or int(scale) != scale or int(scale) <= 0:
        raise BenchmarkError("nearest_upsample expects CHW input and a positive integer scale")
    scale = int(scale)
    return np.repeat(np.repeat(image, scale, axis=-2), scale, axis=-1)


def bicubic_upsample(image: np.ndarray, scale: int) -> np.ndarray:
    """Use PyTorch's deterministic bicubic interpolation when available."""
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise BenchmarkError("bicubic baseline requires PyTorch in this environment") from exc
    image = np.asarray(image)
    if image.ndim != 3 or isinstance(scale, bool) or int(scale) != scale or int(scale) <= 0:
        raise BenchmarkError("bicubic_upsample expects CHW input and a positive integer scale")
    with torch.no_grad():
        tensor = torch.from_numpy(image.astype(np.float32, copy=False))[None]
        output = F.interpolate(tensor, scale_factor=int(scale), mode="bicubic", align_corners=False)
    return output[0].cpu().numpy()


def _validate_scene(scene: Mapping[str, Any], scale: int) -> None:
    for key in ("scene_id", "lr", "target"):
        if key not in scene:
            raise BenchmarkError(f"scene is missing required field {key!r}")
    lr = np.asarray(scene["lr"])
    target = np.asarray(scene["target"])
    if lr.ndim != 3 or target.ndim != 3:
        raise BenchmarkError(f"scene {scene['scene_id']!r}: lr and target must be CHW")
    expected = (lr.shape[0], lr.shape[1] * scale, lr.shape[2] * scale)
    if target.shape != expected:
        raise BenchmarkError(f"scene {scene['scene_id']!r}: target {target.shape} != expected {expected}")
    if not np.isfinite(lr).all() or not np.isfinite(target).all():
        raise BenchmarkError(f"scene {scene['scene_id']!r}: lr and target must be finite")
    if "valid_mask" in scene and scene["valid_mask"] is not None:
        mask = np.asarray(scene["valid_mask"], dtype=bool)
        if mask.shape not in (target.shape[-2:], target.shape):
            raise BenchmarkError(f"scene {scene['scene_id']!r}: valid_mask shape {mask.shape} is incompatible")
        if not np.any(mask):
            raise BenchmarkError(f"scene {scene['scene_id']!r}: valid_mask has no valid pixels")


def _evidence_label(metadata: Mapping[str, Any], claim_real: bool) -> str:
    if not claim_real:
        return "development-fixture"
    if metadata.get("data_status") != "verified_real":
        raise BenchmarkError("real benchmark requires data_status='verified_real'")
    for key in ("provenance_manifest", "scene_split_manifest"):
        if not metadata.get(key):
            raise BenchmarkError(f"real benchmark requires {key}")
    return "verified-real"


def run_benchmark(
    scenes: Sequence[Mapping[str, Any]],
    *,
    scale: int,
    sr_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    extra_methods: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    claim_real: bool = False,
    data_range: float = 1.0,
) -> dict[str, Any]:
    """Evaluate SR, nearest, bicubic, and optional methods on identical scenes."""
    if not scenes:
        raise BenchmarkError("at least one scene is required")
    evidence = _evidence_label(metadata or {}, claim_real)
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "nearest": lambda lr: nearest_upsample(lr, scale),
        "bicubic": lambda lr: bicubic_upsample(lr, scale),
    }
    if sr_fn is not None:
        methods["sr"] = sr_fn
    methods.update(extra_methods or {})
    per_method: dict[str, dict[str, Any]] = {name: {} for name in methods}

    for scene in scenes:
        _validate_scene(scene, scale)
        mask = scene.get("valid_mask")
        target = np.asarray(scene["target"])
        lr = np.asarray(scene["lr"])
        for name, method in methods.items():
            prediction = np.asarray(method(lr))
            if prediction.shape != target.shape:
                raise BenchmarkError(
                    f"scene {scene['scene_id']!r}/{name}: prediction {prediction.shape} != target {target.shape}"
                )
            per_method[name][str(scene["scene_id"])] = metric_report(
                prediction, target, data_range=data_range, valid_mask=mask
            )

    summary: dict[str, Any] = {}
    for method, scene_metrics in per_method.items():
        summary[method] = {
            metric: summarize_scene_metric(
                {scene_id: values[metric] for scene_id, values in scene_metrics.items()},
                confidence=0.95,
                seed=0,
            )
            for metric in ("psnr_db", "ssim", "sam_degrees", "alignment_ncc")
        }
    result: dict[str, Any] = {
        "benchmark_version": 1,
        "evidence": evidence,
        "scale": int(scale),
        "scene_count": len(scenes),
        "methods": sorted(methods),
        "per_scene": per_method,
        "summary": summary,
    }
    if "sr" in summary and "bicubic" in summary:
        result["sr_minus_bicubic"] = {
            metric: summary["sr"][metric]["mean"] - summary["bicubic"][metric]["mean"]
            for metric in ("psnr_db", "ssim", "sam_degrees", "alignment_ncc")
        }
    return result


if __name__ == "__main__":
    print("benchmark harness ready; call run_benchmark with verified scene records")
