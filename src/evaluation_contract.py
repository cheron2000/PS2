"""Validation and loading for the frozen SIH26142 evaluation contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class EvaluationContractError(ValueError):
    """Raised when an evaluation contract violates a frozen policy."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise EvaluationContractError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise EvaluationContractError(f"{name} must be a non-empty list")
    return value


def validate_evaluation_contract(contract: Mapping[str, Any]) -> None:
    """Validate schema and invariants of the versioned evaluation policy."""
    if not isinstance(contract, Mapping):
        raise EvaluationContractError("contract root must be an object")
    if contract.get("contract_version") != 1:
        raise EvaluationContractError("unsupported contract_version")
    if contract.get("status") != "frozen-policy":
        raise EvaluationContractError("contract status must be 'frozen-policy'")

    task = _require(contract, "task", "contract")
    bands = _require_list(_require(task, "input_bands", "task"), "task.input_bands")
    if bands != ["blue", "green", "red", "nir"]:
        raise EvaluationContractError("task.input_bands must use canonical blue/green/red/nir order")
    for key in ("input_gsd_m", "primary_output_gsd_m", "secondary_output_gsd_m"):
        value = _require(task, key, "task")
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise EvaluationContractError(f"task.{key} must be finite and positive")

    datasets = _require(contract, "datasets", "contract")
    for dataset_id in ("primary", "secondary", "india_holdout"):
        dataset = _require(datasets, dataset_id, "datasets")
        for key in ("id", "route", "required_provenance"):
            _require(dataset, key, f"datasets.{dataset_id}")
        if dataset["required_provenance"] != "sealed_sha256_manifest":
            raise EvaluationContractError(
                f"datasets.{dataset_id}.required_provenance must be sealed_sha256_manifest"
            )
    if _require(datasets["primary"], "scale_factor", "datasets.primary") != 4:
        raise EvaluationContractError("primary dataset scale_factor must be 4")
    if _require(datasets["secondary"], "scale_factor", "datasets.secondary") != 2:
        raise EvaluationContractError("secondary dataset scale_factor must be 2")

    split = _require(contract, "split_policy", "contract")
    if split.get("unit") != "scene" or not split.get("scene_disjoint") or not split.get("sample_disjoint"):
        raise EvaluationContractError("split_policy must enforce scene- and sample-disjoint splits")
    fractions = _require(split, "fractions", "split_policy")
    values = [fractions.get(name) for name in ("train", "val", "test")]
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in values):
        raise EvaluationContractError("split fractions must be finite and positive")
    if not np.isclose(sum(values), 1.0):
        raise EvaluationContractError("split fractions must sum to 1")
    if split.get("manifest_schema_version") != 1:
        raise EvaluationContractError("split manifest schema version must be 1")
    if split.get("test_used_for_model_selection") is not False:
        raise EvaluationContractError("test_used_for_model_selection must be false")

    preprocessing = _require(contract, "preprocessing_policy", "contract")
    if preprocessing.get("canonical_band_order") != bands:
        raise EvaluationContractError("preprocessing canonical band order must match task input bands")
    if preprocessing.get("sentinel2_reflectance_divisor") != 10000.0:
        raise EvaluationContractError("Sentinel-2 divisor must remain 10000.0")
    if preprocessing.get("naip_reflectance_divisor") != 255.0:
        raise EvaluationContractError("NAIP divisor must remain 255.0")
    alignment = _require(preprocessing, "alignment", "preprocessing_policy")
    if alignment.get("reject_non_scale_aligned_shift") is not True:
        raise EvaluationContractError("alignment must reject non-scale-aligned shifts")

    masks = _require(contract, "mask_policy", "contract")
    if masks.get("required_for_real_evaluation") is not True:
        raise EvaluationContractError("masks must be required for real evaluation")
    required_mask_stages = {"normalization", "loss", "image_metrics", "downstream_metrics", "uncertainty_metrics"}
    if not required_mask_stages.issubset(set(_require_list(masks.get("invalid_pixels_excluded_from"), "mask_policy.invalid_pixels_excluded_from"))):
        raise EvaluationContractError("mask policy must exclude invalid pixels from all evaluation stages")
    valid_fraction = masks.get("minimum_valid_fraction_per_patch")
    if not isinstance(valid_fraction, (int, float)) or not 0.0 <= valid_fraction <= 1.0:
        raise EvaluationContractError("minimum_valid_fraction_per_patch must be in [0, 1]")
    if masks.get("empty_valid_set") != "reject":
        raise EvaluationContractError("empty valid sets must be rejected")

    baselines = _require(contract, "baselines", "contract")
    required_baselines = _require_list(baselines.get("required"), "baselines.required")
    required_ids = {item.get("id") for item in required_baselines if isinstance(item, Mapping)}
    if not {"sr", "bicubic"}.issubset(required_ids):
        raise EvaluationContractError("SR and bicubic must be required baselines")
    if baselines.get("same_input_and_mask_across_methods") is not True or baselines.get("same_test_scene_ids_across_methods") is not True:
        raise EvaluationContractError("baselines must use identical inputs, masks, and test scenes")

    metrics = _require(contract, "metrics", "contract")
    required_metric_groups = {
        "image_quality_required": {"psnr_db", "ssim", "sam_degrees", "alignment_ncc"},
        "downstream_required": {"mean_iou", "overall_accuracy", "per_class_iou"},
        "uncertainty_required": {"gaussian_nll", "mean_standardized_squared_error", "coverage_0.50", "coverage_0.80", "coverage_0.90", "coverage_0.95"},
    }
    for field, expected in required_metric_groups.items():
        actual = set(_require_list(metrics.get(field), f"metrics.{field}"))
        if not expected.issubset(actual):
            raise EvaluationContractError(f"metrics.{field} is missing {sorted(expected - actual)}")
    aggregation = _require(metrics, "aggregation", "metrics")
    if aggregation.get("primary_unit") != "scene" or aggregation.get("report_per_scene") is not True:
        raise EvaluationContractError("metrics must report per-scene results")
    confidence = aggregation.get("bootstrap_confidence")
    if confidence != 0.95:
        raise EvaluationContractError("bootstrap confidence must be 0.95")

    gates = _require(contract, "minimum_improvement_gates", "contract")
    if gates.get("comparison") != "sr_minus_bicubic":
        raise EvaluationContractError("minimum gates must compare SR against bicubic")
    iou_gate = _require(gates, "primary_downstream_mean_iou_delta", "minimum_improvement_gates")
    if iou_gate.get("minimum_absolute_delta", 0) <= 0:
        raise EvaluationContractError("primary IoU gate must require a positive improvement")
    uncertainty = _require(gates, "uncertainty_calibration", "minimum_improvement_gates")
    if uncertainty.get("coverage_absolute_error_maximum", 0) <= 0:
        raise EvaluationContractError("uncertainty coverage gate must be positive")

    reproducibility = _require(contract, "reproducibility", "contract")
    for key in ("manifest_required", "record_model_checkpoint", "record_code_revision", "record_environment", "record_config_snapshot", "resume_fingerprint_must_match"):
        if reproducibility.get(key) is not True:
            raise EvaluationContractError(f"reproducibility.{key} must be true")
    if reproducibility.get("manifest_checksum") != "sha256":
        raise EvaluationContractError("reproducibility manifest checksum must be sha256")


def load_evaluation_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON evaluation contract."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"failed to load evaluation contract: {exc}") from exc
    validate_evaluation_contract(contract)
    return contract


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs" / "evaluation_contract.json"


def load_default_evaluation_contract() -> dict[str, Any]:
    return load_evaluation_contract(DEFAULT_CONTRACT_PATH)


if __name__ == "__main__":
    load_default_evaluation_contract()
    print(f"evaluation contract valid: {DEFAULT_CONTRACT_PATH}")
