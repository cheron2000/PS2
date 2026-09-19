# PS2 Prototype — Genuine Gap Audit

**Date:** 2026-09-19  
**Method:** Five parallel reviews covering code correctness, real-data/geospatial readiness, scientific validity, reproducibility, and inference/deployment robustness, followed by a deduplication and prioritization pass.

## Executive conclusion

The repository is a well-tested **synthetic-array prototype scaffold**, but it is not yet a scientifically reproducible, real-product geospatial super-resolution system or a deployment-safe inference service. The strongest gaps are concrete and reachable through the current public APIs; they are not merely requests for polish.

## Prioritized genuine gaps

| Rank | Gap | Severity | Why it is genuine | Recommended next step |
|---:|---|---|---|---|
| 1 | **No independent scene/geographic split** | Critical | `src/train.py` builds one dataset and calls `fit()` without a validation loader; best-checkpoint selection falls back to training loss. The integration test uses the same loader for training and validation. | Add a persisted scene/AOI/date split before patch extraction, with blind test data and per-scene reporting. |
| 2 | **Real sensor ingestion and spectral schemas are missing** | Critical | SEN2NAIPv2 is documented as TACO-format but only `lr/*.npy`/`hr/*.npy` is supported. SEN2Vénus is unverified. Cartosat can repeat one band or truncate extra bands. | Implement product-specific adapters, explicit band/wavelength/unit maps, rejection of ambiguous channels, and versioned manifests. |
| 3 | **Pairing is shape-based, not CRS/grid-based** | Critical | Pairing does not read or validate CRS, affine transform, bounds, GSD, rotation, or overlap. `dy // scale` can leave a phase mismatch for non-multiple HR shifts. | Reproject to a declared reference grid, preserve transforms through crops, and test non-integer shifts, CRS differences, rotation, and partial overlap. |
| 4 | **Invalid-pixel masks are disconnected from training/inference** | High | SCL utilities exist, but the primary datasets do not load SCL/nodata masks or return `valid_mask`; finite-value checks do not catch finite sentinel nodata values. Edge pixels can also be dropped without provenance. | Carry SCL/QA/nodata masks through ingestion, pairing, loss, metrics, inference, and GeoTIFF outputs. Return patch transforms and source provenance. |
| 5 | **Training and inference units are inconsistent** | Critical | Training defaults to reflectance normalization, while inference defaults to no normalization. Inference scales the mean by 10,000 but does not scale variance by 10,000². | Persist a versioned preprocessing/unit contract in the checkpoint; fail on missing metadata and test mean/variance unit conversions. |
| 6 | **Checkpoint loading is unsafe and under-validated** | Critical | `torch.load()` uses unrestricted deserialization; model configuration is not strongly validated before construction, and raw state dictionaries can silently fall back to default architecture. | Use safetensors or restricted tensor loading, validate schema/config/parameter budgets on CPU, and require explicit checkpoint markers. |
| 7 | **Full-scene inference has no resource or input/output safety** | Critical | Inference reads the full raster and runs one full-scene tensor. There is no tiling, overlap blending, pixel/byte quota, finite-value preflight, path-collision check, or atomic output transaction. | Implement bounded tiled/streaming inference, strict input validation, correct overlap handling, temporary-file atomic writes, and output manifests. |
| 8 | **Public APIs permit unsupported or fragile configurations** | High | The model supports only scale 2 or 4 while datasets accept arbitrary scales; `batch_size > 1` can fail on variable-size crops; custom models saved without `model_config` may be unloadable. | Restrict unsupported parameters with clear errors or implement general support; add API round-trip and heterogeneous-batch tests. |
| 9 | **Experiments are not reproducible or resumable** | High | No resume path exists; deterministic backend settings and RNG state are not persisted; dependencies are broad lower bounds; datasets are filesystem-discovered without immutable manifests; logs omit resolved configuration and environment. | Add locked environment metadata, dataset/config hashes, deterministic mode, RNG persistence, structured logs, and resume-capable atomic checkpoints. |
| 10 | **Evaluation and uncertainty claims lack a valid independent protocol** | High | Metrics are global rather than per-scene, lack bootstrap intervals, and do not distinguish native vs registered scores. Uncertainty calibration is not evaluated. The three-class NDVI proxy is not semantically compatible with 11-class WorldCover without an explicit ontology. | Add scene-level confidence intervals, calibration/coverage metrics, label-schema checks, temporal/resolution rules, and same-protocol learned baselines. |

## Proposed next tasks

### T14 — Real-product ingestion, canonical band schemas, and provenance manifests

**Output:** TACO/SEN2NAIP, SEN2Vénus/SAFE-or-COG, and Cartosat adapters with sample IDs, band maps, wavelengths, units, scale/offset, acquisition metadata, masks, and source checksums.

**Acceptance:** Representative fixtures load end-to-end; reordered, missing, QA/alpha, panchromatic, and unsupported channels are rejected or explicitly mapped; arrays and metadata match expected schemas.

### T15 — CRS/grid-aware pairing and mask propagation

**Depends on:** T14.

**Output:** Reference-grid reprojection, transform-preserving crops, documented registration policy, SCL/nodata/QA propagation, valid-mask-aware loss/metrics/inference, and patch provenance.

**Acceptance:** Tests cover differing CRS, affine origins, rotations, partial overlap, non-multiple/subpixel shifts, mixed-resolution masks, and edge coverage.

### T16 — Scene-level splits and semantic evaluation protocol

**Depends on:** T14 and T15.

**Output:** Persisted train/validation/test scene splits, validation-driven checkpoint selection, blind-test evaluation, per-scene metrics with confidence intervals, uncertainty calibration, label-schema checks, and same-protocol baselines.

**Acceptance:** The CLI uses disjoint persisted IDs; best checkpoints use validation loss only; incompatible WorldCover label schemas fail fast; metrics report registered/native definitions and per-scene uncertainty.

### T17 — Safe, metadata-driven, bounded inference

**Depends on:** T14.

**Output:** Restricted/tensor-only checkpoint loading, architecture/resource validation, persisted normalization and unit transforms, finite/range/size preflight, tiled streaming inference, correct variance scaling, and metadata/mask-preserving atomic exports.

**Acceptance:** Malicious, oversized, malformed, and non-finite inputs are rejected before allocation; valid checkpoints round-trip on CPU; tiled output matches a reference within tolerance; output artifacts are not partial after injected write failure.

### T18 — Auditable, resumable experiments

**Depends on:** T14 and T16.

**Output:** Locked environment/constraints, deterministic mode and RNG persistence, resolved config/run manifest, dropped-pair and dataset hashes, structured logs, and atomic resume-capable checkpoints.

**Acceptance:** Interrupted and uninterrupted runs match under deterministic settings; repeated CLI runs record identical manifest/config hashes and documented-equivalent metrics; evaluation refuses missing provenance.

## What was deliberately excluded

The audit did not count already-documented limitations as newly discovered bugs merely because they remain open. It also excluded environment-only observations such as missing `pytest` or `rasterio` in a particular sandbox. Those affect local execution, but they are not repository-design gaps by themselves.

## Recommended order

Implement **T14 → T15 → T16**, while implementing **T17 in parallel after T14**. T18 should follow once the real-product and evaluation contracts are stable. Until at least T14–T16 are complete, claims about India-domain generalization, numerical superiority, or calibrated uncertainty should be framed as unverified prototype hypotheses.
