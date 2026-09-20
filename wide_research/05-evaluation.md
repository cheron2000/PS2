# Evaluation and Reproducibility Audit

**Track:** Evaluation and reproducibility audit  
**Repository:** `/home/ubuntu/PS2`  
**Audit focus:** `metrics.py`, `eval_downstream.py`, `scene_protocol.py`, training and inference paths, tests, logs, documentation, and external evaluation practice.

## Executive conclusion

The repository is a **well-commented prototype scaffold with useful evaluation primitives**, but it does not yet contain a completed scientific evaluation of Sentinel-2 super-resolution. The implementation can compute several full-reference image metrics, compare an SR image with bicubic and optional HR through a simple downstream classifier, create scene-level split manifests, and calculate bootstrap confidence intervals and basic Gaussian interval coverage. Those are meaningful building blocks. They are not evidence that the proposed model improves reconstruction quality, downstream remote-sensing utility, or calibrated uncertainty on real imagery.

The most important distinction is between **implemented code**, **planned protocol**, and **reported evidence**. The code implements synthetic-array utilities and tests. `solution-draft.md` describes a stronger protocol involving SEN2NAIP, SEN2VENµS, WorldCover, OpenSR-style metrics, downstream tasks, uncertainty calibration, and external baselines, but the repository contains no completed real-data run, metric table, per-scene result artifact, baseline comparison, ablation result, or calibration plot. The project statement itself requires validation against high-resolution references and explicitly calls for analytical utility and uncertainty management (`problem-statement.md:18-23`). Those requirements remain unverified.

The highest-risk scientific issue is **evaluation validity under spatial, cross-sensor, and registration dependence**. The current CLI training path creates an index-level random train/validation split, not a persisted scene/AOI split (`src/train.py:191-220`, `src/train.py:250-272`). The separate scene protocol is useful but is not wired into the training CLI or an end-to-end real-data evaluator. Remote-sensing observations are spatially autocorrelated, so random patch splitting can make validation optimistic and can select models that fail on new locations [3]. Cross-sensor pairs also have residual registration and radiometric differences; a single integer-shift NCC check does not establish pixelwise correspondence.

The second major issue is **metric incompleteness and aggregation**. `src/metrics.py` reports PSNR, a custom uniform-window SSIM, mean spectral-angle error, and an alignment diagnostic. These metrics are not sufficient to establish spectral consistency, hallucination control, or task utility. They are global aggregates rather than scene-stratified results, and the alignment search is reported but not used to produce a separate registered score. The OpenSR benchmark argues that conventional PSNR/SSIM/LPIPS-style metrics are inadequate for real-world remote-sensing SR under luminance changes and spatial misalignment, and separates consistency, synthesis, and correctness metrics [1]. More general SR research also documents disagreement between full-reference metrics and perceptual quality, including sensitivity to reference-image quality [2].

The third major issue is **uncertainty claims without calibration evidence**. The model emits a heteroscedastic variance and `scene_protocol.py` computes Gaussian NLL, standardized squared error, and marginal interval coverage. However, no real held-out predictions are evaluated, no reliability diagram or ENCE-style binning is produced, no sharpness/dispersion analysis is reported, and no calibration is compared across scenes, land-cover classes, bands, or error regimes. Regression calibration literature recommends comparing predicted uncertainty with observed error in uncertainty bins and reporting a reliability diagram plus calibration scores such as ENCE [5]. A variance tensor being finite, non-negative, or mathematically well-scaled is not evidence that it is calibrated.

## What is actually implemented

### Image-fidelity metrics

`src/metrics.py` implements:

- mean squared error and PSNR with a caller-supplied `data_range` (`src/metrics.py:42-54`);
- a channelwise SSIM-like score using a reflected-border, uniform box window (`src/metrics.py:56-109`);
- mean per-pixel spectral-angle error, in degrees by default (`src/metrics.py:111-135`);
- integer-pixel normalized cross-correlation alignment search with best shift and overlap diagnostics (`src/metrics.py:137-186`);
- a convenience `metric_report` returning PSNR, SSIM, SAM, and alignment fields (`src/metrics.py:188-198`).

The module has a useful finite-value and shape contract (`src/metrics.py:16-27`), explicit mask handling, and a smoke test covering identity, a known shift, noise, and finite outputs (`src/metrics.py:200-223`). These are implementation strengths, not experimental results.

The limitations are material. `metric_report` does not report ERGAS, reflectance consistency, spectral-distance alternatives, high-frequency synthesis, hallucination, omission, improvement, perceptual quality, or no-reference quality. The alignment result is a **diagnostic**: the function finds a best integer shift, but `metric_report` computes PSNR, SSIM, and SAM on the original unregistered arrays (`src/metrics.py:188-198`). Consequently, a high alignment score does not mean that the reported fidelity metrics are registration-corrected, and a low score is not converted into an exclusion or uncertainty flag. The code also does not provide native-grid versus registered-grid scores, which would be important for distinguishing true reconstruction error from co-registration error.

The SSIM implementation is a project-specific variant rather than an explicitly documented reproduction of a standard implementation. It uses a uniform filter, reflected borders, constants based on the supplied data range, and no sample-covariance correction (`src/metrics.py:56-80`). This can be valid as a defined metric, but comparisons with published SSIM numbers require matching window, border, channel aggregation, dynamic range, scaling, and masking conventions. The caller can provide an arbitrary `data_range`; there is no recorded range in the output manifest. The valid-mask implementation retains only windows that are entirely valid (`src/metrics.py:90-101`), which changes the estimand near clouds, nodata, and scene edges and can make different scenes incomparable if valid-window fractions are not reported.

SAM excludes pixels with zero prediction or target spectral norm (`src/metrics.py:127-131`). That is numerically sensible, but the excluded fraction is not reported. All valid pixels are pooled, so large homogeneous scenes dominate a global mean. No per-band or per-scene weighting is available.

### Downstream utility

`src/eval_downstream.py` implements a confusion matrix, per-class IoU, overall accuracy, a three-class NDVI threshold classifier, and a side-by-side SR/bicubic/optional-HR comparison (`src/eval_downstream.py:30-135`). Tests cover perfect predictions, range checks, missing classes, basic NDVI behavior, shape mismatch, and a synthetic case in which SR beats bicubic (`test_eval_downstream.py:20-157`).

This is a useful harness, but it is not yet a valid remote-sensing downstream benchmark. The default classifier maps every pixel to **water, non-vegetated, or vegetation** using fixed thresholds (`src/eval_downstream.py:65-87`). That ontology is not equivalent to the 11-class ESA WorldCover ontology mentioned in the project materials. A three-class NDVI proxy can be a diagnostic, but it cannot support a claim of WorldCover-class improvement without an explicit label remapping and a task-specific validation design. The repository has a schema validator for integer labels and a configurable class count (`src/scene_protocol.py:145-158`), but schema-range validation is not semantic validation.

`compare_downstream_utility` applies the same externally supplied classifier to SR, bicubic, and HR (`src/eval_downstream.py:90-135`). This is appropriate for a fixed analytic rule, but it is not the same as training a downstream model fairly on each input condition or testing transfer from LR/SR/HR. There is no learned classifier, fixed train/validation/test task split, tuning protocol, class-imbalance treatment, calibration of class probabilities, macro/micro averaging policy, or statistical test of the SR-minus-bicubic difference. A global mean IoU difference is particularly vulnerable to scene composition and class prevalence.

There is also an edge-case weakness: `confusion_matrix` calls `min()` and `max()` on flattened labels (`src/eval_downstream.py:30-42`), so an empty valid array raises an incidental reduction error rather than a clear evaluation error. This is minor compared with the protocol gaps, but it should be covered before batch evaluation.

### Scene-level protocol

`src/scene_protocol.py` is the strongest recent addition. It can create deterministic scene-disjoint train/validation/test manifests (`src/scene_protocol.py:24-65`), validate scene and sample leakage and complete sample coverage (`src/scene_protocol.py:68-92`), save manifests atomically (`src/scene_protocol.py:94-109`), compute bootstrap intervals over per-scene scores (`src/scene_protocol.py:112-142`), validate a declared integer label schema (`src/scene_protocol.py:145-158`), evaluate per-scene macro IoU for SR/bicubic/HR (`src/scene_protocol.py:161-206`), and compute Gaussian NLL, standardized squared error, and empirical interval coverage (`src/scene_protocol.py:209-242`).

These functions are **available**, not evidence that the repository has run the protocol on real data. They are imported as public exports from `src/eval_downstream.py:138-151`, but there is no CLI path that builds or consumes a scene manifest, loads authoritative scene/AOI metadata, generates per-scene SR predictions, and writes a complete result artifact. The protocol also bootstraps the mean over scenes but does not provide a paired bootstrap for the SR-minus-bicubic contrast, a permutation test, a hierarchical model, or correction for multiple downstream tasks and metrics. With few scenes, percentile bootstrap intervals can be unstable; the report must include the number of independent scenes, not just the number of patches or pixels.

`uncertainty_calibration` assumes independent Gaussian per-element predictive distributions and evaluates pooled elements (`src/scene_protocol.py:209-242`). It does not estimate epistemic uncertainty, account for spatial correlation, or report calibration conditional on predicted standard deviation. It also has no scene-level aggregation, no per-band breakdown, no sharpness, and no reliability diagram. The implementation is therefore a reasonable first diagnostic, not a calibration result.

### Training path

The training code has several good engineering choices. `set_seed` seeds Python, NumPy, and Torch (`src/train.py:31-38`), validation loss can drive best-checkpoint selection (`src/train.py:142-174`), history is written to JSON (`src/train.py:175-176`), and the CLI exposes a seed and validation fraction (`src/train.py:223-272`). The split helper uses a seeded Torch generator (`src/train.py:191-220`).

The scientific evaluation problem is that the split is over dataset indices after filesystem discovery, not over scenes, geographic regions, dates, or acquisition groups. The implementation comments explicitly acknowledge this limitation (`src/train.py:199-208`). The CLI does not create a blind test set and does not invoke `build_scene_splits`. If multiple patches from one scene occur in both train and validation, the result estimates interpolation among nearby correlated patches rather than geographic generalization. Spatial cross-validation research shows that random splits are misleading when observations are spatially autocorrelated and that block size should reflect the application and correlation range [3].

The included training integration test intentionally passes the **same loader** as both training and validation (`test_train.py:44-47`). That verifies that `fit()` can execute and write checkpoints, but it cannot validate generalization, checkpoint selection against an independent validation set, or absence of leakage. The test uses two synthetic scenes whose HR image is exactly nearest-neighbor upsampling of LR (`test_train.py:23-27`), which is a convenient shape fixture but not a realistic cross-sensor reconstruction problem.

Reproducibility is incomplete. The seed does not enable deterministic Torch backend behavior, does not record CUDA/device/library versions, and does not persist Python/NumPy/Torch RNG states. Dataset ordering depends on filesystem contents and `glob` discovery. The checkpoint contains model and optimizer state plus metrics, but not the resolved training configuration, normalization contract, dataset manifest/hash, code revision, dependency lock, or RNG state (`src/train.py:101-119`). Checkpoint and history writes are direct rather than transactional. There is no resume implementation. `requirements.txt` specifies only lower bounds (`requirements.txt:1-4`), so a clean environment is not pinned to a reproducible dependency set.

### Inference path

`src/infer.py` safely loads Torch checkpoints with `weights_only=True` (`src/infer.py:102-155`), checks input channel count (`src/infer.py:158-173`), exports NumPy or georeferenced GeoTIFF data, and correctly scales variance by the square of the reflectance divisor when the output is denormalized (`src/infer.py:280-295`). The geotransform test checks CRS propagation and the expected 4x pixel-size change (`test_infer.py:81-135`).

Inference still does not constitute an evaluated deployment path. It reads the complete input raster and performs one full-scene forward pass (`src/infer.py:48-82`, `src/infer.py:158-173`); there is no bounded tiling, overlap blending, seam test, memory budget, or large-scene benchmark. The CLI defaults to no input normalization (`src/infer.py:255-265`), while the SEN2NAIP dataset defaults to reflectance normalization (`src/datasets/sen2naip.py:160-168`, `src/datasets/sen2naip.py:240-251`). The output unit choice is CLI-controlled rather than enforced from checkpoint metadata. A result can therefore be numerically valid while being in the wrong physical unit system for evaluation.

The uncertainty output is a pointwise variance from a single forward pass. It is not validated against targets by the CLI, and there is no calibration or coverage report generated by inference. GeoTIFF export preserves the input CRS and scales the affine resolution, but it does not emit a provenance manifest containing source checksum, model hash, preprocessing parameters, band names, acquisition time, registration status, nodata mask, or output-unit semantics.

## Tests, logs, and documentation

The repository has substantial synthetic unit coverage for array contracts, band schemas, pairing arithmetic, WorldCover remapping, inference helpers, and scene-protocol arithmetic. The torch-free checks that were runnable in this audit passed: `test_eval_downstream.py` reported 12 passes, `test_audit_fixes.py` reported all checks passing, and `src/metrics.py` printed `metrics smoke test passed`. These results demonstrate basic function behavior only.

The full `pytest` command could not run in the audit environment because `pytest` is not installed. Direct execution of `test_infer.py` and `test_train.py` failed before tests began because `torch` is not installed in that environment. Direct execution of `tests/test_scene_protocol.py` failed because its import path assumes a test-runner/package context (`ModuleNotFoundError: No module named 'src'`). These are environment and harness observations, not claims that the underlying Torch code is incorrect. They do mean that the repository does not currently provide a single, verified, environment-independent test command in the audited environment.

The tests do not include real SEN2NAIP/TACO, SEN2VENµS, Cartosat, or WorldCover tiles. They do not test independent scene splits through the training CLI, metric behavior under realistic masks and partial overlap, registered versus native scoring, bandwise radiometric errors, tiling equivalence, downstream classifier training, calibration curves, or repeated-seed variance. There are no committed result tables, predictions, evaluation manifests, tensorboard/W&B runs, or machine-readable metric outputs. The logs and `external_gap_audit.md` correctly describe many of these limitations; documentation should not be mistaken for execution evidence.

## Missing baselines and ablations

A defensible SR study should compare at least the following under the **same input bands, preprocessing, registration policy, crop sizes, scene split, and output unit convention**:

1. **Nearest-neighbor and bicubic interpolation.** Bicubic is already present in the downstream comparison API, but there is no committed benchmark run. Both should be measured for image fidelity, spectral consistency, spatial alignment, and downstream utility.
2. **A deterministic learned reconstruction baseline.** SRCNN, EDSR/RCAN, or a similarly established CNN baseline should be trained and evaluated under the same data contract. A model-specific number copied from another paper is not a controlled baseline.
3. **A spectral-consistency or low-frequency-constrained baseline.** The benchmark literature includes explicit reflectance and spectral consistency checks; a low-frequency constraint or bandwise reconstruction baseline would test whether improvements are due to sharpening versus physically consistent reconstruction [1].
4. **Relevant published Sentinel-2 systems.** The solution draft names OpenSR/SEN2SR and diffusion systems, but the repository contains no adapters, reproduced checkpoints, or apples-to-apples result table. If external models cannot be run, report that as a limitation and separate published numbers from measurements made in this repository.
5. **Real HR upper reference and LR lower reference.** Where available, report real HR, bicubic/LR, and SR in the same downstream task. GeoSR-Bench explicitly evaluates original lower-resolution imagery, SR outputs, and original higher-resolution imagery and repeats downstream training across independent runs [6].

Required ablations are also missing. At minimum, isolate the heteroscedastic uncertainty head and NLL term, reconstruction loss choice, registration/QC filtering, normalization and unit conversion, input-band subset, model capacity, scale factor, and any perceptual/high-frequency loss. Each ablation should use the same split and compute budget. A single best run cannot identify which component caused an observed change.

## Statistical rigor that is missing

The current code reports means over pixels or scenes but no completed uncertainty around the primary SR metrics. The evaluation should:

- define the independent unit as scene/AOI or acquisition, not pixel or overlapping patch;
- report per-scene values before aggregation and include scene count and valid-pixel fraction;
- use paired scene-level bootstrap intervals for SR minus bicubic and SR minus HR, with a fixed, recorded seed;
- report repeated training seeds for learned baselines and separate training variance from test-scene sampling variance;
- predefine primary metrics and the direction of improvement;
- report class-balanced and prevalence-sensitive downstream metrics, including per-class IoU/F1, macro averages, and confusion matrices;
- use paired permutation or bootstrap tests for downstream differences rather than interpreting a single mean-IoU delta;
- disclose multiple datasets, scales, bands, metrics, and downstream tasks to avoid selective reporting;
- show failure cases and stratify by land-cover composition, cloud/valid fraction, registration quality, season, geography, and radiometric regime.

For uncertainty, report NLL, RMSE or MAE, predicted standard-deviation sharpness, standardized residual diagnostics, reliability plots by uncertainty bin, ENCE or a clearly defined equivalent, and empirical coverage at nominal 50%, 80%, 90%, and 95% levels. Calibration must be assessed on an untouched test set. Spatially correlated pixels should not be treated as millions of independent observations; scene-level or block-level resampling is more defensible.

## Threats to validity

**Spatial leakage.** Patch-level random splitting can put adjacent or overlapping information in train and validation. This inflates estimates and undermines geographic transfer. The scene manifest utility reduces this risk only when authoritative scene IDs are present and the training/evaluation path actually uses it.

**Cross-sensor registration error.** SEN2NAIP-style Sentinel-2/NAIP pairs are cross-sensor and may differ in viewing geometry, acquisition time, point-spread function, and radiometry. Integer NCC on one band cannot remove subpixel misregistration or establish spectral correspondence. A score may reward blur or common low-frequency structure while the pixelwise target remains shifted.

**Synthetic-to-real transfer.** Synthetic degradation can make the inverse problem easier and does not reproduce sensor-specific blur, noise, atmospheric effects, compression, or temporal change. The draft correctly states that synthetic data should not be treated as independent real validation (`solution-draft.md:28-34`), but no real-data result is present to resolve the threat.

**Reference quality and metric validity.** Full-reference metrics assume the HR reference is an adequate target. Recent SR work shows that reference quality can change metric values and model rankings [2]. Real HR and LR may not be temporally or spectrally identical, so a lower PSNR can reflect legitimate scene change rather than model failure.

**Hallucination and false detail.** A sharp output can invent structures absent from the LR observation. PSNR, SSIM, and SAM do not directly quantify whether high-frequency details are correct. OpenSR-test therefore separates consistency, synthesis, spatial alignment, and correctness/hallucination measures [1]. The repository currently has no hallucination/omission metric or qualitative error taxonomy.

**Downstream label mismatch.** WorldCover class codes, three-class NDVI thresholds, and any segmentation labels represent different ontologies and spatial supports. A downstream improvement is only interpretable if labels are aligned, masks are propagated, class mapping is explicit, and the classifier is evaluated on independent AOIs. Global land-cover datasets also emphasize the importance of label quality and geographic/biogeographic representation [7].

**Uncertainty misspecification.** The model's Gaussian, per-pixel heteroscedastic variance may not represent multimodal reconstruction uncertainty, epistemic uncertainty, spatial dependence, or systematic cross-sensor bias. Good NLL or marginal coverage alone would not establish useful uncertainty maps; conditional calibration and sharpness are needed [5].

**Selection and reproducibility bias.** Without locked dependencies, data manifests, configuration hashes, checkpoint provenance, repeated seeds, and blind-test outputs, another researcher cannot distinguish a robust effect from a favorable run or data ordering. The absence of a committed result artifact also prevents independent audit of claims.

## Recommended acceptance criteria

Before claiming that the solution improves SR or downstream utility, require the following end-to-end artifact set:

1. A versioned data manifest with source IDs, acquisition dates, scene/AOI IDs, band mapping, units, CRS/grid metadata, mask statistics, and source checksums.
2. A persisted scene-disjoint train/validation/test split, with a declared spatial blocking rationale and no patch overlap across splits.
3. A resolved run manifest containing code revision, dependency lock, hardware, seed, deterministic settings, model configuration, normalization, scale, metric conventions, and checkpoint hash.
4. Predictions and machine-readable metrics for every test scene, including native and registered scores, per-band scores, valid fractions, alignment diagnostics, and failure flags.
5. Results for bicubic, nearest-neighbor, at least one established learned baseline, the proposed model, and real HR where available, all under one protocol.
6. Ablation results isolating uncertainty loss/head, registration/QC, normalization, band selection, and perceptual or high-frequency terms.
7. Paired scene-level confidence intervals and repeated-seed results for learned methods. Report the number of scenes, not only pixels or patches.
8. A downstream protocol with an explicit label ontology, fixed classifier/training procedure, class-balanced metrics, confusion matrices, and SR-minus-bicubic paired uncertainty.
9. Held-out uncertainty calibration with NLL, coverage, reliability/ENCE diagnostics, sharpness, and stratification by scene, band, and error regime.
10. A reproducible command that runs all tests and a small fixture evaluation without network access, plus a documented command for the full real-data benchmark.

## Bottom line

The repository's **engineering primitives are stronger than its evidence base**. The metrics, scene manifest, downstream harness, and uncertainty calculations are reasonable starting points, and the existing documentation is unusually candid about unresolved data and geospatial issues. However, the central claims remain plans: there is no demonstrated real-scene superiority, no controlled baseline table, no ablation evidence, no independent downstream validation, and no calibrated uncertainty result. Until the scene-disjoint real-data protocol, controlled baselines, statistical reporting, and provenance artifacts are executed, the scientifically accurate description is **prototype evaluation infrastructure**, not a validated super-resolution system.

## References

[1]: https://github.com/ESAOpenSR/opensr-test "OpenSR-test: A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution"

[2]: https://arxiv.org/html/2503.13074v2 "Rethinking Image Evaluation in Super-Resolution"

[3]: https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1531097/full "Choosing blocks for spatial cross-validation: lessons from a marine remote-sensing application"

[4]: https://www.tensorflow.org/datasets/catalog/eurosat "TensorFlow Datasets: EuroSAT"

[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9330317/ "Evaluating and Calibrating Uncertainty Prediction in Regression Tasks"

[6]: https://arxiv.org/html/2605.00310v1 "Beyond Visual Fidelity: Benchmarking Super-Resolution for Remote Sensing Downstream Tasks"

[7]: https://www.nature.com/articles/s41597-023-02798-5 "A global land cover training dataset from 1984 to 2020"

## Evidence index

- **Problem requirements:** `problem-statement.md:18-23` requires useful fine-scale reconstruction, analytical utility, uncertainty management, and validation against HR references.
- **Implemented fidelity metrics:** `src/metrics.py:42-109`, `src/metrics.py:111-198`.
- **Alignment is diagnostic, not a registered score:** `src/metrics.py:159-198`.
- **Three-class NDVI proxy and global downstream comparison:** `src/eval_downstream.py:65-135`.
- **Scene protocol implementation:** `src/scene_protocol.py:24-109`, `src/scene_protocol.py:112-206`.
- **Basic uncertainty diagnostics only:** `src/scene_protocol.py:209-242`.
- **Index-level training split and explicit scope limitation:** `src/train.py:191-220`.
- **CLI integration does not consume scene manifests:** `src/train.py:223-272`.
- **Same-loader train/validation integration test:** `test_train.py:44-55`.
- **Training checkpoint/history contents:** `src/train.py:101-119`, `src/train.py:165-176`.
- **Training normalization default:** `src/datasets/sen2naip.py:160-168`, `src/datasets/sen2naip.py:240-251`.
- **Inference normalization and full-scene path:** `src/infer.py:48-93`, `src/infer.py:158-173`, `src/infer.py:255-303`.
- **Synthetic downstream tests:** `test_eval_downstream.py:20-157`.
- **Synthetic geospatial inference test:** `test_infer.py:81-163`.
- **Documented planned rather than completed evaluation:** `solution-draft.md:25-54`.
- **Documented external baselines without repository runs:** `solution-draft.md:56-60`.
- **Existing audit's matching conclusion:** `external_gap_audit.md:6-23`, `external_gap_audit.md:41-63`.
- **Dependency lower bounds only:** `requirements.txt:1-4`.
- **Audit execution observations:** torch-free tests ran successfully; `pytest` was unavailable; Torch-dependent tests could not start because Torch was unavailable; direct scene-protocol execution had an import-path failure. These observations were collected without modifying source code.

**Audit status:** No source code was modified. Only this report was created.
