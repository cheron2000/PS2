# Operational and Competition-Readiness Gaps

## Executive assessment

The repository is a **well-documented, synthetic-array prototype scaffold**, not yet a usable end-to-end super-resolution product or a competition-ready evidence package. It has meaningful engineering primitives: a CPU-executable model, an uncertainty head, explicit array-shape checks, alignment diagnostics, a training loop, georeferenced GeoTIFF export, a scene-level evaluation module, and a test suite built around synthetic fixtures. Those are real strengths.

The central gap is operational closure. The stated problem asks for a robust framework that ingests medium-resolution satellite imagery, preprocesses it, reconstructs finer imagery while preserving geographic and spectral consistency, supports analysis, manages uncertainty, and validates against high-resolution references (`problem-statement.md:18-23`). The current implementation instead requires a pre-converted `lr/<id>.npy` and `hr/<id>.npy` directory, does not download or convert official products, runs whole input rasters in memory, has no service or batch orchestration layer, and has not produced a real-data benchmark. The project documentation says this plainly: setup verifies synthetic tests but does not train or predict, and real data access remains a manual step (`SETUP_GUIDE.md:16-24`, `SETUP_GUIDE.md:134-145`).

For a technical review or hackathon, the solution can be presented credibly as **evaluation-aware prototype infrastructure**. It should not yet be presented as a validated India-domain super-resolution system, a calibrated uncertainty product, or a deployable operational pipeline. The highest-priority blockers are: (1) official-product ingestion and provenance, (2) CRS/grid-aware pairing and mask propagation, (3) bounded tiled inference with an explicit output contract, (4) a persisted scene-disjoint benchmark with real references, and (5) reproducible, monitored experiment artifacts. These are not cosmetic improvements; they determine whether a judge or user can run the stated workflow and trust its output.

## What the problem requires

The problem statement is broader than image sharpening. It identifies target uses including change detection, agriculture, land-cover mapping, disaster monitoring, and urban planning, then requires reconstruction that retains geographic and spectral consistency (`problem-statement.md:12-20`). The expected solution explicitly includes preprocessing, paired-data training, accuracy assessment, high-resolution validation, analytical utility, and uncertainty/error accounting (`problem-statement.md:22-23`).

That requirement creates an end-to-end contract:

1. **Input contract:** identify sensor/product, bands, units, masks, CRS, grid, acquisition metadata, and valid pixels.
2. **Pairing contract:** align low- and high-resolution observations on a declared reference grid and preserve provenance.
3. **Model contract:** persist architecture and preprocessing configuration, constrain resource use, and produce a defined mean/uncertainty representation.
4. **Output contract:** emit imagery, uncertainty, spatial metadata, masks, units, and a machine-readable manifest without partial artifacts.
5. **Evidence contract:** use scene/AOI-disjoint data, real references where claimed, appropriate baselines, downstream-task tests, and uncertainty calibration.
6. **Operational contract:** fail safely, expose progress and diagnostics, support repeatability/resume, and make resource requirements visible.

The repository implements parts of contracts 3 and 4 for small, pre-converted arrays. Contracts 1, 2, 5, and most of 6 remain incomplete.

## What is implemented versus what is claimed

| Area | Implemented fact | Claim or intended capability | Readiness judgment |
|---|---|---|---|
| Installation | `setup.py` creates a virtual environment, installs three broad lower-bound dependencies, and runs `test_*.py` scripts (`setup.py:183-220`). | A working project setup is implied by the setup guide. | **Prototype-ready**, but environment and CUDA reproducibility are not pinned. |
| Data | Loaders consume paired `.npy` arrays in a local `lr/` and `hr/` layout, with shape, finite-value, band-schema, and NCC checks (`src/datasets/sen2naip.py:146-230`). | README describes SEN2NAIP, SEN2Vénus, and Cartosat as real-data routes. | **Not real-product-ready**; README itself says official formats are not supported (`README.md:121-129`). |
| Model | CNN attention, fixed-window transformer, PixelShuffle upsampling, and mean/log-variance heads are implemented (`src/model.py:95-175`, `src/model.py:191-257`). | The model is framed as 10 m to 2.5 m Sentinel-2 SR. | **Runnable prototype**; scale is restricted to 2× or 4× and model quality is unverified. |
| Training | Seed calls, train/validation split option, validation-based best checkpoint selection, `last.pt`, `best.pt`, and `history.json` exist (`src/train.py:31-38`, `src/train.py:101-181`). | Reproducible training and resumability are implied by checkpoint language. | **Partly implemented**; no resume path, RNG-state persistence, environment lock, or persisted data manifest. |
| Inference | Safe tensor-oriented checkpoint load with `weights_only=True`, channel validation, CPU/GPU selection, mean and variance prediction, `.npy` or GeoTIFF output (`src/infer.py:102-173`, `src/infer.py:241-308`). | README describes a usable tile inference workflow (`README.md:102-112`). | **Small-tile ready, full-scene unsafe**; no tiling, quotas, streaming, or atomic transaction. |
| Geospatial output | CRS and affine transform are copied and scaled by the configured factor for GeoTIFF writing (`src/infer.py:181-225`). | “Preserve geospatial meaning” is a presentation claim. | **Partial**; metadata is minimal and input grid/pixel semantics are not validated. |
| Uncertainty | The model emits per-band log variance and inference converts it to variance; an evaluation helper computes Gaussian NLL and interval coverage (`src/infer.py:158-173`, `src/scene_protocol.py:209-242`). | Presentation says it exposes a calibrated uncertainty product (`presentation/defense_script.md:57-59`). | **Uncalibrated diagnostic**, not demonstrated calibrated uncertainty. |
| Downstream utility | A simple NDVI threshold classifier and SR-versus-bicubic IoU harness exist (`src/eval_downstream.py:65-135`). Scene-level summaries and bootstrap intervals are implemented (`src/scene_protocol.py:112-206`). | The presentation promises classification, crop, urban, and disaster utility (`presentation/defense_script.md:35-39`, `69-71`). | **Harness exists; evidence does not**. The classifier is a three-class proxy, not a validated application model. |
| Demo | There is a CLI and a defense script, but no server, UI, live-data fetch, or packaged demo artifact (`SETUP_GUIDE.md:16-24`). | Slides describe an analytical system rather than an architecture-only prototype (`presentation/defense_script.md:47-49`). | **Presentation-ready only if claims are narrowed**. |

## Prioritized operational gaps

### 1. Official product ingestion is the primary end-to-end blocker — critical

The stated workflow starts with Sentinel-2 imagery, but the main loader starts after a human or separate script has already converted data into paired NumPy arrays. It discovers `lr/*.npy`, requires matching `hr/*.npy`, and applies project-specific band checks (`src/datasets/sen2naip.py:146-230`). README explicitly says this is **not** the official SEN2NAIP release format and that no converter is built (`README.md:93-100`, `121-129`). The setup guide further states that no real Sentinel-2, NAIP, Cartosat, or WorldCover data are downloaded and that every test uses generated data (`README.md:77-83`, `SETUP_GUIDE.md:134-145`).

This is a competition risk because a judge cannot reproduce the claimed path from a public product. It is also a scientific risk: official Sentinel-2 products contain product XML, `manifest.safe`, granules, quality indicators, auxiliary data, and metadata in SAFE packaging [1]. A loader that accepts only arrays has no opportunity to validate processing baseline, acquisition time, tile, band wavelength, scale/offset, cloud/SCL status, or source checksum. A generic `.npy` file can therefore satisfy the shape contract while violating the semantic contract.

**Required closure:** implement adapters for the actual release formats used in the benchmark, or provide a checked-in conversion command and representative fixtures. The adapter should emit a versioned sample manifest containing source IDs and checksums, acquisition metadata, band map and wavelengths, units and scale/offset, CRS/transform, GSD, masks, and conversion version. Ambiguous, missing, reordered, or extra bands should fail fast rather than be silently mapped.

### 2. Pairing is not yet a geospatial registration contract — critical

The pairing code has useful shape and NCC checks, but the default SEN2NAIP path matches arrays and estimates shifts. The dataset contract does not require CRS, affine transform, bounds, GSD, or overlap metadata (`src/datasets/sen2naip.py:146-230`). The Cartosat adapter can accept optional grid metadata, but it still has a shape-first interface and uses a simple integer shift/crop flow (`src/datasets/cartosat_pairing.py:168-225`). This is insufficient for real products whose grids may differ in CRS, affine origin, rotation, pixel alignment, resolution, and partial overlap.

A shape-compatible pair can be geographically wrong. In particular, a low-resolution shift that is not an exact multiple of the scale factor can leave a subpixel phase error at the high-resolution grid. The resulting training target can reward registration artifacts and make PSNR or downstream scores misleading.

**Required closure:** declare a reference grid; reproject/resample each source onto that grid; preserve the exact output transform; record the registration method and residual error; propagate overlap and edge coverage. Add fixtures for differing CRS, affine origins, rotation, partial overlap, non-multiple shifts, and mixed-resolution masks. Report both native-grid and registered-grid metrics so a registration step is not hidden inside the score.

### 3. Invalid pixels and quality masks are not connected to the primary pipeline — high/critical

Finite-value checks reject NaN and infinity, but they do not reject finite nodata sentinels or cloud-contaminated pixels. The primary SEN2NAIP sample returned by `__getitem__` contains `lr`, `hr`, normalization statistics, and alignment metadata, but no `valid_mask` (`src/datasets/sen2naip.py:235-255`). The training loop can consume a mask if one is supplied (`src/train.py:47-53`), yet the actual primary dataset does not supply one. Consequently, the loss and metrics can treat invalid, cloud, edge, or no-overlap pixels as valid.

The output writer copies one nodata value into the GeoTIFF profile when present, but there is no output validity mask or provenance sidecar (`src/infer.py:201-225`). Sentinel-2 documentation identifies quality indicators such as defective-pixel masks and quality reports as part of the product [1]. Omitting those signals makes uncertainty and quality claims hard to interpret.

**Required closure:** carry SCL/QA/nodata masks from ingestion through pairing, training loss, metrics, inference, and output. Emit a valid-data mask and mask statistics. Track dropped samples and cropped edge fractions in the run manifest rather than only keeping an in-memory `dropped_pairs` list.

### 4. Full-scene inference can exceed memory and has no resource guard — critical for deployment

The inference path reads the entire raster into a NumPy array (`src/infer.py:67-82`), converts the entire array into a batch tensor (`src/infer.py:158-169`), runs one forward pass, and writes the whole output (`src/infer.py:273-303`). There is no tile size, overlap, blending rule, stream-to-disk path, pixel quota, byte quota, timeout, or VRAM estimate. A 10 m Sentinel-2 tile expanded 4× multiplies output pixels by 16, and the transformer attention cost is affected by the number of window tokens. The code is therefore appropriate for a synthetic patch or small tile, not an unrestricted satellite scene.

The current setup documentation confirms that GPU-scale training and real memory behavior have not been verified (`README.md:134-137`). The architecture itself is modest in parameter count, but parameter count is not a sufficient deployment bound: activation memory, attention windows, output arrays, and uncertainty arrays all matter.

**Required closure:** add preflight validation before allocation; require explicit maximum input pixels/bytes and output pixels/bytes; implement overlapped tiled inference with a documented border/blending policy; stream outputs through a temporary file; report peak CPU/GPU memory, latency, tile count, and throughput. The demo should show the same tiled path used for a real product, not a special small-array path.

### 5. Input and output contracts are underspecified and can produce semantically wrong artifacts — high

The CLI accepts a `.npy` array or a raster readable by rasterio. For NumPy input, optional JSON metadata may contain CRS and transform (`src/infer.py:48-82`). There is no mandatory product identifier, band order, unit, acquisition time, GSD, mask, or preprocessing version. `--input-normalization` defaults to `none`, while the training dataset defaults to reflectance normalization (`src/datasets/sen2naip.py:160-168`, `235-251`; `src/infer.py:85-93`, `255-265`). A user can therefore run a valid-looking inference with units inconsistent with training.

The checkpoint contains architecture and optimizer state but not a versioned preprocessing contract, data schema, band map, normalization statistics, or model provenance (`src/train.py:101-120`). The loader filters configuration keys and constructs a default model when a raw state dictionary lacks `model_config` (`src/infer.py:131-155`). That convenience can turn a malformed or incomplete checkpoint into an architecture mismatch rather than a clear contract error. The current security improvement—`weights_only=True`—is good, but PyTorch still warns never to load untrusted data and recommends restricted tensor loading [2]. Security is not the same as schema validation.

Output writing also lacks a sidecar manifest. A GeoTIFF receives CRS, scaled transform, dtype, and optional nodata, but not the source ID, original transform, scale, bands, units, normalization, model hash, uncertainty semantics, mask statistics, or software version (`src/infer.py:181-225`). Writes happen directly to the final path, so an interrupted write can leave a partial artifact.

**Required closure:** define and validate a JSON schema for input, checkpoint, and output. Persist preprocessing and unit transforms with the checkpoint and reject missing or incompatible metadata. Require explicit band maps and units. Emit an output manifest and mask. Write to a temporary path and atomically rename only after close and validation.

### 6. Training is not resumable or fully reproducible — high

The code seeds Python, NumPy, and Torch (`src/train.py:31-38`) and makes an index-based train/validation split when requested. However, the source itself notes that this is not a persisted scene/AOI-aware split and does not record file-level provenance (`src/train.py:191-220`). The checkpoint stores epoch, model state, optimizer state, metrics, and model configuration, but not RNG states, scheduler state, resolved CLI configuration, dependency versions, dataset hashes, dropped-pair list, or source manifest (`src/train.py:101-120`). There is no `--resume` implementation in the CLI.

PyTorch’s own reproducibility guidance cautions that identical seeds do not guarantee identical results across releases, platforms, or CPU versus GPU, and recommends explicit deterministic controls where needed [3]. The repository’s broad dependency ranges (`requirements.txt:1-4`) and absent lockfile therefore matter in a benchmark setting. A rerun may use different Torch, NumPy, rasterio, CUDA, or driver behavior while appearing to be the same experiment.

**Required closure:** add a resolved run manifest with Git commit, Python/Torch/CUDA versions, hardware, all arguments, data and model hashes, split manifest, dropped-pair report, and normalization contract. Persist Python/NumPy/Torch/CUDA RNG states and deterministic settings. Add atomic, resume-capable checkpoints and an interrupted-versus-uninterrupted equivalence test. Pin or lock the environment used for the reported numbers.

### 7. Evaluation infrastructure is ahead of the evidence, but not yet proof — high

The scene protocol can create scene-disjoint splits, validate leakage, calculate per-scene summaries, bootstrap confidence intervals, and compute Gaussian NLL and interval coverage (`src/scene_protocol.py:24-142`, `209-242`). This is a strong direction. The operational problem is that the training CLI still uses its own index split and does not consume the scene manifest (`src/train.py:223-272`). The evaluation helper is an API, not a completed benchmark run.

The downstream implementation uses a three-class NDVI threshold proxy—water, non-vegetated, vegetation—and compares SR, bicubic, and optional HR using IoU and accuracy (`src/eval_downstream.py:65-135`). It does not establish that the SR output improves a real target task such as 11-class WorldCover, crop-boundary mapping, change detection, or disaster assessment. A three-class proxy and an 11-class land-cover label schema are not interchangeable without an explicit ontology and resampling policy.

This matters because recent remote-sensing SR work finds that visual fidelity metrics do not reliably imply downstream performance and that spurious reconstructed detail can harm decisions [4]. The repository’s own presentation appropriately promises to measure utility rather than claim it (`presentation/defense_script.md:35-39`, `69-71`), but no reported real-scene table is present in the repository.

**Required closure:** make the persisted scene manifest a required input to train and evaluation CLIs; reserve a blind test set; report scene-level paired deltas and confidence intervals; state whether scores are native or registered; add a real downstream task with an explicit label ontology, class handling, resampling, and fixed classifier/training procedure. Treat uncertainty evaluation as a held-out calibration report, not merely an NLL/coverage function call.

### 8. Failure handling is local, not system-level — high

Individual functions raise useful errors for wrong shapes, missing files, invalid channels, unsupported scales, missing georeferencing, and malformed labels. Dataset constructors record dropped pairs in `dropped_pairs` (`src/datasets/sen2naip.py:178-230`). These are good unit-level behaviors.

At the system level, failures are not yet observable or recoverable. There is no structured log, error code taxonomy, retry policy for transient I/O, quarantine directory for rejected products, run manifest, progress checkpoint, or partial-output cleanup. A failed pair can be excluded while the caller receives no persisted accounting unless it inspects object state. A direct output write can leave an unusable file. A full-scene out-of-memory failure is not converted into a smaller-tile retry.

**Required closure:** define fatal versus recoverable errors. Persist rejected sample IDs and reasons. Add structured JSON logs with stage, scene, tile, elapsed time, memory, and error code. Use atomic output transactions and cleanup. For batch jobs, continue independent scenes while returning a nonzero summary status when any item fails. Add fault-injection tests for corrupt arrays, malformed metadata, out-of-memory preflight, and interrupted writes.

### 9. Monitoring and explainability are absent beyond printed diagnostics — high for operational use

The CLI prints one completion line containing input/output shapes, uncertainty presence, and device (`src/infer.py:305-308`). Training prints epoch losses (`src/train.py:177-180`) and writes a history JSON. Neither path emits latency, throughput, peak memory, tile-level failures, input distributions, mask fractions, uncertainty quantiles, or drift indicators.

The uncertainty head provides an interpretable numerical field—per-band variance—but the repository does not show calibration curves, reliability plots, error-versus-variance analysis, or thresholds that tell a user when an output should not be trusted. The presentation’s statement that the system provides “calibrated uncertainty” (`presentation/defense_script.md:57-59`) is therefore a claim, not an implemented result. Calibration should be measured on held-out data; interval coverage alone is not enough unless reported by scene, band, error regime, and sample count. The regression calibration literature emphasizes comparing predicted uncertainty with empirical error and coverage rather than treating a variance head as calibrated by construction [5].

Explainability is similarly limited. There is no per-pixel provenance linking a generated detail to input evidence, no registration residual map, no cloud/quality mask overlay, no input-versus-output spectral diagnostic, and no warning when the model is outside its trained scale, band schema, or domain. A useful demo should make uncertainty and invalid areas visible, not just show a sharper RGB image.

**Required closure:** log operational metrics and data-quality metrics for every run. Produce a report containing uncertainty quantiles, empirical coverage, error-stratified calibration, input mask fraction, registration residuals, and out-of-domain warnings. Add visual panels for input, SR, uncertainty, validity mask, and reference/bicubic difference. State clearly that uncertainty is predictive model variance, not a guarantee that every hallucinated detail has been detected.

### 10. Competition positioning overstates the current proof level — high reputational risk

The README headline says “validation against real Indian ground truth (ISRO Cartosat-2S/3)” (`README.md:1-6`), while the setup and README sections say real Cartosat access, real downloads, and real training have not been executed (`README.md:121-137`, `SETUP_GUIDE.md:134-145`). The defense script is more careful in places—it labels Cartosat as optional if access is secured (`presentation/defense_script.md:61-63`) and says differentiation “must be measured” (`presentation/defense_script.md:41-45`). Nonetheless, a judge reading the README headline could reasonably infer that the validation already exists.

The presentation also says the system produces enhanced imagery, preserves geospatial meaning, exposes uncertainty, tests real Indian transfer, and reports quality–utility–compute trade-offs (`presentation/defense_script.md:47-49`). Those are appropriate **acceptance goals**, but they are not demonstrated repository outcomes. The competition-ready version should distinguish three labels in every slide and report: **implemented**, **measured**, and **planned**.

**Required closure:** change the headline and demo language to “prototype pipeline and evaluation plan” until real evidence is committed. If the team secures Cartosat data, include the order/product identifiers, acquisition dates, processing level, pairing manifest, and blind-test results. Do not use a visual Indian AOI screenshot as evidence of India-domain generalization without a high-resolution reference and a predeclared metric protocol.

## Practical competition-readiness scorecard

| Review question | Current answer | Minimum credible demonstration |
|---|---|---|
| Can a reviewer install it? | Usually, assuming compatible Python and binary wheels; dependencies are broad lower bounds. | One locked environment or container, exact command, and clean-machine log. |
| Can a reviewer obtain and ingest the stated data? | No; official formats and portal access are outside the repository. | Public fixture plus official-format converter/adapter and provenance manifest. |
| Can it process an ordinary scene? | Not safely; it loads the full raster and performs one full forward pass. | Bounded tiled inference with memory/latency report and atomic GeoTIFF output. |
| Does output retain spatial meaning? | CRS and affine transform are copied/scaled for GeoTIFF. | Reference-grid contract, mask, band/unit metadata, registration diagnostics, and sidecar manifest. |
| Can results be reproduced? | Seeds and history exist; split, environment, RNG, and dataset provenance are incomplete. | Locked environment, persisted scene split, hashes, RNG state, and resume test. |
| Is SR better than bicubic for a real task? | No repository result table demonstrates this. | Scene-level paired test on real references with confidence intervals and a fixed downstream task. |
| Is uncertainty trustworthy? | Variance and basic coverage functions exist. | Held-out calibration plots/tables, coverage by scene/band/error regime, and abstention guidance. |
| What happens on bad input or failure? | Local exceptions and some pair dropping. | Preflight rejection, structured errors, quarantine, retry/continue policy, and no partial outputs. |
| Can a judge see the system rather than code? | CLI and scripts exist; no application/server is provided. | Reproducible demo command or UI using the same production path and showing masks/uncertainty. |

## Recommended implementation order

**First, close the data and geospatial contracts.** Build the official-product adapters, canonical band/unit schema, provenance manifest, reference-grid reprojection, registration residuals, and mask propagation. Without this, model and metric numbers are not attached to a trustworthy physical product.

**Second, make inference bounded and auditable.** Add finite/range/size preflight, model resource validation, persisted preprocessing metadata, tiled streaming inference, overlap blending, output masks, sidecar manifests, and atomic writes. Add CPU round-trip tests and an intentionally oversized-input rejection test.

**Third, bind training and evaluation to immutable scene manifests.** Use scene/AOI/date-disjoint train, validation, and blind test IDs. Persist the manifest and require it in the CLI. Select checkpoints from validation only. Report per-scene metrics, paired SR-minus-bicubic deltas, confidence intervals, and registered/native status.

**Fourth, produce one real downstream proof.** Choose one task with a declared ontology and fixed classifier or segmentation procedure. Use the same preprocessing for SR, bicubic, and real-HR where available. Report class-balanced results, confusion matrices, uncertainty, and failure cases. Do not generalize one land-cover result to all agriculture, urban, disaster, and change-detection uses.

**Fifth, add operational observability and a truthful demo.** Emit structured run manifests and logs, latency/throughput/peak-memory data, mask and uncertainty summaries, and a visual panel showing input, SR, uncertainty, mask, and reference. Mark every result as implemented/measured/planned. This will make the presentation stronger, not weaker, because the problem statement itself prioritizes scientific reliability and analytical utility.

## Bottom line

The repository has enough implemented structure to support a strong engineering demo, but it does not yet support the strongest headline claims. Its honest current description is: **a tested prototype of an uncertainty-aware Sentinel-2 SR model and evaluation scaffold that requires official-data ingestion, geospatial registration, bounded inference, and real benchmark execution before deployment or competition claims are validated**.

The strongest evidence-backed strengths are the explicit model/output shapes, safe restricted checkpoint loading, useful local validation errors, optional mask-aware loss interface, GeoTIFF transform handling, and the newer scene-level evaluation utilities. The most consequential gaps are not missing neural-network layers; they are missing product semantics, provenance, resource bounds, failure transactions, monitoring, and executed real-data evidence.

## References

[1]: https://sentiwiki.copernicus.eu/web/s2-products "Copernicus Sentinel-2 Products: SAFE structure, metadata, image data, and quality indicators"

[2]: https://docs.pytorch.org/docs/stable/generated/torch.load.html "PyTorch torch.load documentation and security warning"

[3]: https://docs.pytorch.org/docs/stable/notes/randomness.html "PyTorch Reproducibility and Deterministic Algorithms"

[4]: https://arxiv.org/html/2605.00310v1 "Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration"

[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9330317/ "Evaluating and Calibrating Uncertainty Prediction in Regression Tasks"

## Repository evidence index

- Problem requirements: `problem-statement.md:12-23`.
- Installation and explicit non-features: `SETUP_GUIDE.md:16-24`, `SETUP_GUIDE.md:134-145`.
- Synthetic-only tests and official-format gap: `README.md:57-83`, `README.md:93-100`, `README.md:116-141`.
- Model scale and output heads: `src/model.py:151-175`, `src/model.py:191-257`.
- Dataset shape/schema/QC assumptions: `src/datasets/sen2naip.py:146-255`.
- Training seeds, checkpoints, validation, and split limitation: `src/train.py:31-38`, `src/train.py:101-181`, `src/train.py:191-220`.
- Safe checkpoint load, prediction, normalization, and output writing: `src/infer.py:85-173`, `src/infer.py:181-225`, `src/infer.py:241-308`.
- Downstream proxy and metrics: `src/eval_downstream.py:65-135`.
- Scene splits, bootstrap summaries, and uncertainty diagnostics: `src/scene_protocol.py:24-142`, `src/scene_protocol.py:161-242`.
- Demo/claim language: `presentation/defense_script.md:29-71`.
- Existing repository audit, used as a cross-check rather than as independent external evidence: `external_gap_audit.md:6-23`, `external_gap_audit.md:27-71`.

No source code was modified; only this report was created at the requested path.
