# Executive conclusion

**Repository verdict: credible, candid prototype scaffold; not yet a validated end-to-end Sentinel-2 super-resolution solution.** The repository contains a meaningful four-band, 4× super-resolution model, a heteroscedastic mean/log-variance head, reconstruction plus Gaussian negative-log-likelihood loss, array-level pairing and quality checks, GeoTIFF export, scene/evaluation utilities, and unusually candid documentation of limitations. Those are real engineering foundations.

The evidence does **not** yet support the strongest project-level claims: official-format Sentinel-2/SEN2NAIP/SEN2Vénus/Cartosat ingestion; real-data training or a completed trained checkpoint result; Indian Cartosat validation; scene/AOI/date-disjoint generalization; calibrated uncertainty; demonstrated downstream benefit; robust full-scene operation; or superiority and novelty relative to established work such as ESA OpenSR. The current implementation is best described as an **evaluation-infrastructure and research-prototype baseline** whose intended differentiator—rigorous Indian cross-sensor validation plus useful, calibrated outputs—remains aspirational.

The most urgent correctness risk is product handling: the active SEN2NAIP path uses a local paired-NumPy convention rather than the official release format, and the analysts identify a likely unit mismatch in which Sentinel-2 is divided by 10,000 while NAIP is declared at 1/255. The next most consequential risk is scientific validity: `src/train.py` performs a seeded random sample-index split rather than consuming the available persisted scene/AOI protocol, so cropped samples can leak geography or scenes across partitions. These issues must be fixed before any real benchmark number is used to defend the solution.

The recommended decision is **conditional continuation, not acceptance as complete**. Freeze claims to “prototype plumbing and unvalidated research implementation”; then execute the P0 gates below: product-specific units and band order, official-format adapters and provenance manifests, mask-aware CRS/grid registration, mandatory scene-disjoint splits, reproducible baseline evaluation, held-out uncertainty calibration, and one real downstream task. Only after these gates pass should the repository claim Indian validation, calibrated uncertainty, analytical utility, or robust deployment.

## 1. Scope, method, and evidence standard

This report synthesizes six independent audits covering the problem statement, literature and novelty, datasets and sensors, code and systems, evaluation and reproducibility, and deployment readiness. It distinguishes four evidence states:

| Evidence state | Meaning in this report |
|---|---|
| **Implemented** | Present in repository code or documentation and supported by cited paths. It is not automatically scientifically validated. |
| **Partially implemented** | A reusable primitive or prototype exists, but important contracts, integrations, or tests are missing. |
| **Planned/aspirational** | Described in plans, solution materials, or task notes, but no executed evidence is present. |
| **Not evidenced** | The audits found no real run, artifact, result, or reproducible record supporting the claim. |

The assessment uses the analyst-provided repository evidence and cited external sources. It does not treat the existence of a function, test, or plan as evidence that a real-scene experiment succeeded. Where audit execution was blocked by missing dependencies, that is reported as a limitation rather than converted into a pass or fail.

## 2. Problem restatement and requirement coverage

The repository addresses SIH26142 as more than an image-to-image model task. The stated problem in `problem-statement.md:12-23` requires a pipeline that takes approximately 10 m Sentinel-2 imagery toward less-than-4 m output while preserving spectral and geospatial consistency, learns from paired high-resolution references, validates against high-resolution data, demonstrates analytical/downstream utility, and manages uncertainty explicitly.

That requirement is inherently a **data, sensor, geospatial, model, uncertainty, evaluation, and operations** problem. A visually plausible image or a high synthetic PSNR is not sufficient. The reference sensor, acquisition gap, spectral response, registration, masks, units, split policy, and downstream decision protocol determine whether a result is scientifically meaningful.

### Requirement coverage summary

| Requirement | Repository evidence today | Coverage | Consequence |
|---|---|---:|---|
| Four-band Sentinel-2-to-high-resolution SR model | `src/model.py:191-257` implements a four-channel model with 2×/4× output options, PixelShuffle reconstruction, mean output, and log-variance output. | **Partial / implemented prototype** | Architecture can be run on compatible arrays; it is not evidence of real-data performance or full Sentinel-2 coverage. |
| Spectral consistency | `src/model.py`, `src/losses.py`, and `src/metrics.py` provide channel-wise modeling and PSNR/SSIM/SAM-like diagnostics. | **Partial** | No explicit low-frequency observation constraint, calibrated sensor-response model, or complete spectral/radiometric conservation protocol is demonstrated. |
| Official Sentinel-2 and benchmark ingestion | `src/datasets/sen2naip.py`, `src/datasets/sen2venus.py`, and `src/datasets/cartosat_pairing.py` use local paired arrays and optional JSON; `src/datasets/sen2naip.py:8-20` documents unsupported official TACO format. | **Not met** | Claimed benchmark routes cannot be reproduced from official products without an unverified conversion step. |
| Correct product units and band identity | `src/datasets/band_schema.py:83-105` declares Sentinel-2 B/G/R/NIR and NAIP R/G/B/NIR; `src/preprocessing.py:58-76` and `src/datasets/sen2naip.py:160-168,235-243` apply normalization. | **Critical gap** | Channel order and the reported /10000 versus /255 scaling conflict can invalidate targets and metrics if upstream conversion is not guaranteed. |
| Geospatially valid pairing | NCC and shape checks exist; Cartosat has optional grid contracts and metadata. | **Partial** | Primary pairing is not consistently CRS/affine/bounds/subpixel aware; floor-dividing HR shifts can silently mishandle phase. |
| Quality masks and nodata/cloud handling | Finite-value and shape checks exist; WorldCover has categorical remapping and nearest-neighbor reprojection. | **Partial / disconnected** | Finite nodata or cloud pixels can remain valid; `src/datasets/sen2naip.py:235-255` returns no `valid_mask`. |
| Paired training | Training, checkpointing, and validation-loader wiring exist in `src/train.py:101-181`. | **Partial** | No completed real-data run or evidence of a valid independently held-out scene test. |
| Scene/AOI/date-disjoint generalization | `src/scene_protocol.py:24-109` and `:112-206` provide reusable manifests and leakage checks. | **Not met in active training path** | `src/train.py:191-220,242-260` uses random index splitting and does not persist or consume scene manifests. |
| High-resolution validation | Metrics and optional HR comparison utilities exist. | **Not evidenced** | No real HR reference result, controlled baseline table, or Indian held-out result is present. |
| Uncertainty | Mean/log-variance head, heteroscedastic NLL, and basic coverage diagnostics exist. | **Partial implementation; validation absent** | A variance parameterization is not calibrated uncertainty. No held-out NLL/coverage/reliability/sharpness artifact exists. |
| Downstream utility | `src/eval_downstream.py:65-135` compares SR/bicubic/optional HR with a three-class NDVI proxy. | **Harness only** | No real task result; three-class NDVI is not equivalent to 11-class WorldCover or the broader application claims. |
| Indian Cartosat validation | Cartosat pairing plumbing exists. | **Not met** | No delivered/verified pairs, independent AOIs, real registration/radiometric harmonization, or blind test result is evidenced. |
| Baseline and novelty comparison | Plans name bicubic, CNN, ESA OpenSR/SEN2SRLite and other systems. | **Not met** | No executed controlled benchmark or ablation; uncertainty-aware Sentinel-2 SR overlaps prior art including ESA OpenSR. |
| Robust full-scene inference | `src/infer.py:159-173` predicts a complete raster and writes output. | **Partial** | No bounded tiling, overlap blending, memory quota, atomic transaction, nodata propagation, resume, or seam verification is demonstrated. |
| Reproducibility and deployment contracts | Seeds, checkpoints, restricted loading, and history output exist. | **Partial** | Checkpoints omit resolved data/preprocessing/environment hashes and RNG states; dependencies are lower-bounded rather than locked. |

## 3. What is implemented today versus what remains aspirational

### Implemented today

The repository implements a credible **synthetic/plain-array prototype**. Specifically, it has:

- A four-band CNN/attention/windowed-transformer-style model with 2×/4× upsampling and separate mean and log-variance outputs (`src/model.py:191-257`). The attention windows are fixed and non-shifted (`src/model.py:95-149`).
- A heteroscedastic Gaussian NLL plus reconstruction loss (`src/losses.py:65-91,126-155`) with bounded log-variance behavior.
- Basic array validation for shapes, finite values, channel counts, band schemas, and some malformed-input negative paths.
- Local SEN2NAIP, SEN2Vénus, and Cartosat pairing conventions based primarily on `lr/*.npy`, `hr/*.npy`, and optional metadata files (`src/datasets/sen2naip.py:8-20,31-38`; `src/datasets/sen2venus.py:42-133`; `src/datasets/cartosat_pairing.py:238-283`).
- Integer NCC alignment and dropped-pair diagnostics, plus optional Cartosat grid/affine and mask contracts.
- GeoTIFF output handling and CRS/transform checks in the inference path (`src/infer.py:201-225`).
- PSNR, custom SSIM, SAM, NCC diagnostics, downstream confusion/IoU/accuracy helpers, scene manifest utilities, leakage checks, bootstrap summaries, label-schema checks, and basic Gaussian uncertainty functions (`src/metrics.py`; `src/scene_protocol.py:24-242`; `src/eval_downstream.py:30-135`).
- Validation-driven checkpoint selection and latest/best checkpoint/history artifacts (`src/train.py:101-181`).
- Documentation that openly states synthetic testing, unsupported official formats, missing real downloads, unverified GPU-scale behavior, and open Cartosat work (`README.md:57-83,93-100,116-141`; `SETUP_GUIDE.md:16-24,134-145`).

These are useful foundations. They should be preserved and connected to a stricter experimental contract rather than discarded.

### Aspirational or not yet evidenced

The following remain plans or claims without repository-level proof of completion:

- Official SEN2NAIP/SEN2NAIPv2 TACO, Sentinel-2, SEN2Vénus, Cartosat, and WorldCover ingestion with product metadata, masks, checksums, licenses, and reproducible conversion.
- Real-data training, a real trained checkpoint, and a real metric table.
- Indian Cartosat-2S/3 validation with independent, scene-disjoint AOIs and calibrated cross-sensor registration/radiometry.
- Calibrated predictive uncertainty, including held-out coverage, reliability, sharpness, and error-ranking evidence.
- Demonstrated downstream gain for crop, urban, change-detection, disaster-response, mapping, or WorldCover-like tasks.
- Superiority over bicubic, nearest, a simple supervised CNN, ESA OpenSR/SEN2SRLite, DSen2 or equivalent baselines, and competing methods.
- Complete Sentinel-2 13-band or broader product support; the current model is four-band-first.
- Bounded, overlap-tiled, memory-safe, atomic, resumable full-scene inference.
- Competition/deployment readiness, an app/server, or a production operational monitoring layer. `SETUP_GUIDE.md:16-24` explicitly says there is no app/server and that setup does not download imagery, train, predict, or configure Bhoonidhi access.

## 4. Strengths to retain

### 4.1 Technical and engineering strengths

The model and loss code are more substantial than a placeholder. The mean-plus-variance output and heteroscedastic objective provide a reasonable starting point for uncertainty-aware reconstruction. The model supports scale-2/scale-4 output and uses PixelShuffle reconstruction. The repository also includes finite-value, shape, schema, and range contracts, safer `weights_only=True` checkpoint loading, and metadata-aware GeoTIFF export.

The repository includes useful defensive primitives: malformed-array tests, duplicate-band checks, grid mismatch and missing-CRS checks, alignment diagnostics, and dropped-pair reasons. `src/scene_protocol.py` is particularly valuable because it already contains scene/sample manifest ideas, leakage checks, per-scene macro-IoU summaries, bootstrap confidence intervals, label-schema checks, and uncertainty diagnostics. The weakness is integration into the authoritative training/evaluation path, not absence of all building blocks.

### 4.2 Documentation and scientific honesty

The documentation is unusually candid. `README.md:77-83` states that tests are synthetic; `README.md:93-100` documents the local array layout; `README.md:121-140` lists real-format, GPU, and inference gaps. `external_gap_audit.md:6-23,41-71` independently identifies the same core risks. This honesty is a strength for review and should be retained in the final claim language.

### 4.3 Directionally appropriate benchmark ideas

SEN2NAIP/SEN2NAIPv2, SEN2Vénus/OpenSR-test, ESA WorldCover, and a carefully designed Cartosat route are directionally appropriate. External evidence supports the feasibility of 2.5 m SEN2NAIP-style evaluation and an Indian Cartosat route, but also confirms cross-sensor, geographic, temporal, spectral, and operational caveats. The strongest defensible future differentiator is not the existence of an uncertainty head; it is a completed, independent Indian validation study with honest uncertainty and downstream utility.

## 5. Critical gaps

### 5.1 Product ingestion, units, and wavelength identity — P0

The active loaders expect pre-converted arrays rather than official product packages. `src/datasets/sen2naip.py:8-20` explicitly documents the TACO-format gap. The analysts identify a serious normalization concern: `src/preprocessing.py:58-76` and `src/datasets/sen2naip.py:160-168,235-243` divide both sides by 10,000, while `src/datasets/band_schema.py:95-105` declares NAIP at 1/255 and Sentinel-2 at 1/10000. Unless an external conversion step is guaranteed and documented, the HR target scale is inconsistent.

The band schemas also differ: Sentinel-2 is declared blue/green/red/NIR while NAIP is red/green/blue/NIR (`src/datasets/band_schema.py:83-105`). Channel-wise comparison without explicit canonical reordering can turn a nominally good model into a physically invalid one. Arrays without metadata cannot establish wavelength identity.

**Required acceptance condition:** each source has a product-specific scale/offset and band map derived from metadata; known-value fixtures prove normalization; channel order is canonicalized explicitly; raw 0–255 NAIP values divided by 10,000 fail a test; units and band maps are persisted in each scene manifest and checkpoint.

### 5.2 Official formats and provenance — P0

No official SEN2NAIP/SEN2NAIPv2 TACO reader, published SEN2Vénus tensor/index reader, or verified Cartosat product adapter is evidenced. The repository cannot currently reproduce the claimed benchmark routes from source products. Provenance fields are not mandatory: acquisition dates, product IDs, processing baselines, checksums, licenses, cloud/SCL/QA masks, footprints, and conversion versions can be absent.

**Required acceptance condition:** a checked-in conversion command or adapter ingests official artifacts into a versioned canonical format and writes immutable manifests with source IDs, checksums, dates, bands/wavelengths, scale/offset, CRS/affine/pixel size, masks, license, and conversion code revision.

### 5.3 Pairing, registration, and cross-sensor validity — P0/P1

Integer NCC alignment is useful as a diagnostic but insufficient as the primary geospatial contract. `src/datasets/sen2naip.py:81-143` uses first-band integer NCC and floor-divides HR shifts by scale. A 1–3 pixel phase shift at 4× can be silently mishandled. The primary path does not consistently enforce CRS, affine transform, bounds, subpixel displacement, rotation, footprint overlap, cloud/edge coverage, or local registration residuals.

Cartosat pairing has optional grid contracts, masks, and metadata (`src/datasets/cartosat_pairing.py:151-235`), but it can repeat a one-band PAN image into pseudo-bands or truncate extra channels (`:71-82`), and it retains a shape-only fallback. The analysts correctly recommend rejecting these cases by default. Cartosat PAN is not multispectral truth merely because it can be repeated into four channels.

**Required acceptance condition:** reproject to an explicitly declared reference grid, preserve native and registered transforms, use a validated subpixel/local registration method, report residuals and overlap masks, and reject ambiguous band mappings and PAN repeat/truncate behavior unless an explicit harmonization experiment is separately labeled.

### 5.4 Scene-disjoint experimental design — P0

`src/scene_protocol.py:24-109` and `:112-206` contain the foundations for deterministic scene manifests and leakage checks, but the active training path does not use them. `src/train.py:191-220,242-260` performs seeded random index splitting and explicitly acknowledges that it is not scene/AOI-aware or persisted. Because patches from one scene or nearby AOI can be split across partitions, validation can overstate geographic generalization.

**Required acceptance condition:** split complete scenes/sites/AOIs before cropping; persist one immutable train/validation/test manifest; validate scene, footprint, temporal, and source overlap; reserve blind Indian sites; report scene counts, spatial separation, date rules, and per-scene sample counts.

### 5.5 No completed real-data benchmark or controlled baselines — P0

There is no real-data benchmark table, trained checkpoint result, completed Cartosat result, ablation, repeated-seed result, or blind-scene result. The intended comparison set includes bicubic, nearest, a simple learned CNN, ESA OpenSR/SEN2SRLite, DSen2 or equivalent, but it has not been executed under a common protocol.

This is also a novelty risk. Uncertainty-aware Sentinel-2 SR, 10 m to approximately 2.5 m reconstruction, spectral consistency, and geospatial export overlap established work including ESA OpenSR, SEN2SR, DiffFuSR, and DSen2. `solution-draft.md:40-44` already acknowledges ESA OpenSR overlap. Novelty must therefore rest on a completed and independently credible Indian validation/downstream study, not on the variance head alone.

**Required acceptance condition:** run all baselines with identical bands, units, masks, registration policy, scene splits, compute budget, and reporting. Separate synthetic, cross-sensor, and Indian tracks. Report paired per-scene deltas and confidence intervals rather than a single pooled score.

### 5.6 Uncertainty is parameterized, not calibrated — P0/P1

The log-variance head and functions in `src/scene_protocol.py:209-242` provide a starting point, but there is no held-out real-scene result for NLL, standardized residuals, empirical 50/80/90/95% coverage, reliability/ENCE curves, sharpness, or error-ranking. No epistemic baseline such as an ensemble or conformal method is present. A model that outputs a variance map has not demonstrated calibrated uncertainty.

**Required acceptance condition:** evaluate an untouched test set, stratify by scene, band, land cover, valid fraction, and error regime, report nominal versus empirical coverage, interval width/sharpness, NLL, standardized error, reliability, and the relationship between predicted uncertainty and observed error. State a decision policy for when uncertainty triggers rejection or review.

### 5.7 Downstream evidence is a harness, not utility proof — P0/P1

`src/eval_downstream.py:65-135` contains an NDVI-threshold three-class proxy and compares SR, bicubic, and optional HR inputs under a caller-supplied classifier. This is useful smoke-test infrastructure, but it is not equivalent to ESA WorldCover's 11-class ontology and is not proof for crop, urban, change-detection, disaster-response, or mapping applications. WorldCover is a 10 m Sentinel-derived, imperfect label product and cannot be treated as independent 2.5 m truth.

**Required acceptance condition:** choose one task, declare the label ontology, train the classifier only on an allowed training partition, evaluate LR/bicubic/SR/reference on the same scenes, document resampling and temporal alignment, and report per-class metrics and paired scene-level SR-minus-bicubic intervals.

### 5.8 Whole-scene inference and transaction safety — P1

`src/infer.py:158-173,280-303` runs one full input tensor rather than bounded overlap-tiled inference. It lacks preflight pixel/memory quotas, halo/overlap blending, seam checks, streaming writes, atomic rename, interruption recovery, and explicit nodata/mask propagation. Output metadata is minimal. Inference defaults also differ from the dataset path: dataset normalization is enabled by default while inference defaults to no normalization (`src/datasets/sen2naip.py:160-168`; `src/infer.py:48-93,158-173`).

**Required acceptance condition:** implement bounded tiles with documented halo/blending behavior, preflight size checks, mask-aware processing, peak-memory/latency telemetry, atomic output transaction, resume support, and a tiling-equivalence test against small full-array fixtures.

## 6. Moderate gaps

### 6.1 Model scope and physical constraints

The current model is four-band-first and does not cover the broader 13-band Sentinel-2 product scope implied by a robust framework. The loss does not enforce a low-frequency observation constraint or explicit sensor-response conservation. Fixed, non-shifted windows (`src/model.py:124-149`) limit context and differ from planned shifted-window attention. The optional diffusion branch is not implemented.

These are secondary to data validity. A modest, scientifically controlled baseline with correct data is preferable to a larger architecture trained on invalid pairs.

### 6.2 Metrics are incomplete and registration semantics are ambiguous

`src/metrics.py:42-109,111-198` provides MSE/PSNR, custom uniform-window SSIM, SAM, and NCC diagnostics, but the metric report can calculate scores on unregistered arrays even after reporting a best NCC shift. Native and registered scores must be separated. Add ERGAS or an equivalent spectral consistency measure, reflectance range/low-frequency consistency, valid fraction, per-band scores, and OpenSR-style synthesis/correctness/hallucination/omission metrics.

### 6.3 DataLoader safety and batch behavior

Variable post-crop shapes may make batch sizes above one unsafe under default PyTorch collation (`src/train.py:184-188`). The code should either enforce batch size one, pad/collate with masks, or guarantee fixed dimensions after every adapter.

### 6.4 Checkpoint and run provenance

`src/train.py:101-120` stores model/optimizer/metrics/config, but not a fully resolved dataset manifest/hash, preprocessing contract, code revision, dependency versions, hardware, deterministic backend settings, or RNG states. There is no complete resume path. `requirements.txt` uses lower bounds and does not supply a lockfile.

### 6.5 Test and packaging coverage

`setup.py:198-209` discovers only root `test_*.py` files, omitting `tests/test_scene_protocol.py`. Audit execution found that AST compilation and NumPy-heavy tests passed, while `test_infer.py`, `test_model_runtime.py`, `test_train.py`, and `test_worldcover.py` were blocked by missing `torch` or `rasterio`; direct scene-protocol execution had an import-path failure. This is not evidence that the tests fail logically, but it does mean README-level all-tests claims are not reproducible in the audit environment.

### 6.6 Operational error handling and monitoring

Failure handling is local. There is no structured error taxonomy, quarantine policy, progress/resume state, fault-injection coverage, cleanup guarantee, or independent-batch continuation contract. Monitoring is limited to printed shapes/losses; there are no standard operational metrics for throughput, peak memory, mask fraction, registration residual, uncertainty calibration, or domain shift.

## 7. Nice-to-have improvements

The following should follow the P0/P1 validity work:

1. Add full Sentinel-2 band/product support with explicit spatial-resolution harmonization and sensor-response metadata.
2. Compare shifted-window attention, cross-stage fusion, variance attention, loss weights, degradation models, and model capacity using fixed budgets and identical manifests.
3. Add perceptual/high-frequency terms only after proving they do not increase hallucination or spectral error.
4. Add an optional diffusion or ensemble uncertainty route and compare it to the current aleatoric head.
5. Add dashboard-ready visualizations: LR, bicubic, SR mean, uncertainty, valid mask, reference, native difference, registered difference, and failure cases.
6. Add network-free fixtures and a single canonical CI command that runs every nested test, including import-path checks.
7. Add resume-capable distributed/GPU training only after CPU and small-fixture equivalence is established.
8. Add model cards and dataset cards documenting intended use, prohibited use, geographic limits, licensing, sensor mismatch, and known hallucination modes.

## 8. Prioritized remediation roadmap

### P0 — Make the data and claims scientifically valid

| Order | Concrete repository action | Primary paths/artifacts | Exit criterion |
|---:|---|---|---|
| 1 | Build a canonical product adapter/conversion layer for official SEN2NAIP/SEN2NAIPv2 TACO, Sentinel-2, SEN2Vénus, and the declared Cartosat route. Persist source IDs, product versions, dates, checksums, licenses, band maps, wavelengths, scale/offset, units, CRS/affine/pixel size, QA/SCL/nodata masks, footprints, and conversion revision. | `src/datasets/sen2naip.py`, `src/datasets/sen2venus.py`, `src/datasets/cartosat_pairing.py`; add immutable scene manifests and a conversion CLI. | At least one official-format fixture per route can be converted and reloaded without manual undocumented steps; manifest hashes reproduce the same arrays. |
| 2 | Correct and test normalization and canonical band order. Refuse ambiguous wavelength identity and raw NAIP scale misuse. | `src/preprocessing.py:58-76`; `src/datasets/band_schema.py:83-105`; `src/datasets/sen2naip.py:160-168,235-243`. | Known-value unit tests pass; a 0–255 NAIP fixture is not divided by 10,000; B/G/R/NIR and R/G/B/NIR mappings are explicit in metadata. |
| 3 | Make mask-aware processing mandatory. Propagate cloud/SCL/QA/nodata/overlap masks through normalization, training loss, metrics, inference, and output sidecars. | `src/datasets/*`, `src/losses.py`, `src/metrics.py`, `src/infer.py`. | Invalid pixels cannot affect loss or metrics; output contains the valid mask and valid fraction; empty/fully invalid scenes fail clearly. |
| 4 | Replace shape/NCC-only pairing with a declared reference-grid protocol. Add CRS/affine/bounds checks, subpixel/local registration, residual reporting, phase rejection, and native-versus-registered status. Reject Cartosat PAN repetition/truncation by default. | `src/datasets/sen2naip.py:81-143`; `src/datasets/cartosat_pairing.py:71-97,151-235`. | Every retained pair has an auditable grid, transform, residual, overlap mask, temporal rule, and band harmonization record. |
| 5 | Make persisted scene/AOI/date-disjoint manifests the only supported train/evaluation route. Split before crop extraction and hold out blind Indian sites. | `src/scene_protocol.py:24-109,112-206`; refactor `src/train.py:191-220,223-272`. | Training logs manifest hash; leakage checks pass; test scenes/sites never appear in train or validation; per-scene counts are published. |
| 6 | Freeze an evaluation contract before tuning: datasets/tracks, splits, baselines, masks, registration semantics, metrics, bootstrap unit, minimum improvement, uncertainty targets, and downstream task. | `problem-statement.md`, `solution-draft.md`, new machine-readable evaluation config. | A reviewer can run one command and obtain the same protocol without choosing hidden defaults. |

### P1 — Demonstrate performance, uncertainty, and utility

| Order | Concrete repository action | Primary paths/artifacts | Exit criterion |
|---:|---|---|---|
| 7 | Run nearest, bilinear/bicubic, a simple supervised CNN, the proposed model, and at least one established system such as ESA OpenSR/SEN2SRLite or a documented equivalent under identical conditions. Separate synthetic, cross-sensor, and Indian tracks. | `src/model.py`, `src/metrics.py`, new baseline/evaluation CLI; external references [1]-[7]. | Machine-readable per-scene/per-band results, paired deltas, confidence intervals, compute/resource fields, and registered/native labels exist for every model. |
| 8 | Add complete SR metrics: PSNR/SSIM/SAM plus ERGAS or equivalent, reflectance/low-frequency consistency, valid fraction, synthesis/correctness/hallucination/omission, and per-band/per-scene reports. | `src/metrics.py`. | Scores cannot silently mix registered and native arrays; all metrics state masks, units, grid, and reference status. |
| 9 | Calibrate uncertainty on untouched real scenes. Add NLL, standardized error, 50/80/90/95% coverage, interval width/sharpness, reliability/ENCE plots, error-ranking curves, scene/band/land-cover strata, and ensemble/conformal comparison where feasible. | `src/scene_protocol.py:209-242`; new calibration report. | Coverage and calibration plots are produced from the blind test set; a documented threshold/review policy is stated. |
| 10 | Select one real downstream task with an explicit ontology and fixed classifier. Compare LR, bicubic, SR, and reference/HR on identical scenes and report paired scene-level deltas. | `src/eval_downstream.py:65-135`; `src/scene_protocol.py:161-206`; WorldCover adapter if used. | A real task report, not only the NDVI proxy, shows whether SR changes task performance and where it fails. |
| 11 | Acquire and validate multiple independent Indian Cartosat-2S/3 AOIs. Document temporal gap, orthorectification, sensor response, radiometric harmonization, registration, licenses, and blind-site holdout. | `src/datasets/cartosat_pairing.py`; new Cartosat manifest/QC report. | Indian results are based on actual independent pairs, not pre-arranged arrays or repeated PAN channels; all caveats are reported. |

### P2 — Make experiments and inference reproducible and operationally safe

| Order | Concrete repository action | Primary paths/artifacts | Exit criterion |
|---:|---|---|---|
| 12 | Implement bounded overlap-tiled inference with halo blending, preflight quotas, mask propagation, streaming/atomic writes, retry/resume, seam checks, and peak-resource telemetry. | `src/infer.py:158-225,241-308`; new tiling tests. | Tiled output matches a small full-array reference within a declared tolerance and oversized inputs fail before allocation. |
| 13 | Persist resolved configuration, dataset/config/model hashes, code revision, dependency lock, hardware, seeds, deterministic settings, RNG states, preprocessing contract, and model hash; implement true `--resume`. | `src/train.py:31-38,101-120`; `requirements.txt`; checkpoint schema. | A run can be resumed or audited on another machine with no hidden preprocessing assumptions. |
| 14 | Fix test discovery/import paths and add network-free fixtures for loaders, masks, units, scene splits, registration, tiling, calibration, and independent train/validation/test execution. | `setup.py:198-209`, `tests/`, CI configuration. | One canonical test command runs the complete suite in a clean environment with pinned dependencies. |
| 15 | Add structured logs, quarantine and failure taxonomy, fault-injection tests, output cleanup guarantees, and monitoring for latency, memory, masks, registration residuals, uncertainty, and domain shift. | `src/infer.py`, training/evaluation entry points, new operational modules. | Interrupted, corrupt, oversized, and partially failing jobs leave no misleading final artifacts and produce actionable diagnostics. |

### P3 — Extend scope and presentation

After the validity gates pass, consider 13-band support, shifted windows, ablations, diffusion/ensemble models, broader tasks, deployment packaging, model/dataset cards, and a polished demo. Do not use these extensions to substitute for correct products, splits, or held-out evidence.

## 9. Claim/evidence matrix

| Current or implied claim | Evidence that exists | Evidence missing or contradictory | Safe wording now | Evidence required before stronger wording |
|---|---|---|---|---|
| “Uncertainty-aware 10 m-to-2.5 m SR is implemented.” | `src/model.py:191-257` has mean/log-variance outputs; `src/losses.py:65-91,126-155` has heteroscedastic NLL; 4× output is implemented. | No real calibration, no uncertainty result, and prior art overlaps this capability. | “The prototype implements a heteroscedastic uncertainty parameterization for four-band SR.” | Held-out real-scene NLL, coverage, sharpness, reliability, and error-ranking results; comparison to prior baselines. |
| “Real Indian Cartosat validation is a differentiator.” | `src/datasets/cartosat_pairing.py` contains pairing/harmonization plumbing; project materials plan the route. | No delivered real pairs, independent AOIs, blind test, calibrated registration/radiometry, or result. | “Indian Cartosat validation is planned and the repository contains preliminary pairing utilities.” | Multiple verified independent Indian pairs, scene-disjoint blind evaluation, full provenance, cross-sensor QC, and results. |
| “The repository supports SEN2NAIP.” | `src/datasets/sen2naip.py` reads local paired arrays, checks shapes/finite values, performs alignment diagnostics. | Official TACO format is unsupported; no real integration fixture; product-specific units/masks/provenance are not enforced. | “The repository supports a local pre-converted paired-array convention inspired by SEN2NAIP.” | Official-format adapter, conversion manifest, unit/band tests, masks, and real benchmark run. |
| “The model preserves spectral/geospatial consistency.” | Band schemas, shape/grid checks, GeoTIFF export, SAM/SSIM/PSNR, and optional Cartosat grid contracts exist. | Channel-order conflict, normalization risk, NCC-only primary alignment, no low-frequency/sensor-response constraint, and no real metric evidence. | “The prototype includes spectral/geospatial validation primitives; preservation is unverified on real data.” | Correct product contracts, reference-grid pairing, masks, native/registered metrics, spectral consistency results, and error analysis. |
| “The system generalizes geographically.” | Scene manifest and split utilities exist in `src/scene_protocol.py`. | Active `src/train.py` split is random by sample index and not persisted or scene/AOI-aware. | “Scene-disjoint evaluation infrastructure exists but is not the active training protocol.” | Mandatory persisted scene/site/date split, leakage report, blind sites, and per-scene confidence intervals. |
| “SR improves downstream applications.” | `src/eval_downstream.py` has a three-class NDVI proxy and SR/bicubic/HR comparison harness. | No real task result; proxy is not WorldCover's 11-class ontology and does not validate stated applications. | “A downstream smoke-test harness exists; utility is not demonstrated.” | One independently trained, real downstream task with paired scene-level deltas and confidence intervals. |
| “The method is superior to existing SR systems.” | Plans name bicubic, learned CNN, OpenSR/SEN2SRLite, DSen2 and related systems. | No completed controlled benchmark, ablation, repeated seeds, or resource report. | “Competitive comparison is planned; superiority is unsubstantiated.” | Common protocol, executed baselines, confidence intervals, ablations, and compute/resource comparison. |
| “The system supports robust full-scene inference.” | `src/infer.py` can process a full array and write `.npy`/GeoTIFF; CPU/GPU selection and restricted loading exist. | No bounded tiling, quotas, atomic writes, mask propagation, seam check, resume, or large-scene evidence. | “Full-array prototype inference and GeoTIFF export are implemented.” | Tiled equivalence, quota/preflight, resource benchmark, fault-injection, atomic transaction, and large-scene tests. |
| “The repository is reproducible.” | Seeds, checkpoints, history, tests, and some deterministic utilities exist. | No locked dependencies, manifest/config/code hashes, RNG persistence, true resume, or fully runnable suite in audit environment. | “The repository has partial reproducibility scaffolding.” | Immutable manifests, lockfile, run metadata, resume, complete CI, and independently rerunnable artifacts. |
| “The method is novel because it is uncertainty-aware Sentinel-2 SR.” | The project combines a model, loss, geospatial code, and intended application. | ESA OpenSR and other literature overlap uncertainty-aware or probabilistic SR, 2.5 m reconstruction, and related goals. | “Novelty, if any, must be established through completed Indian validation, calibration, and utility rather than uncertainty alone.” | Literature-grounded differentiation plus completed evidence on the declared unique protocol. |

## 10. Recommended acceptance gates

A review-ready release should not be marked complete until all of the following are true:

1. **Data gate:** official-format or fully documented conversion paths load at least one real fixture per claimed route, with checksums, units, band maps, CRS/affine metadata, masks, licensing, and product versions.
2. **Pairing gate:** every pair has declared temporal, spatial, spectral, grid, registration, overlap, and radiometric contracts; ambiguous PAN/repeated-band cases are rejected.
3. **Split gate:** train, validation, and blind test are persisted at scene/AOI/site/date level before cropping, with automated leakage checks and manifest hashes.
4. **Benchmark gate:** nearest/bicubic and at least one learned and one established external baseline run under identical conditions; results are per-scene, masked, and native-versus-registered explicit.
5. **Uncertainty gate:** held-out calibration report includes NLL, standardized residuals, coverage, sharpness, reliability/ENCE, and a stated decision policy.
6. **Utility gate:** one real downstream task uses a declared ontology and fixed classifier protocol; SR versus bicubic deltas are reported with paired scene-level intervals.
7. **Operations gate:** bounded tiled inference, mask propagation, preflight, atomic outputs, resume/failure behavior, and resource telemetry are tested.
8. **Reproducibility gate:** locked dependencies, code/data/config/model hashes, seeds/RNG states, resolved preprocessing, hardware, and a canonical CI command are recorded.
9. **Claims gate:** README, `SUMMARY.md`, `solution-draft.md`, `presentation/defense_script.md`, and demo materials label each statement as implemented, measured, planned, or unverified.

## 11. Limitations and confidence

This assessment is a repository and evidence audit, not a re-run of a complete real-data scientific campaign. The analysts reported that AST compilation and NumPy-heavy preprocessing, pairing, band-schema, geospatial, and downstream checks passed in the audit environment. They also reported that `test_infer.py`, `test_model_runtime.py`, `test_train.py`, and `test_worldcover.py` could not start because `torch` or `rasterio` were unavailable, that `pytest` was unavailable in at least one audit path, and that direct execution of `tests/test_scene_protocol.py` had an import-path failure. Therefore, implementation findings are high confidence where supported by direct source inspection, while runtime behavior outside the available dependencies is lower confidence.

The report does not independently verify the external datasets, license status, sensor acquisition records, Cartosat availability, or the cited literature. External claims are included only through the analyst-provided sources and are not treated as repository results. No performance number, confidence interval, calibration score, or downstream improvement is invented here. The absence of a result in the audit means “not evidenced in the assessed repository,” not proof that no private or uncommitted experiment exists.

Overall confidence is **high** for the central conclusion that the repository is a synthetic/plain-array prototype rather than a validated end-to-end system; **high** for the priority risks around official ingestion, normalization/band order, scene splits, missing real results, uncertainty calibration, and deployment safety; and **moderate** for details that require installing the missing runtime dependencies or inspecting uncommitted data and external conversion steps.

## References

[1] ESA OpenSR model repository: https://github.com/ESAOpenSR/opensr-model

[2] ESA OpenSR test benchmark: https://github.com/ESAOpenSR/opensr-test

[3] ESA OpenSR test documentation: https://esaopensr.github.io/opensr-test/

[4] SEN2NAIP dataset: https://huggingface.co/datasets/isp-uv-es/SEN2NAIP

[5] SEN2NAIPv2 dataset: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2

[6] SEN2NAIP publication: https://www.nature.com/articles/s41597-024-04214-y

[7] Copernicus Sentinel-2 data documentation: https://documentation.dataspace.copernicus.eu/Data/Sentinel2.html

[8] Copernicus Sentinel-2 data collection: https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2

[9] Copernicus Sentinel-2 products: https://sentiwiki.copernicus.eu/web/s2-products

[10] ESA WorldCover data access: https://esa-worldcover.org/en/data-access

[11] Cartosat-3 mission, EO Portal: https://www.eoportal.org/satellite-missions/cartosat-3

[12] ISRO Cartosat-3: https://www.isro.gov.in/Cartosat_3.html

[13] USGS National Agriculture Imagery Program archive: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip

[14] SEN2NAIP record: https://pmc.ncbi.nlm.nih.gov/articles/PMC11655869/

[15] Zenodo SEN2Vénus-related record: https://zenodo.org/records/6514159

[16] SEN2Vénus-related remote-sensing publication: https://www.mdpi.com/2072-4292/14/14/3281

[17] SEN2Vénus-related Frontiers publication: https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1531097/full

[18] DiffFuSR-related preprint: https://arxiv.org/html/2506.11764v1

[19] Remote-sensing super-resolution publication: https://www.sciencedirect.com/science/article/pii/S0034425725006261

[20] DSen2 publication: https://www.sciencedirect.com/science/article/abs/pii/S0924271618302636

[21] Remote-sensing SR publication: https://ieeexplore.ieee.org/abstract/document/9844267/

[22] IEEE remote-sensing paper PDF: https://ieeexplore.ieee.org/iel8/4609443/10766875/10887321.pdf

[23] PyTorch DataLoader documentation: https://pytorch.org/docs/stable/data.html

[24] PyTorch `torch.load` documentation: https://docs.pytorch.org/docs/stable/generated/torch.load.html

[25] PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html

[26] OpenSR-related preprint: https://arxiv.org/html/2503.13074v2

[27] EuroSAT dataset documentation: https://www.tensorflow.org/datasets/catalog/eurosat

[28] Remote-sensing dataset/publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC9330317/

[29] Remote-sensing publication: https://www.nature.com/articles/s41597-023-02798-5

[30] Additional preprint cited in the evaluation audit: https://arxiv.org/html/2605.00310v1

[31] Additional project-related source: https://www.sciencedirect.com/science/article/pii/S0034425725006261

[32] Additional external source cited by the problem audit: https://www.eoportal.org/satellite-missions/cartosat-3

[33] Additional external source cited by the datasets audit: https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1531097/full

[34] Additional external source cited by the literature audit: https://www.sciencedirect.com/science/article/abs/pii/S0924271618302636

[35] Additional external source cited by the literature audit: https://ieeexplore.ieee.org/abstract/document/9844267/
