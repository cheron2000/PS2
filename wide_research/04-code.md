# Code and systems implementation audit

## Overall conclusion

The repository contains a credible **prototype skeleton** for 4× Sentinel-2-to-NAIP super-resolution: a PyTorch model with a mean and uncertainty head, paired-array loading, alignment checks, preprocessing helpers, GeoTIFF export, downstream metric utilities, and a growing set of geospatial contracts. It is not yet an end-to-end solution to SIH26142. The central gap is not that the code lacks every component; it is that several components are only wired for a synthetic/plain-`.npy` convention, while the problem requires a scientifically reliable pipeline on real medium-resolution imagery with paired high-resolution validation, preserved spectral/geospatial meaning, uncertainty accounting, and downstream utility.

The highest-risk correctness issue is in the default training data path. The loader normalizes both Sentinel-2 and NAIP arrays with the same `reflectance` operation (`/10000` and clipping), even though the repository's own band schema declares NAIP values as DN with a `1/255` scale. The official SEN2NAIP example likewise divides LR by 10000 and HR by 255.[1] [2] Consequently, a normal 8-bit NAIP target is reduced to approximately 0–0.0255 while the Sentinel-2 input is 0–1. The model can still train and the synthetic tests can still pass, but the supervised target is in the wrong units. A second related risk is that the code declares Sentinel-2 bands as blue/green/red/NIR and NAIP bands as red/green/blue/NIR without an explicit reorder before channel-wise loss.

The second major gap is that the official SEN2NAIP release is not consumed. The loader explicitly supports only paired `.npy` files and states that the Hugging Face release is a TACO archive; no TACO-to-loader conversion or real-file integration test exists. The official dataset contains 2,851 real cross-sensor LR-HR pairs and a much larger synthetic component, with Sentinel-2 10 m and NAIP 2.5 m imagery.[1] [2] The implemented loader therefore verifies an internal test convention, not the stated benchmark.

The repository is candid about many of these gaps, which is a strength. However, setup and test orchestration can report a misleadingly healthy state: `setup.py` runs only root-level `test_*.py` files, omits `tests/test_scene_protocol.py`, and does not fail early with a clear dependency matrix for missing PyTorch/rasterio. In this sandbox, pure-NumPy tests passed, while model/inference/WorldCover tests failed at import because `torch` and `rasterio` were unavailable; `tests/test_scene_protocol.py` also failed when invoked as a script because the `src` package was not importable from that script path.

## What is actually implemented

The problem statement calls for a framework that preprocesses 10 m Sentinel-2 input, trains on paired data, produces finer-than-4 m output, preserves geospatial and spectral consistency, validates against high-resolution references, supports analytical applications, and manages uncertainty. The repository implements the following concrete subset:

* `src/model.py` defines a CNN stem, repeated high-order attention blocks, a non-shifted windowed `MultiheadAttention` block, feature fusion, and two PixelShuffle-based output heads. The default is four input/output channels and 4× output, producing 2.5 m-equivalent pixels from 10 m input (`src/model.py:191-257`).
* `src/losses.py` implements masked L1/L2 reconstruction plus a bounded heteroscedastic Gaussian NLL. The uncertainty head predicts log variance, and inference exponentiates a clamped log variance (`src/losses.py:94-155`; `src/infer.py:158-173`).
* `src/datasets/sen2naip.py` loads matching `lr/*.npy` and `hr/*.npy`, rejects malformed arrays, checks finite values, validates declared band schemas, estimates an integer shift by NCC, crops, and returns normalized arrays (`src/datasets/sen2naip.py:146-255`).
* `src/preprocessing.py` provides reflectance, percentile, and z-score normalization, validity masks, and patch extraction. Edge remainders are deliberately dropped rather than padded (`src/preprocessing.py:84-164`).
* `src/infer.py` accepts CHW `.npy` or rasterio-readable input and writes `.npy` or GeoTIFF output. The GeoTIFF path scales the affine transform by the SR factor and retains CRS/transform metadata (`src/infer.py:158-238`).
* `src/eval_downstream.py` implements confusion matrices, IoU, overall accuracy, an NDVI threshold proxy classifier, and SR-versus-bicubic-versus-HR comparisons (`src/eval_downstream.py:1-135`). `src/scene_protocol.py` adds scene-level split manifests and bootstrap-style metric summaries, but these are utilities, not training-pipeline defaults.
* `src/datasets/worldcover.py` reads a single-band GeoTIFF, reprojects it to a reference grid with nearest-neighbor resampling, and remaps the documented WorldCover raw codes to consecutive class IDs (`src/datasets/worldcover.py:39-117`). ESA's official documentation confirms that WorldCover map products are delivered as 3×3-degree COGs in EPSG:4326 with 11 classes.[3]
* `setup.py` creates a virtual environment, installs `requirements.txt`, and executes root-level test scripts. The README and setup guide explicitly state that no satellite data is downloaded and that the loaders do not target the official SEN2NAIP/TACO release (`README.md:57-140`; `SETUP_GUIDE.md:134-145`).

These facts support calling the repository a **tested prototype**, not a validated production system. No model-quality claim can be inferred from the tests because the tests use synthetic arrays and do not report a trained model's PSNR, SSIM, SAM, downstream IoU, calibration, or comparison to a reproducible baseline on real data.

## Findings by severity

### P0: default LR/HR normalization is unit-inconsistent

The loader's default is `normalize_method="reflectance"` (`src/datasets/sen2naip.py:160-167`). That method divides every input array by 10,000 and clips it to [0, 1] (`src/preprocessing.py:84-86). The loader applies it independently to both aligned LR and HR arrays (`src/datasets/sen2naip.py:235-243`). Yet the repository's own canonical schema marks Sentinel-2 as reflectance with scale `1/10000`, while NAIP is DN with scale `1/255` (`src/datasets/band_schema.py:83-105`). The official SEN2NAIP example uses exactly this distinction: HR is divided by 255 and LR by 10000.[1]

For ordinary 8-bit NAIP data, the current code therefore supplies HR values in approximately [0, 0.0255] while LR values are in [0, 1]. This changes the objective, biases the prediction toward near-zero targets, and makes inference normalization ambiguous. The test suite only checks synthetic loader shapes and pair retention; it does not assert physical/unit equivalence. This is a correctness defect, not merely a missing feature.

**Required fix:** define product-specific conversion functions and metadata. Convert Sentinel-2 L2A DN to reflectance with its documented scale and NAIP DN to a comparable declared target space, or explicitly train in sensor-native units with a sensor-aware loss. Record the conversion in the checkpoint and test it with known-value fixtures such as LR=10000 -> 1.0 and NAIP=255 -> 1.0.

### P0: official data format and real-data path are absent

The loader's module docstring says it supports only `lr/<id>.npy` and `hr/<id>.npy`, and explicitly says the official release is a TACO archive requiring a missing adapter (`src/datasets/sen2naip.py:8-20`). This is consistent with the Hugging Face dataset card, which documents downloadable ZIP/TACO artifacts and GeoTIFF demonstrations rather than this paired `.npy` layout.[1] No converter, streaming reader, checksum/provenance import, official sample fixture, or end-to-end real-file test is present.

The same pattern applies to the SEN2Vénus and Cartosat loaders according to their module documentation and the README's known-gaps section. WorldCover has a real GeoTIFF reader, but the repository has only synthetic GeoTIFF round-trips and no downloaded tile validation (`src/datasets/worldcover.py:1-32`; `README.md:121-137`). Thus, “dataset loader implemented” currently means “loader for a local surrogate convention implemented.”

**Required fix:** add an explicit ingestion layer for each official source. It should preserve band names/order, scale/offset, CRS, affine transform, nodata/cloud masks, acquisition dates, and source IDs. Add at least one checked-in metadata-only fixture or a CI-downloadable small sample, then run the actual loader and alignment path against it.

### P0: band order is declared inconsistently and never reconciled

The Sentinel-2 schema is declared as blue, green, red, NIR (`src/datasets/band_schema.py:83-93`), while the NAIP schema is declared as red, green, blue, NIR (`src/datasets/band_schema.py:95-105`). The dataset validates LR against one schema and HR against the other but does not reorder either array before applying a per-channel reconstruction loss (`src/datasets/sen2naip.py:206-216`, `src/datasets/sen2naip.py:235-243`). A four-band count check cannot verify wavelength identity; the schema itself acknowledges that arrays without metadata cannot establish band identity (`src/datasets/band_schema.py:25-33`).

If arrays follow the declared schema, the model is trained to compare LR blue with HR red and LR red with HR blue. The official paper describes the 10 m Sentinel-2 subset as RGBNIR and the NAIP data as RGBNIR, but a reliable implementation still needs one canonical order and an explicit adapter.[2]

**Required fix:** select one canonical order, encode it in each adapter, reorder before pairing, and assert band names in metadata. Add a fixture with distinct constant/gradient signatures per band so a channel swap fails the test.

### P1: alignment crops non-grid-aligned shifts by floor division

`estimate_pair_shift` searches HR-pixel shifts across the full integer range (`src/datasets/sen2naip.py:81-117`). `apply_shift_and_crop` converts an HR shift to LR units with `dy // scale` and `dx // scale` (`src/datasets/sen2naip.py:120-143`). At scale 4, an estimated shift of 1–3 HR pixels becomes zero LR pixels. The HR crop is shifted while the LR crop is not, and the later trim only makes dimensions divisible; it does not restore the missing subpixel/phase registration. The newer geospatial contract correctly rejects non-scale-aligned shifts (`src/datasets/geospatial.py:91-118`), but the primary SEN2NAIP path does not use that contract.

This is especially important for cross-sensor pairs. The SEN2NAIP paper emphasizes temporal and cross-sensor differences, and the dataset construction filters acquisitions by time and quality rather than making the two sensors physically identical.[2] A weak, phase-inconsistent pairing can teach the model registration artifacts.

**Required fix:** either search in LR-grid units only, model HR subpixel offsets explicitly, or reject shifts not divisible by the scale. Propagate the shift into affine transforms and masks, not only array crops.

### P1: training permits batch sizes that its default collate cannot safely handle

The loader crops each sample according to its detected shift; the resulting shapes can differ. `build_loader` documents that batch size 1 avoids this problem but accepts any `batch_size >= 1` and uses default PyTorch collation (`src/train.py:184-188`). The CLI exposes `--batch-size` without a shape-aware collate function (`src/train.py:223-238`). With two samples of different post-crop sizes and `--batch-size 2`, the DataLoader will attempt to stack unequal tensors and fail at runtime.

**Required fix:** either enforce fixed-size patching before dataset output, implement padding plus a valid-pixel mask, or reject batch sizes above one with a clear error. Add a test with two deliberately different crop sizes.

### P1: validation is index-random, not scene/AOI-disjoint

The CLI uses `random_split` over sample indices (`src/train.py:191-220`, `src/train.py:250-255`). The code comments explicitly concede that this is not a persisted scene/AOI-aware split and does not record file provenance. `src/scene_protocol.py` can build and validate scene manifests, but `train.py` never consumes them. If adjacent patches or multiple dates from one ROI are present, train and validation can share scene-specific textures, causing optimistic checkpoint selection and metrics.

This conflicts with the scientific goal of validation against high-resolution references. The official SEN2NAIP construction deliberately uses spatially separated ROIs for its cross-sensor dataset.[2] A project-level split should preserve that property rather than randomly splitting already cropped files.

**Required fix:** ingest scene IDs and acquisition groups, create a persisted manifest before training, validate disjointness, and make `train.py` accept that manifest. Report scene-level confidence intervals and hold out a geographic region for final evaluation.

### P1: inference is whole-image, not a production tile pipeline

`predict` converts the entire CHW input to a single batch and runs the model in one call (`src/infer.py:158-173`). There is no overlap tiling, halo/context handling, stitching, memory cap, or nodata-aware output masking. A full Sentinel-2 scene or large GeoTIFF can therefore exceed GPU/CPU memory, while edge artifacts are likely if users later add naive tiling. The input GeoTIFF profile is read, but output nodata is copied as a tag even though nodata pixels are still sent through the model and receive predicted values (`src/infer.py:201-225`).

**Required fix:** implement overlap-tile inference with configurable tile/halo sizes, weighted blending, and a propagated validity mask. Preserve nodata as nodata in the output rather than merely copying the metadata tag. Add a large synthetic raster test and a seam/edge regression test.

### P1: model architecture is only partially aligned with the “generative” claim

The implemented model is deterministic CNN/window-attention regression with an aleatoric variance head. It is not a GAN, diffusion model, or stochastic conditional generator. That is acceptable because the problem permits CNN/Transformer choices, but the product description should not imply that hallucinated detail is sampled or that the uncertainty head captures all reconstruction ambiguity. The uncertainty head is trained through NLL, but there is no calibrated uncertainty training set, calibration loss, or automatic calibration report in the training CLI.

The window transformer partitions padded features into fixed, non-shifted windows (`src/model.py:124-149`). There is no shifted-window mechanism or positional encoding. Attention is therefore local to fixed windows and has no explicit position signal within a window. The README acknowledges the non-shifted simplification (`README.md:138-140`). This is a defensible prototype trade-off, but it is a real risk for boundary artifacts and spatial correspondence.

### P1: evaluation utilities are not a complete evidence pipeline

`compare_downstream_utility` can compare SR, bicubic, and optional HR images with a caller-provided classifier (`src/eval_downstream.py:90-135`). The built-in NDVI threshold classifier produces only water/non-vegetated/vegetation classes (`src/eval_downstream.py:65-87`), whereas WorldCover provides 11 land-cover classes and requires raw-code remapping (`src/datasets/worldcover.py:39-71`; [3]). There is no trained downstream classifier, no real WorldCover/scene ingestion command, and no script that runs the complete protocol from a checkpoint and a real scene.

The metric code also has two methodological caveats. First, `metric_report` reports PSNR, SSIM, and SAM on the original alignment but separately searches up to ±4 pixels for the best NCC (`src/metrics.py:159-198`); the alignment search is diagnostic rather than applied consistently to the fidelity metrics. Second, the search has no minimum-overlap threshold beyond the image dimensions (`src/metrics.py:167-179`), so on small images a nearly one-pixel overlap can win with a high correlation. The tests cover idealized synthetic cases but not these adversarial cases.

**Required fix:** publish one evaluation command that loads real data, applies the same preprocessing to SR and baseline, computes aligned and unaligned fidelity metrics, applies a semantically justified downstream model, reports scene-level uncertainty intervals, and records all preprocessing/baseline/checkpoint metadata.

### P2: setup and test orchestration overstates verification

`setup.py` discovers only `repo_dir.glob("test_*.py")` (`setup.py:198-209`). It does not run `tests/test_scene_protocol.py`, even though that file contains the scene split, WorldCover schema, and uncertainty calibration tests. The setup guide says it runs every test file (`SETUP_GUIDE.md:96-97`), which is not literally true for the nested test directory.

The setup script installs unpinned lower bounds only (`requirements.txt:1-3`), does not provide a lockfile or platform-specific PyTorch selection, and treats GPU detection as informational (`setup.py:70-90`). A clean environment without PyTorch/rasterio cannot run the model, inference, or WorldCover paths. The shell bootstrap uses `curl -sO` without `--fail` or checksum verification (`setup.sh:21-27`), so a failed or intercepted download can be saved as `setup.py` and executed.

In this audit environment, `python3 -m compileall` succeeded and the NumPy-heavy scripts passed. `test_infer.py`, `test_model_runtime.py`, and `test_train.py` failed immediately because `torch` was absent. `test_worldcover.py` failed because `rasterio` was absent. Direct execution of `tests/test_scene_protocol.py` failed with `ModuleNotFoundError: src`. This is evidence of environment/dependency coverage limits, not proof that the implementation is intrinsically broken; it does prove the documented “run every test” setup path is incomplete and that model-path verification was not available in this environment.

## Strengths worth retaining

The project is unusually explicit about its known gaps. README and module docstrings distinguish synthetic tests from real data and state that official-format adapters are not built (`README.md:116-140`; `src/datasets/sen2naip.py:14-38`). That honesty is better than silently presenting a fake downloader or claiming benchmark results.

The code has several useful defensive checks: finite-value validation, shape validation, band-count and duplicate-band detection, explicit geospatial grid contracts, CRS/transform checks before GeoTIFF output, bounded log variance, and persisted `last.pt`/`best.pt`/`history.json` artifacts. The tests also cover negative cases such as malformed shapes, mismatched grids, duplicated bands, missing CRS, and split fractions. These are good foundations for a production hardening pass.

The WorldCover implementation makes a scientifically appropriate resampling choice: nearest-neighbor for categorical classes rather than interpolation (`src/datasets/worldcover.py:94-115`). The affine/CRS-aware alignment code is materially stronger than a simple array resize. The training CLI has also corrected the earlier omission of a validation loader by defaulting to a validation fraction (`src/train.py:223-255`), even though the split unit remains insufficient.

## Recommended implementation order

1. **Fix physical data contracts first.** Implement official readers/converters, canonical band order, product-specific scale/offset, nodata/cloud masks, and metadata manifests. Add known-value unit tests.
2. **Make pairing scientifically valid.** Reject or explicitly model non-scale-aligned shifts, preserve transforms, use multiband/quality-aware registration, and add real-data co-registration diagnostics.
3. **Make training reproducible.** Use scene/AOI-disjoint persisted manifests, fixed-size or padded batches, checkpoint preprocessing and split metadata, and a complete environment lock/CI matrix.
4. **Build real inference.** Add overlap tiling, halo blending, nodata propagation, large-raster memory tests, and deterministic CPU/GPU comparison tests.
5. **Complete evaluation before claiming utility.** Add bicubic and simple learned baselines, real HR reference scoring, aligned/unaligned PSNR/SSIM/SAM, WorldCover or a trained downstream task, calibration curves, and scene-level confidence intervals.
6. **Only then tune architecture and claims.** Benchmark fixed versus shifted windows, test spectral losses and uncertainty calibration, and state clearly that the current architecture is deterministic regression with aleatoric uncertainty rather than a sampled generative model.

## References

[1]: https://huggingface.co/datasets/isp-uv-es/SEN2NAIP "SEN2NAIP official Hugging Face dataset card"

[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11655869/ "SEN2NAIP: A large-scale dataset for Sentinel-2 Image Super-Resolution, Scientific Data"

[3]: https://esa-worldcover.org/en/data-access "ESA WorldCover official data access and product documentation"

[4]: https://www.nature.com/articles/s41597-024-04214-y "SEN2NAIP: A large-scale dataset for Sentinel-2 Image Super-Resolution, Nature Scientific Data"

[5]: https://pytorch.org/docs/stable/data.html "PyTorch data loading documentation"

## Audit evidence and execution notes

Repository evidence is cited inline as `path:line-range`. The audit did not modify source code. Static AST compilation succeeded for `src/`, `tests/`, and root test scripts. Root-level NumPy/geospatial-contract tests that could run without optional dependencies passed. Model, training, inference, and WorldCover scripts were not executable in the audit environment because `torch` and/or `rasterio` were not installed; this limitation is reported rather than treated as a passing result.
