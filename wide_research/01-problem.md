# SIH26142 Problem Statement and Solution Gap Audit

## Executive conclusion

The repository has a credible **prototype skeleton** for four-band Sentinel-2 super-resolution: preprocessing, pair-quality checks, a CNN/attention/transformer model, a heteroscedastic uncertainty head, georeferenced output helpers, and an evaluation harness. It does **not yet demonstrate the problem statement’s required end-to-end solution**. The central gap is evidentiary rather than architectural: the repository has not trained or evaluated the model on real Sentinel-2/NAIP, Sentinel-2/Venµs, Cartosat, or WorldCover downloads. Its adapters consume a project-specific `.npy` convention, while the official SEN2NAIP release uses GeoTIFF/TACO-style archives and metadata [1]. The current tests are predominantly synthetic; in this environment, the PyTorch, rasterio, and scene-protocol test entry points also could not run because their dependencies or import path were unavailable.

The strongest claimed differentiator—validated performance on real Indian high-resolution ground truth—is therefore **not implemented as an evidenced result**. Cartosat pairing code exists, but it consumes pre-arranged `.npy` pairs and optional JSON metadata. No Cartosat data order, real pair, Indian test result, or uncertainty-calibration result is present in the audited files. The downstream-task code also exists, but the evidence is a proxy NDVI threshold classifier and synthetic tests, not a demonstrated crop, urban, change-detection, or disaster-response improvement.

The repository is unusually candid about several limitations. README.md explicitly says that real-format converters are missing, tests do not download real data, GPU-scale training is unverified, and the transformer uses fixed windows [2]. That honesty is a strength, but it also means the current state should be described as **validated plumbing plus an unvalidated research prototype**, not as a completed robust SRM system.

## 1. What the problem actually asks for

The supplied problem statement identifies NTRO as the organization and frames the task as deep-learning-based super-resolution mapping from medium-resolution satellite imagery [3]. Its operative requirements are:

1. **Input and target:** take 10 m Sentinel-2 imagery and generate an output finer than 4 m ground sampling distance (GSD).
2. **Image validity:** preserve both geospatial consistency and spectral consistency; the output must not merely look sharper.
3. **Preprocessing and training:** include preprocessing and model training with paired medium/high-resolution data.
4. **Validation:** assess accuracy and validate against high-resolution reference data.
5. **Analytical utility:** improve interpretability and utility for classification, change detection, crop monitoring, urban mapping, and disaster response.
6. **Uncertainty:** explicitly account for uncertainty and error because sub-10 m detail is partly inferred rather than directly observed.
7. **Robustness:** provide a robust framework, although the statement does not define a robustness test, threshold, or acceptance criterion.

The requirement is therefore broader than a model checkpoint. It is a data, geospatial, spectral, uncertainty, evaluation, and application claim. The statement does not prescribe GANs, diffusion, CNNs, or transformers; it leaves the architecture to the team [3]. It also does not specify an official dataset, a fixed test split, required metric values, a required number of scenes, a latency or memory budget, or what qualifies as a successful downstream improvement.

### Requirement interpretation and ambiguity

| Requirement in the statement | What it means operationally | Ambiguity or missing acceptance criterion |
|---|---|---|
| “<4m” output | At least one target product should have GSD below 4 m; a 4× 10 m→2.5 m route satisfies this numerically. | It does not say whether the target is exactly 2.5 m, whether non-integer ratios are allowed, or whether all input bands must be output at that GSD. |
| “Preserving geospatial consistency” | Correct CRS, affine transform, pixel grid, alignment, nodata/mask semantics, and ideally no spatial shift or seam artifacts. | No tolerance is specified for registration error, reprojection error, boundary seams, or metadata loss. |
| “Preserving spectral consistency” | Preserve band identity, reflectance units, spectral angle/shape, and radiometric behavior relative to Sentinel-2. | No band subset, unit convention, radiometric tolerance, or metric threshold is specified. |
| “Accuracy assessment” | Report reproducible fidelity, alignment, spectral, and error metrics against held-out HR references. | No mandatory metrics, confidence intervals, statistical test, or baseline is defined. |
| “Uncertainty and error components” | Produce calibrated uncertainty and show whether uncertainty predicts actual error. | A variance map alone is insufficient, but the statement does not specify calibration metrics, coverage levels, or a decision policy. |
| “Analytical utility” | Show a downstream task improves when using SR rather than LR/bicubic, with a fair task protocol. | The listed applications are not narrowed to one measurable task, and no classifier, labels, split, or minimum improvement is given. |
| “Robust framework” | Handle real product formats, clouds, nodata, different scenes, and resource constraints safely. | No threat model, supported product version, scale limit, throughput target, or failure behavior is specified. |

## 2. Stakeholders and their likely interests

The explicit stakeholders are NTRO, the Ministry of Education’s Innovation Cell, and the problem creator identified in the statement [3]. The practical stakeholders are broader:

- **Remote-sensing analysts and mission users** need outputs whose spatial detail is useful without silently inventing features.
- **Indian government or security users** need Indian-geography validation, provenance, controlled uncertainty, and reproducible evidence rather than a US-only benchmark.
- **Data providers and custodians** include Copernicus/ESA, the SEN2NAIP maintainers, ISRO/NRSC/Bhoonidhi, and ESA WorldCover. Their product formats, licenses, metadata, and access conditions constrain reproducibility.
- **Judges and evaluators** need a concise comparison against a defensible baseline and a clear explanation of what is actually demonstrated.
- **Engineering operators** need an executable path from real input products to output GeoTIFF/COG, with bounded memory and safe failure behavior.

The solution draft correctly identifies that Indian ground-truth validation would be a meaningful differentiator, but it treats this as a future action rather than a result. It also acknowledges that ESA OpenSR already overlaps the proposed uncertainty-aware 10 m→2.5 m positioning [4]. Official OpenSR documentation states that its model produces 4-channel 128×128→512×512 SR and an uncertainty map, while full-file geospatial processing is handled by a separate utility package [5]. This makes “uncertainty-aware Sentinel-2 SR” alone an unsafe novelty claim.

## 3. Assumptions made by the repository and solution documents

The documents make several reasonable but consequential assumptions:

1. **Four-band scope is sufficient initially.** The model and training CLI default to RGB+NIR. The solution draft says 20 m bands will be added later [4, lines 26–36]. That is a scope choice, not evidence that the full Sentinel-2 product requirement is met.
2. **SEN2NAIP is an acceptable primary proxy for the target.** The peer-reviewed dataset contains 2,851 real Sentinel-2/NAIP cross-sensor pairs and a synthetic component [1]. It is valuable for a 4× benchmark, but it is US-focused and cross-sensor. It cannot by itself establish Indian deployment validity.
3. **A simple paired-array representation is enough for the prototype.** The official release provides GeoTIFFs, metadata, and archive structure [6], whereas the repository expects matching `lr/<id>.npy` and `hr/<id>.npy` files [2, lines 93–100]. The converter and metadata-preserving ingestion step are assumed rather than delivered.
4. **WorldCover labels can stand in for downstream ground truth.** ESA WorldCover is a real global 10 m product with 11 classes and independently reported validation [7]. However, its map date, class errors, mixed pixels, and 10 m label scale create a noisy proxy for measuring 2.5 m SR utility. The repository does not show a real-scene experiment that quantifies these limitations.
5. **Cartosat can provide the Indian reference.** Cartosat-3’s MX camera is reported at 1 m across four visible/NIR bands by the EO Portal [8]. That establishes plausibility of a sub-4 m Indian reference, not the existence of a synchronized, co-registered, radiometrically compatible pair. Acquisition date, look angle, orthorectification, coverage, licensing, cost, and delivery remain operational dependencies.
6. **NCC plus simple harmonization is sufficient for cross-sensor pairing.** The code estimates shifts and applies gains/offsets, but no real Cartosat pair demonstrates that this handles different optics, PSFs, atmospheric conditions, seasonal change, viewing geometry, or band response functions.
7. **Fixed-seed index splits are adequate for model selection.** The training code now has a train/validation split, but its own comments state that it is not scene/AOI-aware and does not persist file-level provenance [9, lines 191–220]. Spatial leakage remains possible when multiple patches come from one scene.
8. **A predicted variance map is meaningful uncertainty.** The model does predict log variance and the loss includes a heteroscedastic NLL [10, lines 177–189; 11, lines 94–155]. Calibration must still be demonstrated on held-out real data; uncertainty head existence is not calibration evidence.

## 4. Comparison with what is actually implemented

### 4.1 Input, preprocessing, and output

**Implemented fact:** the repository contains preprocessing, normalization, masks, tiling utilities, band schemas, geospatial grid checks, and a GeoTIFF/NumPy inference interface. The README lists these components and states that GeoTIFF CRS/transform are preserved in output [2, lines 30–47; 2, lines 102–112]. The inference code scales the affine transform for the SR factor and writes an optional variance output [12, lines 181–225; 12, lines 280–303].

**Claim or limitation:** README also states that SEN2NAIP, SEN2Vénus, and Cartosat loaders do not read the official releases; they consume plain `.npy` conventions, and no real portal downloads were used [2, lines 116–133]. The official SEN2NAIP page shows that the real dataset is distributed as ZIP archives containing GeoTIFFs and metadata, with `rioxarray` examples rather than the repository’s two-directory array layout [6]. Thus the repository does not yet provide the required real-product ingestion path.

**Gap:** inference loads the complete input into one tensor and has no demonstrated bounded, tiled, overlap-blended processing. ESA’s established OpenSR workflow explicitly separates tensor-level model inference from geospatial tiling and stitching [5]. For large scenes, the current implementation’s output-preservation claim is narrower than an operational full-scene framework.

### 4.2 Model and uncertainty

**Implemented fact:** `SRModel` has a CNN stem, repeated high-order attention blocks, a windowed transformer fusion block, two PixelShuffle stages for 4× output, and separate mean and log-variance heads [10, lines 191–257]. The model supports scale 2 or 4 and defaults to four input/output channels [10, lines 204–223]. The loss combines reconstruction with heteroscedastic NLL [11, lines 94–155].

**Gap:** the fixed, non-shifted window implementation creates independent attention windows; the README openly flags this simplification [2, lines 138–140]. More importantly, the repository contains no real trained model result, no ablation, and no evidence that the architecture improves over bicubic, a supervised CNN baseline, SEN2SRLite, OpenSR, or a simpler model. It also does not implement the solution draft’s optional diffusion branch; the draft itself correctly says diffusion is a comparison target rather than a build target [4, lines 26–36].

**Uncertainty gap:** inference converts log variance to positive variance and corrects the units when denormalizing [12, lines 170–173; 12, lines 280–295]. The repository also includes a generic calibration function with NLL, standardized error, and interval coverage [13, lines 209–242]. These are useful mechanisms, but no report, artifact, or real held-out evaluation shows calibrated coverage. The problem statement’s uncertainty requirement is therefore only **partially implemented as code**, not satisfied as evidence.

### 4.3 Training and reproducibility

**Implemented fact:** the training loop supports validation-driven best-checkpoint selection, saves `last.pt`, `best.pt`, model configuration, optimizer state, metrics, and a history JSON [14, lines 101–181]. The CLI exposes seed, validation fraction, batch size, and device arguments [9, lines 223–272].

**Gap:** the split is sample-index-based rather than scene/AOI-based, and the code explicitly says it does not persist which source files went to which split [9, lines 191–208]. There is no demonstrated dataset hash manifest tied to a run, no full environment lock, no resume CLI, and no real-data training log. The source task board also records auditable/resumable experiments and safe bounded inference as unfinished work [15, lines 28–29].

### 4.4 Pairing and datasets

**Implemented fact:** SEN2NAIP and SEN2Vénus adapters validate shape, estimate an NCC-based shift, crop aligned pairs, normalize, and return arrays; Cartosat pairing adds optional grid metadata, masks, and harmonization [16]. Band schema validation rejects wrong counts, repeated/near-duplicate, and constant bands, while acknowledging that array data cannot prove wavelength identity [17, lines 149–159].

**Gap:** the adapters glob `.npy` files and look for matching array names [16]. Cartosat’s dataset class reads `lr/*.npy`, `hr/*.npy`, and optional JSON metadata [16]. This is not an official Cartosat/Bhoonidhi or SEN2NAIP product parser. It also leaves the hardest scientific work—cross-sensor registration, PSF and spectral-response differences, acquisition-date compatibility, orthorectification quality, and independent validation—unproven on real imagery.

### 4.5 Evaluation and analytical utility

**Implemented fact:** `eval_downstream.py` compares SR, bicubic, and optional HR with accuracy and mean IoU, and offers a three-class NDVI threshold classifier [18, lines 65–135]. `scene_protocol.py` adds scene-disjoint manifests, per-scene summaries, bootstrap intervals, WorldCover schema checks, and uncertainty calibration helpers [13, lines 24–65; 13, lines 128–206]. The official OpenSR benchmark reinforces the need for reflectance, spectral, spatial, synthesis, hallucination, omission, and improvement measures rather than relying on PSNR/SSIM alone [19].

**Gap:** the tests exercise synthetic arrays, and no result file shows a real SR-versus-bicubic downstream improvement. The NDVI threshold classifier is a transparent proxy, not a crop/urban/change/disaster model. WorldCover is a 10 m land-cover map, so comparing 2.5 m predictions against it requires a clearly documented aggregation and temporal/geospatial protocol. The repository has not shown such a protocol on downloaded scenes. The problem statement’s “analytical utility” requirement is therefore currently a harness capability, not a demonstrated outcome.

## 5. Verification performed and evidence limits

The repository README says all tests generate synthetic data and require no real downloads [2, lines 57–83]. I ran the available plain-script tests in the current environment. NumPy-only pairing, preprocessing, band-schema, geospatial, and downstream utility tests passed. The inference, model-runtime, training, and WorldCover tests could not start because `torch` or `rasterio` was not installed in this environment. The scene-protocol test also could not import `src` when invoked from its nested path. These failures do not prove the implementation is incorrect; they do prove that the repository’s advertised all-tests verification was not reproducible in the audit environment without first installing dependencies or fixing invocation context.

No checked-in model weights, real downloaded scenes, real metrics table, uncertainty reliability diagram, Cartosat order/delivery, or real end-to-end evaluation report was found in the requested four documents or the inspected source tree. Claims in SUMMARY.md and solution-draft.md about future Cartosat validation, expected differentiating value, or comparison numbers must therefore be labeled **planned or researched claims**, not achieved results.

## 6. Missing or weakly supported requirements, prioritized

### P0 — blocks a defensible “solution completed” claim

1. **Real-data end-to-end proof is missing.** Build converters/loaders for official SEN2NAIP/SEN2NAIPv2 and at least one real Sentinel-2 product. Record product version, bands, units, CRS, dates, masks, and provenance.
2. **Indian validation is unexecuted.** Acquire legally usable Cartosat-2S/3 MX data or document a credible fallback. Produce a held-out Indian test set with real registration, radiometric harmonization, and independently reviewed pair-quality criteria.
3. **No quantitative success criteria are operationalized.** Freeze metrics, baselines, split rules, minimum sample/scene counts, and pass/fail or confidence criteria before tuning.
4. **No uncertainty calibration result exists.** Report NLL, error-versus-uncertainty ranking, and empirical interval coverage on held-out real scenes. Include calibration plots and failure cases.
5. **Analytical utility is not demonstrated.** Select one primary task, such as land-cover segmentation or crop classification, and compare LR, bicubic, SR, and HR-or-reference conditions using the same scene-disjoint test set.

### P1 — required for scientific credibility and reproducibility

6. **Replace patch-index splitting with scene/AOI splitting.** Persist a manifest containing scene IDs, source files, dates, geographies, checksums, and split assignment. Reject any spatial or temporal leakage.
7. **Add official-format and metadata handling.** Preserve per-band identity, scale/offset, nodata, cloud/SCL masks, CRS, transform, acquisition time, and source license. Do not treat a four-channel array as proof of band identity.
8. **Add strong, reproducible baselines.** At minimum include bicubic, a supervised simple CNN, and one current open baseline such as SEN2SRLite/OpenSR. Report both quality and resource cost.
9. **Add real-scene spatial and spectral tests.** Use alignment-aware metrics and the OpenSR-style consistency/correctness framing [19]. Avoid claiming PSNR/SSIM alone establishes scientific fidelity.
10. **Clarify WorldCover use.** Pin WorldCover version and map date, document label remapping and nodata, quantify temporal mismatch, and state that it is a noisy downstream proxy rather than HR truth.

### P2 — operational robustness and deployment

11. **Implement tiled/streaming inference with bounded memory.** Include overlap blending, border handling, atomic writes, and a test showing tiled output agrees with a reference on manageable inputs.
12. **Add preflight validation.** Reject malformed, non-finite, oversized, unsupported-band, unsupported-scale, or inconsistent-georeferencing inputs before large tensor allocation.
13. **Make training resumable and auditable.** Persist RNG states, resolved configuration, package versions, dataset manifest/hash, code revision, and checkpoint provenance.
14. **Measure compute and latency.** Report parameter count, peak memory, patch/tile latency, and hardware for training and inference. Compare against OpenSR/SEN2SRLite rather than implying novelty from lightweight execution alone.
15. **Document application limits.** State that SR reconstructs plausible detail and cannot recover information absent from the input. Provide uncertainty-aware guidance for analysts rather than presenting all fine structures as observed facts.

## 7. Recommended claim language for the current repository

A defensible current description would be:

> “A four-band Sentinel-2 SR prototype with preprocessing, alignment/QC utilities, a CNN-attention-transformer model, heteroscedastic variance output, georeferenced I/O, and evaluation scaffolding. The code is unit-tested primarily on synthetic data and is not yet validated end-to-end on official real datasets or Indian Cartosat ground truth.”

The following claims should **not** be made yet: “validated against Cartosat,” “improves analytical utility,” “uncertainty is calibrated,” “robust full-scene inference,” “works on official SEN2NAIP/Bhoonidhi formats,” or “novel uncertainty-aware Sentinel-2 SR.” The first five lack result evidence; the last overlaps with existing ESA OpenSR work [5]. The strongest eventual positioning is narrower and testable: **real Indian-ground-truth validation, with scene-disjoint evaluation and calibrated uncertainty, if and only if those experiments are actually completed**.

## References

[1]: https://www.nature.com/articles/s41597-024-04214-y "Aybar et al., A large-scale dataset for Sentinel-2 Image Super-Resolution"
[2]: https://github.com/ManusAI/PS2/blob/main/README.md "Repository README.md (local audit path: /home/ubuntu/PS2/README.md)"
[3]: https://github.com/ManusAI/PS2/blob/main/problem-statement.md "Repository problem-statement.md (local audit path: /home/ubuntu/PS2/problem-statement.md)"
[4]: https://github.com/ManusAI/PS2/blob/main/solution-draft.md "Repository solution-draft.md (local audit path: /home/ubuntu/PS2/solution-draft.md)"
[5]: https://github.com/ESAOpenSR/opensr-model "ESA OpenSR model documentation and inference repository"
[6]: https://huggingface.co/datasets/isp-uv-es/SEN2NAIP "Official SEN2NAIP dataset page and download examples"
[7]: https://esa-worldcover.org/en/data-access "ESA WorldCover data access, classes, resolution, and validation"
[8]: https://www.eoportal.org/satellite-missions/cartosat-3 "EO Portal Cartosat-3 mission and sensor specifications"
[9]: https://github.com/ManusAI/PS2/blob/main/src/train.py "Repository training implementation (local audit path: /home/ubuntu/PS2/src/train.py)"
[10]: https://github.com/ManusAI/PS2/blob/main/src/model.py "Repository model implementation (local audit path: /home/ubuntu/PS2/src/model.py)"
[11]: https://github.com/ManusAI/PS2/blob/main/src/losses.py "Repository loss implementation (local audit path: /home/ubuntu/PS2/src/losses.py)"
[12]: https://github.com/ManusAI/PS2/blob/main/src/infer.py "Repository inference implementation (local audit path: /home/ubuntu/PS2/src/infer.py)"
[13]: https://github.com/ManusAI/PS2/blob/main/src/scene_protocol.py "Repository scene and uncertainty evaluation protocol (local audit path: /home/ubuntu/PS2/src/scene_protocol.py)"
[14]: https://github.com/ManusAI/PS2/blob/main/src/train.py "Repository checkpoint and training loop implementation"
[15]: https://github.com/ManusAI/PS2/blob/main/tasks.md "Repository task board and explicitly unfinished robustness work"
[16]: https://github.com/ManusAI/PS2/blob/main/src/datasets/sen2naip.py "Repository SEN2NAIP adapter (local audit path: /home/ubuntu/PS2/src/datasets/sen2naip.py)"
[17]: https://github.com/ManusAI/PS2/blob/main/src/datasets/band_schema.py "Repository band-schema and provenance validation"
[18]: https://github.com/ManusAI/PS2/blob/main/src/eval_downstream.py "Repository downstream evaluation implementation"
[19]: https://github.com/ESAOpenSR/opensr-test "ESA OpenSR benchmark documentation and metrics"

*Repository evidence in this report uses local file paths and line ranges in the body. The GitHub-style reference links for local files are navigational; the audited files were read directly from `/home/ubuntu/PS2`.*
