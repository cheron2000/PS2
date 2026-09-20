# Remote-sensing super-resolution literature: AstraSR/AegisSRM gap analysis

**Scope.** This report analyzes the repository as it exists at inspection time and compares its proposed AstraSR/AegisSRM system with established and recent methods for satellite and remote-sensing super-resolution. It distinguishes **implemented facts**, **project claims or plans**, and **external evidence**. Repository references use file paths and line numbers; external literature is defined in the References section.

## Executive conclusion

The repository is a credible, well-tested **prototype scaffold**, but it is not yet evidence for a validated Sentinel-2 super-resolution method. The code implements a four-band CNN/attention/transformer model with a heteroscedastic log-variance head, synthetic-array tests, basic pair-quality checks, and a useful collection of fidelity, downstream, geospatial, and uncertainty utilities. It does **not** yet demonstrate numerical superiority, calibrated uncertainty, downstream benefit, Indian-domain generalization, or even execution on the official SEN2NAIP/SEN2Vénus/Cartosat products.

The central scientific issue is that 10 m Sentinel-2 to approximately 2.5 m reconstruction is an ill-posed inverse problem. A model can generate a plausible high-resolution image, but pixels below the native sensor sampling scale are partly inferred. A defensible contribution therefore needs a declared observation/degradation model, strict scene-level separation, real cross-sensor validation, metrics that separate spectral consistency from invented detail, uncertainty calibration, and task-level utility. The problem statement itself requires useful fine-scale reconstruction, spectral/geospatial consistency, uncertainty management, and validation against high-resolution references (`problem-statement.md:12-23`).

The repository's proposed architecture is **not presently a safe novelty claim**. Its building blocks—CNN reconstruction, attention, windowed self-attention, sub-pixel upsampling, heteroscedastic regression, and spectral-consistency objectives—are standard or already present in Sentinel-2-specific systems. ESA OpenSR provides an especially close prior-art comparator: a released Sentinel-2 10 m-to-2.5 m latent-diffusion model with uncertainty-related outputs and a production-oriented geospatial utility stack [5] [6]. SEN2SR further reports 2.5 m Sentinel-2 reconstruction with CNN, Mamba, and Swin variants and a low-frequency hard constraint for radiometric consistency [7]. DiffFuSR addresses all Sentinel-2 bands through a modular RGB diffusion plus multispectral fusion pipeline, using native Sentinel-2 data and Wald-protocol simulations [4]. The defensible novelty space is therefore empirical: for example, a carefully documented Indian Cartosat-2S/3 validation set, or a reproducible study demonstrating a meaningful quality/utility/compute trade-off under the same protocol. Neither is implemented in this repository yet.

## 1. What the task actually is

Sentinel-2 MSI has 13 bands at native 10 m, 20 m, and 60 m sampling. Its Level-2A product is surface reflectance and includes scene-classification, aerosol, and water-vapour products [1]. The repository currently narrows the primary task to four 10 m bands—red, green, blue, and NIR—and a nominal 4x output, 10 m to 2.5 m (`src/model.py:18-20`, `src/model.py:195-223`). That is a reasonable first benchmark, but it is not the same as super-resolving all Sentinel-2 bands. It is also not classical pansharpening: Sentinel-2 has no single high-resolution panchromatic band. The relevant formulations are single-image SR, multi-resolution Sentinel-2 band fusion, and cross-sensor translation/fusion with an external HR image.

A scientifically precise problem statement should separate three regimes:

1. **Reduced-resolution or synthetic SR.** A high-resolution reference is deliberately blurred/downsampled, often using a Wald-style protocol, so exact targets exist. This is scalable but tests the assumed degradation model.
2. **Real cross-sensor SR.** Sentinel-2 is paired with NAIP, SPOT, VenµS, WorldView, or another HR sensor. This tests real sensor differences, but temporal, spectral-response, geometric, and atmospheric mismatch make pixel losses imperfect.
3. **Operational enhancement.** A single Sentinel-2 scene is enhanced without a contemporaneous HR target. This is useful for deployment, but cannot by itself establish correctness; it requires external validation and uncertainty analysis.

The repository's plans mention all three, but its executable data path currently supports only plain paired `.npy` arrays. The official SEN2NAIPv2 release follows the TACO columnar archive specification and offers 62,242 LR/HR pairs across synthetic, cross-sensor, histogram-matched, and temporal variants [3]. The repository explicitly acknowledges this mismatch (`src/datasets/sen2naip.py:8-20`; `README.md:116-129`).

## 2. What is implemented versus what is claimed

### Implemented facts

**Model.** `SRModel` contains a convolutional stem, six default residual high-order-attention blocks, one fixed-window multi-head self-attention block, a 1x1 concatenation fusion, and separate mean and log-variance pixel-shuffle heads (`src/model.py:191-257`). The attention is not a full channel covariance module: it computes per-channel spatial mean and variance and adds them before a small gating network (`src/model.py:41-75`). The transformer pads to fixed square windows and crops back, but explicitly does not shift windows (`src/model.py:95-149`). The model accepts only scale 2 or 4 (`src/model.py:151-175`, `src/model.py:215-223`).

**Uncertainty objective.** The log-variance head is trained with a masked heteroscedastic Gaussian NLL plus L1 or L2 reconstruction loss (`src/losses.py:41-91`, `src/losses.py:94-155`). This establishes a predictive variance parameterization and an aleatoric-error objective. It does not, by itself, establish calibrated probabilities or distinguish aleatoric uncertainty from distribution shift, registration error, or epistemic uncertainty.

**Data-side checks.** The SEN2NAIP loader validates array shape, finite values, channel count, a four-band schema, expected scale, and an integer HR-pixel NCC alignment search (`src/datasets/sen2naip.py:81-117`, `src/datasets/sen2naip.py:181-230`). It records dropped pairs and returns alignment diagnostics (`src/datasets/sen2naip.py:178-180`, `src/datasets/sen2naip.py:235-255`). The SEN2Vénus loader reuses this plain-array convention and a 2x route (`src/datasets/sen2venus.py:4-17`, `src/datasets/sen2venus.py:42-133`).

**Metrics and protocol utilities.** NumPy metrics cover MSE, PSNR, SSIM, spectral-angle error, and an integer-shift NCC diagnostic (`src/metrics.py:42-198`). The repository also contains a scene split manifest, bootstrap intervals over scene metrics, WorldCover label-schema validation, downstream per-scene mIoU, and Gaussian interval coverage diagnostics (`src/scene_protocol.py:24-242`). These are useful infrastructure, but their existence is not evidence that a real-data experiment has been run.

**Inference.** Inference can load a checkpoint with `weights_only=True`, produce mean and variance arrays, and write NumPy or georeferenced GeoTIFF output (`src/infer.py:102-173`, `src/infer.py:181-238`). It scales the variance by the square of the reflectance divisor when denormalizing, which is mathematically correct (`src/infer.py:280-303`).

### Claims or plans not demonstrated by the repository

The README presents “validation against real Indian ground truth (ISRO Cartosat-2S/3)” as a product description (`README.md:1-6`), but the same README states that the tests use synthetic data and that Cartosat access and conversion are not implemented (`README.md:116-137`). The summary calls Indian ground truth the actual contribution (`SUMMARY.md:7-12`, `SUMMARY.md:38-44`), but no Cartosat scene, paired manifest, result, or trained checkpoint is present in the repository. This must be reported as a **target differentiator**, not a completed contribution.

The solution draft says it will report PSNR/SSIM, spectral angle, alignment, hallucination/omission, useful detail, and calibrated uncertainty (`solution-draft.md:25-45`). OpenSR-test is a strong reason to retain those categories, but there are no AstraSR/AegisSRM results, confidence intervals, baseline table, ablations, or uncertainty reliability plots in the repository. The draft itself correctly labels its external baselines as planned comparisons (`solution-draft.md:56-74`).

## 3. Literature and baseline comparison

| Method or resource | What it establishes | Relevance to AstraSR/AegisSRM | Required comparison |
|---|---|---|---|
| **Bicubic/nearest and classical interpolation** | Necessary lower bounds; nearest is already used only for alignment search in the loader (`src/datasets/sen2naip.py:63-66`). | A learned method must beat these under exactly the same registered/native protocol. | Include bicubic, bilinear, and nearest, with per-scene metrics and paired confidence intervals. |
| **DSen2 (Lanaras et al.)** | Established Sentinel-2 CNN super-resolution of lower-resolution bands using synthetically downsampled Sentinel-2 data; compared against interpolation and pansharpening and reported spectral and reconstruction improvements [8]. | A canonical learned Sentinel-2 baseline. The repository's four-band 4x cross-sensor task is different, so claims must not compare numbers across protocols. | Reproduce or use published results only as contextual prior art; preferably implement a controlled same-data baseline. |
| **SEN2NAIP/SEN2NAIPv2** | Peer-reviewed real cross-sensor dataset: 2,851 Sentinel-2/NAIP pairs and a large synthetic subset; the paper uses temporal/cloud filtering, spatial separation, harmonization, and explicit degradation modeling [2]. The current v2 card documents 62,242 pairs and TACO access [3]. | Best-fit primary 10 m-to-2.5 m dataset family, but real pairs are US-focused and cross-sensor. | Use official cross-sensor and synthetic variants separately; never pool them into one unexplained score. Implement TACO ingestion and preserve source metadata. |
| **SEN2VENµS/OpenSR-test Venµs route** | OpenSR-test provides a 2x Venµs benchmark and 4x NAIP, SPOT, Spain crops, and urban sets. Its metrics explicitly assess reflectance, spectral consistency, spatial alignment, synthesis, hallucinations, omissions, and improvements [5]. | Directly exposes the weakness of only reporting PSNR/SSIM. | Run the repository model and all baselines through the same OpenSR-test implementation and report metric distributions, not only means. |
| **ESA OpenSR / LDSR-S2** | Released latent-diffusion Sentinel-2 RGB-NIR model; the official repository says it performs 128-to-512 tensor SR and relies on `opensr-utils` for tiling, overlap blending, stitching, CRS/transform preservation, and SAFE/GeoTIFF workflows [6]. | Closest prior art to “Sentinel-2 10m-to-2.5m + uncertainty + spectral/geospatial reliability.” A generic uncertainty-aware claim is not novel against this baseline. | Benchmark its released weights where licensing/compute permit; compare quality, calibration, memory, latency, and tile-scale behavior. |
| **SEN2SR/SEN2SRLite** | Recent framework for 2.5 m Sentinel-2 RGB/NIR and 20 m bands, evaluating CNN, Mamba, and Swin architectures. It introduces a low-frequency hard constraint to preserve native spectral content and reports near-zero reflectance deviation [7]. | Directly overlaps the proposed fidelity-first CNN/transformer story and offers a lightweight practical comparator. | Include SEN2SRLite as an executable baseline. Test whether AstraSR's learned uncertainty adds value beyond its deterministic spectral constraint. |
| **DiffFuSR** | Modular RGB diffusion followed by fusion of remaining 10/20/60 m bands; uses harmonization, blind-kernel modeling, and Wald-protocol native Sentinel-2 simulations. It evaluates against Gram-Schmidt pansharpening and reports spectral/reconstruction metrics [4]. | Strong recent baseline for all-band output and explicit cross-resolution fusion. Its RGB-first design clarifies that all-band SR needs a band-specific observation model, not only a 4-band head. | Compare on the subset and scale that can be reproduced. Do not claim superiority from paper numbers generated under a different split or target. |
| **Deep-learning pansharpening benchmarks** | Pansharpening has mature reduced-resolution and full-resolution protocols, common datasets, classical and variational baselines, and computational analysis [9]. | Relevant when using a high-resolution spatial prior, but Sentinel-2 itself has no PAN band. Calling the proposed model “pansharpening” without a PAN or explicit cross-sensor prior is technically imprecise. | If Cartosat PAN or another HR source is fused, report a pansharpening-style protocol separately from single-image SR. |

The external state of the art also sets a higher bar for the word **trustworthy**. OpenSR-test states that ordinary PSNR, LPIPS, and SSIM are inadequate under luminance changes and spatial misalignment and therefore separates consistency, synthesis, and correctness metrics [5]. The repository currently implements PSNR/SSIM/SAM and an alignment diagnostic, but not the OpenSR-test hallucination/omission/improvement definitions (`src/metrics.py:188-198`).

## 4. Methodological novelty: what survives scrutiny

### Novelty that is not supported

The following are useful engineering choices but are not defensible as standalone research novelty:

- **“High-order attention.”** The implementation is a low-cost per-channel variance gate, not a full second-order covariance attention mechanism (`src/model.py:41-75`). It may be a valid ablation, but the name should not imply a new MHAN method.
- **CNN plus transformer fusion.** This is a familiar hybrid architecture. The code itself calls the blocks MHAN-style and SPIFFNet-style and labels the transformer simplified (`src/model.py:4-9`, `src/model.py:95-99`).
- **Sub-pixel reconstruction.** PixelShuffle is an established SR head (`src/model.py:151-175`).
- **Heteroscedastic uncertainty.** Predicting log variance with Gaussian NLL is established probabilistic regression (`src/losses.py:65-91`) and closely overlaps the uncertainty positioning of Sentinel-2 trustworthy-SR prior art [6].
- **Georeferenced output.** CRS and affine propagation are essential engineering requirements, not novelty (`src/infer.py:181-225`).
- **NCC pair registration.** Integer shift search is a useful quality-control heuristic, but it is not CRS/grid-aware co-registration (`src/datasets/sen2naip.py:81-117`).

### Potentially defensible contribution

The strongest remaining contribution is a **reproducible, scene-disjoint, Indian-domain evaluation against real Cartosat imagery**, provided it is actually executed and documented. A publishable claim would need: contemporaneous or carefully justified temporal pairing; explicit band-response and radiometric harmonization; orthorectified CRS/grid registration; independent Indian scenes held out by geography and date; comparison against bicubic, DSen2 or another established CNN, SEN2SRLite, and OpenSR; OpenSR-test-style correctness metrics; uncertainty calibration; and a downstream task evaluated on labels with a declared ontology.

A second possible contribution is an empirical **quality–calibration–compute trade-off**. This is weaker as novelty because lightweight Sentinel-2 packages now exist, but it can still be useful if measured rigorously: parameter count, peak memory, throughput, tile latency, quality, spectral deviation, hallucination rate, and uncertainty coverage under one reproducible hardware and data protocol. “Our model is lighter” is not enough; the frontier must be demonstrated.

## 5. Specific methodological gaps and unsupported claims

### 5.1 Data and observation model

The official SEN2NAIPv2 archive is TACO, while the loader expects paired `.npy` folders (`src/datasets/sen2naip.py:8-20`, `src/datasets/sen2naip.py:146-155`). This is a blocking reproducibility gap. The loader also assumes LR and HR have identical channel counts and maps the same four-channel schema to Sentinel-2 and NAIP (`src/datasets/sen2naip.py:200-216`). A real cross-sensor benchmark needs an explicit spectral-response map, units, acquisition times, cloud masks, sensor metadata, and harmonization policy.

The code aligns arrays by integer HR-pixel NCC and converts the resulting shift to LR coordinates with floor division (`src/datasets/sen2naip.py:120-143`). This is not a substitute for reprojection to a declared reference grid. It can mishandle subpixel or non-multiple-of-four offsets, rotations, different affine origins, and partial overlap. The external SEN2NAIP paper's filtering and harmonization pipeline is materially more specific than this loader [2].

The project mentions cloud and invalid-pixel masking, but the primary pair loaders do not return a valid mask. Finite-value checks do not reject finite sentinel nodata codes, and edge crops have no source-footprint provenance. A valid-mask-aware NLL exists, but the main SEN2NAIP sample does not populate `valid_mask` (`src/datasets/sen2naip.py:235-255`; `src/losses.py:27-39`).

### 5.2 Splits and leakage

The CLI creates a deterministic random index split, not a persisted scene/AOI/date split (`src/train.py:191-220`). The code comment explicitly says this does not provide an AOI-aware manifest (`src/train.py:199-208`). Patches from the same scene can therefore cross train and validation if the input directory contains multiple crops per scene. This is especially dangerous for synthetic NAIP-derived pairs, where neighboring or related patches can share textures and degradation parameters. The scene manifest utilities are not wired into the training CLI (`src/scene_protocol.py:24-109`; `src/train.py:223-272`).

### 5.3 Spectral consistency and target semantics

The loss is a pixelwise L1/L2 reconstruction plus NLL; it does not enforce a low-frequency observation constraint, spectral-angle loss, bandwise radiometric conservation, or a sensor response model (`src/losses.py:41-155`). That does not make it wrong, but it means spectral consistency remains an evaluation hope rather than a model guarantee. SEN2SR's hard low-frequency constraint is a direct comparator [7].

The model defaults to four output channels and scale 4. The project statement asks for a product below 4 m and mentions Sentinel-2 imagery broadly (`problem-statement.md:22-23`), but the implementation does not support the full 13-band native-resolution problem. Claims should say **RGBNIR 10 m-to-2.5 m prototype** unless and until 20 m/60 m bands and their distinct point-spread functions are implemented.

### 5.4 Uncertainty

The model's output is a conditional variance map, not automatically a calibrated confidence map. The NLL objective may learn relative error weighting, but calibration can fail under domain shift, misregistration, non-Gaussian residuals, clipping, and multimodal reconstruction ambiguity. The repository has a useful Gaussian NLL/coverage function (`src/scene_protocol.py:209-242`), but no held-out results, reliability diagrams, expected calibration error, negative log-likelihood table, sharpness-versus-coverage analysis, or stratification by scene, band, land cover, and error regime. The README's wording “honest confidence map” (`SUMMARY.md:7-12`) is therefore stronger than the current evidence.

The uncertainty head also shares the same deterministic backbone and does not represent epistemic uncertainty. A stronger study would add deep ensembles, MC dropout, or a deterministic baseline with calibrated conformal intervals, then report whether uncertainty predicts actual absolute spectral or spatial error on unseen scenes.

### 5.5 Inference and geospatial safety

The inference path reads the complete input raster and predicts one full-scene tensor (`src/infer.py:48-82`, `src/infer.py:158-173`). It does not implement the tiled, overlap-blended workflow that OpenSR's utility package documents [6]. Full-scene inference risks memory exhaustion and edge artifacts for 110 km Sentinel-2 tiles. The code propagates an affine transform and scales pixel size, but it does not carry masks, source provenance, band names, normalization contract, or atomic output manifests. The checkpoint loader is safer than a default unrestricted `torch.load`, but raw state dictionaries can still fall back to default architecture parameters (`src/infer.py:131-155`), so checkpoint schema validation remains needed.

## 6. Missing experiments required for a credible paper or competition claim

The following experiments are prioritized by scientific importance.

### P0: Establish a valid benchmark before claiming improvement

1. **Official data ingestion.** Implement the TACO reader/converter for SEN2NAIPv2, and a verified adapter for SEN2VENµS/OpenSR-test. Record sample ID, scene ID, acquisition date, CRS, affine transform, band map, units, scale/offset, masks, and source checksum.
2. **Scene-level split.** Create immutable train/validation/test manifests before cropping. Ensure geographic and temporal separation. Report the number of scenes, not only patches or pixels.
3. **Baseline suite.** Run nearest, bilinear, bicubic, a simple residual CNN, DSen2 or another established Sentinel-2 CNN, SEN2SRLite, and OpenSR where feasible. Keep synthetic, real cross-sensor, and Indian data as separate test tracks.
4. **Registered and native metrics.** Report PSNR/SSIM/SAM only on explicitly registered pairs, alongside OpenSR-test reflectance, spectral, spatial, synthesis, hallucination, omission, and improvement metrics [5]. Report per-scene means, medians, standard deviations, and paired bootstrap intervals.

### P1: Test the claimed mechanisms

5. **Ablation.** Remove the variance attention, remove the transformer, remove cross-stage fusion, compare shifted versus non-shifted windows, compare L1/L2/NLL weights, and test a low-frequency hard constraint. Report parameter count, training time, VRAM, and metrics.
6. **Data ablation.** Compare synthetic-only, real cross-sensor-only, mixed training, harmonization variants, and degradation kernels. Evaluate on a test distribution whose degradation generator is not identical to training.
7. **Scale and band ablation.** Test 2x VenµS and 4x NAIP separately. If adding 20 m bands, use band-specific degradation and resampling rather than treating all channels as identical.

### P1: Validate uncertainty and usefulness

8. **Uncertainty calibration.** On held-out scenes, report Gaussian NLL, RMSE normalized by predicted standard deviation, empirical coverage at 50/80/90/95%, interval width, reliability plots, and calibration error. Compare the proposed head against ensemble or conformal baselines. Stratify by cloud/edge/land-cover/error regime.
9. **Downstream task.** Use a declared label ontology. ESA WorldCover is an 11-class 10 m product, not a three-class NDVI label source [10]. The repository's current NDVI classifier creates water/non-vegetated/vegetation labels (`src/eval_downstream.py:65-87`), so it is a proxy and cannot be presented as WorldCover semantic validation. Train or freeze one classifier consistently and compare LR/bicubic/SR/HR on the same scenes, with class-balanced IoU/F1 and paired intervals.
10. **Indian generalization.** Acquire actual Cartosat-2S/3 data, create a documented cross-sensor pairing and harmonization procedure, and reserve geographically distinct Indian sites for a blind test. An Indian visual demo without HR ground truth is not validation.

### P2: Operational reproducibility

11. Persist preprocessing configuration and units in checkpoints. Training uses reflectance normalization by default (`src/datasets/sen2naip.py:160-168`, `src/datasets/sen2naip.py:240-251`), while inference defaults to no normalization (`src/infer.py:255-265`); a user can therefore run a mathematically valid but scientifically mismatched pipeline unless the contract is explicit.
12. Add tiled inference with overlap blending, memory quotas, finite/range checks, mask propagation, and atomic GeoTIFF/COG writes. Validate that tiled output agrees with a small full-image reference.
13. Record environment versions, RNG state, split manifest hash, dropped-pair list, dataset checksums, model configuration, and complete metric configuration. Add resume tests and repeated-seed results.

## 7. Recommended claim language

**Supported now:** “We implement and unit-test a four-band Sentinel-2-style 4x SR prototype with a CNN/attention/windowed-transformer backbone, a heteroscedastic log-variance head, basic array-pair alignment diagnostics, and evaluation utilities.”

**Supported as a plan:** “We propose to evaluate on SEN2NAIP/SEN2VENµS/OpenSR-test and, if acquired and paired rigorously, Cartosat imagery.”

**Not supported yet:** “The system produces calibrated confidence,” “improves analytical utility,” “generalizes to India,” “preserves spectral/geospatial consistency on real products,” “outperforms state of the art,” or “reconstructs trustworthy <4 m detail.” Those require executed real-data experiments and statistical evidence.

The most honest novelty statement is: **the repository is an evaluation-oriented prototype pursuing a real Indian cross-sensor validation gap; the architecture itself is a hybrid baseline, not a novel SR algorithm.** This positioning is consistent with the project's own candid literature review (`solution-draft.md:62-74`, `SUMMARY.md:38-44`) and avoids claiming to have invented uncertainty-aware Sentinel-2 SR when ESA OpenSR and related work already exist [4] [6] [7].

## References

[1]: https://documentation.dataspace.copernicus.eu/Data/Sentinel2.html "Copernicus Data Space Sentinel-2 documentation"

[2]: https://www.nature.com/articles/s41597-024-04214-y "SEN2NAIP: A large-scale dataset for Sentinel-2 Image Super-Resolution"

[3]: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2 "Official SEN2NAIPv2 TACO dataset card"

[4]: https://arxiv.org/html/2506.11764v1 "DiffFuSR: Super-Resolution of all Sentinel-2 Multispectral Bands using Diffusion Models"

[5]: https://esaopensr.github.io/opensr-test/ "OpenSR-test: A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution"

[6]: https://github.com/ESAOpenSR/opensr-model "ESA OpenSR LDSR-S2 model and inference repository"

[7]: https://www.sciencedirect.com/science/article/pii/S0034425725006261 "SEN2SR: A radiometrically and spatially consistent super-resolution framework for Sentinel-2"

[8]: https://www.sciencedirect.com/science/article/abs/pii/S0924271618302636 "Super-resolution of Sentinel-2 images: Learning a globally applicable deep neural network"

[9]: https://ieeexplore.ieee.org/abstract/document/9844267/ "Machine learning in pansharpening: A benchmark, from shallow to deep networks"

[10]: https://esa-worldcover.org/en/data-access "ESA WorldCover data access and product documentation"

[11]: https://ieeexplore.ieee.org/iel8/4609443/10766875/10887321.pdf "Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion"

## Repository evidence index

- Problem requirements: `problem-statement.md:12-23`.
- Proposed architecture and intended 4x RGBNIR scope: `solution-draft.md:8-23`; `src/model.py:18-20`, `src/model.py:191-257`.
- Simplified variance attention and non-shifted windows: `src/model.py:41-75`, `src/model.py:95-149`.
- Heteroscedastic loss: `src/losses.py:65-91`, `src/losses.py:126-155`.
- Plain-array SEN2NAIP loader and official-format gap: `src/datasets/sen2naip.py:8-20`, `src/datasets/sen2naip.py:146-155`.
- Integer NCC registration and floor-division crop conversion: `src/datasets/sen2naip.py:81-143`.
- Primary loader validation and missing valid-mask return: `src/datasets/sen2naip.py:181-255`.
- Index split and explicit lack of scene manifest: `src/train.py:191-220`; CLI wiring: `src/train.py:223-272`.
- Metrics: `src/metrics.py:42-198`.
- Scene split, downstream, and uncertainty utilities: `src/scene_protocol.py:24-242`.
- Three-class NDVI downstream proxy: `src/eval_downstream.py:65-87`.
- Inference normalization, prediction, GeoTIFF path, and variance scaling: `src/infer.py:85-99`, `src/infer.py:158-173`, `src/infer.py:181-225`, `src/infer.py:255-303`.
- Synthetic-only execution and known real-data gaps: `README.md:57-83`, `README.md:116-141`.
- Existing audit's conclusion and prioritized gaps: `external_gap_audit.md:6-23`, `external_gap_audit.md:41-71`.

**Bottom line:** the repository has useful engineering foundations and unusually candid documentation. Its scientific evidence base is still empty where it matters most: no real-product benchmark, no controlled learned-baseline table, no ablation, no independent scene split actually used by the CLI, no calibrated uncertainty result, and no Cartosat validation. Until those experiments are executed, AstraSR/AegisSRM should be described as a prototype and a proposed Indian-domain evaluation study, not as a validated or novel remote-sensing SR method.

*Report prepared by Manus AI.*
