# Solution Draft — v2
Status: IN-PROGRESS
Last edited by: AstraSR, round 1

## Problem Restatement
SIH26142 (NTRO): build a deep-learning super-resolution framework that takes medium-resolution Sentinel-2 imagery (10m bands, with 20m bands as an extension) and produces a sharper product targeting <4m GSD while preserving geospatial and spectral consistency. The solution must include pre-processing, paired-data training, quantitative accuracy assessment, validation against real high-resolution references, and explicit uncertainty management because reconstructed details are partly inferred.

## Proposed Approach
- Input: Sentinel-2 L2A surface-reflectance imagery. Start with the four 10m bands (B2/B3/B4/B8); extend to 20m bands after the core pipeline is stable.
- Use a **benchmark-driven hybrid reconstruction model** rather than claiming that MHAN+SPIFFNet is intrinsically state of the art. A lightweight CNN/high-order-attention stage handles local texture and a transformer fusion stage handles broader context.
- Add an optional diffusion refinement branch only if controlled ablation demonstrates a measurable benefit on real Sentinel-2 benchmarks. DiffFuSR is particularly relevant prior art because it targets all 12 Sentinel-2 L2A bands and 2.5m GSD using diffusion plus learned multispectral fusion.
- Produce the SR image **and a calibrated uncertainty/confidence product**. Uncertainty should be evaluated against observable failure modes such as hallucination, spectral inconsistency, and spatial misalignment, rather than treated as a generic extra channel.

## Technical Architecture
1. Sentinel-2 L2A ingestion, cloud/mask handling, reflectance normalization, and band-wise resampling.
2. Geospatial co-registration / quality filtering for every LR-HR pair; avoid naive pixel-wise loss on weakly aligned cross-sensor pairs.
3. Shallow convolutional feature extraction.
4. MHAN-style high-order attention blocks for local high-frequency structure.
5. SPIFFNet-style cross-spatial/cross-stage transformer fusion for wider context.
6. Reconstruction and upsampling head.
7. Optional diffusion refinement branch, activated only after ablation against the deterministic baseline.
8. Uncertainty head, preferably calibrated against validation residuals and benchmark correctness metrics.
9. Geospatial output writer preserving CRS, transform, band metadata, and reflectance units.

## Evaluation Protocol
Do not judge success primarily by visual sharpness or PSNR/SSIM.

Use a held-out-site/held-out-pair protocol and report:
- reflectance consistency;
- spectral consistency, including spectral-angle error;
- spatial alignment;
- synthesis/high-frequency detail;
- hallucination and omission rates;
- improvement/correct-detail rate;
- conventional reconstruction metrics such as PSNR/SSIM where an aligned HR reference is available;
- uncertainty calibration/error correlation.

The public **OpenSR-test** framework provides a real-Sentinel-2-oriented evaluation protocol spanning reflectance, spectral, spatial, synthesis, hallucination, omission, and improvement metrics. It should be adopted or reproduced as the primary evaluation layer where compatible. This is important because OpenSR research shows that stronger spatial synthesis can trade off against spectral fidelity.

## Novelty / Differentiation
The defensible novelty is **trustworthy SR rather than an unverified claim of a novel backbone**:
- spectral/geospatial consistency is treated as a hard design objective;
- uncertainty is calibrated and linked to hallucination/error risk;
- evaluation explicitly separates useful detail from invented detail;
- architecture selection is empirical, with deterministic and diffusion variants compared under the same protocol;
- the system should expose confidence/quality metadata alongside the SR product.

MHAN is established remote-sensing SR prior art, and SPIFFNet is established transformer-based RSISR prior art. They should therefore be described as components/baselines rather than as novelty by themselves. DiffFuSR is a 2025 Sentinel-2-specific diffusion pipeline and must be considered in the comparison set.

## Feasibility & Data Sources
- **SEN2VENµS**: same-day, spatially registered Sentinel-2/VENµS pairs across 29 locations, with 132,955 patches and 5m references for eight Sentinel-2 bands. This is a strong supervised dataset for 10/20m → 5m experiments.
- **OpenSR-test datasets**: real-world benchmark data including Venµs, NAIP, SPOT, Spain Crops, and Spain Urban, designed to reduce spatial/spectral misalignment and evaluate correctness beyond conventional SR metrics.
- **WorldStrat / SEN2NAIP** can be used as additional data where licensing and alignment are appropriate, but cross-sensor pairs require explicit registration/quality control.

**Resolution caveat:** SEN2VENµS directly validates 5m reconstruction, not <4m. A credible <4m claim therefore requires an additional sub-4m reference/benchmark or must be presented explicitly as an extrapolation rather than as directly validated performance.

## Known Risks & Open Questions
1. **No direct MHAN+SPIFFNet vs DiffFuSR comparison was found.** Do not declare a winner without an experiment.
2. **<4m validation remains the largest data gap.** Training/evaluating only against 5m VENµS cannot substantiate a <4m claim.
3. **Diffusion compute and inference cost need measurement.** DiffFuSR/EDiffSR provide relevant evidence, but project-specific latency/memory must be benchmarked.
4. **Uncertainty calibration is unresolved.** Compare a practical heteroscedastic/ensemble or MC-dropout approach against held-out residuals and hallucination metrics.
5. **India-specific validation is not yet established.** It is useful for contextual relevance but is not mandated by the PS.
6. **The team's original AegisSRM design rationale is not fully present in this repository.** Later agents should not assume missing details.

## Sources
- SEN2VENµS dataset: https://zenodo.org/records/6514159
- EDiffSR: https://arxiv.org/abs/2310.19288
- DiffFuSR: https://arxiv.org/abs/2506.11764
- SPIFFNet: https://arxiv.org/abs/2307.02974
- MHAN: https://ieeexplore.ieee.org/document/9151234/
- OpenSR-test: https://github.com/ESAOpenSR/opensr-test
- OpenSR: https://opensr.eu/
