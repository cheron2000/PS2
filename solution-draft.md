# Solution Draft — v3
Status: IN-PROGRESS
Last edited by: Sonnet5, round 1

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
- **SEN2VENµS**: same-day, spatially registered Sentinel-2/VENµS pairs across 29 locations, with 132,955 patches and 5m references for eight Sentinel-2 bands. Strong for 10/20m → 5m experiments, but does not by itself support the <4m target.
- **SEN2NAIP (v2)**: real cross-sensor Sentinel-2/NAIP pairs with HR reference downloaded at **2.5m GSD** (2,851 real pairs; 62,242 in v2 including the degradation-model-generated subset). This is genuinely sub-4m, not an extrapolation — it is also the dataset DiffFuSR itself trains and evaluates on for its 2.5m claim. **This is now the primary recommended dataset for substantiating the <4m requirement.** Coverage is limited to the continental United States, so it supports the resolution claim but not geographic generalization. Source: https://www.nature.com/articles/s41597-024-04214-y
- **MuS2**: independent real-world benchmark pairing multi-temporal Sentinel-2 with real WorldView-2 HR imagery at **~3.3m GSD (3x)**, 91 scenes, ~2500 km². Useful as a second, independent sub-4m cross-check outside the SEN2NAIP data family, though it is built for multi-image (multi-revisit) SR — usable for single-image evaluation by scoring one revisit at a time. Source: https://doi.org/10.7910/DVN/1JMRAT
- **OpenSR-test datasets**: real-world benchmark data including Venµs, NAIP, SPOT, Spain Crops, and Spain Urban, designed to reduce spatial/spectral misalignment and evaluate correctness beyond conventional SR metrics.
- **WorldStrat** can be used as additional training data where licensing and alignment are appropriate, but cross-sensor pairs require explicit registration/quality control.

**Resolution caveat (updated):** the <4m target is no longer an unvalidated extrapolation. SEN2NAIP provides a real 2.5m reference for both training and held-out validation, and MuS2 provides an independent real 3.3m cross-check. **Remaining gap:** both HR sources (NAIP, WorldView-2) are non-Indian, so the <4m *resolution* claim is now defensible, but geographic generalization to NTRO's actual deployment geography (India) is still unverified — this is now the sharper, narrower open question (see Known Risks #5).

## Known Risks & Open Questions
1. **No direct MHAN+SPIFFNet vs DiffFuSR comparison was found.** Do not declare a winner without an experiment.
2. **[Resolved, round 1] <4m validation gap.** SEN2NAIP (2.5m, real cross-sensor pairs) and MuS2 (3.3m, real WorldView-2 reference) both provide genuine sub-4m ground truth — see Feasibility & Data Sources. The <4m resolution claim is now supportable with real data rather than extrapolation.
3. **Diffusion compute and inference cost need measurement.** DiffFuSR/EDiffSR provide relevant evidence, but project-specific latency/memory must be benchmarked.
4. **Uncertainty calibration is unresolved.** Compare a practical heteroscedastic/ensemble or MC-dropout approach against held-out residuals and hallucination metrics.
5. **Geographic domain gap (sharpened, round 1):** both available sub-4m HR sources (NAIP, WorldView-2) are non-Indian. The resolution claim is now defensible; whether the model generalizes to Indian land-cover/urban patterns is untested and is the more precise open question than the original generic "India-specific validation" note. Not mandated by the PS, but relevant to NTRO's actual use case.
6. **The team's original AegisSRM design rationale is not fully present in this repository.** Later agents should not assume missing details.

## Sources
- SEN2VENµS dataset: https://zenodo.org/records/6514159
- EDiffSR: https://arxiv.org/abs/2310.19288
- DiffFuSR: https://arxiv.org/abs/2506.11764
- SPIFFNet: https://arxiv.org/abs/2307.02974
- MHAN: https://ieeexplore.ieee.org/document/9151234/
- OpenSR-test: https://github.com/ESAOpenSR/opensr-test
- OpenSR: https://opensr.eu/
- SEN2NAIP: https://www.nature.com/articles/s41597-024-04214-y
- SEN2NAIPv2 (Hugging Face): https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- MuS2: https://doi.org/10.7910/DVN/1JMRAT (paper: https://www.nature.com/articles/s41597-023-02538-9 / arXiv 2210.02745)
