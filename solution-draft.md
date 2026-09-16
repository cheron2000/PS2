# Solution Draft — v4
Status: IN-PROGRESS
Last edited by: Sonnet5, round 2

## Problem Restatement
SIH26142 (NTRO): build a deep-learning super-resolution framework that takes 10m Sentinel-2 imagery and produces an enhanced product targeting <4m GSD while preserving geospatial and spectral consistency. The solution must include preprocessing, paired-data training, quantitative assessment, validation against real high-resolution references, and explicit uncertainty management because reconstructed detail is partly inferred.

## Proposed Approach
- Input: Sentinel-2 L2A, initially RGB+NIR 10m bands; extend to 20m/other bands after the core pipeline is stable.
- Primary reconstruction path: CNN high-order attention + transformer cross-stage fusion, treated as a fidelity-oriented baseline rather than claimed architectural novelty.
- Optional diffusion refinement is an experimental branch, not assumed to be superior. DiffFuSR is a key external baseline because it targets all 12 Sentinel-2 L2A bands at 2.5m GSD.
- Output: SR image plus a calibrated per-pixel uncertainty/confidence product.

## Technical Architecture
1. L2A ingestion, cloud/invalid-pixel masking, normalization and band resampling.
2. Pair quality control and co-registration.
3. CNN feature stem.
4. MHAN-style local high-frequency attention.
5. SPIFFNet-style cross-spatial/cross-stage transformer fusion.
6. Sub-pixel reconstruction head.
7. Heteroscedastic uncertainty head predicting mean and variance.
8. Optional lightweight perceptual/high-frequency loss, validated by ablation.
9. Geospatial output preserving CRS, affine transform and band metadata.

## Evaluation Protocol
Use geographically separated train/validation/test regions to prevent spatial leakage.

### Primary 4x target
Use **SEN2NAIP's real cross-sensor subset** as an external 10m → 2.5m test route. The dataset contains 2,851 Sentinel-2/NAIP pairs and deliberately represents HR NAIP at 2.5m for a 4x task. The authors apply spatial/spectral quality filtering and visual inspection. Source: https://www.nature.com/articles/s41597-024-04214-y

Use the synthetic SEN2NAIP/SEN2NAIPv2 data for scalable training, but do not treat synthetic targets as equivalent to independent real-world validation.

### Secondary 5m target
Use SEN2VENµS for same-day registered 10/20m → 5m validation. This provides a useful lower-scale check with reduced cross-sensor uncertainty.

### Benchmark metrics
Report:
- PSNR/SSIM where aligned references permit;
- spectral-angle error and reflectance consistency;
- spatial alignment;
- hallucination and omission;
- useful-detail/improvement measures;
- uncertainty calibration and correlation with reconstruction error.

OpenSR-test-style metrics should be preferred over visual inspection alone.

### External baseline
Where reproducible, compare with DiffFuSR. It reports 2.5m Sentinel-2 output and evaluation on OpenSR, making it directly relevant to the requested scale.

## Novelty / Differentiation
Do not claim that MHAN+SPIFFNet itself is novel. The defensible contribution is a **trustworthy SR pipeline** combining:
- fidelity-first reconstruction;
- explicit uncertainty;
- hallucination-aware evaluation;
- spectral/geospatial consistency constraints;
- empirical comparison of deterministic and diffusion alternatives.

## Feasibility & Data Sources
- **SEN2NAIP:** 2,851 real Sentinel-2/NAIP pairs, with 10m RGBNIR input and 2.5m HR representation for a 4x task; additionally provides synthetic training data.
- **SEN2NAIPv2:** current public release reports 62,242 LR/HR pairs and an x4 2.5m/10m synthetic setup, using blur/downsampling, reflectance harmonization and noise degradation. Source: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- **SEN2VENµS:** same-day registered Sentinel-2/VENµS pairs with 5m reference data for a secondary validation scale.
- **WorldStrat/MuS2:** optional additional benchmarks, subject to alignment and licensing checks. (MuS2 specifically: real Sentinel-2 vs. real WorldView-2 pairs at ~3.3m GSD/91 scenes/~2500 km², published with a full evaluation protocol — a genuine real-world cross-check independent of the SEN2NAIP family, not synthetic. Source: https://doi.org/10.7910/DVN/1JMRAT)
- **Cartosat-2S / Cartosat-3 (ISRO):** a concrete answer to the India/generalization gap below. Cartosat-3 delivers ~1.12m multispectral (0.25–0.28m panchromatic); Cartosat-2S delivers ~2m multispectral (~0.65m panchromatic) — both genuinely sub-4m and captured over Indian territory. Source: https://www.eoportal.org/satellite-missions/cartosat-3
  - Access is via ISRO/NRSC's **Bhoonidhi portal** (bhoonidhi.nrsc.gov.in): account registration + end-user license agreement; Cartosat-2S/3 MX products are listed as available for order, not instant free download like Sentinel-2. Source: https://bhoonidhi.nrsc.gov.in/
  - **(assumption, unverified):** no ready-made paired Cartosat↔Sentinel-2 LR-HR dataset exists publicly — building one (co-registration + harmonization, similar in spirit to SEN2NAIP's pipeline) is new work. Order lead time and licensing cost for Cartosat should be checked early against the project timeline rather than assumed trivial.

### Resolution claim
The earlier concern that the project lacked any sub-4m reference is now resolved. SEN2NAIP provides a direct 2.5m reference route for 4x Sentinel-2 SR. However, this is US-focused and cross-sensor. Therefore the evidence supports a **2.5m benchmark target**, not a universal claim that every geographic scene can reliably be reconstructed at 2.5m.

## Known Risks & Open Questions
1. **Architecture comparison:** literature supports testing diffusion rather than assuming it wins or loses. DiffFuSR is directly relevant and should be an external baseline.
2. **Cross-sensor domain gap:** SEN2NAIP's 2.5m real pairs are valuable but not same-sensor and are US-focused.
3. **Synthetic-data bias:** training on S2-like synthetic degradation can produce a model that performs well on its generator but transfers poorly to real Sentinel-2.
4. **Uncertainty calibration:** heteroscedastic variance is practical, but calibration must be measured on held-out real data.
5. **India/generalization (updated, round 2):** ISRO's Cartosat-2S/3, accessible via the Bhoonidhi portal, provide a credible Indian-geography sub-4m source — this answers the "does one exist" question raised since round 1. What remains open: Cartosat access is order/licence-based rather than instant download, no ready-made Cartosat↔Sentinel-2 paired dataset currently exists, and building one is unverified new work — order lead time and pairing effort should be checked early rather than assumed feasible within a hackathon timeline. This is a logistics/engineering gap now, not a data-existence gap.
6. **Perceptual loss:** adding perceptual/high-frequency loss may improve sharpness while harming spectral fidelity, so it must be ablated rather than assumed beneficial.

## Sources
- SEN2NAIP: https://www.nature.com/articles/s41597-024-04214-y
- SEN2NAIPv2: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- DiffFuSR: https://arxiv.org/abs/2506.11764
- OpenSR: https://opensr.eu/
- OpenSR-test: https://github.com/ESAOpenSR/opensr-test
- SEN2VENµS: https://zenodo.org/records/6514159
- MuS2: https://doi.org/10.7910/DVN/1JMRAT (paper: https://www.nature.com/articles/s41597-023-02538-9)
- Cartosat-3 specs: https://www.eoportal.org/satellite-missions/cartosat-3
- Bhoonidhi portal (ISRO/NRSC EO data hub): https://bhoonidhi.nrsc.gov.in/
