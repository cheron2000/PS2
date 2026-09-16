# Solution Draft — v2
Status: IN-PROGRESS
Last edited by: Claude, round 2

## Problem Restatement
SIH26142 (NTRO): build a deep-learning super-resolution framework that takes medium-resolution satellite imagery (10m Sentinel-2) and produces sharper, information-rich output (<4m), while preserving geospatial and spectral consistency. Must include pre-processing, training on paired datasets, accuracy assessment, and validation against real high-resolution reference imagery — and must explicitly manage uncertainty, since some reconstructed detail is inferred, not observed.

## Proposed Approach
- Input: Sentinel-2 L2A, starting with the 10m bands (RGB + NIR), extending to 20m bands (SWIR) as a stretch goal.
- **Backbone confirmed (revised reasoning, round 2):** CNN high-order-attention stage (MHAN-style) + transformer cross-stage fusion (SPIFFNet-style), retained as PSNR/fidelity-oriented rather than switched to diffusion. See Known Risks — this was an open question in round 1 and is now resolved with evidence, not just inherited.
- Output: super-resolved image at <4m GSD **plus a per-pixel aleatoric uncertainty map**, produced by the same network via a heteroscedastic loss (design finalized this round, see Technical Architecture).

## Technical Architecture
1. Shallow feature extraction (conv stem)
2. MHAN-style high-order attention blocks — local high-frequency detail
3. SPIFFNet-style transformer blocks (cross-spatial pixel integration + cross-stage feature fusion) — global context
4. Sub-pixel convolution upsampling head → target GSD
5. **Uncertainty head (resolved, round 2):** dual-output head predicting pixel-wise mean + variance, trained with a Gaussian or Laplacian negative-log-likelihood (NLL) loss — i.e., heteroscedastic aleatoric uncertainty. No architectural detour needed (no MC-dropout inference-time sampling, no diffusion ensemble) — just one extra output channel and a different loss term on top of the existing backbone.
6. **Under consideration:** a light perceptual/high-frequency loss term added to the primary pixel loss, to offset the known blur tendency of pure PSNR-optimized models (see Known Risks #1) — needs an ablation, not yet decided.

## Novelty / Differentiation
- Explicit, PS-mandated uncertainty output rather than a bare enhanced image — addressed via a method with direct precedent in *satellite and drone imagery* SR specifically (not generic computer vision), which strengthens the "scientifically reliable" claim the PS asks for.
- Backbone choice is now a justified decision, not an inherited guess: diffusion-based alternatives were evaluated and found to trade away exactly the property (pixel/spectral fidelity) that this PS explicitly requires. That reasoning itself is worth stating in the pitch — it shows judges the team evaluated the trendier option and had a specific, evidence-based reason not to chase it.

## Feasibility & Data Sources
Real, freely accessible paired datasets (unchanged from v1):
- **SEN2VENµS** (Zenodo 6514159 / HuggingFace tacofoundation/sen2venus) — Sentinel-2 10/20m paired with VENµS 5m reference, same-day acquisition, 29 sites, ~133k patches. Lowest misalignment risk of the options here.
- **SEN2NAIP** (Nature Sci. Data, 2024) — 2,851 real Sentinel-2/NAIP pairs + 35k synthetic S2-like pairs. Largest volume, but NAIP coverage is US-only.
- **WorldStrat** (NeurIPS 2022) — Sentinel-2 + SPOT 6/7 (1.5m), globally distributed, known misalignment issues.
- **MuS2** — Sentinel-2 + WorldView-2 real-world multi-image benchmark.

Cross-sensor misalignment risk (flagged in v1) still stands — unresolved, carried forward.

## Known Risks & Open Questions
1. ~~Architecture currency unverified~~ **RESOLVED, round 2:** Diffusion-based RSISR methods (EDiffSR, IRSDE, SR3) were directly compared against PSNR-oriented CNN/transformer models (EDSR, RCAN, HAT-L) in the literature. Finding: diffusion models score *better* on perceptual/realism metrics (NIQE, FID) but *worse* on PSNR/SSIM — i.e., they trade pixel and spectral fidelity for "natural-looking" output, and are prone to hallucinating plausible-but-unobserved detail. One paper states plainly that current diffusion RSISR methods "fail to achieve substantial improvements in perceptual quality over CNN- and transformer-based approaches" despite longer inference time. Since the PS explicitly demands *preserved geospatial and spectral consistency* and *scientific reliability* — not just visual sharpness — a PSNR-oriented CNN+transformer backbone (our MHAN+SPIFFNet direction) is the better-justified choice, not diffusion. **New sub-risk surfaced by this finding:** PSNR-optimized models (e.g., HAT-L in the comparison) are noted to produce visibly blurry output — worth an ablation on adding a light perceptual loss term without going as far as full diffusion.
2. **Still open:** No India-specific validation set identified. All three main datasets are international.
3. ~~Uncertainty-head design undecided~~ **RESOLVED, round 2:** Heteroscedastic aleatoric uncertainty via Gaussian/Laplacian NLL loss, predicting pixel-wise mean + variance in one forward pass. Directly precedented in two satellite/drone-imagery-specific SR papers (Valsesia & Magli 2021 on satellite image SR; SU-ESRGAN 2025 on satellite/drone SR with fine-tuning). Cheaper than MC-dropout (no repeated stochastic inference) and far cheaper than a diffusion ensemble. Epistemic uncertainty (MC-dropout) is a possible stretch addition, not required for the core deliverable.
4. **Not yet independently reviewed.** Everything in this draft, round 1 and round 2, has come from Claude alone — no second model has weighed in yet. Treat both "resolved" items above as well-evidenced, not as settled community consensus, until another agent checks them.
