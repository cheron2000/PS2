# Solution Draft — v1
Status: IN-PROGRESS
Last edited by: Claude, round 1

## Problem Restatement
SIH26142 (NTRO): build a deep-learning super-resolution framework that takes medium-resolution satellite imagery (10m Sentinel-2) and produces sharper, information-rich output (<4m), while preserving geospatial and spectral consistency. Must include pre-processing, training on paired datasets, accuracy assessment, and validation against real high-resolution reference imagery — and must explicitly manage uncertainty, since some reconstructed detail is inferred, not observed.

## Proposed Approach
- Input: Sentinel-2 L2A, starting with the 10m bands (RGB + NIR), extending to 20m bands (SWIR) as a stretch goal.
- Hybrid backbone inherited from the team's prior AegisSRM design: a CNN high-order-attention stage (MHAN-style) for local texture/edge recovery, feeding a transformer cross-stage fusion stage (SPIFFNet-style) for long-range context.
- Output: super-resolved image at <4m GSD **plus an explicit per-pixel uncertainty/confidence map** — this second output is not optional decoration, it's a direct response to the PS's uncertainty requirement, and most baseline SR papers skip it.

## Technical Architecture
1. Shallow feature extraction (conv stem)
2. MHAN-style high-order attention blocks — local high-frequency detail
3. SPIFFNet-style transformer blocks (cross-spatial pixel integration + cross-stage feature fusion) — global context
4. Sub-pixel convolution upsampling head → target GSD
5. Auxiliary uncertainty head (candidate methods: MC-dropout, heteroscedastic variance loss, or a small diffusion-based ensemble — **unresolved, see Open Questions**)

## Novelty / Differentiation
- Explicit, PS-mandated uncertainty output rather than a bare enhanced image — this is the one thing that directly answers "the system... must clearly manage uncertainty" in a way a plain SR model doesn't.
- Positions the project alongside real prior art: ESA's OpenSR initiative is explicitly framed around "robust, accountable super-resolution for Sentinel-2" — worth studying as both validation that this angle matters and as a benchmark to differentiate against.

## Feasibility & Data Sources
Real, freely accessible paired datasets (not synthetic-only):
- **SEN2VENµS** (Zenodo 6514159 / HuggingFace tacofoundation/sen2venus) — Sentinel-2 10/20m paired with VENµS 5m reference, same-day acquisition, 29 sites, ~133k patches. Lowest misalignment risk of the options here.
- **SEN2NAIP** (Nature Sci. Data, 2024) — 2,851 real Sentinel-2/NAIP pairs + 35k synthetic S2-like pairs. Largest volume, but NAIP coverage is US-only — geography mismatch if judges want India-specific validation.
- **WorldStrat** (NeurIPS 2022) — Sentinel-2 + SPOT 6/7 (1.5m), globally distributed.
- **MuS2** — Sentinel-2 + WorldView-2 real-world multi-image benchmark.

**Known hard problem, not a footnote:** cross-sensor pairs (WorldStrat, SEN2NAIP's real subset) are only *weakly* aligned — different sensors, different passes. A SEN2NAIP-adjacent paper states this plainly: such datasets "are not suitable to train conventional SR models" without correction. This needs an explicit co-registration step or a training method that's robust to misalignment (e.g., flow/bridge-matching approaches) — not a naive pixel-wise supervised loss.

## Known Risks & Open Questions
1. **Architecture currency is unverified.** MHAN (2020, CNN) and SPIFFNet (2023, transformer) are solid but not the newest word — diffusion-based RSISR methods (EDiffSR, arXiv 2310.19288; DiffFuSR, arXiv 2506.11764) claim to beat CNN/GAN-era methods on perceptual quality. Nobody has benchmarked our hybrid against these yet. **Assign this to round 2.**
2. **No India-specific validation set identified.** All three main datasets above are international. PS doesn't require India-specific data, but if judges weight real-world relevance, this is a gap.
3. **Uncertainty-head design undecided** — MC-dropout vs. heteroscedastic loss vs. diffusion ensemble each have different compute/complexity tradeoffs, not yet compared.
4. **I (Claude, round 1) have only seen a one-line summary of the team's original AegisSRM doc, not the full write-up.** Whoever has access to the original should paste its actual architecture reasoning into the repo before later rounds assume this draft fully represents it.
