# AstraSR — Round 1 Research Notes

## Focus
Determine whether the current MHAN + SPIFFNet backbone is defensible for SIH26142, or whether newer Sentinel-2-specific diffusion approaches should replace it.

## Findings

1. **There is no evidence in the sources checked of a direct MHAN+SPIFFNet vs DiffFuSR head-to-head.** MHAN is a 2020/2021 remote-sensing SR CNN and SPIFFNet is a 2023 transformer. Their papers report gains over their contemporaneous baselines, but that does not establish superiority against current Sentinel-2-specific methods.

2. **DiffFuSR is materially closer to the actual problem than generic RSISR diffusion papers.** Its 2025 paper targets all 12 Sentinel-2 Level-2A bands at a unified 2.5 m GSD. It uses diffusion SR for RGB plus a learned fusion network for the remaining multispectral bands, and reports evaluation on the OpenSR benchmark. Source: https://arxiv.org/abs/2506.11764

3. **EDiffSR is useful prior art but is not a direct Sentinel-2 solution.** Its published experiments cover four remote-sensing datasets and its public implementation describes AID/DOTA/DIOR/NWPU-RESISC45 rather than a Sentinel-2 paired benchmark. Therefore it should be treated as architectural evidence for efficient diffusion, not as direct evidence that diffusion beats the proposed hybrid on SIH26142. Source: https://arxiv.org/abs/2310.19288

4. **OpenSR-test changes what should count as success.** The benchmark evaluates real Sentinel-2 SR using separate dimensions for reflectance consistency, spectral consistency, spatial alignment, synthesis, hallucination, omission, and improvement. Its current public benchmark includes Venµs, NAIP, SPOT, Spain Crops, and Spain Urban. Source: https://github.com/ESAOpenSR/opensr-test

5. **The current draft's uncertainty framing needs refinement.** OpenSR explicitly emphasizes trustworthy SR and confidence metrics, and recent OpenSR analysis reports a trade-off between preserving native Sentinel-2 measurements and introducing high-frequency spatial content. Thus a confidence/uncertainty map should be tied to measurable failure modes such as hallucination and spectral inconsistency, rather than presented only as a generic auxiliary output. Sources: https://opensr.eu/ and https://opensr.eu/news/spectral_usecases/

6. **SEN2VENµS is appropriate for supervised training but only provides 5 m reference imagery.** The dataset contains same-day, spatially registered Sentinel-2/VENµS pairs across 29 locations and 132,955 patches, for 8 Sentinel-2 bands. It therefore supports a strong 10/20 m -> 5 m training/evaluation stage, but it does not by itself validate the requested <4 m product. Source: https://zenodo.org/records/6514159

## Decision for the draft
Do not replace MHAN+SPIFFNet outright based on the available evidence. Instead, change the architecture proposal to a **benchmark-driven hybrid**: retain a lightweight deterministic CNN/transformer backbone as the primary reconstruction path, add a diffusion refinement branch only if ablation on OpenSR-test shows a real gain, and evaluate against DiffFuSR/EDiffSR where reproducible. The central novelty should move from “our backbone is new” to **spectral/geospatial consistency + calibrated uncertainty + hallucination-aware evaluation**.

## Biggest practical correction
The draft currently says “<4m” while its principal real paired dataset is 5 m. Training on SEN2VENµS can establish 5 m reconstruction, but a credible <4 m claim requires an additional reference/benchmark or a clearly labeled extrapolation. This must not be hidden behind visual quality.
