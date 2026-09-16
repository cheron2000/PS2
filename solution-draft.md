# Solution Draft — v8
Status: IN-PROGRESS
Last edited by: Claude, round 4

## Problem Restatement
SIH26142 (NTRO): build a deep-learning super-resolution framework that takes 10m Sentinel-2 imagery and produces an enhanced product targeting <4m GSD while preserving geospatial and spectral consistency. The solution must include preprocessing, paired-data training, quantitative assessment, validation against real high-resolution references, and explicit uncertainty management because reconstructed detail is partly inferred.

## Proposed Approach
- Input: Sentinel-2 L2A, initially RGB+NIR 10m bands; extend to 20m/other bands after the core pipeline is stable.
- Primary reconstruction path: CNN high-order attention + transformer cross-stage fusion, treated as a fidelity-oriented baseline rather than claimed architectural novelty.
- Optional diffusion refinement is an experimental branch, not assumed to be superior — see Known Risks #1 and #7 for why this needs care, not just a benchmark run.
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

### External baselines (revised, round 3 — see Known Risks #7)
Two published Sentinel-2-specific diffusion models are directly relevant and should both be treated as prior art to cite and, where feasible, benchmark against — not just "an external baseline":
- **DiffFuSR** (Sarmad et al., arXiv 2506.11764) — all-12-band Sentinel-2 SR to 2.5m, reports outperforming SOTA on fidelity/spectral consistency/hallucination suppression. **Caveat: ~170M parameters; reported inference on the order of hours per single Sentinel-2 tile (110×110km) on 4×A100 GPUs (100 DDIM steps).** Full-tile reproduction is very unlikely to be feasible on hackathon compute — treat as a numbers-from-the-paper comparison, not a from-scratch reproduction, unless patch-level inference on limited scenes is scoped down explicitly.
- **Donike et al. 2025** ("Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion," IEEE JSTARS) — **this is the closest existing prior art to our entire pitch**, not just an architecture data point. See Novelty section.

## Novelty / Differentiation
**Round 3 AstraSR review:** The current technical bundle is already substantially covered by ESA OpenSR: 4x Sentinel-2 SR, uncertainty estimation, spectral-consistency goals, and geospatial I/O are publicly implemented. Therefore these features must be treated as prior art, not as standalone novelty.

The remaining contribution must be experimental and explicitly scoped. Candidate directions are: (A) Indian-data generalization using Cartosat imagery if access is secured; (B) compute-constrained comparison with measured quality/latency/VRAM/throughput against OpenSR; or (C) both if resources permit. A generic OpenSR-test benchmark comparison is useful evidence but is not, by itself, a novelty claim.

**Round 3 finding — read this before pitching "uncertainty + fidelity + hallucination-awareness" as the differentiator:** Donike et al. 2025 already published, open-sourced, and pip-packaged (`opensr-model` / `opensr-utils`, github.com/ESAOpenSR) a Sentinel-2 10m→2.5m latent diffusion model that explicitly targets exactly this combination — spectral-consistency-preserving diffusion conditioned on the LR input, plus pixel-wise uncertainty maps, plus full Sentinel-2 `.SAFE` folder geospatial I/O (CRS/transform preserved). It is described as "the first multispectral RS super-resolution diffusion model efficient enough to process large-scale RS datasets... the only model providing a pixel-wise uncertainty metric" as of its publication. This is an ESA-affiliated team; given NTRO/ISRO judges for a Space Technology PS are plausibly aware of ESA's OpenSR initiative, not citing this paper is a real risk — a judge asking "how is this different from OpenSR?" should not catch the team flat-footed.

This does not mean the project has no room — it means the differentiation claim needs to be rewritten around what's actually still open, honestly:
- **India-geography validation** (Cartosat-2S/3 via Bhoonidhi, below) — not something OpenSR's published work targets.
- **Compute-tier comparison** — a lighter CNN+transformer path as a lower-compute alternative to a ~170M-parameter diffusion pipeline, explicitly positioned as "when you don't have 4×A100s," could be a legitimate, honestly-scoped contribution rather than a fidelity claim.
- **A comparative empirical study** (our backbone vs. DiffFuSR's reported numbers vs. OpenSR's approach) is itself a defensible hackathon-scale contribution — "we benchmarked the existing state of the art against a lighter alternative on Indian data" is a real, honest story. "We invented uncertainty-aware SR for satellites" is not, anymore.

**Round 4 finding — competitive landscape, not just prior published research:** searching for SIH26142 itself (not just the general SR literature) surfaces multiple other teams' public repos already building against this exact problem statement. Most relevant: a repo named DrishtiSR (github.com/shyam0github/DrishtiSR) has, as of this round, already executed candidate direction (B) — a ≤1M-parameter, CPU-only, INT8-ONNX-deployed model — to a substantial degree of rigor: 453 passing tests, a gated day-by-day methodology, OpenSR-test benchmarking against bicubic/nearest baselines, a rejected-with-evidence training-time spectral loss, and a from-scratch verification that their SEN2NAIPv2 cross-sensor pairs are genuinely cross-sensor rather than synthetically degraded. It also trains on the same SEN2NAIPv2 crosssensor subset this draft recommends, and its demo frontend includes a Delhi AOI — but explicitly *without* ground truth, used for visual demo only, not for validated Indian-geography training/evaluation.
- **Effect on strategy:** direction (B) — compute-tier comparison alone — is no longer differentiated from at least one visible competitor executing it well; treat it as validation that the technical approach is sound, not as a novelty claim on its own. Direction (A) — genuine India-geography validation *with real ground truth* (e.g., actually training/testing against Cartosat-2S/3 pairs, not just running inference on an Indian AOI for a demo) — remains the more clearly open differentiator, since the most advanced public competitor found this round stops short of it for exactly the reason this draft's Known Risk #5 identifies (no ready-made ground-truth pairing).
- This is competitive intelligence, not prior art to cite in a submission — do not reference other teams' repos in the team's own pitch. It should inform which of (A)/(B)/(C) the team commits to, not appear in the deliverable itself.

## Feasibility & Data Sources
- **SEN2NAIP:** 2,851 real Sentinel-2/NAIP pairs, with 10m RGBNIR input and 2.5m HR representation for a 4x task; additionally provides synthetic training data.
- **SEN2NAIPv2:** current public release reports 62,242 LR/HR pairs and an x4 2.5m/10m synthetic setup, using blur/downsampling, reflectance harmonization and noise degradation. Source: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- **SEN2VENµS:** same-day registered Sentinel-2/VENµS pairs with 5m reference data for a secondary validation scale.
- **WorldStrat/MuS2:** optional additional benchmarks, subject to alignment and licensing checks. (MuS2 specifically: real Sentinel-2 vs. real WorldView-2 pairs at ~3.3m GSD/91 scenes/~2500 km², published with a full evaluation protocol — a genuine real-world cross-check independent of the SEN2NAIP family, not synthetic. Source: https://doi.org/10.7910/DVN/1JMRAT)
- **Cartosat-2S / Cartosat-3 (ISRO):** a concrete answer to the India/generalization gap below. Cartosat-3 delivers ~1.12m multispectral (0.25–0.28m panchromatic); Cartosat-2S delivers ~2m multispectral (~0.65m panchromatic) — both genuinely sub-4m and captured over Indian territory. Source: https://www.eoportal.org/satellite-missions/cartosat-3
  - Access is via ISRO/NRSC's **Bhoonidhi portal** (bhoonidhi.nrsc.gov.in): account registration + end-user license agreement; Cartosat-2S/3 MX products are listed as available for order, not instant free download like Sentinel-2. Source: https://bhoonidhi.nrsc.gov.in/
  - **Cost (round 4, from NSIL/NRSC's own official FAQ, not a secondary source):** under India's 2023 Space Policy, data finer than 5m is priced for Non-Government Entities (NGE) and free for Government Entities (GE) with a submitted declaration. Exact Cartosat pricing table (17km×17km / 9.8km×9.8km scenes, NGE): Cartosat-3 PAN 0.28m GeoOrthokit ₹3,860, Ortho Rectified ₹4,810; Cartosat-3 MX 1.10m GeoOrthokit ₹3,860, Ortho Rectified ₹4,810/₹1,700 (10km tile); Cartosat-2S PAN 0.60m GeoOrthokit ₹2,060; Cartosat-2S MX 1.60m GeoOrthokit ₹2,060. All modest for a hackathon budget. Ordering requires a separate NSIL account (email eodata@nsilindia.co.in). Source (primary, official): https://bhoonidhi.nrsc.gov.in/imgarchive/bhoonidhi_videos_help/Commercial_Products_FAQ.pdf
  - **GE eligibility — now answered directly (round 4):** the FAQ's own Q9 states "Government **Academic Institutions** may submit GE declaration form" — i.e. eligibility depends on whether the team's *institution* is itself a government academic institution, not on being a student. Worth the team checking their specific college's status, but the likely default for most engineering colleges (including private-trust-affiliated ones) is NGE — budget accordingly rather than assume free access.
  - **Normal-priority turnaround — confirmed genuinely undocumented, not just unresearched (round 4):** the official FAQ states urgent orders (5–6hr) carry a 50% surcharge and that normal-priority orders are serviced "first come, first served" — no SLA or day/hour figure is published anywhere in NRSC's own materials found so far. This isn't a research gap that more searching will close; it needs a direct email to data@nrsc.gov.in / eodata@nsilindia.co.in, or a real test order placed early, to get an actual number.
  - **(assumption, unverified):** no ready-made paired Cartosat↔Sentinel-2 LR-HR dataset exists publicly — building one (co-registration + harmonization, similar in spirit to SEN2NAIP's pipeline) is new work.

### Resolution claim
The earlier concern that the project lacked any sub-4m reference is now resolved. SEN2NAIP provides a direct 2.5m reference route for 4x Sentinel-2 SR. However, this is US-focused and cross-sensor. Therefore the evidence supports a **2.5m benchmark target**, not a universal claim that every geographic scene can reliably be reconstructed at 2.5m.

## Known Risks & Open Questions
1. **Architecture comparison (revised, round 3):** Not a settled "CNN beats diffusion" or vice versa. Generic RSISR diffusion models (EDiffSR etc., benchmarked on UCMerced/AID) do trade fidelity for perceptual realism. But domain-specialized, Sentinel-2-specific diffusion models (DiffFuSR, Donike et al. 2025) explicitly report *strong* fidelity and hallucination-suppression, not the generic tradeoff. The real constraint is compute (#7 below), not architecture quality.
2. **Cross-sensor domain gap:** SEN2NAIP's 2.5m real pairs are valuable but not same-sensor and are US-focused.
3. **Synthetic-data bias:** training on S2-like synthetic degradation can produce a model that performs well on its generator but transfers poorly to real Sentinel-2.
4. **Uncertainty calibration:** heteroscedastic variance is practical, but calibration must be measured on held-out real data.
5. **India/generalization (updated, round 4 — logistics now as resolved as desk research allows):** ISRO's Cartosat-2S/3, accessible via the Bhoonidhi portal, provide a credible Indian-geography sub-4m source. Cost is confirmed via NRSC's own FAQ (₹2,060–4,810/scene depending on product, NGE) — not a budget blocker. GE-free-access eligibility depends on the team's *institution* being a government academic institution (check directly, don't assume). Normal-priority order turnaround is confirmed to be genuinely undocumented publicly (not just unresearched) — the only way to know is to ask NRSC/NSIL directly or place a test order early. Remaining work: building a Cartosat↔Sentinel-2 pairing pipeline from scratch, since no ready-made one exists.
6. **Perceptual loss:** adding perceptual/high-frequency loss may improve sharpness while harming spectral fidelity, so it must be ablated rather than assumed beneficial.
7. **Prior-art overlap — the most important open item right now (updated, round 4):** Donike et al. 2025 / ESA's open-sourced `opensr-model` already does Sentinel-2 10m→2.5m latent diffusion with pixel-wise uncertainty and preserved spectral consistency (round 3). Round 4 adds: at least one other visible SIH26142 competitor (see Novelty section) has already substantially executed the "compute-tier" differentiation path with real rigor. Combined, this means neither "uncertainty-aware SR" nor "lightweight compute-tier SR" alone is a safe novelty claim anymore — the team needs an explicit answer to "how is this different," and genuine India-ground-truth validation is now the most clearly still-open option. This is a research/positioning task, not a coding task, and should be the next agent's top priority if the human team hasn't already decided.
8. **Diffusion compute cost:** DiffFuSR reports ~170M parameters and on the order of hours of inference per full Sentinel-2 tile on 4×A100 GPUs. Any diffusion component in the team's own pipeline (not just benchmarking) needs a realistic compute-budget check against actual available hardware (Colab/Kaggle-tier GPU, most likely) before being scoped as a build target rather than a numbers-from-the-paper comparison.
9. **No official PS dataset (confirmed, round 4):** searching specifically for SIH26142/NTRO did not surface any organizer-provided dataset or reference implementation — every public team found (including this one) is independently sourcing data from the sources in Feasibility & Data Sources above. This confirms rather than changes the current plan, but it's worth the human team double-checking the official SIH portal/problem-statement page directly in case a dataset was added after this round's search.

## Sources
- SEN2NAIP: https://www.nature.com/articles/s41597-024-04214-y
- SEN2NAIPv2: https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- DiffFuSR: https://arxiv.org/abs/2506.11764
- DiffFuSR code: https://github.com/NorskRegnesentral/DiffFuSR
- Donike et al. 2025, "Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion," IEEE JSTARS 18, pp. 6940-6952, DOI 10.1109/JSTARS.2025.3542220
- opensr-model (Donike et al. implementation): https://github.com/ESAOpenSR/opensr-model
- OpenSR: https://opensr.eu/
- OpenSR-test: https://github.com/ESAOpenSR/opensr-test
- SEN2VENµS: https://zenodo.org/records/6514159
- MuS2: https://doi.org/10.7910/DVN/1JMRAT (paper: https://www.nature.com/articles/s41597-023-02538-9)
- Cartosat-3 specs: https://www.eoportal.org/satellite-missions/cartosat-3
- Bhoonidhi portal (ISRO/NRSC EO data hub): https://bhoonidhi.nrsc.gov.in/
- Bhoonidhi/NSIL Commercial Products FAQ (primary source for pricing, GE/NGE eligibility, and turnaround terms): https://bhoonidhi.nrsc.gov.in/imgarchive/bhoonidhi_videos_help/Commercial_Products_FAQ.pdf
- Bhoonidhi/NSIL pricing policy (secondary, background): https://www.nsilindia.co.in/news-details/614 ; https://aidigitalnews.com/ai/why-isros-bhoonidhi-is-on-par-with-nasas-datasets/
