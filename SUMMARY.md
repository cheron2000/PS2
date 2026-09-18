# SIH26142 — Sentinel-2 Super-Resolution: Solution Summary

*NTRO Problem Statement 26142 — Deep-learning super-resolution of Sentinel-2 imagery to sub-4m resolution*

---

## Executive Summary

Sentinel-2 gives us free, global, frequently-updated satellite imagery — but only at 10m resolution, too coarse for fine-grained crop monitoring, urban mapping, or disaster response. We're building a deep-learning model that reconstructs Sentinel-2 imagery at **under 4 meters resolution**, with two things most super-resolution projects skip: an **honest confidence map** showing where the model is inferring detail versus reconstructing it reliably, and **real validation against Indian ground truth**, not just imagery from other countries.

That second point is our actual contribution. High-quality Sentinel-2 super-resolution already exists in published, open-source form (ESA's OpenSR and SEN2SR projects, and at least one strong competing hackathon submission). What none of them do — and what we could not find anywhere in the public literature — is validate against real high-resolution Indian satellite imagery (ISRO's Cartosat). We can, because it's available, affordable, and unused for this purpose. That's the pitch.

---

## 1. The Problem

Sentinel-2's 10m bands are too coarse for tasks the PS names directly: crop monitoring, urban mapping, change detection, disaster response. The ask is a model that:
- Takes 10m Sentinel-2 imagery as input
- Outputs a reconstruction at **<4m GSD**
- Preserves geospatial and spectral accuracy (it must still be *true* imagery, not just sharper-looking)
- Comes with **explicit uncertainty accounting** — since detail below 10m is partly inferred, not observed, the model must say where it's confident and where it's guessing
- Demonstrably improves **interpretability and analytical utility** for downstream tasks, not just pixel-level image quality

## 2. Our Approach

**Architecture** — a fidelity-first backbone, not an experimental one:
1. Sentinel-2 L2A ingestion → cloud/invalid-pixel masking, normalization, band resampling
2. Cross-sensor pair quality control and co-registration
3. CNN feature extraction
4. High-order local attention (MHAN-style) for fine spatial detail
5. Cross-stage transformer fusion (SPIFFNet-style)
6. Sub-pixel reconstruction head
7. **Heteroscedastic uncertainty head** — predicts a variance alongside the mean, giving a genuine per-pixel confidence map, not a bolted-on heuristic
8. Geospatial-correct output (CRS, affine transform, band metadata preserved)

We're starting with RGB+NIR 10m bands and extending to 20m bands once the core pipeline is stable. An optional diffusion-based refinement branch is being evaluated separately — it's the highest-fidelity option in the published literature, but at ~170M parameters and hours of inference per tile on 4×A100s, it's realistically a **benchmark comparison**, not something we'll train from scratch on hackathon compute.

## 3. What Makes This Different

Be upfront about this with judges: **uncertainty-aware, spectrally-consistent Sentinel-2 super-resolution to ~2.5m already exists and is open-source.** ESA's `opensr-model` (Donike et al. 2025) does almost exactly what a naive pitch of "SR + uncertainty + spectral fidelity" would claim as novel. ESA's newer `SEN2SR`/`SEN2SRLite` package is pip-installable and runs on a free Colab GPU today. At least one other public SIH26142 team has already built a rigorous, lightweight (≤1M parameter, CPU-only) version of this. Citing these upfront is safer than a judge catching the gap — and it sets up the real differentiator:

**None of them validate against Indian ground truth.** Every benchmark in the published ecosystem (SEN2NAIP, SEN2VENµS, MuS2, SEN2NEON) uses US or European high-resolution imagery as the reference. For a problem statement issued by NTRO, for Indian deployment, that's a real gap. ISRO's own Cartosat-2S/3 satellites deliver genuinely sub-4m imagery (down to 0.25m panchromatic) over Indian territory, and it's accessible for a modest fee through ISRO/NRSC's Bhoonidhi portal. **Training and evaluating against real Cartosat ground truth — not a demo inference run, but actual validated accuracy numbers on Indian geography — is the part nobody else has done.**

Secondary differentiator: we measure **downstream task utility** (e.g., land-cover/crop classification accuracy using the SR output vs. the raw LR input), not just image-quality metrics. Recent research shows these can be decoupled — a model can score well on PSNR and still not help a real classification task. This directly answers the PS's "analytical utility" requirement, which pure fidelity metrics don't.

## 4. Evaluation Plan

| Validation route | Resolution | Real or synthetic | Purpose |
|---|---|---|---|
| SEN2NAIP (real cross-sensor pairs) | 2.5m | Real | Primary 4× benchmark |
| SEN2VENµS | 5m | Real | Secondary benchmark |
| **Cartosat-2S/3 (via Bhoonidhi)** | **~1.1m (MX)** | **Real** | **India-specific validation — the differentiator** |
| ESA WorldCover-based classification | 10m labels | Real | Downstream-task utility check |

Metrics reported: PSNR/SSIM (where alignment permits), spectral-angle error, hallucination/omission, uncertainty calibration against held-out error, and at least one downstream-task accuracy number (e.g., IoU on land-cover classes).

Baselines we cite and benchmark against (numbers-from-paper where full reproduction isn't feasible): DiffFuSR, ESA's OpenSR / SEN2SR / SEN2SRLite.

## 5. Data Access — What the Team Needs to Do

| Source | Cost | Access |
|---|---|---|
| Sentinel-2 L2A | Free | Instant, Copernicus/GEE |
| SEN2NAIP / SEN2NAIPv2 | Free | Instant, HuggingFace |
| SEN2VENµS | Free | Instant, Zenodo |
| ESA WorldCover | Free | Instant |
| **Cartosat-2S/3** | **₹2,060–4,810/scene** (Non-Government pricing) | **Bhoonidhi portal — requires NSIL account (email eodata@nsilindia.co.in), order-based, turnaround undocumented** |

**Action: place a real Bhoonidhi test order this week.** Cost and eligibility are known; actual delivery turnaround isn't published anywhere and can only be learned by asking or ordering.

**Action: build the Cartosat↔Sentinel-2 pairing pipeline.** No ready-made one exists publicly — this is genuine engineering work (co-registration + reflectance harmonization), not a data-download task.

## 6. Immediate To-Do List

1. **This week:** Bhoonidhi/NSIL account + test Cartosat order (learn real turnaround before it blocks the timeline)
2. Stand up the SEN2NAIP/SEN2NAIPv2 + SEN2VENµS training/eval pipeline; get a baseline model training
3. `pip install sen2sr` and run SEN2SRLite as a reference baseline — five minutes of work, gives an immediate comparison point
4. Start the Cartosat↔Sentinel-2 co-registration pipeline in parallel — it's the long pole
5. Wire ESA WorldCover into the eval harness for the downstream-task metric
6. Prepare a one-slide, judge-facing answer to "how is this different from OpenSR" — the answer is in Section 3 above; don't let it be improvised on the spot

## 7. Known Risks (honest, not blockers)

- **Diffusion refinement is compute-heavy** — treat as a cited comparison, not a build target, unless patch-level scope is deliberately narrowed.
- **No off-the-shelf Cartosat↔Sentinel-2 pairing pipeline** — budget real time for this.
- **Cartosat order turnaround is unknown** — resolve early, don't assume.
- **Perceptual/high-frequency loss can hurt spectral fidelity** — ablate before adopting.
- **Two of three research reviewers on this project (during the research phase) were the same underlying AI model** — worth knowing that the "independent" convergence on this strategy had some correlation, even though the conclusions held up under direct stress-testing.

## 8. Key Sources

- SEN2NAIP: nature.com/articles/s41597-024-04214-y · SEN2NAIPv2: huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- SEN2VENµS: zenodo.org/records/6514159
- DiffFuSR: arxiv.org/abs/2506.11764
- ESA OpenSR (Donike et al. 2025, IEEE JSTARS): github.com/ESAOpenSR/opensr-model
- SEN2SR / SEN2SRLite: github.com/ESAOpenSR/SEN2SR · pypi.org/project/sen2sr
- Cartosat-3 specs: eoportal.org/satellite-missions/cartosat-3
- Bhoonidhi portal: bhoonidhi.nrsc.gov.in · pricing/eligibility FAQ: bhoonidhi.nrsc.gov.in/imgarchive/bhoonidhi_videos_help/Commercial_Products_FAQ.pdf
- ESA WorldCover: esa-worldcover.org
- GeoSR-Bench (downstream-task benchmarking): arxiv.org/abs/2605.00310

---

*This is a condensed, presentation-ready summary of the full research process (8 rounds, 3 independent reviewers, v11 final). Full citations, round-by-round reasoning, and every source checked are in `solution-draft.md` in the repo, for anyone who wants the underlying detail.*
