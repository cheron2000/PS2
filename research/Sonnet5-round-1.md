# Research Notes — Sonnet5, Round 1

**Open question addressed:** the standing question in status.md — find a credible sub-4m validation strategy so the <4m claim in the PS can be substantiated without overstating results.

## Finding 1: SEN2NAIP already provides real, cross-sensor sub-4m pairs
- SEN2NAIP's cross-sensor component pairs Sentinel-2 L2A (10m) with NAIP imagery downloaded at **2.5m** GSD (NAIP's native ~1m resolution, retrieved from a Google Earth Engine pyramid level chosen specifically to support 4x SR). 2,851 real LR-HR pairs, each covering 1.46 km², across the contiguous United States.
- A degradation model converts NAIP into S2-like LR imagery, producing a second, larger synthetic subset (35,314 pairs) for additional training volume.
- SEN2NAIPv2 (Hugging Face, tacofoundation/SEN2NAIPv2) extends this to 62,242 pairs, still at the 2.5m/10m pairing.
- Source: https://www.nature.com/articles/s41597-024-04214-y ; https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2
- **This directly resolves the gap**: unlike SEN2VENµS (5m only), SEN2NAIP's HR reference is already finer than the PS's <4m target, so both training and validation against a real (not synthetic-degraded) sub-4m reference is achievable today.
- Caveat: NAIP coverage is limited to the continental United States. Using it for training is fine, but it does not provide India-specific ground truth — this reinforces (does not solve) known risk #5.

## Finding 2: MuS2 is an independent, real-world sub-4m benchmark using WorldView-2
- MuS2 (Kowaleczko et al., Scientific Data 2023; arXiv 2210.02745) pairs real Sentinel-2 multi-temporal stacks with real WorldView-2 imagery as HR reference, at **3x magnification (~3.3m GSD)** — genuinely sub-4m, no synthetic degradation.
- 91 real scenes, ~2500 km² total coverage, published with a full evaluation protocol (including a note that PSNR/SSIM alone don't correlate well with perceptual quality against a true cross-sensor reference — consistent with why the draft already avoids PSNR/SSIM as the primary metric).
- Data: https://doi.org/10.7910/DVN/1JMRAT ; paper: https://www.nature.com/articles/s41597-023-02538-9
- Important architectural note: MuS2 is built for **multi-image** SR (MISR), fusing multiple revisits, not single-image SR. The current draft's backbone (MHAN+SPIFFNet, DiffFuSR) is single-image. MuS2 can still be used for single-image evaluation (score against one revisit at a time), but using it as a MISR benchmark would require an architecture extension — flagged as a future option, not a requirement.

## Finding 3: DiffFuSR's own 2.5m claim is already trained/evaluated on SEN2NAIP
- Confirms DiffFuSR's <4m (2.5m) output claim in the current draft is not hypothetical — it is grounded in the same SEN2NAIP data now recommended here, which strengthens using DiffFuSR as a directly comparable baseline rather than an untested reference.
- Source: https://arxiv.org/pdf/2506.11764 (Datasets section, confirms SEN2NAIP as primary training data)

## Net effect on the draft
- The <4m validation gap (status.md's standing question, and Known Risk #2) is now addressable with real data: SEN2NAIP for training + held-out validation at 2.5m, MuS2 as an independent real-world sub-4m cross-check at ~3.3m.
- This does not eliminate all risk — NAIP/WorldView-2 are both non-Indian sources, so geographic generalization to NTRO's actual use case remains unverified (existing risk #5, unchanged).
