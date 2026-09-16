# Research Notes — Sonnet5, Round 2

**Note on process:** I did round-1 work independently (see research/Sonnet5-round-1.md) which was pushed and merged. While preparing a follow-up push, a separate session (also identifying as "Claude," plus AstraSR) had independently run round 2, rewriting solution-draft.md and arriving at similar SEN2NAIP-based conclusions for the <4m gap. Rather than overwrite that work, I rebased onto their v3 and am contributing only the delta they don't yet have: India-specific sub-4m data. My earlier MuS2 finding (round 1) also wasn't reflected in their rewrite, so I re-added a short mention of it here too.

**Open question addressed:** round 2's standing question ("India-specific validation dataset gap — open since round 1 — still unaddressed").

## Finding: ISRO's Cartosat-2S/3 closes the India-geography data-existence question
- **Cartosat-3**: panchromatic 0.25–0.28m, multispectral (4-band, incl. NIR) ~1.12–1.13m. Already used in published research over Indian cities (LULC segmentation study, 6 Indian cities). Source: https://www.eoportal.org/satellite-missions/cartosat-3 ; https://arxiv.org/pdf/2409.05494
- **Cartosat-2S**: panchromatic ~0.65m, multispectral ~2m, longer archive depth.
- **Access**: ISRO/NRSC's Bhoonidhi portal (bhoonidhi.nrsc.gov.in) — registration + end-user license agreement required; Cartosat-2S/3 MX products explicitly listed as available for order (not instant free download like Sentinel-2). Source: https://bhoonidhi.nrsc.gov.in/ ; ISRO Bhoonidhi Newsletter 2024 Ed. 1.
- **Caveat (assumption, unverified):** no public Cartosat↔Sentinel-2 paired LR-HR dataset exists yet, unlike SEN2NAIP. Building one (co-registration + reflectance harmonization) is real, likely-necessary new work. Cartosat's order-based access model (vs. Sentinel-2's instant download) means lead time and per-scene cost should be checked directly against the project timeline — this is a logistics question, and I did not find a documented typical turnaround time in this pass, so it stays flagged rather than assumed short.

## Secondary note: MuS2 (carried over from round 1, not present in the round-2 rewrite)
- Real Sentinel-2 vs. real WorldView-2 benchmark at ~3.3m GSD (3x), 91 scenes, ~2500 km², published evaluation protocol. Independent of the SEN2NAIP data family — useful as a second real-world (non-synthetic) cross-check. Source: https://doi.org/10.7910/DVN/1JMRAT
- Built for multi-image SR; usable for single-image evaluation by scoring one revisit at a time, but that's a repurposing, not its native design — flagged as-is, not overstated.

## Net effect on the draft
- Known Risk #5 (India/generalization) moves from "no India-specific validation is currently established" to "a concrete source exists (Cartosat via Bhoonidhi); the open item is now access lead time + building the pairing pipeline." This is a narrower, more actionable risk for whoever picks it up next.
