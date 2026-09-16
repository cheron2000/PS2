## Sonnet5 — Round 2
**Draft version reviewed:** v3

**Note:** I did round-1 work in parallel with a separate session that also ran rounds 1 and 2 under the name "Claude" plus AstraSR — a genuine race condition in how this multi-session protocol was operated (two chats editing concurrently). Their v3 independently reached similar SEN2NAIP-based conclusions to my round-1 push and is a coherent, well-cited draft, so rather than reconcile text line-by-line I rebased onto it and am contributing only what it's still missing.

**Critique:** v3 still leaves the India/generalization gap (risk #5) exactly as open-ended as it was at the end of round 1 — "no India-specific validation is currently established" — despite this being flagged as the round 2 standing question. It also drops MuS2 (present in round-1 AstraSR/Sonnet5 research) without comment; not wrong, but worth re-surfacing since it's an independent real-world cross-check the SEN2NAIP-only plan doesn't have.

**Research findings:**
- ISRO's Cartosat-3 (~1.12m MX, 0.25–0.28m PAN) and Cartosat-2S (~2m MX, ~0.65m PAN) are genuinely sub-4m and captured over Indian territory — already used in published research over Indian cities. Source: https://www.eoportal.org/satellite-missions/cartosat-3 ; https://arxiv.org/pdf/2409.05494
- Access route: ISRO/NRSC's Bhoonidhi portal (registration + license agreement; Cartosat-2S/3 MX listed as order-based, not instant free download). Source: https://bhoonidhi.nrsc.gov.in/
- (assumption, unverified) No public Cartosat↔Sentinel-2 paired dataset exists yet; building one is new work, and Bhoonidhi's order-based access means lead time/cost need an early check against the timeline — I did not find a documented typical turnaround and am not assuming one.
- MuS2 (real Sentinel-2 vs. real WorldView-2, ~3.3m, 91 scenes) re-added as a second real-world cross-check independent of the SEN2NAIP family. Source: https://doi.org/10.7910/DVN/1JMRAT

**Edits made to solution-draft.md:** v3 → v4. Added Cartosat-2S/3 + Bhoonidhi and a restored MuS2 mention to Feasibility & Data Sources; updated Known Risk #5 from open-ended to a concrete source + a narrower logistics caveat; expanded Sources.

**Vote:** NEEDS-REVISION
- Feasibility check: an Indian sub-4m source now has a name and access path; whether it's usable within a hackathon timeline (order lead time, building a pairing pipeline) is unverified, not resolved.
- Novelty check: unchanged — trustworthy, uncertainty-aware, hallucination-aware SR evaluation remains the defensible contribution, not the backbone.
- Biggest remaining risk: this draft has now been shaped by two sessions both claiming variations of "Claude" plus one "AstraSR" and one "Sonnet5," without a clean single review chain — the next agent should treat v4 as needing a full fresh read, not a diff against v3, since the history here is genuinely non-linear. Concretely unresolved: diffusion inference-cost measurement, uncertainty-calibration validation on held-out real data, and the Cartosat access-timeline question above.
