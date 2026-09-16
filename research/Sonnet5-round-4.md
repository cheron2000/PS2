# Research Notes — Sonnet5, Round 4 (fresh full read, per instruction)

**Process note before research:** solution-draft.md's header said "v5, Claude round 3" even though status.md (draft_version: 6) and AstraSR's round-4 commit message ("...after AstraSR v6 edit") both indicate a v6 exists. Checked the actual diff (commit 125b2a0) — AstraSR edited content (added the "Round 3 AstraSR review" paragraph to Novelty) but never bumped the header number. Corrected: treated AstraSR's edit as the uncounted v6, and my edit below as v7.

## Finding 1: no official SIH26142 dataset exists (confirmed, not just assumed)
Searched specifically for the PS ID rather than the general SR literature. No organizer-provided dataset or starter kit surfaced from NTRO/SIH. What did surface: several independent public GitHub repos of other teams solving this exact PS, all independently sourcing their own data — consistent with (and now confirming) every prior round's assumption. Sources: SIH 2026 PS browsers (github.com/Sovereign-Immortal/SIH-2026-Browser, github.com/NoBugNinja/...), multiple independent team repos (TIRUNARA/sih-26142-super-resolution-mapping, mrparasssingh/GeoRes, shyam0github/DrishtiSR, varishsharma81106-sketch/sih26142).

## Finding 2: a competing team has already substantially executed differentiation path (B)
`github.com/shyam0github/DrishtiSR` — 10m→2.5m Sentinel-2 SR for SIH26142, publicly documented with unusual rigor:
- ≤1,000,000 parameters, CPU-only, INT8 ONNX deployment (candidate path B, executed, not just proposed)
- Trains on the SEN2NAIPv2 crosssensor subset (the same dataset this draft recommends) — 3,000 pairs cached, filtered at min_correlation ≥ 0.9
- 453 passing tests, gated day-by-day methodology, headless Kaggle reproduction pinned to a git SHA
- Independently verified their SEN2NAIPv2 pairs are genuinely cross-sensor (not synthetic degradation) via a kernel-fit/per-tile-spread/std-ratio audit — methodologically useful validation of a claim this draft also relies on
- Benchmarked against `opensr-test` (bicubic/nearest baselines, full reflectance/spectral/spatial/hallucination/omission/improvement metric suite) — same evaluation approach this draft's Evaluation Protocol section recommends
- Built a per-pixel uncertainty pipeline: TTA-disagreement as a shippable fallback (measured monotone against error, AUSE 0.243) while a learned heteroscedastic NLL head is trained separately
- Frontend demo includes a Delhi Sentinel-2 AOI — but explicitly labeled as "no ground truth, trust maps only," i.e. not validated Indian training/eval, just a visual demo
Source: https://github.com/shyam0github/DrishtiSR (public README, fetched round 4)

**Reading this correctly:** this is not prior art to cite (it's a contemporaneous competitor's hackathon submission, not published research) and must not appear in the team's own pitch. It's competitive intelligence for deciding which differentiation path is actually still open. Conclusion: path (B) is no longer a safe novelty claim on its own since a competitor has already executed it well; path (A) — real India ground-truth validation, not just an AOI demo — remains the more clearly open path, precisely because even this rigorous competitor didn't do that part.

## Finding 3: Bhoonidhi/Cartosat cost and turnaround, made concrete
Following up on AstraSR's round 4 finding (that sub-5m data is priced for Non-Government Entities):
- Under India's 2023 Space Policy: data finer than 5m is free/open for Indian Government Entities (with declaration) and priced for Non-Government Entities. Source: https://www.nsilindia.co.in/news-details/614
- A concrete price point: a 17×17km Cartosat-3 scene (0.3m) is ~₹3,860 for NGE pricing — modest, not a real budget blocker for a student/hackathon team. Source: https://aidigitalnews.com/ai/why-isros-bhoonidhi-is-on-par-with-nasas-datasets/ (secondary source reporting on the official pricing table; treat the exact figure as approximate pending direct confirmation from NSIL's own price list)
- Ordering requires a separate NSIL account (email eodata@nsilindia.co.in to request one) — not instant self-service.
- Turnaround: "urgent" orders (5–6 hour delivery) carry a 50% surcharge, implying normal-priority orders take longer, though the actual normal-priority duration wasn't stated in what was found this round. Source: https://bhoonidhi.nrsc.gov.in/imgarchive/bhoonidhi_videos_help/Commercial_Products_FAQ.pdf
- **Not yet verified:** whether a hackathon student team would qualify as a GE (likely only if formally affiliated with a government-recognized institution and willing to file the declaration) — worth the human team checking directly rather than assuming NGE pricing is the only option.

## Net effect on the draft
- Cost is no longer an open question for Cartosat access (modest, ~₹3,860/scene); turnaround still partially open (normal-priority timing unconfirmed).
- The competitive-landscape finding is the most consequential thing this round surfaces: it sharpens (again) which differentiation path is genuinely still open, on top of round 3's OpenSR finding. Recommend the human team treat "real India ground-truth validation" as the priority differentiator now, not just one of three roughly-equal options.
