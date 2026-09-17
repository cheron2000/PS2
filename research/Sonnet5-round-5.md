# Research Notes — Sonnet5, Round 5 (independent review of v8)

**Approach:** per the standing instruction, reviewed v8 independently rather than assuming AstraSR's READY vote was the final word. Re-read problem-statement.md fresh (Research Standards says to do this before searching) rather than relying on memory of it from earlier rounds.

## Gap found: downstream-task utility, never addressed
Grepped the entire draft for crop/urban/disaster/classification/downstream/interpretability/analytical utility — zero matches. Yet problem-statement.md's Description and Expected Solution sections explicitly say:
- "The system should improve feature visibility, support better classification, change detection, crop monitoring, urban mapping, and disaster response."
- "The final outcome should improve in terms of interpretability and analytical utility, while clearly accounting for uncertainty and error components."

Every round so far (including AstraSR's round 5 "PS requirements" checklist) restated the PS's requirements as: preprocessing, paired training, accuracy assessment, HR-reference validation, uncertainty accounting — and left out the applications/utility line entirely. This is a genuine gap, not a rehash of the novelty-vs-OpenSR conversation.

## Evidence this isn't a "nice to have" that follows automatically from good SR
- GeoSR-Bench (Li et al., arXiv 2605.00310, 2026): built specifically because "improvements in traditional SR metrics often do not correlate with gains in [downstream] task performance, and the correlations can be negative." 36,000 locations, resolutions 500m–0.6m, tasks include land cover segmentation, infrastructure mapping, biophysical variable estimation.
- A separate land-cover segmentation study (arXiv 2606.25128) found the same decoupling: a model with much worse FID than baseline nonetheless improved downstream segmentation F1, and human-preferred outputs sometimes scored worse on the actual task.
- Net: fidelity/spectral/hallucination metrics (already in the draft) are necessary but demonstrated-insufficient for the "analytical utility" claim the PS actually asks for.

## Feasible path to close it (checked before proposing, not assumed)
- GeoSR-Bench itself may or may not cover the exact Sentinel-2 10m→2.5m/4m case — its stated range (500m–0.6m) suggests broad sensor coverage but I did not confirm Sentinel-2/NAIP-equivalent inclusion this round; flagged as "check before assuming" in the draft rather than asserted.
- Lower-effort alternative I'm more confident is directly usable: ESA WorldCover (10m, Sentinel-1/2-derived, global including India, free, standard land-cover classes including cropland and built-up) can be co-registered to the same AOIs already used for SEN2NAIP/Cartosat pairs, giving a simple classification/segmentation check without new data acquisition.

## Decision
This is a real, sourced, previously-missed gap directly tied to the PS's own wording — not a manufactured edit. Added a new Evaluation Protocol subsection and Known Risk #10. Did not touch the novelty/differentiation conversation (rounds 3-5's convergence on India-validation-as-priority-differentiator stands; this is a separate, additive completeness item).
