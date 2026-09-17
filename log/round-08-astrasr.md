## AstraSR — Round 8
**Draft version reviewed:** v11

**Critique:** v11 now addresses the specific Round 7 baseline gap by adding SEN2SR/SEN2SRLite and by closing lightweight-compute differentiation as a novelty path. I checked for a further substantive gap rather than manufacturing an edit.

**Research findings:**
- ESAOpenSR's current SEN2NEON benchmark provides 2,269 spatially aligned tiles pairing observed Sentinel-2 L2A with AVIRIS-NG-derived high-resolution references, with a canonical 2.5 m HR product covering all Sentinel-2-equivalent bands. It also provides adapters for SEN2SR, LDSR-S2 and SRGAN and integrates OpenSR-Test-style metrics. Source: https://github.com/ESAOpenSR/SEN2NEON
- This strengthens the prior-art/validation landscape but does not remove the India-ground-truth gap: SEN2NEON is based on NEON/US geography, not Indian Cartosat references.

**Edits made to solution-draft.md:** none. This is the final scheduled round under the protocol's 8-round stall guard; another draft edit would reset consensus at the boundary without materially changing the conclusion.

**Vote:** READY
- Feasibility check: v11 has a realistic public-data route through SEN2NAIP/SEN2VENµS and an optional Cartosat acquisition route; SEN2NEON further strengthens the available validation landscape.
- Novelty check: 2.5m SR, uncertainty, geospatial consistency, lightweight SR, and standard benchmarking are already represented in existing work; real India-specific high-resolution validation remains the clearest unresolved contribution identified by the review process.
- Biggest remaining risk: actual Cartosat acquisition, Sentinel-2↔Cartosat pairing, co-registration/harmonization, and turnaround are execution risks that desk research cannot fully resolve.
