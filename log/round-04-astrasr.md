## AstraSR — Round 4
**Draft version reviewed:** v5

**Critique:** v5 correctly identifies ESA OpenSR as the closest prior art, but it still listed a generic comparative benchmark as one of the main differentiation paths. OpenSR-test already supplies the real Sentinel-2 x4 benchmark ecosystem, so merely running another model on it is evidence, not novelty.

**Research findings:**
- ESAOpenSR's public model performs 4x RGB-NIR Sentinel-2 SR and its demo produces uncertainty maps; the surrounding utilities preserve geospatial metadata and support tiled inference. Source: https://github.com/ESAOpenSR/opensr-model ; https://github.com/ESAOpenSR/opensr-model/blob/main/demo.py
- OpenSR-test already contains 2.5m NAIP/SPOT x4 and Spain x4 data, so a generic benchmark comparison is not itself a new contribution. Source: https://github.com/ESAOpenSR/opensr-test
- Bhoonidhi currently lists CartoSat-2S/3 MX products for ordering. Its registration policy states that sub-5m remote-sensing data are priced for non-government entities. Source: https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html ; https://bhoonidhi.nrsc.gov.in/bhoonidhi/registration.html
- The practical alternative is a measured compute-constrained comparison: latency, VRAM, throughput and quality against OpenSR on the actual hardware tier used by the team. This is only a contribution if experimentally demonstrated.

**Edits made to solution-draft.md:** Added an explicit Round 3 AstraSR review under Novelty/Differentiation and narrowed the contribution candidates to (A) Indian-data generalization, (B) compute-constrained SR, or (C) both. Removed the implication that a generic benchmark study alone is novel.

**Vote:** NEEDS-REVISION
- Feasibility check: the existing CNN/transformer research path is feasible; OpenSR is also publicly runnable, while Indian sub-5m data acquisition remains subject to access/cost.
- Novelty check: the current technical bundle is substantially covered by OpenSR, so novelty must come from a measured, concrete experimental contribution.
- Biggest remaining risk: the human team has not yet selected and experimentally validated its differentiation path.
