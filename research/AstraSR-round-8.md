# AstraSR — Round 8 Research

## Open question
Does v11 now contain the necessary current prior-art coverage, and is there a remaining research gap that justifies another draft edit?

## Finding
The current ESAOpenSR ecosystem is broader than the older OpenSR/DiffFuSR comparison. In particular, ESAOpenSR's SEN2NEON benchmark is now publicly documented as a 2,269-tile dataset pairing observed Sentinel-2 L2A data with AVIRIS-NG-derived high-resolution references at a canonical 2.5 m resolution, covering all Sentinel-2-equivalent bands. It includes adapters for SEN2SR, LDSR-S2 and SRGAN and integrates OpenSR-Test-style metrics. Source: https://github.com/ESAOpenSR/SEN2NEON

This is important prior art/validation infrastructure, but it does not invalidate the current India-ground-truth direction: the benchmark is based on NEON/US geography, not Indian Cartosat reference data. It also strengthens the draft's warning that generic 2.5 m SR and uncertainty are not novelty claims.

## Decision
No further solution-draft edit is justified in the final scheduled round. The v11 baseline update already incorporates the directly relevant SEN2SR/SEN2SRLite finding. SEN2NEON should be considered during implementation/benchmark selection, but adding another draft edit in round 8 would reset consensus at the stall-guard boundary and would not materially change the research conclusion.

## Vote
READY
- Feasibility: v11 has a realistic public-data route through SEN2NAIP/SEN2VENµS plus optional Cartosat acquisition; the new SEN2NEON benchmark further strengthens available validation options.
- Novelty: the research establishes that 2.5 m SR, uncertainty, geospatial consistency, lightweight SR, and standard benchmarking are already covered by existing work; India-specific real high-resolution validation remains the clearest unresolved contribution identified by this process.
- Biggest remaining risk: actual Cartosat↔Sentinel-2 pairing, acquisition turnaround, and co-registration/harmonization remain execution risks and cannot be resolved by further desk research alone.
