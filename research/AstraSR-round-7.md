# AstraSR — Round 7 Research

## Open question
Does the current draft's baseline/positioning account for the newest Sentinel-2-specific open-source SR tooling, or is it implicitly comparing only against older OpenSR/DiffFuSR work?

## Finding
The current ESAOpenSR ecosystem includes **SEN2SR**, a Python package explicitly designed to super-resolve Sentinel-2 imagery up to 2.5 m. Its current README documents a lightweight `SEN2SRLite`, a `SEN2SR` model using a Mamba architecture, an LDSR-S2 latent-diffusion path, and a 10 m/20 m Sentinel-2 workflow. It also documents a 10 m RGB+NIR 4x non-reference model and 20 m→10 m reference model. Source: https://github.com/ESAOpenSR/SEN2SR

This matters because the living draft currently names Donike/OpenSR and DiffFuSR as the key external baselines, while its proposed CNN+MHAN+SPIFFNet path could otherwise be evaluated against an incomplete current baseline set. SEN2SR is not evidence that the proposed architecture is inferior, but it is directly relevant prior art and a practical baseline candidate, especially if the team wants to make any lightweight/compute-efficiency claim.

## Scope decision
Add SEN2SR/SEN2SRLite to the external baseline plan. Do not claim that Mamba is superior, faster, or more accurate without an experiment. Do not add another architecture or redesign the proposed model based solely on this finding.

## Remaining risk
The India-ground-truth path remains the strongest unresolved differentiation. Round 7 does not establish that SEN2SR provides Indian validation or a novel research contribution for this project.
