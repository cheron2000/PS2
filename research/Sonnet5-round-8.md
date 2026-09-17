# Research Notes — Sonnet5, Round 8 (final scheduled round, stall guard)

**Approach:** given this is the last round before the automated loop stops regardless, the highest-value check isn't re-verifying individual citations already checked by Claude/AstraSR — it's stress-testing the single claim the entire 8-round consensus now rests on: "Path A (India/Cartosat ground-truth validation) ... Nothing found across 8 rounds does this." That's a strong, load-bearing claim. No round had run a direct search for it — prior searches targeted the OpenSR ecosystem specifically, or SIH26142 competitors specifically, not "Sentinel-2 SR validated against real India ground truth" as its own query.

## Stress test
Searched directly: "Sentinel-2 super-resolution India Cartosat validation ground truth" and a second pass on ISRO/NRSC-specific SR research. Found:
- Extensive general Sentinel-2 SR literature (DSen2/Lanaras et al., S2DR3, GINet, various 2025-2026 papers) — none uses Indian high-resolution ground truth; several explicitly discuss the general difficulty of obtaining any real HR ground truth for Sentinel-2 SR at all, which is consistent with (not contradicting) the draft's framing.
- One adjacent-but-distinct hit: a 2025 Indian paper combining Cartosat-2 and Sentinel-2 imagery for **road extraction** (a downstream segmentation task, not SR, not validation of an SR model against Cartosat ground truth). Source: sciencedirect.com/science/article/pii/S2352340925008625. This is useful supporting evidence that Cartosat+Sentinel-2 joint use is precedented and tractable in Indian research generally, but it does not touch this project's actual claim (SR model quality validated against Cartosat ground truth) and should not be cited as if it does.
- No hit for an existing Sentinel-2-SR-model validated against Cartosat or other Indian sub-4m ground truth.

**Conclusion: the claim holds up under a direct, independent stress test.** This doesn't newly prove a negative (absence of evidence isn't proof of absence), but it's the most direct check this specific claim has received across 8 rounds, and it didn't break.

## On AstraSR's round 8 SEN2NEON finding
AstraSR found SEN2NEON (real AVIRIS-NG-derived 2.5m validation benchmark) and correctly chose not to edit the draft with it, reasoning that editing at the stall-guard boundary would reset consensus without changing the conclusion (SEN2NEON is US/NEON-geography, doesn't touch the India gap). I agree with that call and am not re-opening it to fold SEN2NEON into solution-draft.md myself — it's already well-logged in research/AstraSR-round-8.md for the human team to read, and forcing it into the main document now would repeat exactly the boundary-manufacturing both other agents correctly avoided, for a fact that doesn't change the recommendation.

## Decision
No edit. The central claim survives independent stress-testing, AstraSR's round-7 concern (SEN2SR/SEN2SRLite) is already incorporated into v11, and no new substantive gap was found. Voting READY.
