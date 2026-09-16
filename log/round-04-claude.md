## Claude — Round 4
**Draft version reviewed:** v7 (fresh read — a lot happened since my round 3: AstraSR did rounds 3–4, Sonnet5 did round 4, both converging hard on "India ground-truth validation is now the priority differentiator")

**Critique:** v7 correctly flagged two remaining logistics unknowns (GE eligibility, normal-priority turnaround) but hadn't gone to NRSC's own primary source to actually try to close them — both were sourced secondhand (a news article) rather than the official FAQ.

**Research findings:**
- Fetched NRSC/NSIL's own official Commercial Products FAQ directly. Q9 answers GE eligibility precisely: "Government Academic Institutions may submit GE declaration form" — eligibility is about the *institution's* status, not the student's. Worth the team checking their college's actual status rather than assuming either way. Source: bhoonidhi.nrsc.gov.in/imgarchive/bhoonidhi_videos_help/Commercial_Products_FAQ.pdf
- Same FAQ gives the full official pricing table (not just one approximate figure): Cartosat-3 PAN/MX GeoOrthokit ₹3,860, Ortho Rectified ₹4,810; Cartosat-2S PAN/MX GeoOrthokit ₹2,060 — all NGE pricing, all modest.
- Normal-priority turnaround: confirmed this is genuinely **not published anywhere** in NRSC's own materials — the FAQ only states urgent (5–6hr, +50% surcharge) and says normal priority is "first come, first served" with no SLA. This is worth recording as "confirmed undocumented" rather than "not yet found" — more web searching won't resolve it; only a direct email or test order will.

**Edits made to solution-draft.md:** v7 → v8. Replaced the GE-eligibility and turnaround bullets in Feasibility & Data Sources and Known Risk #5 with the primary-source-grounded versions above. Updated Sources to point at the official FAQ as the primary citation.

**Vote:** NEEDS-REVISION
- Feasibility check: cost and GE-eligibility criteria are now as resolved as desk research allows; turnaround genuinely requires direct contact with NRSC/NSIL, not more searching.
- Novelty check: unchanged from round 4 (AstraSR/Sonnet5) — India ground-truth validation is the clearest remaining differentiator, and that assessment holds.
- Biggest remaining risk: **this is now a team decision, not a research gap.** Every open technical/logistics question the community can resolve from open sources has been resolved. What's left — commit to path A, place a real Bhoonidhi test order, decide the compute-tier tradeoff — needs the human team, not another research round. Also flagging again: `sonnet5` is confirmed to be Claude Sonnet 5, same as me — worth the team deciding whether a genuinely distinct third model (not just AstraSR, whose underlying model is still unconfirmed) should weigh in before treating this as community-reviewed.
