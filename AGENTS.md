# AGENTS.md — Multi-Model Research Deliberation Protocol

## Purpose

This repo is a shared workspace where **multiple different AI models act as an independent research community** analyzing one problem statement (PS). Each model reads the current state, does its own research, critiques and corrects the existing draft, and votes on whether it's ready. The loop repeats until every participating agent votes **READY** on the *same* draft version.

**Scope boundary — read this twice:** This process produces a research-backed **solution document**, not code, not a prototype, not a slide deck. If you are an agent reading this, do not write implementation code, do not scaffold a repo structure for building the thing, do not create a pitch deck. Your only output is research findings, critique, and edits to `solution-draft.md`.

---

## Repo Structure

```
/AGENTS.md              ← this file (protocol — do not edit during rounds)
/problem-statement.md   ← the official PS text, pasted verbatim (immutable reference)
/solution-draft.md      ← the ONE living document all agents refine together
/status.md              ← machine-parseable round/vote state (YAML frontmatter + notes)
/log/
   round-01.md           ← append-only, one file per round
   round-02.md
   ...
/research/
   <agent>-round-<n>.md  ← raw research notes/sources an agent gathered that round
```

Nothing outside `solution-draft.md`, `status.md`, `/log/`, and `/research/` should change during a run.

---

## Participants

List every agent taking part before round 1 starts, so everyone knows who has to agree for the loop to end.

| Agent (model)        | GitHub identity used | Notes                        |
|-----------------------|-----------------------|-------------------------------|
| e.g. Gemini 2.5 Pro    | gemini-bot            |                                |
| e.g. Llama 3.3 70B (Groq) | groq-llama-bot     |                                |
| e.g. DeepSeek-R1       | deepseek-bot          |                                |
| e.g. Claude            | claude-bot            |                                |

Fill this table in `status.md`, not here — this file is the protocol, not the run state.

---

## The Round Protocol

A **round** = every listed agent takes exactly one turn, in a fixed order. Turns within a round are sequential (one agent finishes and commits before the next starts), which is what keeps this safe to run without merge conflicts even though nothing is truly running "in the background."

Each agent, on its turn, does the following **in order**:

### 1. Orient
Read, in this order: `status.md` (what round is it, what did the last agent say, what's still open), `solution-draft.md` (current state of the solution), the most recent 1–2 files in `/log/` (recent critiques/votes — don't reread the whole history every time).

### 2. Identify the gap
Before researching anything, write down (in your own head, not the repo) what specifically is weak, unsupported, missing, or contested in the *current* draft. Don't research the whole topic from scratch each round — research the open question.

### 3. Research
Follow the **Research Standards** below. Save raw notes to `/research/<your-agent-name>-round-<n>.md`.

### 4. Critique
In your log entry, state plainly what you think is wrong or weak in the current draft, and why — cite what you found. Agreeing without a specific reason is not a valid critique entry (see Anti-Rubber-Stamp Rule below).

### 5. Edit
If your research changes the picture, edit `solution-draft.md` directly. Keep edits scoped to the section(s) your research actually informs — don't rewrite the whole document every round. Bump the version number in the draft's header.

### 6. Log and vote
Append a new entry to the current round's file in `/log/` using the template below, ending in a vote: `READY` or `NEEDS-REVISION`. Then update `status.md`: your vote, and whether you changed the draft version.

If you changed the draft version, every other agent's vote from this round is invalidated automatically (see Termination Rule) — say so explicitly in `status.md`.

---

## Research Standards — "how to research"

- **Start from the PS text itself.** Re-read `problem-statement.md` before searching anything — most weak drafts drift from what was actually asked.
- **Prioritize primary sources**: papers (arXiv, Papers with Code), official datasets/benchmarks named or implied by the PS, prior hackathon-winning approaches to similar problems, vendor/API docs for anything you're proposing to depend on.
- **Every non-obvious technical claim needs a source or an explicit "(assumption, unverified)" tag.** Do not present a plausible-sounding number or method as fact without one.
- **Don't repeat research already logged.** Check `/research/` first. If you find the same paper someone already cited, only add a note if you're adding a *new* reading of it (e.g., "the 20% figure in round 2 is for a different sensor resolution than ours — flagging").
- **Look for reasons the current approach fails, not just reasons it works.** A round that only confirms the existing draft is a wasted round — actively try to break it: What data won't actually be available? What's the compute/time cost in reality vs. on paper? What would a judge ask that the draft doesn't answer?
- **Keep it scoped.** One round = one or two specific open questions, not a full literature review. This also matters practically — most of you are running on free-tier rate limits.

---

## Anti-Rubber-Stamp Rule

A `READY` vote must state, in one line each, what you checked:
- Feasibility (can this actually be built with realistic data/compute?)
- Novelty (what does this add beyond the obvious baseline approach?)
- Unresolved risk (what's the biggest remaining weakness, even in a READY draft?)

If you can't fill in that third line, you haven't looked hard enough — go back to step 2.

---

## Termination Rule

The loop ends when, **within a single round**, every listed agent in `status.md` votes `READY` on the *same* draft version (i.e., no one edited `solution-draft.md` that round). If even one agent edits the draft or votes `NEEDS-REVISION`, a new round begins and every agent's vote resets — the whole community re-reviews the changed draft.

**Stall guard:** cap at **8 rounds**. If round 8 ends without full consensus, stop, and a human reviews `solution-draft.md` plus the open disagreements in the latest `/log/` entries and makes the final call manually. This exists so the loop can't quietly burn everyone's free-tier quota forever.

---

## File Templates

### `solution-draft.md` header
```markdown
# Solution Draft — v<N>
Status: IN-PROGRESS | READY
Last edited by: <agent>, round <n>

## Problem Restatement
## Proposed Approach
## Technical Architecture
## Novelty / Differentiation
## Feasibility & Data Sources
## Known Risks & Open Questions
```

### `/log/round-<n>.md` entry (one per agent per round)
```markdown
## <agent name> — Round <n>
**Draft version reviewed:** v<N>

**Critique:** <specific problem with the current draft, with reasoning>

**Research findings:**
- <finding> — source: <link/citation> (or "assumption, unverified")

**Edits made to solution-draft.md:** <section(s) changed, or "none">

**Vote:** READY | NEEDS-REVISION
- Feasibility check: <one line>
- Novelty check: <one line>
- Biggest remaining risk: <one line>
```

### `status.md`
```markdown
---
round: <n>
draft_version: <N>
terminate: false
votes_this_round:
  <agent-1>: pending
  <agent-2>: pending
---

Next to act: <agent>
Last draft change: <agent>, round <n>, sections: <...>
Open disagreements: <brief note, or "none">
```

---

## Kickoff Prompt (paste this into each agent's chat when it's their turn)

```
Read AGENTS.md in this repo (via your GitHub connector/token), then status.md,
solution-draft.md, and the most recent file(s) in /log/. Follow the round
protocol in AGENTS.md exactly — research the current open gap, critique the
draft with reasons, edit solution-draft.md if warranted, then add your
log entry and vote, and update status.md. Do not write code or build
anything. Stay scoped to one or two open questions this round.
```

---

## Security Note on the Shared Access Token

- Create a **fine-grained Personal Access Token** scoped to **this one repo only**, with **Contents: Read and write** and nothing else (no other repos, no admin, no account-wide scopes).
- Every agent you connect this to can read and write anything in the repo — treat the token as shared/semi-public the moment you hand it to a third-party AI service, not as a secret only you hold.
- Set an expiration (30–60 days is plenty for a hackathon research loop) and revoke it manually once the deliberation is done, rather than leaving it live.
- Don't reuse this token for any other repo or purpose.

---

## Human Orchestrator Checklist

Since these are separate chat sessions and not autonomous background processes, you (the human) are the scheduler:
1. Confirm whose turn it is via `status.md`.
2. Paste the Kickoff Prompt into that agent's chat.
3. Let it finish its commit(s).
4. Move to the next agent in the `Participants` list.
5. When `status.md` shows `terminate: true`, the current `solution-draft.md` is the community's final answer.
