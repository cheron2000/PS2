# BUILD_AGENTS.md — Prototype Build Coordination Protocol

## Purpose & scope

Research is done (`solution-draft.md` v11, `terminate: true`). This document governs a **new, separate phase**: building the actual prototype code. Different rules apply here than in `AGENTS.md` — that document governs research/critique and is now closed; this one governs implementation.

**The roster is open and size-unknown.** Any number of agents may join, in any order, over however long this takes. Nothing here assumes a fixed headcount.

---

## The core mechanism: named turn + open task board

Two files drive this:
- **`tasks.md`** — the task board. What needs building, current status, who's doing it.
- **`build-status.md`** — exactly one field that matters most: `next_agent`. Whoever is named there is the only one who should act.

This separation matters: the task board can grow or shrink (tasks get added, split, or finished) without ever making turn order ambiguous, because turn order lives in one explicit named field, not in a position you have to calculate.

---

## Step 0 — before doing anything, check whose turn it is

Read `build-status.md`. Look at `next_agent`.

**If you are already in the roster (Participants table below) and `next_agent` does not name you:** stop immediately. Do not read `tasks.md`. Do not do any work. Reply only with:

> Not my turn — `build-status.md` says it's `<next_agent>`'s turn. Please call `<next_agent>` instead.

**If you have never appeared in the roster before:** you may join — see "Onboarding a new agent" below. This is the one exception to the rule above, because the human is the one choosing who to bring in, and the roster can't predict that in advance.

**If `next_agent` is literally `open`:** any registered agent may take the turn (use this only when the previous agent genuinely couldn't determine who should go next — it should be rare).

---

## Onboarding a new agent

First turn ever for you:
1. Add yourself to the Participants table below (see format there) — a real name for the underlying model, not a persona name alone, learned from the identity confusion in the research phase.
2. Add yourself to the end of `build-status.md`'s `roster:` list.
3. Proceed straight to "Taking a turn" below — no need to wait an extra cycle.

---

## Taking a turn

1. **Read `tasks.md`.** Find the highest-priority task that is `TODO` (not `BLOCKED`, not already `IN-PROGRESS`, and whose `Depends on` tasks are all `DONE`). Take the *next* eligible one in the list, top to bottom — don't cherry-pick the easy or interesting one. That's what keeps distribution actually equal as the roster grows or shrinks.
2. **Claim it:** mark it `IN-PROGRESS`, set `Assigned to` to your name, commit that alone first (so two agents can't silently claim the same task if turns ever overlap).
3. **Do the work.** Write real, runnable code — not a stub, not pseudocode, unless the task explicitly says otherwise. Include a short usage comment or a minimal test at the bottom of the file where the task allows it.
4. **State what you could and couldn't verify.** If you can't execute/test something in your own environment (missing package, no GPU, no data access), say so explicitly in your commit message and in `tasks.md`'s notes column — don't claim untested code is tested. Future agents and the human need to know which pieces are hand-verified vs. execution-verified.
5. **Mark the task `DONE`** (or `BLOCKED` with a one-line reason if you genuinely can't finish it — don't leave it silently `IN-PROGRESS`).
6. **Hand off:** update `build-status.md` —
   - Append one short log line: your name, task ID, what you did, what's still unverified.
   - Set `next_agent` to the next name in the roster after your own position, wrapping to the first name if you're last. If the roster has only you so far, set it to `open`.
7. **If there is no eligible `TODO` task** (everything is `DONE` or genuinely `BLOCKED` on something outside any agent's control), say so plainly in `build-status.md` rather than inventing busywork — same principle as the research phase's anti-rubber-stamp rule, applied to building instead of critiquing.

---

## Task sizing — keep distribution actually equal

When adding tasks to `tasks.md`, tag each `S` / `M` / `L`. One turn = one task, regardless of size, so lumpy sizing breaks the "equal distribution" goal even with perfect rotation. Prefer splitting an `L` task into 2-3 `M` ones over letting sizes drift apart.

---

## What "done" means for a code task

- It runs, or the task notes explicitly say what's blocking execution and why (missing data, missing package, no GPU here).
- It matches the interfaces/shapes other tasks depend on (check `tasks.md`'s `Depends on` column for what you're expected to produce for downstream tasks).
- No secrets, tokens, or credentials committed — ever.

---

## Participants

| Name (used in `build-status.md`) | Underlying model | Notes |
|---|---|---|
| `claude1` | Claude Sonnet 5 (Anthropic) | First agent, bootstrapped the task board |

Add a row here the first time you join. Use a real model identification, not just a persona label — this repo already had one round of confusion in the research phase over two identities turning out to be the same underlying model.
