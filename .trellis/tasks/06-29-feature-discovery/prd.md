# PRD: Automatic Feature Discovery

## Goal

Build the first working slice of the spec §15 feature-discovery loop on top of
the just-completed FeatureRecipe registry: propose candidate features from
observed failures, test them cheaply, and recommend which to promote into the
registry — all without auto-promotion (the project's manual-promotion
invariant holds).

## User value

- Turns the `failure_analysis.md` output (already produced per eval) from a
  passive report into actionable feature candidates.
- Gives the registry (just built) its first real consumer: candidates flow in,
  get tested, and the good ones land in `feature_recipes.yaml`.
- Establishes the ablation backbone that HPO-driven finalist ablation will plug
  into later (deferred).

## Confirmed facts (from inspection)

- Spec §15.2 discovery loop: select failure cluster → compare correct vs fail →
  identify discriminating difference → generate candidate FeatureRecipe →
  compute offline → ablation → evaluate → promote/reject/quarantine → write to
  feature memory.
- Spec §15.3: full ablation = A/B/C/D cells (baseline±candidate × baseline±tuned
  params); promotion should consider `D - B`. Requires HPO.
- Spec §15.4: LLM may propose hypotheses ("try merchant_velocity_24h"), not
  bypass validation. Unstructured feature code with no recipe entry is forbidden.
- CODEBUDDY.md: scout stage = fixed strong baseline params OR small-budget HPO
  to reject weak features cheaply; finalist stage = full HPO for shortlisted.
- `failure_analysis.md` (produced by `eval_reranker.py`) already has per-case
  Feature Contrast (required vs forbidden delta for each feature) + Suggested
  Next Actions — natural input for candidate generation.
- FeatureRecipe registry (shipped `2ab3ca1`/`243e30e`) validates candidates at
  load time — any generated candidate MUST satisfy `ALLOWED_INPUTS`,
  `online_safe`, required fields or it won't load.
- HPO adapter (shipped `0db0bba`, child #1) + A/B/C/D ablation framework
  (shipped `5e821e3`, child #2) are done — the candidate-testing half of the
  loop is complete.
- Project invariants: promotion is always manual; train never reads cases as
  training rows; `FeatureMemory` is deferred.

## Resolved decisions (scope split)

This is a **parent task**. The full LLM-driven + Full A/B/C/D scope spans
multiple independently verifiable children. Per the multi-deliverable workflow,
split into parent + children; dependencies written here, not implied by tree
position.

**Child tasks:**

| # | Child | Status | Depends on | Independently verifiable? |
|---|---|---|---|---|
| 1 | HPO adapter (Optuna backend) | **done** (`0db0bba`) | — | yes |
| 2 | A/B/C/D ablation framework | **done** (`5e821e3`) | #1 | yes |
| 3 | LLM candidate generation (reads `DEEPSEEK_API_KEY` env) | pending | user rotates leaked key | code yes; end-to-end user runs locally |
| 4 | Discovery orchestration | **deferred** (see below) | — | — |

**Start order**: child 1 → child 2 (both done). Next: child 3 (blocked on key
rotation). Child 4 is deferred.

**Scope of THIS parent task**: track the overall goal + child dependency graph.
The end-to-end discovery loop closes once child 3 lands: `#3 generates
candidates → run_ablation.py (child 2) per candidate → human reads reports +
promotes manually`. No separate orchestrator needed for V0 (see child 4
deferral below).

## Child task dependency graph

```
[1 HPO adapter] ──done──> [2 A/B/C/D ablation] ──done
                                                   │
[3 LLM candidate gen] (needs key rotation) ────────┴──> manual loop:
   #3 output  ->  run_ablation.py per candidate  ->  human promotes
```

Each child's PRD restates the dependency on its parent + earlier children.

## Why child 4 (orchestration) is deferred

Child 4 was originally "Discovery orchestration (generate → ablate →
recommend)". On inspection, its only V0-relevant responsibility was a for-loop
over #3's candidates calling #2's `run_ablation.py` — a ~50-line convenience
wrapper, not an independently verifiable deliverable. The project's
manual-promotion invariant means there is no auto-promote to automate, and:

- Candidate volume is single-digit → running `run_ablation.py` per candidate
  manually is fine (each run is seconds).
- Scout-vs-finalist staging only pays off at higher candidate volume.
- `FeatureMemory` is a separate concern (spec §15.2 step 9), not needed for the
  loop to close.

The loop is already complete with #3 (generate) + #2 (ablate, done) + human
(promote). Child 4 becomes worth a task when candidate volume grows or
FeatureMemory is needed. Marked **deferred until volume justifies**.

## Out of scope (deferred)

- Child 4 discovery orchestration (deferred — see above).
- Automatic promotion (project invariant: promotion is manual).
- `FeatureMemory` (institutional memory of promoted/rejected/quarantined).
- Scout-vs-finalist staging (optimization for high candidate volume).
- Online serving of discovered features.
