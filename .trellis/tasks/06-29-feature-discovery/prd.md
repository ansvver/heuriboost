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
- FeatureRecipe registry (just shipped) validates candidates at load time — any
  generated candidate MUST satisfy `ALLOWED_INPUTS`, `online_safe`, required
  fields or it won't load.
- HPO adapter does NOT exist yet (separate checklist item) → full A/B/C/D
  ablation is NOT possible in this task.
- Project invariants: promotion is always manual; train never reads cases as
  training rows; `FeatureMemory` is deferred.

## Resolved decisions (scope split)

This is a **parent task**. The full LLM-driven + Full A/B/C/D scope spans multiple
independently verifiable children with two external blockers (HPO adapter not
yet built; LLM key needs rotation). Per the multi-deliverable workflow, split
into parent + children; dependencies written here, not implied by tree position.

**Child tasks (in dependency order):**

| # | Child | Depends on | Independently verifiable? |
|---|---|---|---|
| 1 | HPO adapter (wrap external backend, e.g. Optuna) | — | yes (small-budget local run) |
| 2 | A/B/C/D ablation framework | #1 | yes (uses HPO adapter) |
| 3 | LLM candidate generation (reads `DEEPSEEK_API_KEY` env) | user rotates leaked key | code yes; end-to-end user runs locally |
| 4 | Discovery orchestration (generate → ablate → recommend) | #2 + #3 | partial (#3 part user-run) |

**Start order**: child 1 (HPO adapter) — no blockers, unblocks child 2.

**Scope of THIS parent task**: track the overall goal + child dependency graph.
Implementation happens in child tasks. The parent is "done" when all children
land and the end-to-end discovery loop runs (generate → ablate → recommend).

## Child task dependency graph

```
[1 HPO adapter] ──unblocks──> [2 A/B/C/D ablation] ─┐
                                                     ├──> [4 Discovery orchestration]
[3 LLM candidate gen] (needs key rotation) ──────────┘
```

Each child's PRD must restate the dependency on its parent and any earlier
children it consumes.

## Out of scope (likely deferred)

- Finalist-stage HPO + full A/B/C/D ablation cells (needs HPO adapter).
- Automatic promotion (project invariant: promotion is manual).
- `FeatureMemory` (institutional memory of promoted/rejected/quarantined).
- Online serving of discovered features.
