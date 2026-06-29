# Case State Machine (V1, step 1)

## Goal

Turn HeuriBoost's regression cases from a static snapshot (2 enforced gates + 4
hard cases buried in comments) into a **stateful, multi-round ledger** that can
carry the iterative "attack the failure set" loop agreed in the 2026-06-29
grilling.

This task implements ONLY the state-machine layer (grilling decisions 1-6). The
`case_sets` inbox and corpus-mining settlement (decisions 7-8) are a SEPARATE
follow-up task and are out of scope here.

## Background / why

The current `examples/fiqa/regression_cases.yaml` enforces only the 2 cases the
reranker already passes; the 4 unsolved hard cases live in a header comment. The
user wants those 4 to be first-class "gaps to attack" tracked across rounds, and
wants gate promotion to stay manual and honest. See memory
`project_autosearch_loop.md` for the full agreed design.

## Anti-leak invariant (must hold)

Regression/gate cases are exam questions; never train on them. This task does NOT
add any training path for cases — it only evaluates and tracks them. (Settlement
into training is the separate follow-up, and even then only via mined samples.)

## Requirements

### R1. Three-state case schema
Each case in `regression_cases.yaml` carries `status`: one of
- `gate` — attacked & frozen. Failure must block (exit non-zero).
- `pending` — a known gap to attack. Evaluated and reported, but failure does
  NOT block (exit 0).
- `retired` — invalidated (corpus/label drift). Not evaluated; kept for history.

Backward compatibility: a case with no `status` defaults to `gate` (preserves
current behavior for any external user files).

### R2. Per-case local metric (grilling "A")
A case may declare a local pass condition beyond the doc-hit check, e.g.
`require_rank: 3` (the must_include doc must reach rank <= 3, stricter than just
being within top_k) and/or `min_ndcg10` for that query. "Case hit" = must_include
in top_k AND must_not_include out of top_k AND local metric satisfied.

### R3. Status-aware evaluation in eval_reranker.py (plan Y, light change)
`eval_reranker.py` stays the single-round evaluator. It must:
- evaluate ALL non-retired cases and compute hit + local-metric (A);
- split results by status: `gate` failures block (exit != 0); `pending`
  failures are reported only (exit 0 if no gate failed);
- skip `retired` cases;
- print a clear per-status summary (gate pass/fail counts, pending pass/fail
  counts, which pending cases passed this round = promotion candidates).

### R4. Cross-round ledger module (plan Y, new file)
A new module (e.g. `regression_ledger.py`) owns cross-round memory, separate
from the evaluator:
- record a per-round snapshot: round id/timestamp, global metrics, per-case
  status + pass/fail, and which round/anchor it is;
- store and read the **B2 anchor** (a frozen snapshot's global metrics) used for
  the global-no-regression check (grilling "B");
- support **manual anchor refresh** (promote current round's global metrics to
  the new anchor when the user confirms gains);
- emit a progress summary: gates X, pending Y, retired Z, pending-passed-this-
  round (promotion candidates), and the B comparison vs anchor.
- ledger storage is a COMMITTED file `examples/fiqa/ledger.json` (version
  controlled, cross-machine consistent, traceable). `.gitignore` must NOT ignore
  it. Scripts must not auto-commit it.

### R5. Global-no-regression check (grilling "B", B2 anchored)
The ledger compares this round's global nDCG@10 / MRR@10 against the anchor
snapshot (NOT the previous round). Report whether global metrics regressed below
anchor. In V1 this is a REPORTED check surfaced at manual promotion time, not an
automatic blocking gate (consistent with manual promotion).

### R6. Manual promotion workflow
Promotion pending -> gate is manual: the tooling surfaces "case X passed this
round (hit + local metric), global vs anchor = ...; promote? " and the user edits
`status` (or a helper command flips it). No auto-promotion.

### R7. Migrate the FiQA demo data
Rewrite `examples/fiqa/regression_cases.yaml` to the new schema: the 2 current
gates become `status: gate`; the 4 header-comment hard cases become
`status: pending` (with their hand-confirmed must_include/must_not_include from
the prior mining). Keep the provenance notes.

## Acceptance Criteria

- [ ] `regression_cases.yaml` uses three-state `status`; missing status defaults to gate.
- [ ] Per-case local metric (`require_rank` / `min_ndcg10`) is honored.
- [ ] `eval_reranker.py`: gate failure -> non-zero exit; pending failure -> exit 0 with report; retired skipped.
- [ ] A per-status summary and promotion-candidate list are printed.
- [ ] Ledger module records per-round snapshots and stores/reads the B2 anchor in committed `examples/fiqa/ledger.json` (not gitignored, not auto-committed).
- [ ] Global-vs-anchor (B) comparison is reported.
- [ ] Manual anchor refresh and manual promotion are supported (no auto-promotion).
- [ ] FiQA demo migrated: 2 gates + 4 pending; `eval_reranker.py` exits 0 on the demo (only pending cases fail).
- [ ] No training path touches cases (anti-leak preserved).
- [ ] `python3 -m py_compile skills/heuriboost-rag/scripts/*.py` passes.

## Out of scope (separate follow-up task)

- `case_sets` pending inbox feeding training.
- Corpus-mining settlement of abstracted samples into train/valid (approach 3).
- LLM-mode labels / LLM pattern generation (approach 2).
- HPO automation, feature registry, horizontal task profiles.

## Open Questions

- None blocking. (Ledger storage resolved: committed `examples/fiqa/ledger.json`,
  version-controlled, per the user's decision on 2026-06-29.)
