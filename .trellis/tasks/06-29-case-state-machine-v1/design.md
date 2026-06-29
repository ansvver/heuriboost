# Design: Case State Machine (V1, step 1)

## Plan Y: evaluator stays thin, ledger is separate

- `eval_reranker.py` remains the single-round evaluator. It gains: status-aware
  case evaluation (A) + status-split exit behavior + a per-status summary. It
  does NOT own cross-round memory.
- New module `skills/heuriboost-rag/scripts/regression_ledger.py` owns the
  cross-round ledger, the B2 anchor, the B comparison, the progress summary, and
  the manual promotion/anchor helpers.

## Storage (resolves PRD open question)

Ledger + anchor are **committed to version control**:
`examples/fiqa/ledger.json` (NOT under the gitignored output/). Consequence: the
anchor and round history travel with the repo — cross-machine consistent and
traceable. Trade-off: each round's `record` mutates ledger.json and produces a
git diff (acceptable; round snapshots are the point).

Behavior still degrades gracefully before the first anchor exists (e.g. on a
brand-new repo before any anchor was set):

- The B (global-no-regression) check must report "no anchor yet — this round can
  be set as the anchor" instead of crashing or treating it as a regression.
- The first `set_anchor` (or `--set-anchor`) freezes the current round's global
  metrics as the committed anchor.

To avoid noisy per-round commits, the per-round snapshot append and the anchor
are in the SAME committed file, but the maintainer decides when to commit
ledger.json (it is not auto-committed by the scripts).

Ledger schema (JSON):

```json
{
  "anchor": {
    "round_id": "2026-06-29T..Z",
    "global": {"ndcg@10": 0.83, "mrr@10": 0.85},
    "set_by": "manual"
  },
  "rounds": [
    {
      "round_id": "2026-06-29T..Z",
      "split": "validation",
      "global": {"ndcg@10": 0.83, "mrr@10": 0.85},
      "cases": [
        {"case_id": "...", "status": "gate", "passed": true},
        {"case_id": "...", "status": "pending", "passed": false}
      ],
      "vs_anchor": {"ndcg@10": +0.00, "regressed": false}
    }
  ]
}
```

## Case schema (regression_cases.yaml)

```yaml
cases:
  - case_id: fiqa_financial_security_definition
    query_id: 10087
    status: gate                 # gate | pending | retired ; default gate if absent
    must_include_doc_ids: ["375368"]
    must_not_include_doc_ids: ["83183"]
    top_k: 3
    require_rank: 3              # A: must_include must reach rank <= 3 (optional)
    # min_ndcg10: 0.5           # A: optional per-query metric floor
    failure_type: semantic_hard_negative
    notes: "..."
```

- `status` absent -> treated as `gate` (back-compat).
- `require_rank` (optional int): the first must_include doc must reach rank
  <= this value. If absent, the existing top_k membership check is the bar.
- `min_ndcg10` (optional float): per-query nDCG@10 floor. If absent, skipped.
- "Case hit" = all must_include within top_k (and within require_rank if set)
  AND no must_not_include within top_k AND min_ndcg10 satisfied if set.

## eval_reranker.py changes (light)

In `run_regression_cases` (currently treats all cases equally):
- read `status` (default `gate`); skip `retired`.
- compute hit including the A checks (require_rank, min_ndcg10). Need the
  per-query nDCG@10 for the case's query_id and the model rank of the
  must_include doc — both already derivable from `model_ranked` / existing
  metric helpers; pass them in.
- return per-case result objects: `{case_id, status, passed, missing_required,
  forbidden_present, rank_of_required, query_ndcg10}`.

In `main`:
- split results into gate vs pending.
- write the per-status summary into eval_report.md (extend the existing
  "Regression Gate" section into: Gates (pass/fail), Pending (pass/fail +
  promotion candidates)).
- exit non-zero ONLY if a `gate` case failed. Pending failures never change the
  exit code.
- after evaluation, call the ledger to append this round's snapshot and print
  the B-vs-anchor line + progress summary.

## regression_ledger.py (new)

Functions (CLI subcommands + importable). Ledger path defaults to
`examples/fiqa/ledger.json` (committed), overridable via `--ledger`:
- `record(report_metrics, case_results, split, ledger_path)` -> append a round
  snapshot; compute vs_anchor if an anchor exists.
- `set_anchor(ledger_path, round_id=latest)` -> manual anchor refresh.
- `summary(ledger_path)` -> print gates/pending/retired counts, this round's
  promotion candidates, and B-vs-anchor.
- `promote(cases_file, case_id)` -> helper that flips a case's status
  pending->gate in the YAML after the user confirms (manual; prints the
  hit+A+B evidence first, asks for confirmation, then edits). No auto-promotion.

eval_reranker calls `record` automatically; `set_anchor` / `promote` / `summary`
are user-invoked.

## B check (B2 anchored, reported not blocking)

- Compare round global metrics to `anchor.global`.
- `regressed = round.ndcg@10 < anchor.ndcg@10 - eps` (eps small, e.g. 0.0;
  configurable). Reported in summary and at promotion time.
- No anchor -> "no anchor yet" message; not a regression; offer to set one.
- B never changes exit code in V1 (manual gate philosophy). It is decision
  support for manual promotion.

## Anchor lifecycle

- No anchor on fresh clone. The first `set_anchor` (or first run with
  `--set-anchor`) freezes the current round's global metrics as the anchor.
- Manual refresh: when a round's gains are confirmed, user runs `set_anchor` to
  move the water line up. Old anchors are kept in round history (the snapshot
  that was anchored is identifiable).

## FiQA data migration

Rewrite `examples/fiqa/regression_cases.yaml`:
- 2 current gates -> `status: gate`, add `require_rank: 3` (they already pass at
  top-3, so this is a faithful freeze of the achieved bar).
- 4 header-comment hard cases (10117, 1, 1100, 10653) -> `status: pending` with
  their hand-confirmed must_include/must_not_include from prior mining; keep
  notes. Header comment's "known boundary" prose is replaced by real pending
  cases.
- Result: `eval_reranker.py --regression-cases ...` exits 0 (only pending fail),
  and the summary shows "2 gates pass, 4 pending (0 passed this round)".

## Tradeoffs

- Ledger committed to git gives cross-machine continuity, a traceable round
  history, and a shared anchor, at the cost of per-round diff noise in
  ledger.json. Acceptable and intended: the round history IS the artifact worth
  versioning. Scripts do not auto-commit ledger.json; the maintainer commits it
  alongside the round's other changes.
- B reported-not-blocking keeps the manual, human-in-the-loop philosophy and
  avoids false auto-blocks on a small noisy demo. Automation can come later.

## py_compile

`python3 -m py_compile skills/heuriboost-rag/scripts/*.py` must pass (incl. the
new ledger module).
