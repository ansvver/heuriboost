# Implementation Plan: Case State Machine (V1, step 1)

## Phase 1: Case schema + status-aware evaluation (eval_reranker.py)

- [ ] Support `status` per case (gate | pending | retired); absent -> gate.
- [ ] Support optional A checks: `require_rank` (must_include rank <= N) and
      `min_ndcg10` (per-query nDCG@10 floor).
- [ ] In `run_regression_cases`: skip retired; compute hit incl. A; return rich
      per-case results (case_id, status, passed, missing_required,
      forbidden_present, rank_of_required, query_ndcg10).
- [ ] In `main`: split gate vs pending; exit != 0 only on gate failure; pending
      failures reported only.
- [ ] Extend eval_report.md "Regression Gate" section into Gates + Pending
      (with promotion candidates = pending that passed this round).

## Phase 2: Ledger module (regression_ledger.py, new)

- [ ] `record(...)` append round snapshot to committed examples/fiqa/ledger.json;
      compute vs_anchor when anchor exists; create file if missing.
- [ ] `set_anchor(...)` freeze current/given round's global metrics as anchor.
- [ ] `summary(...)` print gates/pending/retired counts, promotion candidates,
      B-vs-anchor (or "no anchor yet").
- [ ] `promote(cases_file, case_id)` print hit+A+B evidence, confirm, flip
      pending->gate in YAML. No auto-promotion.
- [ ] CLI subcommands + importable functions; `--ledger` defaults to
      examples/fiqa/ledger.json.
- [ ] B check degrades gracefully with no anchor (no crash, not a regression).

## Phase 3: Wire eval -> ledger

- [ ] eval_reranker.main calls ledger.record after evaluation and prints the
      B-vs-anchor line + progress summary.
- [ ] Do NOT auto-commit ledger.json.

## Phase 4: Migrate FiQA demo data

- [ ] Rewrite examples/fiqa/regression_cases.yaml: 2 gates (status: gate,
      require_rank: 3) + 4 pending (10117, 1, 1100, 10653) with hand-confirmed
      ids/notes.
- [ ] Confirm .gitignore does NOT ignore examples/fiqa/ledger.json.

## Phase 5: Verify

- [ ] python3 -m py_compile skills/heuriboost-rag/scripts/*.py passes.
- [ ] Run eval on validation with migrated cases: exits 0, summary shows
      "2 gates pass, 4 pending (0 passed)", ledger.json written, B reports
      "no anchor yet" on first run; set_anchor then second run reports vs-anchor.
- [ ] No training path references cases (anti-leak grep: cases never loaded by
      train_reranker.py).

## Notes for implementer

- Plan Y: keep eval_reranker thin; cross-round logic in regression_ledger.py.
- Anti-leak invariant is absolute: this task only evaluates/tracks cases.
- See memory project_autosearch_loop.md for the full agreed design rationale.
