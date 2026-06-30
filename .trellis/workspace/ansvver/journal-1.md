# Journal - ansvver (Part 1)

> AI development session journal
> Started: 2026-06-27

---



## Session 1: FiQA pivot: replace toy financial demo with single FiQA-2018 demo

**Date**: 2026-06-29
**Task**: FiQA pivot: replace toy financial demo with single FiQA-2018 demo
**Branch**: `main`

### Summary

Removed the hand-built examples/financial_rag/ toy demo (could not honestly back the beats-baselines claim: n=1 validation, rigged hard negative). Replaced with a single BEIR/FiQA-2018 demo. Added build_fiqa_csv.py (offline BM25+MiniLM+RRF retrieval + build-time LLM judge for 5-level labels; FiQA ships no candidates/scores so they are manufactured) with deps isolated in requirements-build.txt. Retargeted feature set to FiQA-style non-temporal hard negatives (dropped year/quarter/wrong_year, renamed numeric->number_overlap_count, added entity_overlap_count/important_term_overlap/low_information_density_flag) and rewrote eval_reranker analysis. Updated README/README.zh-CN/CODEBUDDY/CONTEXT and templates. Captured executable contracts in .trellis/spec/backend/fiqa-demo-contracts.md. Follow-up (maintainer, local): run build_fiqa_csv.py to generate examples/fiqa/query_doc_examples.csv, mine+hand-confirm 5-8 regression cases, commit both.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `f371cc0` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 2: Case state machine (gate/pending/retired) + cross-round ledger

**Date**: 2026-06-29
**Task**: Case state machine (gate/pending/retired) + cross-round ledger
**Branch**: `main`

### Summary

Implemented V1 step 1 of the iterative failure-attack loop. regression_cases.yaml now uses three-state status (gate/pending/retired; absent->gate) with optional per-case A checks (require_rank, min_ndcg10). eval_reranker.py is status-aware (plan Y, thin): gate failure exits non-zero, pending failure reports only, retired skipped; report split into Gates + Pending with promotion candidates; --ledger/--no-ledger. New regression_ledger.py owns the committed examples/fiqa/ledger.json: per-round snapshots + B2 anchor, record/set-anchor/summary/promote (manual, no auto-promotion), B global-no-regression reported (not exit-blocking), graceful no-anchor. Migrated FiQA demo to 2 gates + 4 pending. Anti-leak preserved (train never reads cases/ledger). Spec updated with three new contracts. Verified: gate-fail->exit1, demo 4-pending-fail->exit0, py_compile clean, ledger reset to clean 1-round+anchor baseline. Deferred: case_sets inbox + corpus-mining settlement (separate task).

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `5dd09b3` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: case_sets mining + settlement loop (V1 step 2)

**Date**: 2026-06-29
**Task**: case_sets mining + settlement loop (V1 step 2)
**Branch**: `main`

### Summary

Implemented V1 step 2: the textbook path of the failure-attack loop. mine_case_sets.py mines same-pattern samples per pending case via a+b+c intersection (failure_type match + failure shape: -1 at dense_rank<=3 and 3 at dense_rank>=5 + MiniLM cosine top-K similar queries), with B+C isolation (no mined query_id/doc_id matches a case's). case_sets committed under examples/fiqa/case_sets/. train_reranker.py gains --case-sets (merges into train only) + --regression-cases (case ID denylist for defensive B+C re-check at load) — the narrow, documented exception to 'train never reads cases' (IDs for isolation, never case rows as training data). eval/ledger tag rounds with case_sets_used. Step-2 experiment under heuristic labels was NEGATIVE: 0/4 pending attacked, and a gate regressed (-0.0175 nDCG@10, recorded in ledger round 2). Model restored to baseline so demo ships green; negative result stands in ledger history, validating the pipeline-validation caveat. Real attack quality waits for LLM-mode labels (separate task, user must rotate the leaked DeepSeek key first). Anti-leak preserved; spec refined.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `ac6aefc` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 4: Reckless mode repair lane

**Date**: 2026-06-30
**Task**: Reckless mode repair lane
**Branch**: `main`

### Summary

Implemented reckless mode for FiQA reranker acceptance, documented the production-case repair workflow, and archived the task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `8e69c7e` | (see git log) |
| `ef2231a` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 5: Production case repair entrypoint

**Date**: 2026-06-30
**Task**: Production case repair entrypoint
**Branch**: `main`

### Summary

Added the two-table production repair workflow: compile_cases, repair_reranker, promote_repair, repair fixtures, docs, and backend contracts. Verified strict compile, auto-split, hard-fail, controlled success, promotion success/refusal, unchanged repository ledger, and standard FiQA train/eval.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `a146d5a` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
