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
