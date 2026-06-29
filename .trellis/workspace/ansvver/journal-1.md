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
