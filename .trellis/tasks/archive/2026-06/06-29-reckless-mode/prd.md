# 鲁莽模式

## Goal

Add a `--reckless` mode for the FiQA reranker workflow where `case_sets` are fed directly into training, then the round is accepted only if the `case_sets` acceptance checks pass and the test split's `nDCG@10` and `MRR@10` both improve over the anchored baseline.

## Background

- The current loop already supports mining `case_sets` and merging them into the train split.
- Regression cases are still modeled as gates, not training rows.
- Promotion currently requires explicit global and regression thresholds, plus manual promotion for anchored-baseline comparisons.
- The acceptance baseline for reckless mode is the pre-optimization anchored baseline, not the immediately prior round.
- If no ledger anchor exists, or if the test split is missing, reckless mode hard-fails.
- Reckless mode intentionally makes `case_sets` part of the training input instead of only an auxiliary mined-data path.
- Reckless-mode acceptance reuses the original regression case rule for each `source_case_id` in `case_sets`.

## Requirements

- Support a named reckless mode on the existing training/evaluation CLI (`--reckless`).
- In reckless mode, include `case_sets` directly in training for the round.
- In reckless mode, include `case_sets` in the acceptance check by reusing the original regression case definition for each `source_case_id`.
- `--case-sets` may be supplied to both training and evaluation; empty inputs are allowed.
- When `--reckless` is used and `--case-sets` is omitted, default to `examples/fiqa/case_sets`.
- In reckless mode, require the test split's `nDCG@10` and `MRR@10` to improve over the anchored baseline.
- In reckless mode, require a test split to exist.
- Preserve the existing non-reckless behavior unless the mode is explicitly selected.

## Open Questions

- None.

## Acceptance Criteria

- [ ] A reckless-mode entry point exists and can train with `case_sets` folded into the training split.
- [ ] `case_sets` are evaluated as part of the round acceptance.
- [ ] The round is considered acceptable only when the test split's `nDCG@10` and `MRR@10` both improve versus the anchored baseline.
- [ ] If the test split is missing, reckless mode hard-fails.
- [ ] Empty `case_sets` inputs are allowed.
- [ ] `--reckless` defaults `--case-sets` to `examples/fiqa/case_sets` when omitted.
- [ ] Default behavior remains unchanged when reckless mode is not selected.

## Notes

- Keep `prd.md` focused on requirements, constraints, and acceptance criteria.
- Lightweight tasks can remain PRD-only.
- For complex tasks, add `design.md` for technical design and `implement.md` for execution planning before `task.py start`.
