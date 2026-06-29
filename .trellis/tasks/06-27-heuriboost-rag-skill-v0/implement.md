# HeuriBoost RAG Skill V0 Implementation Plan

## Phase 1: Planning Artifacts

- [x] Review this PRD/design with the user.
- [x] Resolve minimum Python version guidance.
- [x] Decide whether V0 includes executable scripts or only skill/templates.
- [x] Decide whether V0 uses skill-local scripts or a formal Python package scaffold.
- [x] Decide whether V0 uses real `xgboost` or a placeholder/demo model.
- [x] Decide dependency installation format.
- [x] Decide V0 skill format and other-agent support boundary.

## Phase 2: Project Skeleton

- [x] Rewrite root `README.md` around agent-skill positioning.
- [x] Add `skills/heuriboost-rag/SKILL.md` with `audit`, `bootstrap`, and `experiment` workflows.
- [x] Add `skills/heuriboost-rag/requirements.txt` with unpinned runtime dependencies.
- [x] Add runtime scripts under `skills/heuriboost-rag/scripts/`; do not add formal package scaffolding in V0.
- [x] Add templates:
  - [x] `query_doc_examples.csv`
  - [x] `regression_cases.yaml`
  - [x] `feature_recipes.yaml`
  - [x] `promotion_gate.yaml`
- [x] Add `examples/financial_rag/` demo data.

## Phase 3: Runtime Scripts

- [x] Add `validate_dataset.py`.
- [x] Add `train_reranker.py`.
- [x] Add `eval_reranker.py`.
- [x] Add dataset validation for required CSV columns and grouped splits.
- [x] Add V0 feature extraction from self-contained CSV rows.
- [x] Add XGBoost LambdaMART training grouped by `query_id`.
- [x] Add startup dependency checks with clear `pip install` guidance.
- [x] Add baseline evaluation for original dense/sparse/RRF order when available.
- [x] Add regression gate evaluation from YAML.
- [x] Emit reports and artifacts under `reports/`, `models/`, and `regression_cases.yaml`.
- [x] Add deterministic failure analysis lite report.

## Phase 4: Verification

- [x] Run the financial RAG demo end to end.
- [x] Confirm reports show before/after ranking changes.
- [x] Confirm a wrong-year hard negative can be represented as a regression case.
- [x] Confirm the skill instructions are actionable without hidden assumptions.

## Validation Commands

```bash
python skills/heuriboost-rag/scripts/validate_dataset.py examples/financial_rag/query_doc_examples.csv
python skills/heuriboost-rag/scripts/train_reranker.py examples/financial_rag/query_doc_examples.csv
python skills/heuriboost-rag/scripts/eval_reranker.py examples/financial_rag/query_doc_examples.csv --regression-cases examples/financial_rag/regression_cases.yaml
```
