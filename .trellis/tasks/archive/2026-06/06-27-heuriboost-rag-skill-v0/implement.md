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

## Phase 4: Verification (original financial demo)

- [x] Run the financial RAG demo end to end.
- [x] Confirm reports show before/after ranking changes.
- [x] Confirm a wrong-year hard negative can be represented as a regression case.
- [x] Confirm the skill instructions are actionable without hidden assumptions.

## Phase 5: FiQA Pivot (2026-06-29)

Reason: the toy financial demo cannot honestly back the "beats baselines" claim
(n=1 validation, rigged ranks). Replace it with a single real-dataset demo
(BEIR/FiQA-2018) and retarget features to FiQA-style hard negatives.

- [ ] Delete `examples/financial_rag/`; update `.gitignore` (drop financial output, add `examples/fiqa/output/` and `examples/fiqa/.cache/`).
- [ ] Replace feature set in `common.py`: remove `year_overlap_count`/`quarter_overlap_count`/`wrong_year_flag`; rename `numeric_overlap_count` -> `number_overlap_count`; add `entity_overlap_count`/`important_term_overlap`/`low_information_density_flag`.
- [ ] Rewrite `eval_reranker.py` `analyze_reason()`, `suggest_next_actions()`, and Feature Contrast block for the new feature set.
- [ ] Add `build_fiqa_csv.py` (BM25 + MiniLM + RRF retrieval, build-time LLM judge, native FiQA split, caps 150/40/40, top-20, doc_text<=400, `--cache-dir`). Author-only; run locally by the user, not in CI/agent.
- [ ] Add `requirements-build.txt` (rank-bm25, sentence-transformers, datasets, LLM client).
- [ ] Add `examples/fiqa/DATA_CARD.md` (source, CC BY-SA 4.0, retriever config, judge model/prompt/date, "weights/corpus/build-deps not committed").
- [ ] Update `README.md`, `README.zh-CN.md`, `CODEBUDDY.md`, `CONTEXT.md` to the FiQA narrative and paths.
- [ ] Update templates `feature_recipes.yaml` (match FEATURE_NAMES), `query_doc_examples.csv`, `regression_cases.yaml` to FiQA style.
- [ ] User runs `build_fiqa_csv.py` locally to produce and commit `examples/fiqa/query_doc_examples.csv`; mine 5-8 failures, hand-confirm as regression cases.

## Validation Commands (after user generates the FiQA CSV)

```bash
python skills/heuriboost-rag/scripts/validate_dataset.py examples/fiqa/query_doc_examples.csv
python skills/heuriboost-rag/scripts/train_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output
python skills/heuriboost-rag/scripts/eval_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --regression-cases examples/fiqa/regression_cases.yaml
```

To (re)generate the demo CSV offline:

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
python skills/heuriboost-rag/scripts/build_fiqa_csv.py --output examples/fiqa/query_doc_examples.csv
```
