# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Repository State

**Project name: HeuriBoost** (local directory: `heuriboost`).

This repository currently ships a V0 Codex-compatible RAG reranking skill with
skill-local Python scripts. It intentionally does **not** have a formal Python
package scaffold yet: no `pyproject.toml`, stable public Python API, or package
publishing flow.

Primary files:

- `README.md` and `README.zh-CN.md` — public project overview and quick start.
- `skills/heuriboost-rag/SKILL.md` — Codex-compatible skill instructions.
- `skills/heuriboost-rag/scripts/` — runnable V0 validation, training, and evaluation scripts.
- `skills/heuriboost-rag/templates/` — user-copyable CSV/YAML templates.
- `examples/fiqa/` — end-to-end FiQA-2018 RAG reranking demo (CSV generated offline by `build_fiqa_csv.py`).
- `docs/specs/ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md` — general adaptive XGBoost framework spec.
- `docs/specs/QD_RERANKER_SPEC.md` — Q-D reranker specialization for RAG retrieval reranking.
- `docs/specs/*_CN.html` — Chinese HTML companion files.

## Commands

Install runtime dependencies:

```bash
python -m pip install -r skills/heuriboost-rag/requirements.txt
```

Validate the demo dataset:

```bash
python3 skills/heuriboost-rag/scripts/validate_dataset.py examples/fiqa/query_doc_examples.csv
```

Train the demo reranker:

```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output
```

Train with mined case_sets folded into the train split (V1 step 2 closed loop):

```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --case-sets examples/fiqa/case_sets --regression-cases examples/fiqa/regression_cases.yaml
```

Evaluate and run regression gates:

```bash
python3 skills/heuriboost-rag/scripts/eval_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --regression-cases examples/fiqa/regression_cases.yaml
```

Set the cross-round ledger anchor (manual, after confirmed gains):

```bash
python skills/heuriboost-rag/scripts/regression_ledger.py set-anchor --ledger examples/fiqa/ledger.json
```

Print a ledger progress summary:

```bash
python skills/heuriboost-rag/scripts/regression_ledger.py summary --ledger examples/fiqa/ledger.json
```

Syntax-check scripts:

```bash
python3 -m py_compile skills/heuriboost-rag/scripts/*.py
```

The FiQA demo CSV (`examples/fiqa/query_doc_examples.csv`) is generated offline
by `build_fiqa_csv.py` and committed; its build dependencies
(`requirements-build.txt`), the downloaded FiQA corpus, and the dense-encoder
weights are NOT committed. Generated demo outputs under `examples/fiqa/output/`
and the build cache `examples/fiqa/.cache/` are ignored.

## Big-Picture Architecture

The specs describe two related systems:

1. **Adaptive XGBoost framework**: a general supervised tabular-learning framework driven by task profiles.
2. **Q-D reranker**: a RAG query-document ranking task profile/specialization of the adaptive framework.

Core loop:

```text
Data snapshot -> feature recipe -> XGBoost model -> evaluation gates -> failure memory -> feature discovery -> controlled promotion
```

Online/batch prediction path:

```text
input request/entity
  -> task profile resolver
  -> data snapshot or online feature fetch
  -> feature extractor L0-L2
  -> XGBoost model
  -> optional post-processor / decision policy
  -> prediction output
  -> feedback / outcome / diagnostic logs
```

Offline learning path:

```text
logs + labels + failure cases
  -> dataset builder
  -> train / validation / regression / test split
  -> feature extraction
  -> XGBoost training
  -> task-specific evaluation
  -> regression gate
  -> shadow or backtest
  -> A/B or staged rollout
```

## Key Domain Objects

From `docs/specs/ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md`:

- `TaskProfile` binds task type, entity/group keys, objective, metrics, gates, slices, and serving behavior.
- `LearningExample` is the canonical supervised row. For ranking, rows share `group_id`; for classification/regression, rows are usually individual entities/events.
- `PredictionContextSnapshot` freezes candidate sets or row-feature context for reproducible evaluation.
- `RegressionCase` captures historical failures and must be used as a gate, not training data.
- `FeatureRecipe` declares generated features in a registry/DSL with version, inputs, type, cost tier, online-safety, leakage risk, and expected slices.

Q-D mapping:

- `QueryDocExample` maps to `LearningExample` with `group_id=query_id`.
- `CandidateSnapshot` maps to `PredictionContextSnapshot` with `context_type=ranking_candidates`.
- `must_include_doc_ids` / `must_not_include_doc_ids` define ranking regression expectations.
- Answer citations and LLM support judgments are labels/diagnostics, never online features.

## Critical Invariants

Follow these when implementing anything from the specs:

- Evaluation snapshots are fixed; compare candidate and baseline models on the same data, feature, and split snapshots.
- Train, validation, regression, and test sets are hard-isolated.
- Never train on regression cases directly.
- Features must be available at prediction time. Do not use post-outcome signals, answer citations, LLM post-hoc judgments, user actions after prediction, or human labels as online model features.
- Implement features through a `FeatureRecipe` registry/DSL, not scattered ad hoc feature code.
- New features require ablation, slice evaluation, latency/cost checks, leakage checks, and a promote/reject/quarantine decision.
- Preserve feature names, feature recipe versions, task profile, model config, training snapshot ID, and evaluation report with each model artifact.
- Ranking tasks must group rows by `query_id`/`group_id`; do not shuffle query-doc pairs across groups.
- Classification threshold policy is separate from raw model score.
- Regression target transforms and inverse transforms must be recorded.
- Metrics must be reported globally, by slice, and by hard example type.

## HPO and Feature Discovery

The framework should expose an `HPOEngine` adapter and use an external HPO backend. Do not implement search algorithms, trial pruning, or distributed scheduling inside this repository unless the spec changes.

HPO must:

- Accept fixed train and validation snapshots.
- Accept bounded search budgets.
- Use deterministic seeds where supported.
- Return tried parameter sets, scores, metrics, artifacts, and failure reasons.
- Preserve objective, eval metric, early stopping config, and feature set version.
- Support cancellation/resumability for long experiments.
- Never read regression cases as part of the optimization objective.

Feature discovery should use two stages:

1. Scout stage: fixed strong baseline params or small-budget HPO to reject weak features cheaply.
2. Finalist stage: full HPO only for shortlisted feature candidates, followed by promotion gates.

For new features, compare tuned baseline against tuned candidate before promotion.

## Suggested Implementation Order

Use this order from the specs when building the first implementation:

1. Decide the V0 task profile first.
2. Create schemas before model code.
3. Build dataset builders around fixed snapshots.
4. Implement V0 features through the registry.
5. Train XGBoost using objective and metrics from the task profile.
6. Add the HPO adapter by calling an existing external tool/backend.
7. Add validation and regression gates before automatic feature discovery.
8. Add feature discovery only after gates exist.
9. Add shadow/backtest before A/B or staged rollout.
10. Add explanation output for score/rank changes.

## Current V0 Layout

```text
skills/heuriboost-rag/
  SKILL.md
  requirements.txt
  requirements-build.txt
  scripts/{common.py,inspect_rag_repo.py,validate_dataset.py,train_reranker.py,eval_reranker.py,regression_ledger.py,mine_case_sets.py,build_fiqa_csv.py,run_hpo.py,run_ablation.py,run_discover_candidates.py}
  scripts/features/{__init__.py,registry.py,primitives.py,recipes.py}
  scripts/hpo/{__init__.py,engine.py,optuna_backend.py}
  templates/{query_doc_examples.csv,regression_cases.yaml,feature_recipes.yaml,promotion_gate.yaml}
examples/fiqa/
  query_doc_examples.csv
  regression_cases.yaml
  ledger.json
  case_sets/{manifest.json,<case_id>.csv}
  DATA_CARD.md
docs/specs/
  ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md
  QD_RERANKER_SPEC.md
```

Generated demo output is written under `examples/fiqa/output/` and ignored by git.
The cross-round ledger (`examples/fiqa/ledger.json`) is committed to git (NOT
ignored) — it is version-controlled round history, not a generated artifact.

## Future Layouts From Specs

Generic framework layout:

```text
adaptive_xgb/
  configs/{task_profiles,feature_sets,training,promotion_gates}/
  data/schemas/
  features/{registry.py,primitives.py,extractors/}
  labels/{build_labels.py,weak_supervision.py,llm_judge.py}
  training/{build_dataset.py,train_xgb.py,evaluate.py}
  regression/{cases/,run_regression_gate.py}
  serving/{predict.py,postprocess.py,explain.py}
  experiments/{ablation.py,feature_discovery.py}
  docs/{ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md,QD_RERANKER_SPEC.md}
```

Q-D reranker layout:

```text
qd_reranker/
  configs/{feature_sets,training,promotion_gates}/
  data/schemas/
  features/{registry.py,primitives.py,extractors/}
  labels/{build_labels.py,llm_judge.py}
  training/{build_dataset.py,train_xgb_ranker.py,evaluate.py}
  regression/{cases/,run_regression_gate.py}
  serving/{rerank.py,explain.py}
  experiments/{ablation.py,feature_discovery.py}
  docs/QD_RERANKER_SPEC.md
```

## V0 Implementation Decisions

- V0 focuses on RAG query-document learning-to-rank, not generic classification/regression.
- V0 is CSV-first and does not directly depend on LangChain, LlamaIndex, a vector database, or a specific retriever framework.
- V0 uses real `xgboost`; missing dependency errors must be clear and actionable.
- V0 keeps runnable scripts under `skills/heuriboost-rag/scripts/` and does not add formal package scaffolding.
- V0 includes deterministic failure analysis lite, not automatic feature discovery.

## V1 Case State Machine (implemented 2026-06-29)

- Three-state case schema: `gate` (blocks on failure), `pending` (reported
  only, non-blocking), `retired` (skipped). Missing `status` defaults to `gate`.
- Per-case local metric checks (A): `require_rank` (first must_include must
  reach rank <= N), `min_ndcg10` (per-query nDCG@10 floor).
- Cross-round ledger in committed `examples/fiqa/ledger.json` (NOT gitignored,
  NOT auto-committed). `regression_ledger.py` owns round snapshots, the B2
  anchor, the B-vs-anchor comparison, and manual `set-anchor`/`promote` helpers.
- B-vs-anchor is REPORTED (not blocking) in V1, consistent with manual
  promotion philosophy.
- Promotion pending -> gate is always manual (interactive confirmation).
- Anti-leak invariant preserved: `train_reranker.py` never loads regression
  cases or the ledger.

## V1 Case Sets Mining (implemented 2026-06-29, step 2)

- `mine_case_sets.py` mines same-pattern training samples for each `pending`
  case using the a+b+c intersection rule (semantic similarity + failure shape
  + failure_type match). B+C isolation is enforced at mining time.
- `train_reranker.py --case-sets <dir-or-file>` merges mined rows into the
  TRAIN split only. A defensive B+C re-check runs at load time using the case
  query_id/doc_id denylist.
- **Refined anti-leak contract**: `train_reranker.py` may read
  `regression_cases.yaml` ONLY for the case query_id/doc_id denylist to
  enforce B+C isolation. Case ROWS never enter training. `case_sets` (mined
  samples) ARE training data, B+C isolated from cases. `train_reranker.py`
  still never reads `ledger.json`.
- `eval_reranker.py --case-sets-used` tags the ledger round; `summary()`
  prints "round N used case_sets" when true.
- `case_sets` are committed under `examples/fiqa/case_sets/` (derived,
  regeneratable via `mine_case_sets.py`, NOT gitignored).

## Future Decisions Before Expanding Beyond V0

Resolve these before creating the initial scaffold:

- Which V0 task profile to implement first: Q-D ranking, binary classification, multi-class classification, regression, count/rate, or survival/time.
- Primary metric, operating point, and critical slices.
- Trusted, weak, and diagnostic label sources.
- Data snapshot and feature snapshot storage approach.
- External HPO backend and budget for scout/finalist stages.
- Latency or batch runtime budget.
- Stable entity/version ID requirements.
- Temporal split requirements.
- Regression-case creation, review, and retirement process.
