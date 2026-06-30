# PRD: HPO Engine Adapter (child of feature-discovery)

## Parent

Child of `.trellis/tasks/06-29-feature-discovery/` (parent PRD has the
dependency graph). This is child #1, no blockers, unblocks child #2
(A/B/C/D ablation framework).

## Goal

Implement the spec §12.5 `HPOEngine` adapter wrapping an external HPO backend
(Optuna), with the V0 Q-D reranker as its first consumer. The adapter searches
XGBoost parameters under a fixed budget on fixed train/validation snapshots,
returns the full trial history, and never reads regression cases. This unblocks
the A/B/C/D ablation framework (child #2).

## User value

- Unblocks A/B/C/D ablation (the `D - B` candidate-gain attribution in spec
  §15.3) — currently impossible because there is no tuned-params baseline.
- Makes parameter search reproducible (deterministic seed) and bounded
  (n_trials + timeout), instead of ad-hoc hand-tuning.
- Gives the feature-discovery loop its scout-stage cheap filter (small-budget
  HPO) and finalist-stage tuner.

## Confirmed facts (from inspection)

- Spec §12.5: `class HPOEngine: def optimize(self, task_profile, feature_set,
  train_snapshot, valid_snapshot, budget) -> TrialResult`.
- Backend requirements (spec): fixed train+valid snapshots; bounded budget;
  deterministic seeds; return tried params/scores/metrics/artifacts/failure
  reasons; preserve objective/eval_metric/early_stopping/feature_set_version;
  support cancellation+resumability; NEVER read regression cases.
- Spec §15.3: HPO must run under the same budget for baseline and candidate
  feature sets; do not tune repeatedly against the regression set; do not let
  HPO optimize only one global metric while ignoring slices/hard examples
  (resolved: HPO optimizes the global validation metric; slices/cases are
  enforced at the PromotionGate layer in child #2, NOT inside HPO — this keeps
  HPO case-blind per the anti-leak rule).
- Current `train_reranker.py:220` uses `xgb.train(params, dtrain,
  num_boost_round, evals=[...])` with fixed params. HPO will search over
  `max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma,
  reg_lambda` (a subset of the spec's common dimensions), keeping
  `objective=rank:ndcg`, `eval_metric=ndcg`, `early_stopping_rounds` fixed.
- FeatureRecipe registry (just shipped) provides `feature_set_name`,
  `feature_set_version`, `names()`, `extract(df)` — the adapter records these
  in `TrialResult` for attribution.

## Requirements

R1. `HPOEngine` adapter class in a new `scripts/hpo/` subpackage, wrapping
    Optuna (external backend — no self-implemented search algorithms).
R2. `optimize(task_profile, feature_set, train_snapshot, valid_snapshot, budget,
    seed) -> TrialResult` where:
    - `train_snapshot` / `valid_snapshot` are PRE-COMPUTED feature matrices
      (X, y, query groups) — the adapter does NOT re-extract features per
      trial, and does NOT touch the raw CSV.
    - `budget` = `n_trials` (int) + optional `timeout_sec` (int).
    - `seed` feeds Optuna's `TPESampler(seed=seed)` for determinism.
R3. Search space (V0 default): `max_depth` (3-10), `min_child_weight` (0.5-5),
    `eta` (0.05-0.3), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0),
    `gamma` (0-5), `reg_lambda` (0.1-10). Fixed: `objective=rank:ndcg`,
    `eval_metric=ndcg`, `early_stopping_rounds=20`, `num_boost_round=200`.
R4. `TrialResult` carries: `best_params`, `best_score`, `trials`
    (list of {params, score, state, failure_reason?}), `feature_set_name`,
    `feature_set_version`, `objective`, `eval_metric`, `early_stopping_rounds`,
    `seed`, `budget`.
R5. Anti-leak: the adapter API has NO parameter for regression cases. Cases
    are never read, imported, or referenced. Enforced by signature.
R6. Cancellation via Optuna `timeout` (V0). Resumability (SQLite-backed study)
    is DEFERRED — V0 trials run in-process; a `KeyboardInterrupt` cleanly
    returns the best-so-far trial.
R7. A `scripts/run_hpo.py` CLI: loads dataset, extracts features via the
    registry, builds snapshots, runs HPO, writes
    `examples/fiqa/output/hpo/hpo_report.md` + `best_params.json` + `trials.json`.
R8. `optuna` added to `requirements-build.txt` (experiment dependency, not
    runtime — the shipped skill's train/eval does not need HPO).

## Acceptance criteria

- A1. `python3 -m py_compile scripts/hpo/*.py scripts/run_hpo.py` passes.
- A2. `python3 scripts/run_hpo.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --n-trials 5 --seed 42` runs to completion, writes `hpo_report.md`, `best_params.json`, `trials.json`.
- A3. `trials.json` has exactly 5 trial entries; `best_params.json` has the
  highest-validation-nDCG@10 params; `best_score` matches the best trial.
- A4. Determinism: two runs with the same `--seed` produce identical `trials.json` (same params + scores in the same order). HPO trials fix `nthread=1` so xgboost is fully deterministic (multi-thread histogram building is the only non-determinism source).
- A5. `TrialResult` carries `feature_set_name=heuriboost_rag_v0`,
  `feature_set_version=1`, `objective=rank:ndcg`, `eval_metric=ndcg@10`.
- A6. Anti-leak: `grep -r "regression_cases\|ledger\|must_include" scripts/hpo/ scripts/run_hpo.py` returns nothing — HPO is case-blind. The HPO SEARCH sees only train+valid snapshots (test-blind); test is used only for post-hoc evaluation, never optimization.
- A7. `--timeout-sec 5` with `--n-trials 1000` returns within ~10s with the
  best-so-far trial (cancellation works).
- A8. `best_params.json` includes `best_iteration`. Retraining with `best_params`
  + `num_boost_round = best_iteration + 1` reproduces `best_score` on validation
  exactly (reproducibility — confirms the adapter + best_iteration contract wire
  end-to-end). The `hpo_report.md` honestly reports val + test + val−test gap.
  On the 40-query FiQA validation, HPO overfits (val > test by ~0.08, test may
  fall below the 0.83 baseline) — this is a real demo-size finding the report
  surfaces, NOT a wiring defect; it motivates scout-vs-finalist staging and CV
  in future children.

## Resolved decisions

- **Backend**: Optuna (de facto Python HPO; deterministic TPESampler; pip-installable; matches "external backend, no self-implemented search").
- **Output contract**: `best_params.json` records `params` + `best_iteration` + `early_stopping_rounds`. Any consumer reproduces the HPO-best model by training `num_boost_round=best_iteration + 1` with `best_params` (no early stopping needed; `best_iteration` is 0-indexed). `train_reranker.py` is NOT modified — a future consumer (child #2 ablation, or a `--tune` flag) uses this contract.
- **Determinism**: HPO trials fix `nthread=1` (xgboost multi-thread histogram building is the only non-determinism source). `TPESampler(seed=...)` is deterministic. A4 (identical trials.json across same-seed runs) is strictly achievable.
- **Overfit protection**: search space NARROWED around the shipped baseline (max_depth 3-6, eta 0.05-0.2, min_child_weight 0.05-3, etc.) — 150 train / 40 validation queries can't support wide bounds. `hpo_report.md` reports both val (search objective) and a post-hoc **test** nDCG@10 (honest estimate). A8 = reproducibility (retrain `best_iteration+1` reproduces `best_score`), NOT "test ≥ 0.83" — on 40-query validation, test may fall below 0.83 baseline and that is a real demo-size finding, not a defect. CV deferred (D-B relative attribution is robust to overfit).
- **test eval location**: `run_hpo.py` does post-hoc test evaluation of the best model after the search (clearly labeled "honest test estimate, not a search objective"). HPO SEARCH stays test-blind (anti-leak); post-hoc eval is a single forward pass, same as `eval_reranker.py`.
- **Objective**: single-objective global validation nDCG@10. Slices + regression
  cases are enforced at the PromotionGate (child #2), NOT inside HPO — keeps
  HPO case-blind per anti-leak.
- **Snapshots**: pre-computed feature matrices (X, y, groups) passed in; no
  per-trial re-extraction, no raw CSV access inside the adapter.
- **Resumability**: DEFERRED. V0 = in-process + timeout cancellation. SQLite
  study persistence can be added when long experiments demand it.
- **Dependency tier**: `optuna` in `requirements-build.txt` (experiment tool,
  not runtime).

## Out of scope (deferred)

- SQLite-backed resumable studies (V0: in-process + timeout).
- Multi-objective HPO (slice/hard-example terms in the objective).
- HPO over feature SETS (adapter searches params for a FIXED feature set; feature-set search is child #2's ablation job).
- Distributed scheduling.
- Integration into `train_reranker.py` as a `--tune` flag (a `run_hpo.py` CLI is enough for V0; `train_reranker.py` can consume `best_params.json` later).
