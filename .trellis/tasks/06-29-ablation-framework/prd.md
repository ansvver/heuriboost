# PRD: A/B/C/D Ablation Framework (child of feature-discovery)

## Parent / dependencies

Child #2 of `.trellis/tasks/06-29-feature-discovery/`. Depends on child #1
(HPO adapter, shipped `0db0bba`) — uses `HPOEngine.optimize` for cells B and D.
Does NOT depend on child #3 (LLM candidate gen) — candidates are provided
manually here; child #3 will feed candidates into this framework later.

## Goal

Build the spec §15.3 A/B/C/D ablation runner: given a candidate feature, run 4
cells (baseline±candidate × baseline±tuned params) under the SAME data/feature
snapshot/split/metric/gate, compute the deltas (B-A, C-A, D-B, D-C), run the
regression gate on each cell, and recommend promote/reject/quarantine.
Promotion stays manual (project invariant).

## User value

- Lets a maintainer test "does candidate feature X actually help, after fair
  tuning?" — the question the spec §15.3 says matters (D-B, not D-A).
- Reuses the HPO adapter (child #1) for B and D under the same budget.
- Produces a single ablation report card that a human can act on.
- Unblocks child #4 (discovery orchestration): the orchestrator calls this
  framework for each LLM-proposed candidate (child #3).

## Confirmed facts (from inspection)

- Spec §15.3: A=baseline features+baseline params, B=baseline features+tuned
  params, C=baseline+candidate+baseline params, D=baseline+candidate+tuned
  params. Promotion primarily considers `D - B`.
- Spec §15.3 rules: same data snapshot, feature snapshot, split, metric,
  regression gate; same HPO budget for baseline and candidate feature sets;
  do not tune repeatedly against the regression set; do not let HPO optimize
  only one global metric while ignoring slices/hard examples.
- HPO adapter (`scripts/hpo/`): `HPOEngine.optimize(task_profile,
  feature_set_name, feature_set_version, train_snapshot, valid_snapshot,
  budget, seed) -> TrialResult`. The adapter is feature-set-agnostic (reads
  `feature_names = list(train_snapshot.X.columns)`), so cells C/D pass a 13-col
  snapshot and it adapts. `TrialResult` carries `best_params`, `best_iteration`,
  `best_score`, `test_score`.
- FeatureRecipe registry uses Option C: ONE shared `extract_all(row) -> dict`
  computes all 12 baseline features. There is NO per-feature dispatch in the
  shipped registry.
- Baseline params (train_reranker.py:210-219): eta=0.08, max_depth=3,
  min_child_weight=0.1, subsample=0.9, colsample_bytree=0.9, seed=42,
  num_boost_round=args.rounds (no early stopping).
- HPO adapter uses nthread=1 + nDCG on raw-label scale (same as baseline 0.853).
- Regression gate: `eval_reranker.py` runs `regression_cases.yaml` (gate
  blocks, pending reports). The gate logic can be reused for per-cell checks.
- On the 40-query FiQA validation, HPO overfits (val > test by ~0.08) — the
  ablation framework MUST report test nDCG@10 per cell, not just val, so
  D-B on val doesn't cherry-pick.

## Requirements (draft — pending grilling)

R1. A `run_ablation.py` CLI that accepts a candidate feature (recipe YAML +
    impl) and runs the 4 cells.
R2. The candidate is wrapped onto the shipped `extract_all` WITHOUT modifying
    the shipped registry: cells C/D use `extract_all_plus(row) = {**extract_all(row),
    candidate_name: candidate_fn(row)}`.
R3. Cells A, C: train with baseline params (fixed). Cells B, D: HPO search
    under the SAME budget + seed.
R4. Per cell: val nDCG@10 + test nDCG@10 (raw-label scale) + regression gate
    (gate cases pass/fail).
R5. Deltas: B-A, C-A, D-B, D-C on both val and test.
R6. Recommendation: promote if D-B > threshold (val) AND D test >= A test AND
    D gates pass; reject if D-B <= 0 OR D regresses gates; quarantine otherwise.
    Promotion is a REPORT recommendation, never automatic.
R7. Output: `output/ablation/ablation_report.md` (cell table + deltas + gate +
    recommendation) + `ablation_result.json` (machine-readable).

## Resolved decisions

- **Candidate provisioning**: `--candidate-recipe <yaml>` + `--candidate-impl
  <pyfile:func>` where fn is `(row) -> float`. The framework wraps
  `extract_all_plus(row) = {**extract_all(row), name: fn(row)}` — the shipped
  `extract_all`/registry are NOT modified; the candidate is a probe. Child #3
  (LLM gen) will output recipe YAML + impl code, consumed directly. Candidate
  `inputs` validated against `ALLOWED_INPUTS` at load (rejects `label` etc.).
- **Promotion criteria**: promote iff `D-B(val) > --promote-threshold` (default
  0.01) AND `D-B(test) > 0` AND D gate cases all pass. reject iff `D-B(val) <=
  0` OR D regresses a gate. quarantine otherwise. Report-only; promotion is
  always manual (project invariant). Dual val+test check avoids cherry-picking
  HPO-overfit validation noise.
- **Training procedure (all 4 cells, unified)**: `xgb.train(params, dtrain,
  200, evals=[dvalid], early_stopping_rounds=20, nthread=1, seed=42)` → predict
  with `iteration_range=(0, best_iteration+1)`. A/C use fixed shipped param
  values; B/D use HPO best params (retrained with early stopping to get the
  model object — deterministic, reproduces HPO-best per A8 of hpo-adapter).
  This isolates the (feature_set, params) effect; B-A is pure param gain.
- **HPO fairness**: B and D use the SAME `n_trials` + `seed` (spec §15.3).
- **Regression gate reuse**: import `load_regression_cases` +
  `run_regression_cases` from `eval_reranker.py` (top-level functions). All 4
  cells run the gate for completeness; only D's gate is promotion-blocking.
- **nDCG scale**: all cells scored via `_ndcg10_from_scores(raw_labels)` — same
  scale as baseline 0.853 + HPO scores.
- **Output**: `output/ablation/{ablation_report.md, ablation_result.json}`
  (gitignored). Single file `scripts/run_ablation.py` (no subpackage; the 4-cell
  flow is cohesive).

## Out of scope (deferred)

- LLM candidate generation (child #3).
- Candidate groups / multiple candidates in one ablation (spec §15.2; V0 = one
  candidate per run; child #4 orchestration loops over candidates).
- FeatureMemory (record promote/reject/quarantine institutionally).
- Slice evaluation beyond gate cases (SliceEvaluator is a future component).
- Scout-vs-finalist staging (this framework IS the finalist stage; scout is a
  cheaper pre-filter for child #4).
- Automatic promotion.
- Configurable baseline params (V0 uses shipped fixed values).
