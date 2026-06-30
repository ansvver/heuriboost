# Design: A/B/C/D Ablation Framework

## Architecture

Single CLI `scripts/run_ablation.py` (inline helpers — the 4-cell flow is
cohesive; no subpackage needed). Reuses:
- `features.REGISTRY` for the baseline 12-feature `extract_all`.
- `hpo.HPOEngine` for cells B and D.
- `eval_reranker.load_regression_cases` + `run_regression_cases` for the gate.
- `common.ndcg_at_k` + `hpo.optuna_backend._ndcg10_from_scores` for raw-label
  nDCG@10 (same scale as baseline 0.853 + HPO scores).

## Candidate feature contract

A candidate = (recipe YAML, impl fn):

```yaml
# candidate_recipe.yaml
name: dense_to_term_ratio
version: 1
description: "Ratio of dense score to term overlap (probe)."
task_profiles: [qd_reranker]
inputs: [dense_score, query_text, doc_text]   # must be in ALLOWED_INPUTS
type: numeric
default_value: 0.0
cost_tier: L0
online_safe: true
leakage_risk: low
expected_slices: []
owner: ablation-probe
impl: candidate            # logical name; the impl fn is loaded from --candidate-impl
```

```python
# candidate_impl.py
def candidate(row) -> float:
    from features.primitives import tokenize, numeric_value
    q = tokenize(row["query_text"]); d = tokenize(row["doc_text"])
    overlap = len(q & d) / max(len(q), 1)
    return numeric_value(row, "dense_score") / max(overlap, 1e-9)
```

CLI: `--candidate-recipe candidate_recipe.yaml --candidate-impl candidate_impl.py:candidate`.

The framework loads the impl via `importlib`, validates the recipe's `inputs`
against `ALLOWED_INPUTS` + `online_safe` + required fields (reusing the
registry's validation logic), then builds the candidate extractor:

```python
def extract_all_plus(row):
    base = extract_all(row)               # the shipped 12 features
    base[candidate_name] = candidate_fn(row)
    return base
```

The shipped `extract_all` and `feature_recipes.yaml` are NOT modified — the
candidate is a probe that exists only for this ablation run.

## The 4 cells

All cells use the SAME training procedure to isolate the (feature_set, params)
effect per spec §15.3:

```text
train with early stopping (num_boost_round=200, early_stopping_rounds=20,
nthread=1, seed=42) -> model with best_iteration
predict with iteration_range=(0, best_iteration+1) on val + test
score = _ndcg10_from_scores(preds, raw_labels, groups)   # raw-label scale
```

| Cell | Feature set | Params |
|---|---|---|
| A | baseline 12 (`REGISTRY.extract`) | fixed shipped values (eta=0.08, max_depth=3, min_child_weight=0.1, subsample=0.9, colsample_bytree=0.9) |
| B | baseline 12 | HPO best (`HPOEngine.optimize` → best_params) |
| C | baseline + candidate 13 (`extract_all_plus`) | fixed shipped values |
| D | baseline + candidate 13 | HPO best on the 13-col snapshot |

For B/D, after HPO returns `best_params` + `best_iteration`, retrain with
`best_params` + early stopping (200, 20) to get the model object, then predict
with `iteration_range=(0, best_iteration+1)`. (HPO's `TrialResult` doesn't
return the model; the retrain is deterministic with nthread=1 + seed=42, so it
reproduces the HPO-best model exactly — A8 of the HPO task guarantees this.)

HPO budget + seed are the SAME for B and D (spec §15.3: "same HPO budget for
baseline and candidate feature sets").

## Per-cell evaluation

For each cell, on BOTH validation and test:
- `val_score` / `test_score` = raw-label nDCG@10 (`_ndcg10_from_scores`).
- `gate_results` = `run_regression_cases(ranked_df, cases)` — gate cases
  pass/fail (status-aware: gate blocks, pending reports, retired skipped).
  All 4 cells run the gate for completeness; only D's gate is promotion-blocking.

`ranked_df` = `rank_by_model(split_df, model.predict(dmatrix))`.

## Deltas + recommendation

```text
B - A = parameter search gain
C - A = feature-only gain
D - B = candidate feature gain after tuning   (primary, per spec)
D - C = tuning gain with the candidate
```

Computed on both val and test. Recommendation (report only, never auto-promote):

```text
promote     iff  D-B(val)  > --promote-threshold (default 0.01)
            AND  D-B(test) > 0
            AND  D gate cases all pass
reject      iff  D-B(val) <= 0  OR  D regresses any gate case
quarantine  otherwise (inconclusive; e.g., D-B(val) small positive but
            D-B(test) <= 0, or test variance too high)
```

The report explicitly labels this as a RECOMMENDATION; promotion is always a
manual decision (project invariant).

## Output

`examples/fiqa/output/ablation/` (gitignored):

- `ablation_report.md`: cell table (A/B/C/D × {val, test, gate pass count,
  best_iteration, params source}) + deltas table (B-A, C-A, D-B, D-C on val +
  test) + recommendation + overfit caveats.
- `ablation_result.json`: machine-readable — per-cell scores + gate results +
  deltas + recommendation + candidate recipe + HPO budget/seed.

## Anti-leak

- HPO (cells B/D) is case-blind + test-blind by the HPO adapter contract.
- The ablation framework passes only train+valid snapshots to HPO; test is used
  ONLY for post-hoc eval, never optimization.
- Regression cases are used ONLY for gate evaluation (never training) — same as
  `eval_reranker.py`.
- The candidate impl is validated against `ALLOWED_INPUTS` before any training;
  a candidate reading `label`/`split`/`query_id`/`doc_id` is rejected.

## Trade-offs

- **Single candidate per run**: spec §15.2 mentions "candidate feature group";
  deferred. V0 = one candidate per `run_ablation.py` invocation. Child #4
  orchestration can loop over multiple candidates.
- **All 4 cells use early stopping + iteration_range predict**: isolates the
  (feature_set, params) effect; B-A is pure param gain, not confounded by
  early-stopping-vs-not.
- **Fixed params for A/C = shipped values**: represents "current production"
  params. A future task could make these configurable, but V0 uses the shipped
  defaults.
- **No slice evaluation beyond gate cases**: spec mentions slices; V0 uses gate
  cases as the slice proxy. SliceEvaluator is a future component.
- **Recommendation threshold default 0.01**: on 40-query val, nDCG@10 has
  ~0.01-0.02 noise; 0.01 is a conservative floor. Configurable via
  `--promote-threshold`.

## Rollback

Purely additive (new `scripts/run_ablation.py`). No existing file's behavior
changes. Rollback = revert the commit.
