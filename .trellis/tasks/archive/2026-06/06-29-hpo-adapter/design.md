# Design: HPO Engine Adapter

## Architecture

New subpackage `skills/heuriboost-rag/scripts/hpo/`:

```text
scripts/hpo/
  __init__.py            # re-exports HPOEngine, TrialResult, Snapshot
  engine.py              # HPOEngine (backend-agnostic) + TrialResult + Snapshot dataclasses
  optuna_backend.py      # OptunaBackend: TPESampler + per-trial xgb.train + trial history
scripts/run_hpo.py       # CLI: dataset -> features -> snapshots -> optimize -> report
```

`optuna` is added to `requirements-build.txt` (experiment dependency; the
shipped skill's runtime train/eval does NOT need it).

## Data flow

```text
run_hpo.py
  | load_dataset -> split_frame(train/validation)
  | REGISTRY.extract(df)  -> feature matrices (reuses the registry, not a copy)
  | build Snapshot(X, y, groups) for train + valid
  v
HPOEngine(backend=OptunaBackend(seed=..., search_space=...))
  .optimize(task_profile, feature_set, train_snap, valid_snap, budget) -> TrialResult
       |
       v  (inside OptunaBackend)
  Optuna study (TPESampler, n_trials, timeout)
    per trial:
      suggest params -> xgb.train(dtrain, dvalid, num_boost_round=200,
                                  early_stopping_rounds=20)
      objective = validation nDCG@10 (maximize)
      record {params, score, state, failure_reason?}
       |
       v
  TrialResult { best_params, best_score, trials[], feature_set_*,
                objective, eval_metric, early_stopping_rounds, seed, budget }
  -> writes hpo_report.md + best_params.json + trials.json
```

## Types (engine.py)

```python
@dataclass(frozen=True)
class Snapshot:
    X: "pd.DataFrame"     # feature matrix (REGISTRY.extract output)
    y: "pd.Series"        # relevance labels (mapped to 0..4)
    groups: list[int]     # query group sizes

@dataclass(frozen=True)
class TrialResult:
    best_params: dict
    best_score: float                  # validation nDCG@10 (search objective)
    best_iteration: int                # xgboost early-stopping best round
    test_score: float | None           # post-hoc honest test nDCG@10 (NOT a search objective)
    trials: list[dict]                 # {params, score, state, failure_reason?}
    feature_set_name: str
    feature_set_version: int
    objective: str                     # "rank:ndcg"
    eval_metric: str                   # "ndcg@10"
    early_stopping_rounds: int
    num_boost_round: int
    seed: int
    n_trials: int                      # requested budget
    timeout_sec: int | None

class HPOEngine:
    def __init__(self, backend): self._backend = backend
    def optimize(self, task_profile, feature_set, train_snapshot,
                 valid_snapshot, budget, seed) -> TrialResult:
        return self._backend.run(train_snapshot, valid_snapshot, budget, seed,
                                  feature_set, task_profile)
```

The `HPOEngine` is a thin façade; `OptunaBackend` does the work. A future
backend (e.g. Ray Tune) implements the same `run(...)` contract.

## OptunaBackend (optuna_backend.py)

- `TPESampler(seed=seed)` for determinism.
- Search space (V0, NARROWED around the shipped baseline to avoid overfitting
  on 150 train / 40 validation queries — confirmed during grilling):
  - `max_depth`: int 3-6 (baseline 3)
  - `min_child_weight`: float 0.05-3.0 (baseline 0.1)
  - `eta`: float 0.05-0.2 (log; baseline 0.08)
  - `subsample`: float 0.6-1.0 (baseline 0.9)
  - `colsample_bytree`: float 0.6-1.0 (baseline 0.9)
  - `gamma`: float 0.0-3.0
  - `reg_lambda`: float 0.1-10.0 (log)
- Fixed per trial: `objective=rank:ndcg`, `eval_metric=ndcg@10`,
  `seed=42`, `nthread=1` (full determinism — confirmed during grilling),
  `num_boost_round=200`, `early_stopping_rounds=20`.
- Per trial: build `xgb.DMatrix(X, label=y, feature_names=...)`, `set_group`,
  `xgb.train(params, dtrain, evals=[(dvalid,"validation")],
  early_stopping_rounds=20, verbose_eval=False)`, read best validation
  nDCG@10 + `best_iteration`. Catch per-trial exceptions → record
  `state="failed"` + `failure_reason`, continue study.
- `study.optimize(objective, n_trials=budget.n_trials, timeout=budget.timeout_sec,
  catch=(Exception,))` — KeyboardInterrupt returns best-so-far.
- Return `TrialResult` with `best_iteration` (so any consumer reproduces the
  HPO-best model by training `num_boost_round=best_iteration` with `best_params`,
  no early stopping needed) and trials in trial-number order.

## run_hpo.py CLI

```bash
python3 scripts/run_hpo.py examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output --n-trials 20 --seed 42 [--timeout-sec 120]
```

Steps: `load_dataset` → `validate_dataset_frame` → `split_frame(train)` +
`split_frame(validation)` + `split_frame(test)` → `REGISTRY.extract` +
`relevance_labels` + `group_sizes` per split → train/valid `Snapshot(...)` for
the HPO search → `HPOEngine(OptunaBackend()).optimize(...)` →
**post-hoc honest test eval**: retrain with `best_params` +
`num_boost_round=best_iteration` on train (no early stopping), evaluate on the
test snapshot, record `test_score` (clearly labeled "honest test estimate, not
a search objective") → write `output/hpo/{hpo_report.md, best_params.json,
trials.json}`.

The HPO SEARCH sees only train+valid snapshots (test-blind, anti-leak). The
post-hoc test eval is a single forward evaluation, not optimization — same
behavior as `eval_reranker.py` evaluating on test.

`hpo_report.md` shows: best params + val score + **test score (honest)** +
overfit gap (val−test) + budget consumed + trial table (rank, params, score,
state) + `feature_set_name/version` for attribution. `best_params.json`
includes `best_iteration` so any consumer reproduces the HPO-best model
exactly by training that many rounds with the best params.

## Anti-leak (by construction)

`HPOEngine.optimize` and `OptunaBackend.run` signatures take ONLY snapshots +
feature_set + task_profile + budget + seed. There is NO `cases` or `ledger`
parameter. `run_hpo.py` loads the dataset and splits, never imports
`regression_cases.yaml` or `ledger.json`. Anti-leak is enforced at the API
boundary, not by convention.

## Determinism

Two `run_hpo.py` invocations with the same `--seed` and same dataset produce
identical `trials.json` (same params + scores in the same order). Optuna's
TPESampler is deterministic given a seed; xgboost is deterministic given
`seed=42` per trial. Verified by A4.

## Trade-offs

- **Optuna vs Ray Tune / Hyperopt**: Optuna — mature, lightweight, deterministic
  TPESampler, simple in-process. Ray Tune is heavier (Ray runtime); Hyperopt is
  less maintained. Optuna is the clear V0 choice.
- **In-process vs SQLite study**: in-process for V0 (no resumability). SQLite
  adds a storage dependency + resumability complexity that V0 doesn't need; the
  ablation framework (child #2) runs 4 fresh studies per ablation, none long
  enough to need resumability yet.
- **Pre-computed snapshots vs raw CSV inside adapter**: snapshots passed in.
  Keeps the adapter case-blind (no CSV access) and avoids per-trial feature
  re-extraction.
- **Early stopping inside HPO**: yes (`early_stopping_rounds=20`). Without it,
  each trial runs the full 200 rounds, wasting budget. The fixed-params
  `train_reranker.py` currently has no early stopping — that's a separate
  concern; HPO trials use it.
- **Single-objective (global nDCG@10) vs multi-objective**: single. Slices +
  cases are the PromotionGate's job (child #2), keeping HPO case-blind.

## Rollback

Purely additive (new `scripts/hpo/` + `scripts/run_hpo.py` + a
`requirements-build.txt` line). No existing file's behavior changes. Rollback =
revert the commit + `pip uninstall optuna`.
