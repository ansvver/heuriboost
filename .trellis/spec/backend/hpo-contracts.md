# HPO Engine Contracts

> Executable contracts for the `HPOEngine` adapter (`scripts/hpo/`).
> Captured 2026-06-29 after the hpo-adapter task (child of feature-discovery).

---

## Scenario: HPO search + post-hoc test evaluation

### 1. Scope / Trigger

Any change to `scripts/hpo/`, `scripts/run_hpo.py`, or the HPO search/training
procedure. Cross-layer contract (snapshots ↔ search ↔ model metadata), so
code-spec depth is mandatory.

### 2. Signatures

```python
# scripts/hpo/engine.py
class HPOEngine:
    def __init__(self, backend=None)  # default OptunaBackend
    def optimize(self, task_profile, feature_set_name, feature_set_version,
                 train_snapshot, valid_snapshot, budget, seed) -> TrialResult

@dataclass(frozen=True)
class Snapshot:
    X: pd.DataFrame          # REGISTRY.extract output
    y: pd.Series             # MAPPED labels 0..4 (for xgboost training)
    raw_labels: list[int]    # ORIGINAL labels -1..3 (for nDCG scoring, baseline-consistent)
    groups: list[int]        # query group sizes

@dataclass(frozen=True)
class TrialResult:
    best_params, best_score, best_iteration, trials,
    feature_set_name, feature_set_version, objective, eval_metric,
    early_stopping_rounds, num_boost_round, seed, n_trials, timeout_sec,
    test_score  # post-hoc, NOT a search objective
```

### 3. Contracts

**Search inputs**: `train_snapshot` + `valid_snapshot` ONLY. The HPO SEARCH is
case-blind (no regression cases) AND test-blind (no test snapshot) by signature.

**Reproducibility contract**: `best_params.json` records `best_iteration`
(xgboost 0-indexed). To reproduce the HPO-best model, retrain with
`num_boost_round = best_iteration + 1` (trains rounds 0..best_iteration). With
`nthread=1` + `seed=42`, the reproduced model's validation nDCG@10 exactly
equals `best_score`.

**nDCG consistency**: HPO objective + post-hoc test eval use
`_ndcg10_from_scores(scores, raw_labels, groups)` — the SAME `ndcg_at_k`
formula as `evaluate_ranked_frame` (raw labels, `max(label, 0)` clamps -1 to 0).
NOT xgboost's internal `model.best_score` (which uses mapped 0..4 labels, a
different gain scale) and NOT `relevance_labels` (mapped). This makes HPO
scores directly comparable to the shipped baseline (0.853).

**Fixed per trial**: `objective=rank:ndcg`, `eval_metric=ndcg@10`, `seed=42`,
`nthread=1` (full determinism), `num_boost_round=200`, `early_stopping_rounds=20`.

**Search space (narrowed around baseline)**: max_depth 3-6, min_child_weight
0.05-3.0 (log), eta 0.05-0.2 (log), subsample 0.6-1.0, colsample_bytree 0.6-1.0,
gamma 0.0-3.0, reg_lambda 0.1-10.0 (log).

### 4. Validation & Error Matrix

| Condition | Behavior |
|---|---|
| `optuna` not installed | `SystemExit: optuna is required...` |
| all trials fail | `SystemExit: HPO failed: no complete trials...` |
| per-trial exception | recorded as `state="failed"` + `failure_reason`; study continues |
| `--timeout-sec` exceeded | returns best-so-far trial |
| `KeyboardInterrupt` | returns best-so-far trial |
| no test split | post-hoc test eval skipped; `test_score=None` |

### 5. Good / Base / Bad Cases

- **Good**: `run_hpo.py ... --n-trials 20 --seed 42` → `trials.json` (20
  entries), `best_params.json` (params + best_iteration + scores),
  `hpo_report.md` (val + test + gap + trial table).
- **Base**: two `--seed 42` runs produce byte-identical `trials.json` (nthread=1
  + TPESampler determinism).
- **Bad**: computing HPO objective with `model.best_score` (xgboost internal,
  mapped 0..4 scale) — incomparable to baseline 0.853 (raw scale). Fixed by
  `_ndcg10_from_scores` with `raw_labels`.

### 6. Tests Required

- A1 py_compile (`scripts/hpo/*.py` + `run_hpo.py`).
- A2 5-trial smoke run writes all three outputs.
- A3 `len(trials)==5`; `best_score == max(complete trial scores)`.
- A4 determinism: two same-seed runs → `diff -q trials.json` identical.
- A5 `best_params.json` carries `feature_set_name/version`, `objective`, `eval_metric`, `best_iteration`.
- A6 anti-leak: `grep -rE "regression_cases|ledger|must_include" scripts/hpo/ scripts/run_hpo.py` → empty.
- A7 `--timeout-sec 5 --n-trials 1000` returns within ~10s with ≥1 complete trial.
- A8 reproducibility: retrain `best_params` + `num_boost_round=best_iteration+1` → validation nDCG@10 == `best_score` (raw-label scale).

### 7. Wrong vs Correct

#### Wrong — xgboost internal best_score

```python
score = float(model.best_score)   # mapped 0..4 scale, gain up to 2^4-1=15
```

Incomparable to the shipped baseline (0.853, raw scale, gain up to 2^3-1=7).
HPO would appear to "beat" baseline on a different scale.

#### Correct — raw-label ndcg@10 via _ndcg10_from_scores

```python
preds = model.predict(dvalid, iteration_range=(0, best_iter + 1))
score = _ndcg10_from_scores(preds, valid_snapshot.raw_labels, valid_snapshot.groups)
```

Same `ndcg_at_k` + raw labels as `evaluate_ranked_frame`. HPO scores directly
comparable to baseline. `iteration_range=(0, best_iter+1)` ensures the score
reflects the best-iteration model (the one a consumer reproduces), not the
full early-stopping-window model.

---

## Design Decision: nthread=1 for determinism

**Context**: xgboost multi-thread histogram building is non-deterministic;
same `seed=42` can produce marginally different models/scores across runs.

**Decision**: HPO trials fix `nthread=1`. Optuna `TPESampler(seed=...)` is
deterministic. Together, same-seed runs produce byte-identical `trials.json`
(A4). V0 demo is small enough that single-thread trials are still sub-second.

**Tradeoff**: slower than multi-thread, but determinism is a spec §12.5 hard
requirement ("Use deterministic seeds where supported") and the D-B attribution
in child #2's A/B/C/D ablation depends on it.

## Design Decision: test-blind search + post-hoc test eval

**Context**: A8 needs an honest test estimate, but HPO must not optimize on
test (anti-leak).

**Decision**: the HPO SEARCH sees only train+valid snapshots (signatures have
no test parameter). `run_hpo.py` does a post-hoc test evaluation of the best
model AFTER the search — a single forward pass, clearly labeled "honest test
estimate, not a search objective", same behavior as `eval_reranker.py`. This
keeps HPO test-blind while making A8 verifiable in-child.

## Finding: HPO overfits on 40-query validation

On the FiQA demo (40 validation queries), even 5-trial HPO overfits: val
nDCG@10 rises to ~0.90 but test falls to ~0.82 (below the 0.83 fixed-params
baseline). This is a real demo-size finding the report surfaces via the
val−test gap, NOT a wiring defect. It motivates scout-vs-finalist staging and
CV in future children. The adapter itself is correct (A8 reproducibility
passes).
