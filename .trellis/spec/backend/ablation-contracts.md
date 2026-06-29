# Ablation Framework Contracts

> Executable contracts for the A/B/C/D ablation framework (`scripts/run_ablation.py`).
> Captured 2026-06-29 after the ablation-framework task (child #2 of
> feature-discovery). Depends on the HPO adapter (`scripts/hpo/`) for cells B/D.

---

## Scenario: 4-cell feature ablation

### 1. Scope / Trigger

Any change to `scripts/run_ablation.py` or the candidate contract. Cross-layer
(candidate recipe ↔ impl ↔ HPO ↔ gate), so code-spec depth is mandatory.

### 2. Signatures

```python
# scripts/run_ablation.py
# CLI: --dataset --candidate-recipe <yaml> --candidate-impl <pyfile:func>
#      --output-dir --n-trials --seed --promote-threshold --regression-cases

def _load_candidate_recipe(path) -> dict       # validates inputs/online_safe/required fields
def _load_impl(spec: "pyfile:func") -> Callable[[row], float]
def _make_extract_plus_df(candidate_fn, name) -> Callable[[df], pd.DataFrame]
def _train_cell(params, train_snap, valid_snap) -> xgb.Booster  # early stopping 200/20
def _score(model, snap) -> float               # raw-label nDCG@10 (best-iteration model)
def _gate(model, snap, df, cases) -> dict      # reuses eval_reranker.run_regression_cases
def _eval_cell(model, valid_snap, test_snap, valid_df, cases) -> dict
```

### 3. Contracts

**Candidate** = (recipe YAML, impl fn `(row) -> float`). The framework wraps it
onto the shipped `extract_all` WITHOUT modifying the registry:

```python
extract_plus_df(df) = pd.DataFrame(
    [{**extract_all(row), candidate_name: candidate_fn(row)} for row in df],
    columns=baseline_names + [candidate_name]
)
```

**Candidate recipe required fields**: `name, version, description, task_profiles
(⊇ qd_reranker), inputs (⊆ ALLOWED_INPUTS), type, cost_tier, online_safe (true),
leakage_risk, owner, expected_slices`. A candidate with `inputs` outside
`ALLOWED_INPUTS` (e.g. `label`) or `online_safe: false` is rejected at load.

**4 cells** (all use the SAME training procedure: `xgb.train(params, 200,
early_stopping=20, nthread=1, seed=42)` → predict `iteration_range=(0,
best_iteration+1)`):

| Cell | Features | Params |
|---|---|---|
| A | baseline 12 | fixed shipped values (eta=0.08, max_depth=3, ...) |
| B | baseline 12 | HPO best (same `n_trials` + `seed` as D) |
| C | baseline + candidate 13 | fixed shipped values |
| D | baseline + candidate 13 | HPO best (same `n_trials` + `seed` as B) |

**Deltas** (on val + test): B-A (param gain), C-A (feature-only), D-B (candidate
gain after tuning — primary), D-C (tuning gain with candidate).

**Recommendation** (report only — promotion is ALWAYS manual):
- `promote` iff `D-B(val) > --promote-threshold` (default 0.01) AND `D-B(test) > 0` AND D gate cases all pass.
- `reject` iff `D-B(val) <= 0` OR D regresses any gate case.
- `quarantine` otherwise (inconclusive).

**nDCG scale**: all cells scored via `_ndcg10_from_scores(raw_labels)` — same
scale as baseline 0.853 + HPO. NOT xgboost internal, NOT mapped labels.

### 4. Validation & Error Matrix

| Condition | Behavior |
|---|---|
| candidate recipe missing field | `SystemExit: Candidate recipe missing required field: <f>` |
| candidate `inputs` ⊄ ALLOWED_INPUTS | `SystemExit: ... not in ALLOWED_INPUTS (leakage/identifier)` |
| candidate `online_safe: false` | `SystemExit: ... is online_safe=false; rejected` |
| `--candidate-impl` not `pyfile:func` | `SystemExit: --candidate-impl must be 'pyfile:func'` |
| impl file missing / no func | `SystemExit: Candidate impl ... has no function '<f>'` |
| no test split | `test_score=None` per cell; deltas `test=None`; promote blocked (D-B(test) check fails) |

### 5. Good / Base / Bad Cases

- **Good**: a candidate that lifts D-B(val) > threshold AND D-B(test) > 0 AND D
  gates pass → `promote` recommendation (human still decides).
- **Base**: a neutral candidate (D-B(val) ≈ 0) → `quarantine` or `reject`.
- **Bad**: computing D-B on val only (ignoring test) — HPO overfits 40-query
  val, would cherry-pick noise. Fixed by the dual val+test check.

### 6. Tests Required

- A1 4 cells, all with val_score (test_score if test split exists).
- A2 deltas B-A, C-A, D-B, D-C on val + test.
- A3 recommendation ∈ {promote, reject, quarantine}.
- A4 anti-leak: no stray `regression_cases`/`ledger` refs in training path;
  candidate `inputs=[label]` rejected at load.
- A5 D gate pass/total present.
- A6 determinism: same-seed re-run → byte-identical `ablation_result.json`.

### 7. Wrong vs Correct

#### Wrong — D-B on val only

```python
if deltas["D-B"]["val"] > threshold:
    recommendation = "promote"
```

HPO overfits the 40-query validation (val > test by ~0.08). A candidate that
cherry-picks val noise would falsely promote.

#### Correct — dual val + test + gate

```python
if db_val > threshold and db_test is not None and db_test > 0 and d_gate_ok:
    recommendation = "promote"
elif db_val <= 0 or not d_gate_ok:
    recommendation = "reject"
else:
    recommendation = "quarantine"
```

val gives the signal strength; test > 0 gives the generalization floor; gate
gives the no-regression floor. All three required for promote; promotion itself
stays manual.

---

## Design Decision: candidate wraps extract_all (probe)

**Context**: the shipped registry uses Option C (one shared `extract_all` for
all 12 features). Adding a candidate must NOT modify the shipped registry
(would pollute the baseline + break the registry contract).

**Decision**: the candidate is a probe — `extract_plus_df` wraps `extract_all`
and appends `candidate_fn(row)`. The shipped `extract_all` and
`feature_recipes.yaml` are untouched. Cells A/B use `extract_features` (12
cols); cells C/D use `extract_plus_df` (13 cols). Child #3 (LLM gen) will
output recipe YAML + impl code, consumed directly by this contract.

## Design Decision: unified training procedure across cells

**Context**: for B-A to be a pure param gain (not confounded by early-stopping
vs not), all 4 cells must use the SAME training procedure.

**Decision**: every cell trains with `xgb.train(params, 200, early_stopping=20,
nthread=1, seed=42)` and predicts with `iteration_range=(0, best_iteration+1)`.
A/C use fixed shipped param values; B/D use HPO best params (retrained with
early stopping — deterministic, reproduces HPO-best per the hpo-adapter A8
contract). B and D use the SAME `n_trials` + `seed` (spec §15.3 fairness).
