# Implement: A/B/C/D Ablation Framework

## Ordered checklist

Single checkpoint (additive — new `scripts/run_ablation.py` only).

1. Create `scripts/run_ablation.py`:
   - CLI: `--dataset`, `--candidate-recipe <yaml>`, `--candidate-impl <pyfile:func>`, `--output-dir`, `--n-trials` (HPO budget for B/D, default 20), `--seed` (default 42), `--promote-threshold` (default 0.01), `--regression-cases <yaml>` (optional, default `examples/fiqa/regression_cases.yaml`).
   - Candidate loader: parse recipe YAML, load impl fn via `importlib`, validate `inputs` ⊆ `ALLOWED_INPUTS` + `online_safe` + required fields (reuse `features.registry` constants or a small inline validator).
   - `extract_all_plus(row)` wrapper: `{**extract_all(row), candidate_name: candidate_fn(row)}`.
   - Snapshot builders: baseline (12 cols via `REGISTRY.extract`) + candidate (13 cols via `extract_all_plus`), each with `raw_labels` + `groups`.
   - `train_cell(params, train_snap, valid_snap)` → model + best_iteration: `xgb.train(params, dtrain, 200, evals=[dvalid], early_stopping_rounds=20, nthread=1, seed=42)`, return model.
   - `eval_cell(model, best_iteration, snap, df)` → `{val_score, test_score, gate_results}`: predict with `iteration_range=(0, best_iteration+1)`, `_ndcg10_from_scores` on raw labels, `run_regression_cases(rank_by_model(df, scores), cases)`.
   - 4-cell runner: A (fixed params, baseline), B (HPO + retrain, baseline), C (fixed params, candidate), D (HPO + retrain, candidate). Same `n_trials` + `seed` for B and D.
   - Deltas: B-A, C-A, D-B, D-C on val + test.
   - Recommendation: promote/reject/quarantine per design.
   - Write `output/ablation/{ablation_report.md, ablation_result.json}`.
2. Update `.trellis/spec/backend/` with `ablation-contracts.md` (candidate contract, 4-cell training procedure, deltas, recommendation rule, anti-leak). Update `index.md`.
3. Update `docs/REFERENCE.md` + `docs/REFERENCE.zh-CN.md`: "Ablation" subsection pointing to `run_ablation.py`.
4. Update README checklist (both): "Automatic feature discovery, ablation, and promotion" — split into ablation (done) vs discovery/promotion (not yet). OR add a new done line for ablation.
5. Update `CODEBUDDY.md` layout: add `run_ablation.py`.

## Validation commands

```bash
# syntax
python3 -m py_compile skills/heuriboost-rag/scripts/run_ablation.py

# smoke: a trivial candidate (dense_score / term_overlap ratio)
# write candidate_recipe.yaml + candidate_impl.py to /tmp
python3 skills/heuriboost-rag/scripts/run_ablation.py examples/fiqa/query_doc_examples.csv \
  --candidate-recipe /tmp/candidate_recipe.yaml \
  --candidate-impl /tmp/candidate_impl.py:candidate \
  --output-dir examples/fiqa/output --n-trials 5 --seed 42 \
  --regression-cases examples/fiqa/regression_cases.yaml

# A1: 4 cells, all scored
python3 -c "import json; r=json.load(open('examples/fiqa/output/ablation/ablation_result.json')); assert set(r['cells'].keys())=={'A','B','C','D'}; print('cells:', {k: round(v['val_score'],4) for k,v in r['cells'].items()})"

# A2: deltas present
python3 -c "import json; r=json.load(open('examples/fiqa/output/ablation/ablation_result.json')); assert all(k in r['deltas'] for k in ['B-A','C-A','D-B','D-C']); print('D-B val:', round(r['deltas']['D-B']['val'],4), 'test:', round(r['deltas']['D-B']['test'],4))"

# A3: recommendation is one of promote/reject/quarantine
python3 -c "import json; r=json.load(open('examples/fiqa/output/ablation/ablation_result.json')); assert r['recommendation'] in ('promote','reject','quarantine'); print('recommendation:', r['recommendation'])"

# A4: anti-leak — no regression_cases/ledger refs in training path; candidate recipe inputs validated
grep -rE "regression_cases|ledger" skills/heuriboost-rag/scripts/run_ablation.py | grep -v "regression-cases" || echo "A4: no stray refs"
# candidate with inputs=[label] must be rejected at load
# (temporarily edit candidate_recipe.yaml inputs to [label], confirm SystemExit, revert)

# A5: D gate status present
python3 -c "import json; r=json.load(open('examples/fiqa/output/ablation/ablation_result.json')); assert 'gate' in r['cells']['D']; print('D gate:', r['cells']['D']['gate'])"

# A6: same-seed determinism (re-run, compare ablation_result.json)
# (capture, re-run, diff)
```

## Risky files / rollback points

- New `scripts/run_ablation.py` — additive, low risk.
- No changes to `common.py`, `train_reranker.py`, `eval_reranker.py`, `hpo/`,
  `features/` — existing behavior untouched.

## Pre-`task.py start` checks

- [ ] Candidate loader validates `inputs` ⊆ `ALLOWED_INPUTS` (rejects `label` etc.).
- [ ] 4 cells run, each with val + test + gate.
- [ ] Deltas B-A, C-A, D-B, D-C computed on val + test.
- [ ] Recommendation is promote/reject/quarantine (report only).
- [ ] D gate pass/fail reported.
- [ ] Anti-leak: HPO sees only train+valid; test is post-hoc only; cases never trained.
- [ ] Same-seed re-run produces identical `ablation_result.json`.
- [ ] py_compile passes; REFERENCE/README/CODEBUDDY/spec updated.
