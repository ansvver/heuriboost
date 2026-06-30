# Implement: HPO Engine Adapter

## Ordered checklist

Single checkpoint (additive — no existing file changes except
`requirements-build.txt`).

1. Add `optuna` to `skills/heuriboost-rag/requirements-build.txt`.
2. Create `scripts/hpo/__init__.py`: re-export `HPOEngine`, `TrialResult`, `Snapshot`.
3. Create `scripts/hpo/engine.py`: `Snapshot`, `TrialResult`, `Budget` dataclasses + `HPOEngine` façade.
4. Create `scripts/hpo/optuna_backend.py`: `OptunaBackend` with `TPESampler`, the V0 default search space, per-trial `xgb.train` with early stopping, trial-history collection, `KeyboardInterrupt` → best-so-far.
5. Create `scripts/run_hpo.py` CLI: `--dataset`, `--output-dir`, `--n-trials`, `--seed`, `--timeout-sec`. Loads dataset, validates, splits, extracts features via `REGISTRY`, builds snapshots, runs `HPOEngine.optimize`, writes `output/hpo/{hpo_report.md, best_params.json, trials.json}`.
6. Update `.trellis/spec/backend/` with a new `hpo-contracts.md` (signatures, anti-leak by construction, determinism, search space, deferred resumability). Update `index.md`.
7. Update `docs/REFERENCE.md` + `docs/REFERENCE.zh-CN.md`: short "HPO" subsection pointing to `run_hpo.py` and noting it's a build/experiment dependency.
8. Update README checklist (both): move "HPO adapter to an external backend" from not-yet to done.
9. Update `CODEBUDDY.md` "Current V0 Layout" to mention `scripts/hpo/` + `run_hpo.py`.

## Validation commands

```bash
# install optuna
python -m pip install -r skills/heuriboost-rag/requirements-build.txt

# syntax
python3 -m py_compile skills/heuriboost-rag/scripts/hpo/*.py skills/heuriboost-rag/scripts/run_hpo.py

# A2: 5-trial smoke run
python3 skills/heuriboost-rag/scripts/run_hpo.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --n-trials 5 --seed 42

# A3: 5 trials, best_params present
python3 -c "import json; t=json.load(open('examples/fiqa/output/hpo/trials.json')); b=json.load(open('examples/fiqa/output/hpo/best_params.json')); assert len(t['trials'])==5; assert b['best_score']==max(x['score'] for x in t['trials'] if x['state']=='complete']); print('A3 OK', b['best_score'])"

# A4: determinism — re-run, compare trials.json
python3 skills/heuriboost-rag/scripts/run_hpo.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --n-trials 5 --seed 42
# diff the two trials.json runs (capture first, compare) — must be identical

# A5: feature_set attribution
python3 -c "import json; b=json.load(open('examples/fiqa/output/hpo/best_params.json')); assert b['feature_set_name']=='heuriboost_rag_v0' and b['feature_set_version']==1 and b['objective']=='rank:ndcg' and b['eval_metric']=='ndcg@10'; print('A5 OK')"

# A6: anti-leak — HPO must not reference cases/ledger
grep -rE "regression_cases|ledger|must_include" skills/heuriboost-rag/scripts/hpo/ skills/heuriboost-rag/scripts/run_hpo.py || echo "A6 OK (no references)"

# A7: cancellation — timeout returns best-so-far
python3 skills/heuriboost-rag/scripts/run_hpo.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --n-trials 1000 --seed 42 --timeout-sec 5
# completes within ~10s with at least 1 complete trial

# A8: reproducibility — retrain best_params + num_boost_round=best_iteration+1,
# validation nDCG@10 (raw-label scale) must == best_score exactly.
python3 -c "import json; b=json.load(open('examples/fiqa/output/hpo/best_params.json')); assert 'best_iteration' in b; print('best_iteration:', b['best_iteration'], 'best_score:', b['best_score'], 'test_score:', b.get('test_score'))"
# hpo_report.md must show val + test + val−test gap (overfit is a real finding, NOT a defect)
```

## Risky files / rollback points

- New `scripts/hpo/` — additive, low risk.
- `scripts/run_hpo.py` — additive, low risk.
- `requirements-build.txt` — one line added.
- No changes to `common.py`, `train_reranker.py`, `eval_reranker.py` — existing
  behavior untouched.

## Pre-`task.py start` checks

- [ ] `optuna` installs cleanly; `TPESampler(seed=...)` is deterministic.
- [ ] 5-trial smoke run completes and writes all three output files.
- [ ] `trials.json` has 5 entries; `best_params.json` best_score = max complete trial score.
- [ ] Same-seed re-run produces identical `trials.json`.
- [ ] `feature_set_name/version`, `objective`, `eval_metric` in `best_params.json`.
- [ ] No `regression_cases`/`ledger`/`must_include` references in `scripts/hpo/` or `run_hpo.py`.
- [ ] `--timeout-sec` returns best-so-far within budget.
- [ ] A8 reproducibility: retrain `best_params` + `num_boost_round=best_iteration+1` → val nDCG@10 == `best_score` (raw-label scale).
- [ ] py_compile passes; REFERENCE/README/CODEBUDDY/spec updated; checklist moved to done.
