# Implement: Feature Recipe Registry DSL

## Ordered checklist

Executed in two commits — checkpoint 1 is additive (common.py untouched, demo
still runs on the old path), checkpoint 2 wires the registry into the main line.

### Checkpoint 1 — additive (commit before touching common.py)

1. Create `scripts/features/__init__.py`: eager-load YAML, register `extract_all` impl, validate, re-export `REGISTRY`, `extract_features`, `FEATURE_NAMES`.
2. Create `scripts/features/primitives.py`: move `tokenize, numbers, entities,
   rank_inverse, numeric_value` verbatim from `common.py`.
3. Create `scripts/features/registry.py`: `Recipe` dataclass, `FeatureRegistry`
   class with `register_impl()`, `load_yaml()`, `validate()`, `names()`,
   `extract()`, `feature_set_name`, `feature_set_version`, `feature_versions()`.
   Define `ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score,
   sparse_rank, sparse_score}`, `ACTIVE_TASK_PROFILE = "qd_reranker"`. Hard-fail
   on all validation failures; `expected_slices` may be empty.
4. Create `scripts/features/recipes.py`: one function `extract_all(row) ->
   dict[str, float]`, body lifted verbatim from current `extract_features`
   per-row block (common.py:207-256).
5. Extend `templates/feature_recipes.yaml`: add `version, task_profiles, inputs,
   cost_tier, expected_slices, owner, impl` per feature via a `defaults` YAML
   anchor. Keep top-level `feature_set.{name,version}`. All 12 features point to
   `impl: extract_all`.

**Checkpoint 1 verify** (demo still on old path, registry loads independently):
- `py_compile features/*.py` passes.
- `python3 -c "import sys; sys.path.insert(0,'skills/heuriboost-rag/scripts'); from features import REGISTRY; print(len(REGISTRY.names()), REGISTRY.feature_set_name, REGISTRY.feature_set_version)"` → `12 heuriboost_rag_v0 1`.
- `train_reranker.py` / `eval_reranker.py` still run unchanged (common.py untouched).
- Commit checkpoint 1.

### Checkpoint 2 — wire main line (commit after A1 + metrics verified)

6. Refactor `common.py`:
   - Remove primitive helper bodies (now in `features.primitives`); re-export
     them for `rank_by_baseline`.
   - Replace `FEATURE_NAMES` with `REGISTRY.names()` re-export (eager).
   - Replace `extract_features` body with `REGISTRY.extract(df)` shim.
   - `from features import REGISTRY, extract_features, FEATURE_NAMES` at top →
     triggers eager registry load + validation on any `import common`.
7. Extend `train_reranker.py`: add `feature_set_name`, `feature_set_version`,
   and `feature_versions` (dict) to `reranker_metadata.json` (metadata dict at
   train_reranker.py:232-242).
8. Update `.trellis/spec/backend/fiqa-demo-contracts.md`: rewrite the
   "FEATURE_NAMES must equal feature_recipes.yaml" contract to note it is now
   enforced at registry load time.
9. Update `docs/REFERENCE.md` + `docs/REFERENCE.zh-CN.md`: add a short
   "Feature registry" subsection pointing to `feature_recipes.yaml` and the
   `features/` subpackage; note required fields and validation.
10. Update README checklist (both): move "Feature registry / recipe DSL" from
    not-yet to done.
11. Update CODEBUDDY.md "Current V0 Layout" to mention `features/` subpackage.

**Checkpoint 2 verify** (A1 strong + metrics + metadata):
- A1 strong: snapshot old `extract_features` output BEFORE step 6, `assert new.equals(old)` AFTER.
- train + eval: metrics unchanged (nDCG@10 val ≈ 0.853, 2/2 gates green).
- metadata carries `feature_set_name`, `feature_set_version`, `feature_versions`.
- Negative test: temporarily set `inputs: [label]` on one feature, `import common` → SystemExit, revert.
- Commit checkpoint 2.

## Validation commands

```bash
# syntax check all scripts incl. new subpackage
python3 -m py_compile skills/heuriboost-rag/scripts/*.py skills/heuriboost-rag/scripts/features/*.py

# contract: registry loads cleanly (validation passes for the shipped 12)
python3 -c "import sys; sys.path.insert(0, 'skills/heuriboost-rag/scripts'); from features import REGISTRY; print(len(REGISTRY.names()), REGISTRY.feature_set_name, REGISTRY.feature_set_version)"

# A1 strong check: snapshot OLD extract_features output BEFORE refactor,
# then AFTER refactor assert bit-for-bit identity on the demo CSV.
# BEFORE (run once, before editing common.py):
python3 -c "import sys; sys.path.insert(0, 'skills/heuriboost-rag/scripts'); import pandas as pd; from common import extract_features, load_dataset; df=load_dataset('examples/fiqa/query_doc_examples.csv'); extract_features(df).to_csv('/tmp/heuriboost_old_features.csv', index=False)"
# AFTER (after refactor):
python3 -c "import sys; sys.path.insert(0, 'skills/heuriboost-rag/scripts'); import pandas as pd; from common import extract_features, load_dataset; df=load_dataset('examples/fiqa/query_doc_examples.csv'); new=extract_features(df); old=pd.read_csv('/tmp/heuriboost_old_features.csv'); assert new.equals(old), new.compare(old); print('A1 bit-for-bit OK')"

# contract: removing a YAML entry or an impl fails loud
#   (temporarily edit, confirm SystemExit, revert)

# behavior unchanged: train + eval produce identical metrics
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output
python3 skills/heuriboost-rag/scripts/eval_reranker.py  examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --regression-cases examples/fiqa/regression_cases.yaml

# metadata now carries feature_set + per-feature versions
python3 -c "import json; m=json.load(open('examples/fiqa/output/models/reranker_metadata.json')); assert 'feature_set_name' in m and 'feature_set_version' in m and 'feature_versions' in m; print(m['feature_set_name'], m['feature_set_version'], len(m['feature_versions']))"
```

## Risky files / rollback points

- `common.py` — heaviest edit (primitive bodies removed, FEATURE_NAMES /
  extract_features become shims). Rollback: revert file.
- `templates/feature_recipes.yaml` — extended with required fields. Rollback:
  revert.
- `train_reranker.py` — one metadata-dict extension (low risk).
- New `features/` subpackage — additive, low risk; if import fails, it fails
  loud at first use.

## Pre-`task.py start` checks

- [ ] All 12 features present in YAML with all spec-required fields.
- [ ] `REGISTRY.names()` returns the same 12 names in the same canonical order.
- [ ] `extract_features(df)` output is bit-for-bit identical (same values).
- [ ] Train + eval still green (2/2 gates, nDCG@10 val ≈ 0.853).
- [ ] `reranker_metadata.json` carries `feature_set_name` + `feature_set_version`.
- [ ] Negative test: a feature with `inputs: [label]` fails registration.
- [ ] py_compile passes for all scripts + subpackage.
- [ ] READMEs + REFERENCE + CODEBUDDY.md updated; checklist item moved to done.
