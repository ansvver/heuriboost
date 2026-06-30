# Implement: Reckless Mode

## Ordered checklist

1. Update `skills/heuriboost-rag/scripts/train_reranker.py`
   - add `--reckless`
   - preserve current `--case-sets` behavior
   - include a metadata field for reckless runs
2. Update `skills/heuriboost-rag/scripts/eval_reranker.py`
   - add `--reckless`
   - evaluate train / validation / test as today
   - in reckless mode, treat every regression case as blocking
   - compare test `nDCG@10` and `MRR@10` against the ledger anchor
   - make the exit / acceptance decision reflect reckless semantics
3. Update `skills/heuriboost-rag/scripts/regression_ledger.py`
   - record reckless-mode runs in round snapshots
   - surface anchor deltas for test `nDCG@10` and `MRR@10`
4. Update docs
   - `docs/REFERENCE.md`
   - `docs/REFERENCE.zh-CN.md`
   - `README.md`
   - `README.zh-CN.md`
   - `CODEBUDDY.md`
5. Run validation
   - syntax
   - one normal eval smoke run
   - one reckless-mode smoke run
   - confirm default behavior is unchanged

## Validation commands

```bash
python3 -m py_compile skills/heuriboost-rag/scripts/*.py
python3 -m py_compile skills/heuriboost-rag/scripts/hpo/*.py

python3 skills/heuriboost-rag/scripts/train_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --case-sets examples/fiqa/case_sets \
  --regression-cases examples/fiqa/regression_cases.yaml

python3 skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split validation \
  --regression-cases examples/fiqa/regression_cases.yaml

python3 skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split test \
  --regression-cases examples/fiqa/regression_cases.yaml \
  --reckless
```

## Risks / rollback points

- `train_reranker.py` CLI changes could affect existing callers if parsing is
  wrong.
- `eval_reranker.py` exit behavior changes are the main risk; keep them behind
  the flag.
- Ledger/report additions should be additive only.

## Pre-start checks

- [ ] `--reckless` is implemented only on the existing CLI surface.
- [ ] Reckless mode blocks on any regression-case failure.
- [ ] Reckless mode requires test `nDCG@10` and `MRR@10` to exceed anchor.
- [ ] Default non-reckless behavior is unchanged.
- [ ] Docs and checklist references are updated.
