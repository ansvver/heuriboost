# Design: case_sets Mining & Settlement (V1, step 2)

## Plan Y continued: new mining script; train gains an optional input

- New script `skills/heuriboost-rag/scripts/mine_case_sets.py` owns mining.
- `train_reranker.py` gains an optional `--case-sets` input (merged into train).
- `eval_reranker.py` unchanged in mechanics; it just sees a retrained model and
  records a round with a `case_sets_used` flag via the ledger.

## Storage (resolves PRD open questions)

- `case_sets` artifacts are COMMITTED under `examples/fiqa/case_sets/`.
- One file per source pending case: `examples/fiqa/case_sets/<case_id>.csv`
  (clear traceability: file name = source case). A small manifest
  `examples/fiqa/case_sets/manifest.json` records mining params + counts per
  case for reproducibility.
- Same column schema as `examples/fiqa/query_doc_examples.csv` (train reads
  them uniformly). A `source_case_id` column is ADDED to preserve traceability
  inside the rows (train ignores it; ledger/inspection can use it).
- `.gitignore` must NOT ignore `examples/fiqa/case_sets/`. Scripts never
  auto-commit.

## Mining rule (a+b+c intersection, per grilling)

For each `pending` case:
1. **(c) failure_type match**: only FiQA queries whose case would share the
   case's `failure_type`. (Today all pending are `semantic_hard_negative`, so
   this is broad now; the filter exists for future multi-type cases.)
2. **(b) failure shape**: a candidate query must have, in the main CSV, both:
   - a hard-negative row (label == -1) with `dense_rank` <= SHAPE_RANK
     (default 3), AND
   - a positive row (label == 3) with `dense_rank` >= SHAPE_POS_GAP
     (default 5).
   This means the query exhibits the same "dense fooled, positive buried"
   shape as the case.
3. **(a) semantic similarity**: among the queries passing (c)+(b), keep the K
   most similar to the case's `query_text` by cosine of all-MiniLM-L6-v2
   embeddings (default K=10).
4. **B+C isolation**: drop any candidate query whose `query_id` is any case's
   query_id, and drop any candidate ROW whose `doc_id` is any case's
   must_include/must_not_include doc_id. (Query-level drop for B; row-level drop
   for C, so a query can still contribute other docs.)
5. Emit the surviving rows (the candidate query's positive + hard-negative rows
   at minimum; optionally all its top-k rows) to
   `examples/fiqa/case_sets/<case_id>.csv` with `source_case_id=<case_id>`.

Defaults: `--shape-rank 3 --shape-pos-gap 5 --top-k-similar 10`. All overridable.

## Encoding reuse

- The mining script needs MiniLM embeddings for FiQA queries. To avoid a
  second encoding pass and a heavy dep at mining time, it tries to reuse the
  build cache: `build_fiqa_csv.py` already encodes the corpus; we extend it (or
  the mining script) to dump query embeddings to
  `examples/fiqa/.cache/query_embeddings.npz` at build time, and the miner loads
  them. If absent, the miner falls back to encoding on the fly (still needs
  sentence-transformers, which is in requirements-build.txt — mining is a build
  activity, not runtime).

## train_reranker.py changes

- New arg: `--case-sets <dir-or-file>` (default: none). If a dir, load all
  `*.csv` under it; if a file, load that one.
- Loaded case_sets rows merge into the TRAIN split ONLY. Their `split` column
  is forced to "train" regardless of source. They never enter validation/test.
- **Defensive B+C re-check at load**: after loading, assert no row's query_id
  is in the set of case query_ids (read from regression_cases.yaml — YES, train
  reads the case FILE here, but ONLY to get the query_id/doc_id denylist for
  isolation, never as training rows). If any leak, fail loud with the
  offending ids. This is the one exception to "train never reads cases": it
  reads the case IDS for isolation, not the case rows for training. The
  anti-leak invariant (never train on case rows) is preserved.
- Without `--case-sets`, behavior is identical to today.
- train still never reads `ledger.json`.

## eval/ledger changes (minimal)

- `eval_reranker.py` gains a `--case-sets-used <note>` flag (or auto-detect by
  checking if case_sets dir is non-empty) that tags the round in the ledger.
- `regression_ledger.py record` stores `case_sets_used: true/false` (and
  optionally the manifest hash) in the round snapshot, so the progress summary
  can show "round N used case_sets".
- No change to gate/pending exit behavior (step 1 contract holds).

## Closed-loop workflow

Documented command sequence (no new driver needed; keep tools composable):

```bash
# 1. mine same-pattern samples for all pending cases
python skills/heuriboost-rag/scripts/mine_case_sets.py \
  --dataset examples/fiqa/query_doc_examples.csv \
  --cases examples/fiqa/regression_cases.yaml \
  --out-dir examples/fiqa/case_sets

# 2. retrain with mined samples folded into train
python skills/heuriboost-rag/scripts/train_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --case-sets examples/fiqa/case_sets

# 3. eval + ledger (records case_sets_used)
python skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split validation \
  --regression-cases examples/fiqa/regression_cases.yaml \
  --case-sets-used

# 4. (manual) if a pending case passed + B ok, promote it
python skills/heuriboost-rag/scripts/regression_ledger.py promote \
  --ledger examples/fiqa/ledger.json --cases examples/fiqa/regression_cases.yaml \
  --case-id <case_id>
```

No auto-promotion anywhere. The loop is run by the maintainer; each step is a
discrete, inspectable command.

## Honest-reporting caveat

- README "Current Status" / DATA_CARD: add a note that step-2 attack results
  are pipeline-validation grade under heuristic labels; credible attack quality
  waits for LLM-mode labels.

## Tradeoffs

- case_sets committed: traceable and reproducible, at the cost of per-round
  diff noise when re-mining. Acceptable (same call as ledger).
- Mining reuses build cache when present; falls back to on-the-fly encoding.
  Either way it needs sentence-transformers (build dep, not runtime).
- train reading regression_cases.yaml for the denylist is a narrow, intentional
  exception to step-1's "train never reads cases" — it reads IDS for isolation,
  not rows for training. Documented and asserted at load.

## py_compile

`python3 -m py_compile skills/heuriboost-rag/scripts/*.py` must pass (incl. the
new mining script).
