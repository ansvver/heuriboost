# Implementation Plan: case_sets Mining & Settlement (V1, step 2)

## Phase 1: Mining script (mine_case_sets.py, new)

- [ ] Argparse: --dataset, --cases, --out-dir (default examples/fiqa/case_sets),
      --shape-rank (3), --shape-pos-gap (5), --top-k-similar (10),
      --cache-dir (default examples/fiqa/.cache).
- [ ] Load main CSV + cases (cases only to get pending cases' query_id,
      query_text, failure_type, and the global denylist of case query_ids and
      case doc_ids).
- [ ] (c) filter candidate queries by failure_type match with the case.
- [ ] (b) shape filter: candidate query has a -1 row with dense_rank <=
      shape_rank AND a 3 row with dense_rank >= shape_pos_gap.
- [ ] (a) semantic similarity: encode case query + candidate queries with
      all-MiniLM-L6-v2 (reuse .cache/query_embeddings.npz if present, else
      encode on the fly and cache); keep top-k-similar by cosine.
- [ ] B+C isolation: drop candidate queries whose query_id is any case's
      query_id; drop candidate rows whose doc_id is any case's doc_id.
- [ ] Write examples/fiqa/case_sets/<case_id>.csv (same schema as main CSV +
      source_case_id column) for each pending case with >0 mined rows; write
      manifest.json with mining params + per-case counts.
- [ ] Also dump query embeddings to .cache if computed fresh (so future runs
      reuse).

## Phase 2: build_fiqa_csv.py cache hook (optional but preferred)

- [ ] When build_fiqa_csv.py encodes queries for retrieval, also dump
      query_embeddings.npz to --cache-dir. (One-time change; mining reuses.)
- [ ] Keep it optional: mining works without the cache (encodes on the fly).

## Phase 3: train_reranker.py consumes case_sets

- [ ] New --case-sets <dir-or-file> arg (default none).
- [ ] Load case_sets rows; force split="train"; merge into train_df.
- [ ] Defensive B+C re-check: load case query_ids + doc_ids from
      regression_cases.yaml (the narrow, documented exception — IDs only,
      never case rows as training data); assert no mined row's query_id is in
      case query_ids and no mined row's doc_id is in case doc_ids; fail loud
      on any leak with offending ids.
- [ ] Without --case-sets: behavior identical to today.
- [ ] train STILL never reads ledger.json; reads cases only for the denylist.

## Phase 4: eval/ledger tag

- [ ] eval_reranker.py: --case-sets-used flag (or auto-detect non-empty
      case_sets dir); pass to ledger.record.
- [ ] regression_ledger.py: record stores case_sets_used (bool) in the round
      snapshot; summary prints "round N used case_sets" when true.
- [ ] No change to gate/pending exit behavior.

## Phase 5: Docs + data

- [ ] README + README.zh-CN: document the closed-loop command sequence + the
      heuristic-label pipeline-validation caveat.
- [ ] CODEBUDDY.md: add mine_case_sets.py and --case-sets to layout/commands;
      note the refined anti-leak contract (train reads case IDs for denylist,
      never case rows for training).
- [ ] DATA_CARD: note case_sets are committed, derived, regeneratable.
- [ ] .gitignore: ensure examples/fiqa/case_sets/ is NOT ignored.
- [ ] .trellis/spec/backend/fiqa-demo-contracts.md: refine the anti-leak
      contract (B+C isolation; train may read case IDs for the denylist, never
      case rows as training data); add the case_sets contract.

## Phase 6: Verify

- [ ] python3 -m py_compile skills/heuriboost-rag/scripts/*.py passes.
- [ ] Run mining on the FiQA demo; confirm case_sets/<case_id>.csv produced per
      pending case with B+C isolation (grep that no mined query_id matches a
      case query_id, no mined doc_id matches a case doc_id).
- [ ] Run train with --case-sets; confirm it merges into train and the
      defensive check passes.
- [ ] Run eval with --case-sets-used; confirm ledger round has
      case_sets_used=true; confirm gate/pending behavior unchanged (gates
      still pass, pending still reported not blocking).
- [ ] Anti-leak grep: train_reranker.py references to cases are ONLY for the
      denylist (no case row enters the training feature matrix).

## Notes for implementer

- Plan Y: mining is a new script; train gains an optional input; eval/ledger
  changes are minimal tags.
- The closed loop is run by the maintainer as discrete commands; no auto-loop,
  no auto-promotion.
- See memory project_autosearch_loop.md for the agreed design rationale.
