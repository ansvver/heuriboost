# Design: Production-case repair entrypoint

## Architecture

Add a user-facing repair layer above the existing FiQA reranker scripts.

The current low-level scripts remain the model training/evaluation primitives:

- `common.py`: canonical dataset loading, validation, feature extraction, metrics.
- `train_reranker.py`: trains a model from canonical query-doc CSV.
- `eval_reranker.py`: ranks a split, evaluates cases, writes reports, records
  ledger rounds.
- `regression_ledger.py`: stores anchors and round snapshots.

The new layer introduces three concepts:

- user-facing input compiler
- strict repair orchestrator
- explicit promote operation

Recommended new modules/scripts:

```text
skills/heuriboost-rag/scripts/repair_cases.py
  Shared compiler, validators, gate snapshot helpers, and repair report helpers.

skills/heuriboost-rag/scripts/compile_cases.py
  CLI: compile user-facing base_dataset + production_cases into internal artifacts.

skills/heuriboost-rag/scripts/repair_reranker.py
  CLI: strict `repair --reckless` orchestration.

skills/heuriboost-rag/scripts/promote_repair.py
  CLI: explicit promotion for an eligible repair run.
```

This keeps the existing train/eval contracts stable while giving users a
smaller, higher-level entrypoint.

## Data Flow

### Compile

Input:

```text
base_dataset.csv | base_dataset.jsonl
production_cases.csv | production_cases.jsonl
```

Output:

```text
output/.heuriboost/compiled/query_doc_examples.csv
output/.heuriboost/compiled/regression_cases.yaml
output/.heuriboost/compiled/case_sets/current_production_cases.csv
output/.heuriboost/compiled/production_cases.json
output/.heuriboost/compiled/compile_report.md
```

Compile steps:

1. Load CSV or JSONL by extension.
2. Normalize column aliases.
3. Normalize labels/verdicts.
4. Fill missing domain with `default`.
5. Generate stable synthetic ids where needed.
6. Scope internal query/doc ids by domain while preserving source ids.
7. Apply user split or deterministic query-group auto-split.
8. Validate base train/validation/test sufficiency for strict repair.
9. Merge production rows by case id.
10. Detect good/bad conflicts.
11. Build production repair samples.
12. Build acceptance cases with full/weak metadata.
13. Write compiled artifacts and a human-readable report.

### Repair

Input:

```text
--base-dataset <path>
--production-cases <path>
--output-dir <path>
--reckless
--acceptance-level full|weak
```

Repair steps:

1. Run the compiler.
2. Load or create `output/.heuriboost/ledger.json`.
3. If no anchor exists:
   - train a temporary baseline model from compiled base data only
   - evaluate compiled base test
   - set global and per-domain anchors from that baseline
   - write `anchor_baseline.json`
   - remove/hide temporary model artifacts unless debugging is enabled
4. Build the candidate training CSV:
   - base train
   - promoted repair memory
   - current production repair samples
5. Train the single user-visible candidate model into `output/models/`.
6. Evaluate:
   - current production cases
   - all historical gates
   - global base test
   - touched-domain base test buckets
7. Write repair report and ledger round.
8. Hard-fail if any strict acceptance condition fails.

### Promote

Input:

```text
--output-dir <path>
```

Promote steps:

1. Read latest repair report / eligibility metadata.
2. Refuse weak or failed runs.
3. Update current model state or pointer to the candidate model.
4. Refresh ledger anchors from the promoted run.
5. Append full production case snapshots to `gates.jsonl`.
6. Append promoted repair samples to `promoted_repair_samples.csv`.
7. Leave user input files untouched.

## Contracts

### User-facing base dataset

Canonical fields after normalization:

```text
domain
source_query_id
query_id
query_text
source_doc_id
doc_id
doc_text
label
split
rank?
score?
dense_rank?
dense_score?
sparse_rank?
sparse_score?
```

Internal `query_id` / `doc_id` are domain-scoped. Source ids are preserved for
audit.

### User-facing production cases

Canonical row fields after normalization:

```text
domain
case_id
query_text
source_doc_id
doc_id
doc_text
verdict: good|bad|unknown
rank?
score?
```

Compiled case fields:

```text
case_id
domain
query
good_doc_ids[]
bad_doc_ids[]
context_doc_ids[]
candidate_pool[]
acceptance_level_required: full|weak
top_k
```

### Gate snapshot

Gate snapshots are self-contained:

```json
{
  "gate_id": "domain::case_id",
  "source_case_id": "case_id",
  "domain": "domain",
  "query": "...",
  "top_k": 3,
  "acceptance_level": "full",
  "source_run_id": "...",
  "promoted_at": "...",
  "candidates": [
    {"doc_id": "...", "source_doc_id": "...", "text": "...", "role": "good"}
  ]
}
```

Evaluation only gates on `good` and `bad`; `context` candidates make the ranking
environment more realistic.

### Ledger

Extend ledger data additively. Existing fields remain valid.

New round metadata should include:

```text
repair_mode: production_cases
acceptance_level
promotion_eligible
global_anchor_deltas
domain_anchor_deltas
touched_domains
case_results
gate_results
compiled_artifact_paths
```

Anchors should support both:

```text
anchor.global
anchor.domains[domain]
```

Existing simple anchor readers must either ignore unknown fields or be updated
to handle the extended shape.

## Acceptance Logic

### Production cases and gates

For each case/gate ranked candidate pool:

- Full:
  - at least one good candidate ranks `<= top_k`
  - every bad candidate ranks `> top_k`
- Weak:
  - every bad candidate ranks `> top_k`
  - `promotion_eligible=false`

Historical gates always use full semantics and must all pass.

### Global metrics

Evaluate compiled base test with the candidate model:

- global `nDCG@10` must be greater than global anchor
- global `MRR@10` must be greater than global anchor

For each touched domain:

- domain `nDCG@10` must be greater than or equal to domain anchor
- domain `MRR@10` must be greater than or equal to domain anchor

## Validation and Error Handling

The compiler owns input normalization and data-quality diagnostics.

Use `SystemExit` with actionable messages for hard failures, matching existing
script style.

Warnings should be collected into compile/repair reports and printed in CLI
output.

Strict repair hard-fails before training when train/validation/test sufficiency
cannot be met.

## Compatibility

- Existing `train_reranker.py`, `eval_reranker.py`, and `regression_ledger.py`
  entrypoints continue to work.
- Existing FiQA demo files remain valid.
- Existing `case_sets` reckless path remains available as a lower-level legacy
  path.
- The new compiler emits canonical files compatible with existing low-level
  scripts where practical.

One deliberate semantic change is introduced only for the new production-case
entrypoint: production case rows may enter reckless training. This does not
change the legacy `--case-sets` B+C isolation rule.

## Trade-offs

- A high-level orchestrator avoids overloading the current train/eval scripts
  with many user-facing input modes.
- Keeping compiled artifacts gives debuggability at the cost of extra files
  under `output/.heuriboost/`.
- Default full acceptance preserves rigor but requires users to provide good
  targets; weak acceptance exists for explicit bad-only repairs but cannot
  promote.
- Domain anchors add bookkeeping but prevent cross-domain repairs from hiding
  local regressions.

## Rollback

The feature is additive:

- Remove the new repair scripts and helper module.
- Leave existing train/eval/ledger behavior intact.
- Any generated `output/.heuriboost/` artifacts can be deleted without affecting
  source inputs.
