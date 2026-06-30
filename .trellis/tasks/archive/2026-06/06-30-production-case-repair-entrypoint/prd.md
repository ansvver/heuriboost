# Production-case repair entrypoint

## Goal

Provide a user-facing production-case repair workflow that lets users run the
strict reckless loop from two simple input tables instead of hand-authoring the
current internal `query_doc_examples.csv`, `regression_cases.yaml`, and
`case_sets/` files.

The workflow should make production incidents cheap to feed into training while
preserving hard acceptance gates: case-level repair checks, historical gate
checks, and global/domain test metrics against anchored baselines.

## User Value

Users should only need to prepare:

- `base_dataset.csv`: stable training, validation, and test data.
- `production_cases.csv`: online incidents or feedback rows to repair.

The system should compile those inputs into internal training, repair, and
acceptance artifacts, then run the strict reckless train/eval flow. Internal
files remain available for audit/debugging but are no longer the primary user
contract.

## User-facing Inputs

### `base_dataset.csv`

Purpose: stable base data for training, validation, and global/domain test
acceptance.

Minimum user-friendly columns:

```csv
query,text,relevance
```

Recommended columns:

```csv
domain,query_id,query,doc_id,text,relevance,split,rank,score
```

Rules:

- `domain` is optional and defaults to `default`.
- `query_id` and `doc_id` are optional; missing ids are generated from stable
  hashes of normalized `query` and `text`.
- `split` is optional; if absent, the compiler auto-splits by query with a
  fixed seed and writes the compiled split to disk.
- If `split` is present, the compiler respects it. It must not re-split unless
  the user explicitly requests `--resplit`.
- Supported relevance aliases should compile to internal integer labels:
  - `good` / `positive` -> `3`
  - `partial` -> `2`
  - `weak` -> `1`
  - `irrelevant` / `negative` -> `0`
  - `bad` / `hard_negative` -> `-1`
- The compiled internal dataset still uses the current canonical columns:
  `query_id,query_text,doc_id,doc_text,label,split`, plus optional retrieval
  metadata when available.

### `production_cases.csv`

Purpose: online incidents or feedback rows that the repair run should absorb
and then verify.

Minimum user-friendly columns:

```csv
query,shown_doc_text,user_verdict
```

Recommended columns:

```csv
domain,case_id,query,shown_doc_id,shown_doc_text,user_verdict,rank,score
```

Rules:

- `domain` is optional and defaults to `default`.
- `case_id` is optional; missing ids are generated from a stable hash of
  normalized `query`, `shown_doc_text`, and `user_verdict`.
- `shown_doc_id` is optional; missing ids are generated from a stable hash of
  normalized `shown_doc_text`.
- `user_verdict` supports exactly:
  - `good`: can become a positive repair sample and a full acceptance target.
  - `bad`: can become a hard-negative repair sample and a rejection target.
  - `unknown`: context only by default; not training or acceptance input.
- `rank` and `score` are optional report metadata. They do not define
  acceptance `top_k`.
- Same `case_id` rows merge into one case:
  - good docs are unioned.
  - bad docs are unioned.
  - unknown docs are retained as context.
  - the same doc marked both good and bad in the same case is a hard failure.

## Domain Semantics

`domain` is an optional but hard boundary once provided.

Rules:

- Missing `domain` compiles to `default`.
- A production case can only use base data, historical gates, promoted repair
  samples, and mined candidates from the same domain.
- Candidate completion, negative augmentation, case similarity, auto-split, and
  report buckets operate within domain boundaries.
- Internal ids are domain-scoped to avoid collisions:
  - `tax::q1`
  - `insurance::q1`
- Original user ids must be preserved as source ids for audit.
- If a production case references a domain absent from `base_dataset`, strict
  repair hard-fails.

## Commands

### `compile-cases`

Audit and compile inputs without training.

Expected behavior:

- Validate user-facing `base_dataset` and `production_cases`.
- Normalize aliases, generate synthetic ids, apply/compile splits, and build
  internal artifacts.
- Write compiled artifacts under `output/.heuriboost/compiled/`.
- Emit warnings for weak data quality issues and hard-fail on structural issues.

### `repair --reckless`

The strict production repair path. There is no separate friendly repair mode.

Expected behavior:

1. Compile user-facing inputs.
2. Auto-initialize the ledger anchor when no anchor exists by training/evaluating
   the baseline on `base_dataset` only.
3. Never overwrite an existing anchor unless an explicit reset option is used.
4. Train one final candidate repaired model using base training data, promoted
   repair memory, and current production repair samples.
5. Evaluate current production cases, historical gates, and base test metrics.
6. Hard-fail unless the selected acceptance level passes and metric gates pass.
7. Output one user-visible candidate model, plus reports and hidden compiled
   artifacts.

### `promote`

Promote an eligible candidate repair run. Promotion is explicit and must not
automatically happen during `repair --reckless`.

Expected behavior:

- Update the current model pointer or model output to the candidate repaired
  model.
- Refresh global/domain anchors from the promoted run.
- Freeze full production cases as historical gates.
- Write promoted repair samples into system-managed repair memory for future
  training.
- Do not deploy online, delete old models, silently mutate user input files, or
  edit the user's original `base_dataset.csv`.

## Internal Artifacts

The system may still produce the existing lower-level representations, but they
are compiled artifacts rather than user-authored inputs:

- `TrainingDataset`: canonical train/validation/test query-doc examples.
- `AcceptanceCases`: current production cases and historical gates.
- `RepairSamples`: production case rows and promoted repair samples used for
  reckless training.

Default output layout:

```text
output/
  models/reranker.json
  models/reranker_metadata.json
  reports/repair_report.md
  .heuriboost/
    compiled/
      query_doc_examples.csv
      regression_cases.yaml
      case_sets/
    anchor_baseline.json
    ledger.json
    gates.jsonl
    promoted_repair_samples.csv
```

`compiled/` artifacts are kept by default for auditability but should be
documented as generated/debug artifacts, not user prerequisites.

## Acceptance Semantics

### Acceptance levels

Default:

```text
--acceptance-level full
```

Full acceptance:

- Every actionable current production case must include at least one `good`
  target.
- At least one good doc must enter `top_k`.
- All bad docs must be outside `top_k`.
- Historical gates must also pass.
- The run can be `promotion_eligible` only at full acceptance.

Weak acceptance:

```text
--acceptance-level weak
```

- Explicit opt-in only.
- Allows cases with only bad evidence.
- A weak case can pass by suppressing bad docs.
- Weak acceptance must set `promotion_eligible=false`.
- Reports must make clear that weak acceptance proves bad-result suppression,
  not full answer repair.

### `top_k`

- Users do not set per-case `top_k` in input files.
- Default case `top_k` is `3`.
- A command-level option may override it for all cases.
- Input `rank` is only original online-display metadata.

### Multiple good/bad docs

- Multiple bad docs: all must be outside `top_k`.
- Multiple good docs: at least one must enter `top_k`.
- Context/unknown docs may be included in candidate pools but do not directly
  determine pass/fail.

## Test and Anchor Semantics

`base_dataset` test split is the metric-level regression suite.

Rules:

- Production cases must not be automatically added to base test.
- Global safety acceptance uses base test `nDCG@10` and `MRR@10` against the
  global ledger anchor.
- Global `nDCG@10` and `MRR@10` must both improve over anchor.
- The touched domains are the domains present in current production cases.
- Every touched domain must have sufficient test coverage and must not regress
  versus its domain anchor.
- Untouched domains participate in global metrics/reporting but do not hard-gate
  per-domain acceptance.
- Missing anchor is initialized automatically on first strict repair run.
- Existing anchors must not be overwritten without an explicit reset.

Test sufficiency defaults:

- Global test: at least 10 test queries, each with at least 2 candidate docs,
  and at least one positive and one negative label overall.
- Touched domain test: at least 3 test queries, each with at least 2 candidate
  docs, and at least one positive and one negative label overall for that domain.

Train/validation sufficiency:

- Train: at least 1 query and at least one positive and one negative label.
- Validation: at least 1 query, each validation query has at least 2 candidate
  docs, and validation has at least one positive and one negative label.

Queries with only one doc are warnings during compilation unless they violate a
strict split sufficiency rule.

## Gate Semantics

Historical gates are case-level regression tests, not metric-level test rows.

Relationship to base test:

- Base test remains stable and drives global/domain `nDCG@10` and `MRR@10`.
- Gates are promoted production full cases that must continue to pass in future
  repair runs.
- Gates are not appended to base test and do not participate in global nDCG/MRR.

Gate snapshot requirements:

- Gates are self-contained snapshots that can be evaluated even if the base
  dataset changes.
- A gate snapshot stores:
  - gate id / source case id
  - domain
  - query
  - full candidate pool when available
  - doc id and text for each candidate
  - candidate role: `good`, `bad`, or `context`
  - `top_k`
  - source run id
  - promoted timestamp
- Gate evaluation uses the snapshot candidate pool. Pass/fail checks only
  consider good and bad roles.
- First version runs all gates every strict repair run.
- The design should leave room for a future `critical` / `archived` gate
  lifecycle, but that lifecycle is not in scope for the first version.

## Repair Samples and Promotion Memory

Production case rows may enter reckless training. This is intentional for the
production repair lane.

Rules:

- Full/weak case checks are not treated as proof of generalization.
- General safety is guarded by base test anchors and touched-domain checks.
- Promoted repair samples are stored separately from the user's original
  `base_dataset`.
- Future repair runs include promoted repair samples by default.
- Promoted repair memory is domain-scoped and must not be used for cross-domain
  mining or candidate completion.
- If a user renames a domain, old promoted samples do not automatically migrate.

## Candidate Completion and Augmentation

Candidate completion is allowed to lower user input burden, but must preserve
test integrity.

Rules:

- Production cases may use same-domain base rows, historical gates, promoted
  repair samples, and negative augmentation to build a candidate pool.
- Negative augmentation can add same-domain bad candidates for training and weak
  evaluation support.
- Negative augmentation cannot synthesize good targets and cannot upgrade weak
  cases to full cases.
- Base train/validation may be candidate-completed with inferred labels.
- Base test must not be silently completed with synthetic labels for strict
  anchor acceptance.

## Validation Behavior

Hard failures:

- `base_dataset` missing query text.
- `base_dataset` missing document text.
- `base_dataset` missing relevance/label/verdict.
- `base_dataset` has no positive label in repair mode.
- `base_dataset` has no negative label in repair mode.
- `production_cases` missing query text.
- `production_cases` missing shown document text.
- `production_cases` missing verdict.
- `production_cases` contains only unknown verdicts.
- Same doc is both good and bad within one case.
- A production case domain is absent from base data.
- Strict `repair --reckless` cannot compile sufficient train/validation/test
  data.
- Any current full production case or historical gate fails.
- Global test `nDCG@10` or `MRR@10` fails to improve over anchor.
- Any touched domain metric regresses versus its domain anchor.

Warnings:

- Synthetic ids generated.
- Split auto-generated.
- Query has only one candidate doc outside strict sufficiency checks.
- Case has only bad evidence and requires weak acceptance.
- Dense/sparse ranks or scores are absent, reducing baseline/report richness.
- Unknown verdict rows are retained as context only.

## Non-goals

- Do not require users to hand-author `regression_cases.yaml` or `case_sets/`
  for the main workflow.
- Do not automatically deploy promoted models online.
- Do not silently mutate the user's original `base_dataset.csv` or
  `production_cases.csv`.
- Do not automatically merge production cases into base test.
- Do not implement a `critical` / `archived` gate lifecycle in the first version.
- Do not treat weak acceptance as promotion-eligible.

## Acceptance Criteria

- [ ] A user can run `compile-cases` from `base_dataset.csv` and
  `production_cases.csv`, including minimal schemas, string label aliases,
  synthetic ids, default domain, and optional auto-split.
- [ ] Compiled artifacts are written under `output/.heuriboost/compiled/` and
  are clearly generated audit/debug artifacts.
- [ ] `repair --reckless` runs the strict flow from the two user-facing inputs,
  auto-initializes a missing anchor, and never overwrites an existing anchor
  unless explicitly requested.
- [ ] `repair --reckless` outputs one user-visible candidate model, not separate
  user-facing baseline and reckless models.
- [ ] Full acceptance is the default and requires good-target hit, bad-target
  suppression, historical gate pass, global test improvement, and touched-domain
  non-regression.
- [ ] Weak acceptance requires explicit opt-in and sets
  `promotion_eligible=false`.
- [ ] Historical full gates are self-contained snapshots and are all evaluated
  on every strict repair run.
- [ ] `promote` is explicit and only updates current model state, anchors,
  full-case gates, and promoted repair memory.
- [ ] Promotion does not mutate user input files, deploy online, or merge
  production cases into base test.
- [ ] Domain boundaries are enforced for ids, candidate completion, negative
  augmentation, promoted repair memory, and touched-domain acceptance.
