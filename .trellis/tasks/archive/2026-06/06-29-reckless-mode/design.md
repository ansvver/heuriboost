# Design: Reckless Mode

## Architecture

Extend the existing FiQA workflow with a `--reckless` flag on `train_reranker.py`
and `eval_reranker.py`.

The mode keeps the current pipeline shape:

`dataset -> train -> eval -> ledger`

but changes two behaviors when the flag is present:

1. `train_reranker.py` still folds `case_sets` into train, but the run is
   explicitly marked reckless in metadata.
2. `eval_reranker.py` treats every regression case as blocking and only accepts
   the round when the test split beats the anchored baseline on both
   `nDCG@10` and `MRR@10`.

## Data flow

### Train

- Load dataset.
- Optionally merge `case_sets` into train, reusing the existing B+C isolation
  re-check.
- Train as before.
- Persist a metadata flag such as `reckless_mode: true`.

### Eval

- Load dataset and model.
- Evaluate train, validation, and test splits when available.
- Evaluate all regression cases with the existing `run_regression_cases`
  helper.
- When `--reckless` is set:
  - block on any case failure, regardless of status
  - compare test `nDCG@10` and `MRR@10` against the ledger anchor
  - mark the round acceptable only if both test metrics improve over anchor
- Keep the existing non-reckless exit behavior unchanged when the flag is
  absent.

## Acceptance rule

The reckless round is successful only when all of the following hold:

- all regression cases pass
- the test split exists
- test `nDCG@10` > anchor `nDCG@10`
- test `MRR@10` > anchor `MRR@10`

## Ledger/reporting

The round snapshot should record:

- `reckless_mode: true/false`
- `case_sets_used`
- per-split metrics already computed by eval
- anchor deltas for test `nDCG@10` and `MRR@10`

No new persistent state format is needed beyond extra booleans / deltas in the
existing ledger snapshot.

## Compatibility

- Default behavior remains unchanged.
- Existing `--case-sets` behavior remains valid outside reckless mode.
- Existing regression case semantics remain valid outside reckless mode.

## Trade-offs

- Using the existing commands keeps the change small and avoids a second
  runner.
- The mode becomes stricter than the current loop, which is intentional.
- If the test split is missing, reckless mode cannot be accepted.

## Rollback

Remove the `--reckless` branches and metadata fields. The default path remains
the current one.
