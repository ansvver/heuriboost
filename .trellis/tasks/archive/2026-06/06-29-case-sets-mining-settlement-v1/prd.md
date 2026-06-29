# case_sets Mining & Settlement (V1, step 2)

## Goal

Close the iterative failure-attack loop's "textbook path": for each `pending`
regression case, mine same-pattern samples from the FiQA corpus and feed them
as TRAINING data (the case itself stays an exam question). This tests whether
attacking a failure pattern with mined training samples can move a pending case
to passing, using the state machine from step 1.

This is step 2 of the V1 loop agreed in memory `project_autosearch_loop.md`.
Step 1 (case state machine) is done and committed.

## Anti-leak invariant (absolute)

Regression/gate cases are exam questions; they NEVER enter training. Settlement
into training is via MINED SAMPLES only, and mined samples must be hard-isolated
from cases by BOTH `query_id` (B) and `doc_id` (C):

- A mined sample's `query_id` must not equal any case's `query_id`.
- A mined sample's `doc_id` must not equal any case's must_include or
  must_not_include `doc_id`.

`case_sets` (the mined samples) are training data, NOT exam questions. They are
physically separate from `regression_cases.yaml` and `ledger.json`. The step-1
contract "train_reranker.py never reads cases/ledger" still holds; train reads
the CSV (as today) PLUS optionally `case_sets` (new, training-only).

## Scope decisions (grilling 2026-06-29)

- **Path X (case_sets inbox)**: mined samples live in an explicit
  `case_sets` artifact, NOT a full CSV rebuild. Traceability: each sample set is
  tied to a source pending case.
- **Mining rule = a+b+c (all three, intersection)**:
  (a) semantic similarity to the case's query (MiniLM embedding top-K), AND
  (b) same failure shape (dense_rank of a -1 doc <= SHAPE_RANK, and a positive
      doc at dense_rank >= SHAPE_POS_GAP), AND
  (c) same `failure_type` as the case.
- **Order P**: ship the closed loop now under heuristic labels; label the
  result a "pipeline validation", NOT a credible attack. Real attack quality
  waits for LLM-mode labels (deferred, separate task).

## Requirements

### R1. Mining script
A new script (e.g. `mine_case_sets.py`) that, for each `pending` case in
`regression_cases.yaml`:
- finds non-case FiQA queries (query_id not in any case) whose docs include
  both a -1-labeled hard negative at dense_rank <= SHAPE_RANK and a positive
  (label 3) at dense_rank >= SHAPE_POS_GAP (failure shape b);
- filters those to the K most semantically similar to the case's query (a,
  via all-MiniLM-L6-v2; reuse build cache when available);
- keeps only same failure_type (c);
- enforces B+C isolation (no case query_id, no case doc_id);
- writes the mined rows to a `case_sets` artifact with the same CSV schema as
  the main dataset (so train can consume them uniformly), tagged with the
  source case_id.

SHAPE_RANK / SHAPE_POS_GAP / K are CLI flags with sensible defaults.

### R2. case_sets artifact location & format
- Lives under `examples/fiqa/case_sets/` (gitignored build output? committed?
  -> see Open Question, resolved in design).
- One file per source case (e.g. `case_sets__<case_id>.csv`) OR a single
  `case_sets.csv` with a `source_case_id` column. Design picks.
- Same column schema as `query_doc_examples.csv` so train reads it uniformly.
- Rows are training samples (label in {3,0,-1} from the existing heuristic).

### R3. train_reranker.py consumes case_sets (optional)
- New flag `--case-sets <path>` (or dir) that, when present, loads those rows
  and merges them into the TRAIN split (only train; never validation/test).
- Mined rows must respect B+C at load time (defensive re-check; fail loud if a
  case query_id/doc_id leaks in).
- Without `--case-sets`, train behaves exactly as today (no behavior change).
- train STILL never reads `regression_cases.yaml` or `ledger.json`.

### R4. Closed-loop workflow support
A way to run one full round: mine -> train with case_sets -> eval -> ledger
record -> see if any pending case turned green. This can be a documented
sequence of commands or a small driver; design picks. Must NOT auto-promote
(step 1 contract: promotion is manual).

### R5. Honest reporting
- The eval/ledger summary must indicate when a round used case_sets (so the
  "pipeline validation" caveat is visible).
- A note in README/DATA_CARD that step-2 attack results under heuristic labels
  are pipeline-validation grade, not benchmark.

## Acceptance Criteria

- [ ] Mining script produces case_sets for each pending case, a+b+c filtered,
      B+C isolated, schema-compatible with the main CSV.
- [ ] `train_reranker.py --case-sets` merges them into train only; defensive
      B+C re-check at load; without the flag, behavior unchanged.
- [ ] `train_reranker.py` still never reads regression_cases.yaml or ledger.json
      (anti-leak grep clean).
- [ ] One full round (mine -> train -> eval -> ledger) runs end-to-end on the
      FiQA demo; ledger records that case_sets were used.
- [ ] After the round, the eval summary reports which pending cases passed
      (promotion candidates) — regardless of whether any actually moved.
- [ ] No auto-promotion; promotion stays manual via step-1 `regression_ledger.py promote`.
- [ ] py_compile clean; no heavy/LLM deps added to runtime requirements.
- [ ] README/DATA_CARD note the heuristic-label pipeline-validation caveat.

## Out of scope (deferred)

- LLM-mode labels (separate task; user must rotate the leaked DeepSeek key).
- Approach 2 (LLM pattern generation).
- HPO, feature registry, horizontal task profiles, online serving.

## Open Questions (resolve in design)

- case_sets committed vs gitignored? (They are derived, regeneratable, and
  potentially large; lean gitignored under examples/fiqa/case_sets/.)
- Single file with source_case_id column vs one file per case?
- Whether the mining script needs its own MiniLM encoding pass or can reuse the
  build cache (examples/fiqa/.cache/).
