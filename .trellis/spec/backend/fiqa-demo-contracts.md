# FiQA Demo & Feature-Set Contracts

> Executable contracts for the HeuriBoost RAG V0 FiQA demo and its feature set.
> Captured 2026-06-29 during the FiQA pivot. These prevent silent-zero metrics,
> data leakage, and broken demo regeneration.

---

## Contract: FiQA-2018 ships no candidates, no scores, no hard negatives

**What**: BEIR/FiQA-2018 provides only `corpus`, `queries`, and binary `qrels`
(relevant docs only). It does NOT provide candidate retrieval lists,
`dense_*`/`sparse_*` scores, graded labels, or `-1` hard negatives.

**Why**: The reranker's CSV contract requires `dense_rank/dense_score/
sparse_rank/sparse_score` as L0 features, and HeuriBoost's headline is
hard-negative (`-1`) suppression. None of these exist in raw FiQA.

**Therefore (build-time obligation)**: `build_fiqa_csv.py` MUST manufacture them:
- Run BM25 (`rank_bm25`) + dense (`all-MiniLM-L6-v2`) over the sliced corpus,
  fuse with RRF, take top-k candidates per query → produces `dense_*`/`sparse_*`.
- Use a build-time LLM judge to expand binary qrels into the 5-level scale
  `{3,2,1,0,-1}` (qrel positives may be seeded as `3` before judging).

**Wrong**: Assuming FiQA "has candidates" and loading qrels directly as the
training label / candidate set. Result: no scores, no hard negatives, demo
cannot show either headline claim.

---

## Contract: LLM-judged labels are build-time only

**What**: Labels produced by the LLM judge in `build_fiqa_csv.py` are baked into
the committed CSV. They may feed training and evaluation.

**Forbidden**:
- Never use an LLM-judged label (or any label field) as an online model feature.
- Never use LLM-judged labels to define a regression gate's
  `must_not_include_doc_ids`. Regression `must_not_include` cases require human
  confirmation (mine observed failures → hand-confirm 5-8).

**Why**: Labels are post-hoc judgments, not prediction-time signals (leakage).
Regression gates are the trusted floor; seeding them from medium-confidence LLM
output undermines the gate's authority.

---

## Contract: FEATURE_NAMES must equal feature_recipes.yaml

**What**: `skills/heuriboost-rag/scripts/common.py::FEATURE_NAMES`, the feature
values computed in `features/recipes.py::extract_all`, and the features listed
in `skills/heuriboost-rag/templates/feature_recipes.yaml` MUST be identical
(names and set).

**Why (silent failure)**: `eval_reranker.py`'s Feature Contrast block and
`feature_lookup` read features via `.get(name, 0.0)`. A name present in the
report/template but absent from `FEATURE_NAMES` does NOT raise — it silently
prints `0.0000`, producing misleading failure analysis with no error.

**Enforcement (registry load-time, since the feature-recipe-registry task)**:
the `FeatureRegistry` (`scripts/features/registry.py`) loads
`feature_recipes.yaml`, registers the shared `extract_all` impl, and runs
`validate()` at `import common` time. The validation hard-fails (SystemExit)
when:
  - a recipe's `impl` does not resolve to a registered implementation,
  - a recipe's `inputs` contains a column outside `ALLOWED_INPUTS`
    (`{query_text, doc_text, dense_rank, dense_score, sparse_rank, sparse_score}`),
  - a recipe is `online_safe: false` for the active task profile,
  - a spec-required field is empty (except `expected_slices`).

So the three-way identity is now a load-time check, not a grep-time check.
Adding/removing/renaming a feature requires editing BOTH the YAML and
`extract_all` (or registering a new impl) — a mismatch refuses to load.

**Current V0 set (12)**: `dense_score, dense_rank_inverse, sparse_score,
sparse_rank_inverse, rrf_score, term_overlap_ratio, number_overlap_count,
entity_overlap_count, important_term_overlap, low_information_density_flag,
doc_length_log, query_length_log`.

---

## Contract: features must be prediction-time computable

**What**: Every feature in `extract_features()` must be computable from
`query_text` + `doc_text` + retriever scores/ranks alone.

**Forbidden**: post-outcome signals as features — answer citations, LLM post-hoc
support judgments, user clicks/actions after prediction, human labels.

**Why**: Online safety / no leakage. A feature unavailable at prediction time
inflates offline metrics and fails in production.

---

## Contract: regression case state machine (gate / pending / retired)

**What** (added 2026-06-29, V1 step 1): each case in `regression_cases.yaml`
carries `status`:
- `gate` — attacked & frozen. Failure MUST block: `eval_reranker.py` exits
  non-zero.
- `pending` — a known gap to attack. Evaluated and REPORTED, but failure must
  NOT change the exit code (exit 0 if no gate failed).
- `retired` — invalidated; skipped entirely, kept for history.
- Absent `status` defaults to `gate` (back-compat).

**"Attacked / pass"** = must_include within top_k AND (if `require_rank` set)
must_include rank <= require_rank AND no must_not_include in top_k AND (if
`min_ndcg10` set) per-query nDCG@10 >= min_ndcg10.

**Why**: only `gate` cases represent achieved, frozen fixes; `pending` cases are
the roadmap of gaps. Auto-blocking on pending would make the demo red by design;
silently passing them would hide the roadmap. Promotion pending->gate is MANUAL.

## Contract: reckless mode hard-fails on anchor/test mismatch

**What** (added 2026-06-30): `train_reranker.py --reckless` folds mined
`case_sets` directly into train and defaults `--case-sets` to
`examples/fiqa/case_sets` when omitted. `eval_reranker.py --reckless` is a
strict acceptance path:
- it only runs on `--split test`
- it requires the ledger anchor to exist
- it loads `case_sets` with raw `source_case_id`
- it maps each `source_case_id` back to the original regression case and
  reuses that case's original pass/fail rule
- it fails if the test split is missing
- it fails unless test `nDCG@10` and `MRR@10` both beat the anchor

**Why**: reckless mode is the production-line repair path. The training data is
the mined case_sets, but acceptance must still prove that the original case is
fixed and that the test holdout improved over the frozen baseline.

**Ledger**: round snapshots record `reckless_mode`,
`reckless_source_case_ids`, and anchor deltas for both `ndcg@10` and
`mrr@10`.

## Scenario: reckless acceptance and round snapshots

### 1. Scope / Trigger
- Trigger: new `--reckless` command path across train/eval plus ledger snapshot
  metadata.

### 2. Signatures
```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py <query_doc_examples.csv> --reckless
python3 skills/heuriboost-rag/scripts/eval_reranker.py <query_doc_examples.csv> --split test --reckless
```

### 3. Contracts
- `--reckless` on train defaults `--case-sets` to `examples/fiqa/case_sets`
  and `--regression-cases` to `examples/fiqa/regression_cases.yaml` when
  omitted.
- `--reckless` on eval defaults `--case-sets` the same way and hard-fails if
  `--split` is not `test`.
- Eval in reckless mode requires a ledger anchor and compares test
  `nDCG@10`/`MRR@10` against it.
- Ledger round snapshots record `reckless_mode`, `reckless_source_case_ids`,
  and anchor deltas for both `nDCG@10` and `MRR@10`.

### 4. Validation & Error Matrix
- missing anchor -> `SystemExit: Reckless mode requires a ledger anchor...`
- missing test split -> `SystemExit: Reckless mode requires a non-empty test split.`
- missing `source_case_id` in case_sets -> `SystemExit: Reckless mode requires case_sets rows to preserve source_case_id.`
- referenced case_id missing from regression cases -> `SystemExit: Reckless mode case_sets reference missing regression case_id(s): ...`
- test `nDCG@10` or `MRR@10` not above anchor -> `SystemExit: Reckless acceptance failed: ...`

### 5. Good/Base/Bad Cases
- Good: train with `--reckless`, then eval `--split test --reckless` after
  setting the ledger anchor; acceptance passes only when the source cases and
  test metrics improve.
- Base: normal `train_reranker.py` / `eval_reranker.py` behavior stays
  unchanged.
- Bad: use `--reckless` without a test split or without an anchor; the command
  hard-fails.

### 6. Tests Required
- `python3 -m py_compile skills/heuriboost-rag/scripts/*.py`
- normal eval smoke on validation
- reckless eval smoke on test with no anchor hard-failing
- reckless eval smoke on test with anchor hard-failing unless both metrics beat anchor

### 7. Wrong vs Correct
#### Wrong
`case_sets` rows are treated as the acceptance unit and the round is accepted
purely because the mined rows pass.

#### Correct
`case_sets` are training input only; acceptance replays the original regression
case rule per `source_case_id` and requires test `nDCG@10` and `MRR@10` to beat
the anchor.

## Contract: the ledger is never a training input (anti-leak extension)

**What**: `examples/fiqa/ledger.json` (committed, NOT gitignored, NOT
auto-committed by scripts) records per-round snapshots + the B2 anchor. Like
regression cases, it is evaluation/tracking state only.

**Forbidden**: `train_reranker.py` must never read `ledger.json`. The ledger
is score-keeping state; training on it turns gates into rubber stamps.

**Refined (2026-06-29, V1 step 2)**: `train_reranker.py` may read
`regression_cases.yaml` ONLY for the case `query_id`/`doc_id` denylist to
enforce B+C isolation when `--case-sets` is set. This is the one narrow,
documented exception: train reads case IDS for isolation, NEVER case ROWS as
training data. Case rows are exam questions; `case_sets` (mined samples) are
the training data, physically separate and B+C isolated from cases.

**Why**: preserves the core "remembers its mistakes" value — gates must stay
exam questions the model has not trained on. The denylist exception exists so
that mined training samples can be defensively checked against case IDs at
load time, preventing accidental leakage.

## Contract: B2 anchor must degrade gracefully when absent

**What**: the global-no-regression check (B) compares a round's global nDCG@10 /
MRR@10 against a frozen anchor snapshot. When no anchor exists yet, the tooling
reports "no anchor yet" and offers `set-anchor`; it must NOT crash and must NOT
report a false regression. A corrupt ledger must fail with a clear message, not
silently.

**Why**: a fresh state (or first run) has no baseline; treating "no baseline" as
a regression would block falsely. B is reported, not exit-blocking, in V1.

## Contract: case_sets are committed, derived, B+C-isolated training data

**What** (added 2026-06-29, V1 step 2): `examples/fiqa/case_sets/` contains
mined training samples for `pending` regression cases. Each `<case_id>.csv`
has the same column schema as `query_doc_examples.csv` plus a
`source_case_id` column. `manifest.json` records mining parameters and
per-case counts.

**Properties**:
- **Committed** (NOT gitignored) — derived and regeneratable via
  `mine_case_sets.py`, committed for traceability (like the ledger).
- **Schema-compatible** — same columns as the main CSV so `train_reranker.py`
  reads them uniformly. The `source_case_id` column is dropped at load; it
  is for traceability, not training.
- **B+C isolated** — no mined row's `query_id` equals any case's `query_id`
  (B), and no mined row's `doc_id` equals any case's `must_include`/
  `must_not_include` `doc_id` (C). Enforced at mining time AND re-checked
  defensively at train load time.
- **Training data, NOT exam questions** — case_sets are physically separate
  from `regression_cases.yaml` and `ledger.json`. They feed the train split
  only (split forced to "train"); they never enter validation/test.

**Mining rule** (a+b+c intersection): semantic similarity to the case's query
(all-MiniLM-L6-v2, top-K), same failure shape (hard negative at
`dense_rank <= SHAPE_RANK`, positive at `dense_rank >= SHAPE_POS_GAP`), and
same `failure_type`. `sentence-transformers` is a build dependency, not
runtime.

**Forbidden**: feeding case ROWS (from `regression_cases.yaml`) into training.
Case rows are exam questions. Only mined case_sets (B+C isolated) may enter
training, and only via `--case-sets`.

**Why**: the closed-loop "textbook path" attacks pending cases with same-pattern
mined samples while keeping the cases themselves as exam questions. B+C
isolation ensures the mined samples don't leak the exam answers.

**Pipeline-validation caveat**: step-2 attack results under heuristic labels
are pipeline-validation grade, not benchmark. They test whether the mechanics
work, not whether the attack credibly moves a pending case.

## Scenario: production-case repair entrypoint

### 1. Scope / Trigger
- Trigger: new user-facing compile/repair/promote command path that intentionally
  lets current production case rows enter training under strict `--reckless`
  acceptance.

### 2. Signatures
```bash
python3 skills/heuriboost-rag/scripts/compile_cases.py \
  --base-dataset <base_dataset.csv|jsonl> \
  --production-cases <production_cases.csv|jsonl> \
  --output-dir <dir> [--strict] [--resplit]

python3 skills/heuriboost-rag/scripts/repair_reranker.py \
  --base-dataset <base_dataset.csv|jsonl> \
  --production-cases <production_cases.csv|jsonl> \
  --output-dir <dir> \
  --reckless [--acceptance-level full|weak] [--reset-anchor]

python3 skills/heuriboost-rag/scripts/promote_repair.py --output-dir <dir>
```

### 3. Contracts
- User-facing base dataset minimal columns are `query,text,relevance`.
  Recommended columns are
  `domain,query_id,query,doc_id,text,relevance,split,rank,score`.
- User-facing production cases minimal columns are
  `query,shown_doc_text,user_verdict`. Recommended columns are
  `domain,case_id,query,shown_doc_id,shown_doc_text,user_verdict,rank,score`.
- Missing `domain` compiles to `default`. Domain is a hard boundary for
  synthetic ids, candidate completion, promoted repair memory, gates, and
  touched-domain acceptance.
- Missing ids are generated from stable hashes and preserved source ids remain
  available for audit.
- If `base_dataset.split` exists, it is respected. Auto-split only happens when
  split is absent or `--resplit` is explicit.
- Compiled internal artifacts are generated under
  `<output>/.heuriboost/compiled/` and are audit/debug output, not user-authored
  prerequisites.
- `repair_reranker.py --reckless` trains one user-visible candidate model using
  base train rows, promoted repair memory, and current production repair
  samples. This is the deliberate exception to the lower-level "case rows are
  exam questions" invariant.
- Base test remains the metric-level regression suite. Production cases and
  historical gates are not appended to base test.
- Missing repair anchor auto-initializes from a base-dataset-only baseline.
  Existing anchors are never overwritten unless `--reset-anchor` is explicit.
- Full acceptance is default and promotion-eligible only when current cases,
  historical gates, global test improvement, and touched-domain non-regression
  all pass.
- Weak acceptance is explicit, supports bad-only suppression, and is never
  promotion eligible.

### 4. Validation & Error Matrix
- production case domain absent from base dataset -> hard fail
- same doc marked both good and bad in one case -> hard fail
- `production_cases` contains only unknown verdicts -> hard fail
- full acceptance case has no good target -> hard fail
- strict validation/test query has fewer than two candidate docs -> hard fail
- strict global test has fewer than required query groups -> hard fail
- touched domain has no anchor metrics -> hard fail
- current case, historical gate, global test, or touched-domain check fails ->
  hard fail from `repair_reranker.py`
- synthetic ids, auto-split, one-doc query outside strict sufficiency, and
  bad-only case -> warning

### 5. Good/Base/Bad Cases
- Good: `base_dataset` includes stable train/validation/test coverage, a full
  production case has good+bad evidence, strict repair passes, and
  `promote_repair.py` freezes the case as a gate.
- Base: lower-level `train_reranker.py --case-sets` and
  `eval_reranker.py --reckless` behavior remains unchanged.
- Bad: weak bad-only repair suppresses the bad doc but `promote_repair.py`
  refuses because `promotion_eligible=false`.

### 6. Tests Required
- `python3 -m py_compile skills/heuriboost-rag/scripts/*.py`
- `compile_cases.py --strict` on the committed repair fixture.
- `compile_cases.py` with no split column to prove deterministic auto-split.
- `repair_reranker.py --reckless` hard-fail path where case/global acceptance
  fails.
- `repair_reranker.py --reckless` success path with a controlled low anchor.
- `promote_repair.py` success on an eligible full run and refusal on weak or
  non-eligible runs.
- Verify `examples/fiqa/ledger.json` is unchanged by the production repair
  fixture smoke tests.

### 7. Wrong vs Correct
#### Wrong
Ask users to maintain `query_doc_examples.csv`, `regression_cases.yaml`, and
`case_sets/` by hand for online production failures.

#### Correct
Users provide `base_dataset` and `production_cases`; the compiler emits the
internal files, repair trains exactly one candidate model, and promotion is an
explicit operation that updates only repair state (`current_model`, anchors,
gates, promoted repair memory), not user input files or online deployment.
