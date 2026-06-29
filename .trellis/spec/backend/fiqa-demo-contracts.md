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

**What**: `skills/heuriboost-rag/scripts/common.py::FEATURE_NAMES`, the dict keys
built in `extract_features()`, and the features listed in
`skills/heuriboost-rag/templates/feature_recipes.yaml` MUST be identical
(names and set).

**Why (silent failure)**: `eval_reranker.py`'s Feature Contrast block and
`feature_lookup` read features via `.get(name, 0.0)`. A name present in the
report/template but absent from `FEATURE_NAMES` does NOT raise — it silently
prints `0.0000`, producing misleading failure analysis with no error.

**Validation point**: After any feature add/remove/rename, grep that no stale
name remains in code or templates, and confirm the three locations match.

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
