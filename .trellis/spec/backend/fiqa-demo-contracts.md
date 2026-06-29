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
