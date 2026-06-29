# HeuriBoost

A CSV-first, Codex-compatible agent skill for failure-driven RAG query-document
reranking. It learns a local XGBoost/LambdaMART reranker, compares it against
retriever baselines, and turns past retrieval failures into durable regression
gates.

## Language

**Reranker**:
The learned XGBoost/LambdaMART model that reorders an already-retrieved
candidate set so answer-supporting evidence rises and misleading documents fall.
It does not retrieve; it only reorders.
_Avoid_: ranker (ambiguous with first-stage retriever), model (too generic)

**Retriever**:
The first-stage system (dense, sparse/BM25, or hybrid) that produces the
candidate set HeuriBoost reranks. Out of scope for V0 — HeuriBoost never
replaces it.
_Avoid_: search, recall stage

**Baseline**:
A non-learned ordering used as the comparison floor for the reranker: dense
rank, sparse/BM25 rank, or reciprocal rank fusion (RRF). "Beats baselines"
means beating these on validation nDCG@10 / MRR@10. For the FiQA demo the
baseline is deliberately a realistic-but-beatable stack: `rank_bm25` +
`all-MiniLM-L6-v2` dense + RRF, run on CPU at build time. If the reranker fails
to beat it, demote the claim rather than swap to a weaker retriever to rig a win.

**Query-Document Example** (`QueryDocExample`):
One CSV row = one (query, candidate document/chunk) pair, carrying retriever
scores/ranks and a relevance label. Rows sharing a `query_id` form one ranking
group.
_Avoid_: row, sample, pair (use when group membership doesn't matter)

**Label scale**:
Five-level relevance: `3` directly supports, `2` partial, `1` weak, `0`
irrelevant, `-1` misleading hard negative. Mapped to non-negative ordered
relevance for XGBoost training; original labels kept for evaluation.

**Hard negative** (`-1`):
A document the retriever ranked high but that must not support the answer
(same financial topic, wrong entity/situation/version). Suppressing these is
HeuriBoost's core differentiator.
_Avoid_: negative (use for plain irrelevant `0`)

**Regression case**:
A historical failure expressed as a gate: `must_include_doc_ids` /
`must_not_include_doc_ids` within a protected `top_k`. A gate test, never a
training row. Requires trusted/human-reviewed labels, not LLM-judged ones.
_Avoid_: test case, golden (golden = the broader trusted eval set)

**LLM-judged label**:
A relevance grade assigned by an LLM at dataset *build time* and baked into the
committed CSV. Medium confidence. May feed training and evaluation; may NOT
define regression-gate `must_not_include` cases; may NEVER be an online model
feature.

**Data card**:
The provenance record for a checked-in derived dataset: source, license, the
retriever config used to produce scores, and the judge model/prompt/date used
to grade labels.

## Relationships

- A **Retriever** produces a candidate set; the **Reranker** reorders it.
- A **Query-Document Example** belongs to exactly one `query_id` group.
- A **Regression case** references documents by ID and is evaluated against a
  reranked `top_k`; it never enters training.
- An **LLM-judged label** may grade a Query-Document Example but may not
  promote that example into a Regression case.

## Example dialogue

> **Dev:** "FiQA qrels are binary — can the LLM-judged `-1`s become our
> `must_not_include` regression cases?"
> **Domain expert:** "No. LLM-judged labels can train and evaluate the
> **Reranker**, but a **Regression case** needs a trusted label. Hand-review a
> small subset for the gate; let the rest stay training labels."

**FiQA demo**:
The single committed real-dataset demo: a slice of BEIR/FiQA-2018 (~150 train /
40 val / 40 test queries, top-20 candidates, doc_text ≤400 chars), retriever
scores precomputed offline, labels LLM-judged at build time. Its job is the
honest "beats baselines on real data" claim with generic LTR + finance
features (term/number/entity/important-term overlap, low-information-density).
The earlier **toy financial demo** (hand-built, ~8 queries, temporal
`wrong_year_flag` hard-negative) was removed in the 2026-06-29 FiQA pivot; FiQA
is now the only live demo.

## Flagged ambiguities

- "V0" was used to mean both a generic adaptive-XGBoost framework and a
  QD-reranking-only CSV skill — resolved 2026-06-29: V0 is the QD-reranking CSV
  skill as shipped (README + code are truth); the generic framework is
  spec-level future vision, not V0.
- "ranking" was ambiguous between document reranking and add-on recommendation
  ranking — resolved: HeuriBoost V0 is query-document reranking only.
