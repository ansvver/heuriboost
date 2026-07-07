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

**Ranking Snapshot**:
A precomputed representation of Query-Document Examples for one split, carrying
feature values, XGBoost-mapped labels, original labels, and query-group sizes.
_Avoid_: training matrix (too narrow), DMatrix (implementation-specific)

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

**Feature**:
A prediction-time signal computed from a Query-Document Example and used by the
Reranker to compare documents within a query group.
_Avoid_: heuristic (too loose), label-derived signal

**Feature Recipe**:
The reviewed metadata contract for a Feature, including its name, safe inputs,
task profile, leakage risk, and ownership.

**Feature Input Boundary**:
The prediction-time fields a Feature may observe: `query_text`, `doc_text`,
`dense_rank`, `dense_score`, `sparse_rank`, and `sparse_score`.
_Avoid_: labels, splits, IDs, regression results, ledger state, clicks, online
LLM judgments

**Candidate Feature**:
A proposed Feature that is not yet part of the shipped feature set and is
evaluated as a temporary probe before promotion.
_Avoid_: discovered feature (sounds already accepted)

**Candidate Feature Discovery**:
The manually triggered process that proposes Candidate Features from pending
Regression cases and failure analysis, ending when those candidates are ready
for human review or explicit Auto-Ablation.
_Avoid_: feature mining, 特征挖掘 (ambiguous with sample mining)

**Auto-Ablation**:
The explicit continuation from Candidate Feature Discovery that runs ablation
for each valid Candidate Feature without an intervening manual review step.
_Avoid_: auto-promotion

**Feature Disposition**:
The post-ablation decision for a Candidate Feature: `promote`, `reject`, or
`quarantine`.
_Avoid_: auto-promotion (promotion is manual)

**case_sets mining**:
The process that mines additional Query-Document Examples for pending Regression
cases to train repair rounds.
_Avoid_: Candidate Feature Discovery, feature mining, 特征挖掘

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
- A **Ranking Snapshot** is built from Query-Document Examples for exactly one
  split and preserves both mapped labels for training and original labels for
  evaluation.
- A **Regression case** references documents by ID and is evaluated against a
  reranked `top_k`; it never enters training.
- A **Feature Recipe** describes exactly one **Feature** or Candidate Feature.
- Every **Feature** and Candidate Feature must stay within the **Feature Input
  Boundary**.
- **Candidate Feature Discovery** proposes Candidate Features from pending
  Regression cases; **case_sets mining** produces training examples instead.
- A **Candidate Feature Discovery** round has one shared candidate budget across
  all pending Regression cases, not one budget per case; the current round
  budget is at most five Candidate Features.
- The Candidate Feature budget is an upper bound; invalid or duplicate
  candidates are dropped and not automatically backfilled.
- A **Candidate Feature Discovery** round succeeds only if at least one
  Candidate Feature is ready for human review or Auto-Ablation.
- **Auto-Ablation** may produce a **Feature Disposition** for every valid
  Candidate Feature immediately after discovery, but it never promotes features.
- **Auto-Ablation** does not pick a winner among promoted candidates; human
  promotion still happens one Candidate Feature at a time.
- **Auto-Ablation** is fail-fast: if one Candidate Feature cannot be ablated,
  the round stops before later candidates run.
- A **Candidate Feature Discovery** round considers all pending Regression
  cases together; it does not target a manually selected subset by default.
- A **Candidate Feature Discovery** round must not silently omit pending
  Regression cases; oversized rounds must be split explicitly.
- **Candidate Feature Discovery** does not decide whether a Candidate Feature
  is useful; ablation produces the **Feature Disposition**.
- A Candidate Feature becomes a shipped **Feature** only after ablation
  recommends `promote` and a human updates the feature set.
- Candidate Features are promoted one at a time; multiple promising candidates
  must not be merged as a batch without re-evaluation.
- An **LLM-judged label** may grade a Query-Document Example but may not
  promote that example into a Regression case.

## Example dialogue

> **Dev:** "FiQA qrels are binary — can the LLM-judged `-1`s become our
> `must_not_include` regression cases?"
> **Domain expert:** "No. LLM-judged labels can train and evaluate the
> **Reranker**, but a **Regression case** needs a trusted label. Hand-review a
> small subset for the gate; let the rest stay training labels."
>
> **Dev:** "Should `case_sets mining` create new **Candidate Features**?"
> **Domain expert:** "No. **case_sets mining** creates training examples for
> repair rounds; **Candidate Feature Discovery** creates feature probes that
> still need ablation."

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
- "feature mining" / "特征挖掘" was ambiguous between **Candidate Feature
  Discovery** and **case_sets mining** — resolved: use Candidate Feature
  Discovery for feature probes and case_sets mining for training examples.
