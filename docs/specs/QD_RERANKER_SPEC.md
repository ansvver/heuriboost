# Q-D Reranker Spec

Generated: 2026-06-26
Status: Draft
Scope: RAG retrieval module reranking, query-document relevance learning, heuristic feature discovery

## 1. One-Line Goal

Build a query-document reranker for RAG that learns which recalled documents can actually support the answer, not merely which documents look semantically similar.

```text
Retriever: high recall, do not miss candidates.
Q-D Reranker: reorder recalled candidates so answer-supporting evidence rises and misleading hard negatives fall.
Generator: consume the final top-k evidence set.
```

## 2. Core Principles

1. Candidate snapshots are fixed during training and evaluation.
   Reranker experiments must compare rankings over the same recalled candidate set. Otherwise changes in retrieval pollute reranker evaluation.

2. Train, validation, regression, and test sets are hard-isolated.
   Regression cases are gate tests, not training material.

3. Features must be online-computable before generation.
   Answer citation, LLM post-hoc support judgment, user click, and human labels are labels or diagnostics, never model input features.

4. Optimize for answer evidence, not topical similarity.
   A good document must support the answer. Same topic but wrong time, wrong entity, wrong version, or no evidence is a negative.

5. New features are experiments.
   Every candidate feature needs ablation, slice evaluation, latency checks, leakage checks, and a promote/reject decision.

6. Every fixed failure becomes a regression case.
   The system should not relearn the same bug every two weeks.

## 3. System Overview

```text
Online path
---------
query
  -> query analyzer
  -> dense/sparse/hybrid retriever topN
  -> candidate snapshot
  -> feature extractor L0-L2
  -> XGBoost LambdaMART reranker
  -> optional diversity pass
  -> topK evidence documents
  -> answer generator
  -> feedback/judgment logs

Offline path
----------
logs + labels + failure cases
  -> dataset builder
  -> train/validation/test/regression split
  -> feature extraction
  -> ranking model training
  -> offline evaluation
  -> regression gate
  -> shadow evaluation
  -> A/B rollout
```

## 4. Non-Goals For V0

- Do not replace the first-stage retriever.
- Do not use LLM scoring in the online reranking hot path.
- Do not train on regression cases directly.
- Do not allow unstructured feature code to grow without a feature registry.
- Do not optimize only global nDCG while ignoring query slices and historical failures.

## 5. Data Model

### 5.1 QueryDocExample

One row represents one query and one candidate document/chunk.

```json
{
  "example_id": "q_001::doc_123::chunk_004",
  "query_id": "q_001",
  "query_text": "2024 Q3 毛利率下降原因是什么？",
  "session_id": "s_789",
  "candidate_doc_id": "doc_123",
  "candidate_chunk_id": "chunk_004",
  "candidate_doc_version": "v2026-06-20",
  "corpus_snapshot_id": "corp_2026_06_26_0001",
  "retrieval_timestamp": "2026-06-26T10:00:00Z",
  "candidate_rank_dense": 3,
  "candidate_rank_sparse": 8,
  "candidate_score_dense": 0.812,
  "candidate_score_sparse": 14.2,
  "candidate_text_ref": "s3://bucket/chunks/doc_123/chunk_004.txt",
  "query_type": ["time_sensitive", "numeric", "cause"],
  "label": 3,
  "label_source": ["human", "answer_citation", "llm_judge"],
  "split": "train"
}
```

Label scale:

```text
3 = directly supports the answer
2 = useful partial evidence
1 = topically related but weak or incomplete
0 = irrelevant or misleading
-1 = explicit hard negative, must not rank high
```

### 5.2 CandidateSnapshot

Candidate snapshots preserve the retriever output used by a reranker experiment.

```json
{
  "query_id": "q_001",
  "corpus_snapshot_id": "corp_2026_06_26_0001",
  "retriever_config_id": "hybrid_v3",
  "candidates": [
    {
      "doc_id": "doc_123",
      "chunk_id": "chunk_004",
      "doc_version": "v2026-06-20",
      "dense_rank": 3,
      "dense_score": 0.812,
      "sparse_rank": 8,
      "sparse_score": 14.2
    }
  ]
}
```

Rule: evaluation must reuse the same `CandidateSnapshot` for baseline and candidate model.

### 5.3 RegressionCase

Regression cases are historical failures and must not enter the training set.

```json
{
  "case_id": "reg_001",
  "query": "2024 Q3 毛利率下降原因是什么？",
  "query_type": ["time_sensitive", "numeric", "cause"],
  "corpus_snapshot_id": "corp_2026_06_26_0001",
  "must_include_doc_ids": ["doc_123"],
  "acceptable_doc_ids": ["doc_456"],
  "must_not_include_doc_ids": ["doc_old_2023"],
  "failure_type": "temporal_hard_negative",
  "expected_evidence": ["2024", "Q3", "毛利率", "原材料成本"],
  "notes": "旧年份文档主题相似，但不能作为答案来源。"
}
```

### 5.4 FeatureRecipe

All generated features must be declared in a registry/DSL.

```yaml
name: temporal_metric_match
version: 1
description: Matches query time expression and metric expression against document text.
inputs:
  - query_text
  - candidate_text
  - query_entities
  - doc_entities
expr: date_overlap(query, doc) * metric_overlap(query, doc)
type: numeric
default_value: 0.0
cost_tier: L1
online_safe: true
leakage_risk: low
expected_slices:
  - time_sensitive
  - numeric
owner: reranker
```

### 5.5 Label Construction

Labels can come from multiple sources. Treat them as different confidence levels, not as equally clean truth.

Recommended priority:

```text
human_label
  Highest confidence. Use for golden queries, regression cases, and audit samples.

answer_citation_label
  High confidence if citation is explicit and the cited chunk actually supports the generated claim.

llm_judge_label
  Medium confidence. Use as scalable supervision, but calibrate against human labels.

user_behavior_label
  Medium to low confidence. Clicks, expands, copies, dwell time, thumbs up/down are useful but biased by exposure.

retrieval_pseudo_label
  Low confidence. Use mainly for bootstrapping and hard-negative mining.
```

Conflict handling:

```text
human_label overrides all other labels.
human must_not_include overrides citation/user behavior.
LLM judge disagreement with citation sends the example to audit or lowers confidence.
User click on a known bad document does not make it positive.
```

Each row should carry label metadata:

```json
{
  "label": 3,
  "label_confidence": 0.95,
  "label_sources": ["human", "answer_citation"],
  "label_rationale": "Chunk directly states Q3 gross margin decline and cost cause.",
  "exposure_position": 4,
  "was_exploration_candidate": false
}
```

Important: label fields are never online model features.

## 6. Dataset Split Rules

Recommended datasets:

```text
train_set       -> model fitting and feature selection
validation_set  -> early stopping, hyperparameter selection, feature promotion
regression_set  -> historical failure gate only
test_set        -> cold holdout, low-frequency final evaluation
```

Hard rules:

- Same `query_id` cannot cross splits.
- Same user session should not cross splits when session behavior is part of label construction.
- Same manually repaired failure family should not be split across train and regression.
- Same candidate snapshot must be reused inside one evaluation run.
- Regression cases are immutable gate examples unless explicitly retired.

Optional aging policy:

```text
Recent data -> validation/shadow emphasis.
Stable historical failures -> regression set.
Old broad logs -> train if labels are still valid.
```

### 6.1 Regression Set Lifecycle

Regression cases should be small, hard, and stable.

Creation triggers:

```text
production bad answer caused by bad rerank
must-include evidence dropped below topK
must-not-include document ranked into topK
manual investigation fixed a failure mode
high-value golden query regressed
```

Lifecycle:

```text
new_case
  -> active regression gate
  -> periodically reviewed for corpus/version validity
  -> either retained, retired, or generalized
```

Do not move raw regression cases into training by default. If the set grows too large, generalize the pattern instead:

```text
Raw case:
  query asks 2024 Q3 margin reason
  bad doc is 2023 Q3 margin report

Generalized training generator:
  for time_sensitive queries, create hard negatives from same topic but wrong year/quarter/version
```

This teaches the pattern without training on the exact exam question.

## 7. Feature Taxonomy

### 7.1 L0 Retrieval Features

Cheap features already available from retrievers.

```text
dense_score
sparse_score
dense_rank
sparse_rank
rrf_score
retriever_overlap_rank_delta
candidate_count_for_query
```

### 7.2 Query Features

```text
query_length_chars
query_length_terms
query_has_date
query_has_number
query_has_entity
query_has_comparison
query_has_current_intent
query_type_definition
query_type_how_to
query_type_numeric
query_type_time_sensitive
query_type_troubleshooting
query_type_policy
query_type_code
```

### 7.3 Document/Chunk Features

```text
chunk_length_chars
chunk_length_terms
chunk_position
doc_age_days
source_type
section_depth
title_length
has_table
has_code
has_list
has_heading
information_density
```

### 7.4 Query-Document Lexical Features

```text
exact_term_overlap_ratio
weighted_term_overlap
title_query_overlap
heading_query_overlap
important_term_overlap
bm25_query_doc_score
rare_term_overlap
```

### 7.5 Evidence Features

These capture answer-supporting evidence.

```text
entity_overlap_count
date_overlap_count
date_granularity_match
quarter_overlap
number_overlap_count
metric_overlap_count
unit_overlap_count
cause_word_overlap
procedure_marker_count
definition_marker_count
citation_like_text_present
```

### 7.6 Anti-Features

These downrank misleading or low-value documents.

```text
boilerplate_score
navigation_score
template_score
duplicate_score
too_short_flag
too_long_flag
wrong_date_flag
wrong_entity_flag
low_information_density_flag
conflicting_version_flag
```

### 7.7 Candidate-Context Features

Features comparing a candidate with other candidates for the same query.

```text
score_gap_to_top_dense
score_gap_to_top_sparse
semantic_redundancy_with_higher_ranked_docs
source_diversity_score
same_doc_as_higher_ranked_candidate
best_chunk_in_doc_flag
```

Use these carefully. They require group-level feature extraction at rerank time.

## 8. Initial V0 Feature Set

Start with 20 to 40 stable features.

```text
dense_score
sparse_score
dense_rank
sparse_rank
rrf_score
query_length_terms
chunk_length_terms
chunk_position
doc_age_days
source_type
exact_term_overlap_ratio
weighted_term_overlap
title_query_overlap
heading_query_overlap
entity_overlap_count
date_overlap_count
number_overlap_count
has_table
has_code
has_list
duplicate_score
boilerplate_score
navigation_score
wrong_date_flag
low_information_density_flag
```

Do not begin with hundreds of features. Add new features through the discovery pipeline.

## 9. Hard Negative Types

Hard negatives should be explicitly tagged.

```text
semantic_hard_negative
  Dense similarity is high, but the document cannot answer the query.

lexical_hard_negative
  Keyword match is high, but context is wrong.

temporal_hard_negative
  Topic matches, time/version does not.

entity_hard_negative
  Same or similar name, wrong entity.

format_hard_negative
  Navigation page, template, table of contents, or boilerplate.

duplicate_negative
  Duplicate or near-duplicate chunk with weaker evidence.

stale_version_negative
  Old policy/spec/product version that should not answer current query.
```

Training should include random negatives and tagged hard negatives. Validation and regression should report metrics by negative type.

## 10. Model Training

Recommended first model:

```yaml
model:
  type: xgboost_ranker
  objective: rank:ndcg
  eval_metric:
    - ndcg@5
    - ndcg@10
  group_key: query_id
  tree_method: hist
  early_stopping_rounds: 100
```

Important:

- Group rows by `query_id`.
- Do not shuffle query-doc pairs across groups.
- Use validation groups for early stopping.
- Preserve feature names and recipe versions in the model artifact.
- Save model, feature config, training data snapshot ID, and evaluation report together.

Baseline comparisons:

```text
baseline_0: dense retriever rank
baseline_1: sparse/BM25 rank
baseline_2: reciprocal rank fusion
baseline_3: existing production reranker, if any
candidate: XGBoost Q-D reranker
```

### 10.1 HPO Integration

The reranker may use an external hyperparameter optimization backend for XGBoost parameter search. The framework should not implement search algorithms, trial pruning, or distributed scheduling itself. Its responsibility is to make parameter search reproducible, bounded, comparable, and correctly attributed.

Responsibility split:

```text
LLM / Feature Discovery Agent
  -> proposes FeatureRecipe candidates from failures and underperforming slices

Feature Engine
  -> computes features from the registry/DSL

AutoML / HPO Engine
  -> searches XGBoost parameters under a fixed budget

Evaluator
  -> runs global, slice, hard negative, latency, and regression checks

Promotion Gate
  -> promote / reject / quarantine

Memory Store
  -> remembers feature decisions and useful parameter priors
```

The framework owns the governance and orchestration layer, not the search algorithms:

```text
ExperimentRunner
  -> owns trial lifecycle, reproducibility, seeds, budgets, and artifacts

FeatureRecipeRegistry
  -> validates feature definitions, versions, online safety, and leakage risk

DatasetSnapshotManager
  -> freezes train, validation, test, and regression snapshots

AblationPlanner
  -> constructs A/B/C/D comparisons and controls budget fairness

RegressionGate
  -> runs historical failure cases without training on them

SliceEvaluator
  -> reports weak query slices, hard negatives, and cost-sensitive metrics

PromotionGate
  -> applies global, slice, regression, latency, and reliability thresholds

FeatureMemoryStore
  -> records promoted, rejected, and quarantined feature ideas plus parameter priors

ReportGenerator
  -> emits experiment cards for humans and AI agents
```

The framework must expose an adapter interface instead of coupling to a specific HPO backend:

```python
class HPOEngine:
    def optimize(
        self,
        task_profile,
        feature_set,
        train_snapshot,
        valid_snapshot,
        budget,
    ) -> TrialResult:
        ...
```

Backend requirements:

```text
- Accept fixed train and validation snapshots.
- Accept a bounded search budget.
- Use deterministic seeds where supported.
- Return all tried parameter sets, scores, metrics, artifacts, and failure reasons.
- Preserve objective, eval metric, early stopping config, and feature set version.
- Support cancellation and resumability for long-running experiments.
- Never read regression cases as part of the optimization objective.
```

Search space examples for XGBoost ranking:

```text
max_depth
min_child_weight
eta / learning_rate
num_boost_round / n_estimators
subsample
colsample_bytree
colsample_bylevel
colsample_bynode
gamma
reg_alpha
reg_lambda
max_bin
objective
eval_metric
early_stopping_rounds
```

Do not run full HPO for every LLM-proposed feature. Use two stages:

```text
Scout stage
  fixed strong baseline params or small-budget HPO
  quickly reject weak features

Finalist stage
  full HPO only for shortlisted feature candidates
  then run promotion gates
```

## 11. Evaluation

### 11.1 Ranking Metrics

```text
nDCG@5
nDCG@10
MRR@10
Recall@5 for must_include_doc_ids
Recall@10 for must_include_doc_ids
BadDoc@5 for must_not_include_doc_ids
BadDoc@10 for must_not_include_doc_ids
```

### 11.2 RAG Outcome Metrics

Computed through answer generation or offline judge pipelines.

```text
answer_support_rate
citation_precision
citation_recall
unsupported_claim_rate
answer_refusal_when_no_evidence_rate
```

### 11.3 Feature Health Metrics

```text
feature_missing_rate
feature_default_rate
feature_latency_p50_ms
feature_latency_p95_ms
feature_drift_score
feature_cardinality
```

### 11.4 Slice Metrics

Always report by query type:

```text
definition
how_to
numeric
time_sensitive
comparison
troubleshooting
policy
code
entity_lookup
multi_hop
```

Also report by source type and hard negative type.

## 12. Promotion Gates

Candidate model can be promoted only if it passes all gates.

```yaml
promotion_gate:
  global:
    ndcg10_min_delta: 0.002
    mrr10_min_delta: 0.000
  regression:
    must_include_recall_at_5_drop_max: 0.000
    must_not_include_at_5_increase_max: 0.000
    golden_mrr10_drop_max: 0.000
  slices:
    max_allowed_ndcg10_drop: 0.010
    critical_slice_max_allowed_drop: 0.000
  latency:
    feature_extraction_p95_ms_max: 20
    model_inference_p95_ms_max: 5
  reliability:
    feature_missing_rate_max: 0.300
    online_safe_required: true
```

Interpretation:

- Global improvement is not enough.
- Regression failures block promotion.
- A feature that helps numeric queries but hurts policy queries should become conditional, not globally applied.

## 13. Automatic Feature Discovery

### 13.1 Discovery Inputs

```text
failed queries
positive documents
hard negative documents
query taxonomy
retriever scores/ranks
answer citations
LLM judge labels
user feedback
human annotations
```

### 13.2 Discovery Loop

```text
1. Select failure cluster.
2. Compare positive docs against hard negatives.
3. Identify discriminating difference.
4. Generate candidate FeatureRecipe in DSL.
5. Compute feature offline.
6. Run ablation:
   - baseline features
   - baseline + candidate feature
   - baseline + candidate feature group
7. Evaluate global, slice, regression, and latency.
8. Promote, reject, or quarantine.
9. Write decision to feature memory.
```

### 13.3 Feature Discovery + HPO Fairness

When combining LLM feature discovery with AutoML/HPO, compare four experiment cells:

```text
A. baseline features + baseline params
B. baseline features + tuned params
C. baseline + candidate feature + baseline params
D. baseline + candidate feature + tuned params
```

Interpretation:

```text
B - A = parameter search gain
C - A = feature-only gain
D - B = candidate feature gain after tuning
D - C = tuning gain with the candidate feature
```

Promotion should primarily consider `D - B`, not `D - A`. Otherwise the system may credit a feature for gains that came from parameter tuning.

Rules:

```text
Use the same data snapshot, candidate snapshot, split, metric, and regression gate.
Run HPO under the same budget for baseline and candidate feature sets.
Do not tune repeatedly against the regression set.
Do not let HPO optimize only global nDCG while ignoring slices and hard negatives.
```

### 13.4 LLM Role

LLM may propose hypotheses, not bypass validation.

Allowed:

```text
"This query requires time precision."
"The negative document is same topic but wrong quarter."
"Try date_granularity_match or quarter_overlap."
```

Not allowed:

```text
LLM online score as the XGBoost feature for all candidates.
Unstructured Python feature code with no recipe entry.
Feature using answer-generation outputs as model input.
```

### 13.5 Feature Candidate Record

```json
{
  "feature_name": "quarter_overlap",
  "hypothesis": "Time-sensitive financial queries need quarter-level matching.",
  "source_failure_cases": ["reg_001", "reg_017"],
  "expected_slices": ["time_sensitive", "numeric"],
  "ablation_result": {
    "global_ndcg10_delta": 0.001,
    "time_sensitive_ndcg10_delta": 0.035,
    "policy_ndcg10_delta": -0.002,
    "regression_failures": 0,
    "latency_p95_ms": 1.2
  },
  "decision": "promote_conditional",
  "decision_reason": "Strong time-sensitive gain with no regression failure."
}
```

## 14. Feature Memory

Maintain a feature memory table.

```text
promoted_features.jsonl
rejected_features.jsonl
quarantined_features.jsonl
feature_ablation_runs.jsonl
```

Rejected features should include a reason so the system does not retry the same weak idea.

Example:

```json
{
  "feature_name": "doc_length_raw",
  "decision": "rejected",
  "reason": "Improved global nDCG by 0.0004 but hurt how_to slice by 0.018.",
  "date": "2026-06-26"
}
```

## 15. Online Serving Flow

```text
1. Query analyzer tags query type and extracts entities/dates/numbers.
2. Retriever returns top100 candidates with dense/sparse scores.
3. Feature extractor computes L0-L2 features for each query-doc pair.
4. XGBoost ranker scores candidates.
5. Optional diversity pass removes duplicate chunks and same-doc crowding.
6. Return top5 or top10 to generator.
7. Log ranking, features hash, model version, citations, judge labels, and user feedback.
```

Online log record:

```json
{
  "query_id": "q_live_001",
  "model_version": "qd_ranker_2026_06_26_01",
  "feature_set_version": "features_v12",
  "retriever_config_id": "hybrid_v3",
  "corpus_snapshot_id": "corp_live_2026_06_26",
  "ranked_candidates": [
    {
      "doc_id": "doc_123",
      "chunk_id": "chunk_004",
      "old_rank": 8,
      "new_rank": 1,
      "score": 1.732,
      "feature_hash": "abc123"
    }
  ],
  "served_top_k": 5
}
```

## 16. Shadow And A/B Rollout

Rollout stages:

```text
offline pass
  -> regression pass
  -> shadow traffic, no user impact
  -> limited A/B
  -> full rollout
```

Shadow metrics:

```text
rank_delta_distribution
must_include_lift_if_labels_available
answer_citation_overlap_with_old_model
LLM_judge_support_delta
latency_delta
top_doc_source_distribution_shift
```

A/B metrics:

```text
answer_helpfulness
user_followup_rate
citation_click_rate
thumbs_down_rate
unsupported_claim_rate
latency_p95
```

## 17. Exploration

To avoid feedback loops, reserve a small amount of exploration.

Recommended:

```text
Top 5: conservative production ranking.
Rank 6-10: allow limited exploration for high-uncertainty candidates.
```

Exploration candidates should be tagged so downstream labels do not overstate production confidence.

## 18. Explainability

Each reranked result should be explainable.

Minimum debug output:

```json
{
  "doc_id": "doc_123",
  "rank_change": "+7",
  "top_positive_features": [
    ["date_overlap_count", 0.31],
    ["metric_overlap_count", 0.24],
    ["title_query_overlap", 0.18]
  ],
  "top_negative_features": [
    ["duplicate_score", -0.09]
  ],
  "reason_template": "Promoted because it matches the query's quarter, metric, and title terms."
}
```

Use SHAP or model contribution APIs offline. Online explanations can be approximate and template-based.

## 19. Suggested Repository Layout

```text
qd_reranker/
  configs/
    feature_sets/
    training/
    promotion_gates/
  data/
    schemas/
  features/
    registry.py
    primitives.py
    extractors/
  labels/
    build_labels.py
    llm_judge.py
  training/
    build_dataset.py
    train_xgb_ranker.py
    evaluate.py
  regression/
    cases/
    run_regression_gate.py
  serving/
    rerank.py
    explain.py
  experiments/
    ablation.py
    feature_discovery.py
  docs/
    QD_RERANKER_SPEC.md
```

## 20. Minimal Implementation Plan

### Milestone 1: Data and Baselines

- Define `QueryDocExample`, `CandidateSnapshot`, `RegressionCase`.
- Export candidate snapshots from current retriever.
- Build dense/sparse/RRF baselines.
- Create 100 to 300 golden queries and 50 failure regression cases.

### Milestone 2: Feature Extraction

- Implement V0 feature set.
- Store feature matrix with feature set version.
- Add leakage checks and online-safe flags.
- Add feature health report.

### Milestone 3: Ranking Model

- Train XGBoost `rank:ndcg` grouped by `query_id`.
- Compare against retriever baselines.
- Add an `HPOEngine` adapter with a pluggable external backend.
- Run tuned-parameter baselines before attributing gains to new features.
- Produce global and slice metrics.
- Add regression gate.

### Milestone 4: Feature Discovery

- Mine failed query clusters.
- Generate FeatureRecipe candidates.
- Use scout-stage HPO for cheap filtering and finalist-stage HPO for shortlisted features.
- Compare A/B/C/D ablation cells: baseline features/params, tuned params, candidate feature, candidate feature plus tuned params.
- Run ablation and promotion gates.
- Maintain promoted/rejected/quarantined feature memory.

### Milestone 5: Serving

- Add online rerank endpoint/function.
- Run shadow evaluation.
- Add explanation output.
- Start limited A/B.

## 21. AI Implementation Checklist

When an AI agent implements this, follow this order:

1. Create schemas first.
2. Build dataset builder around fixed candidate snapshots.
3. Implement V0 features through a registry, not scattered code.
4. Train ranking model with `query_id` groups.
5. Add an HPO adapter by calling an existing tool, not by implementing search algorithms.
6. Add validation and regression gates before any automatic feature discovery.
7. Add feature discovery only after gates exist.
8. For new features, compare tuned baseline against tuned candidate before promotion.
9. Add shadow serving before A/B.
10. Never train on regression cases directly.
11. Never use post-generation signals as online model features.
12. Always report metrics by query slice and hard negative type.

## 22. Open Decisions

These should be decided before implementation:

- Current retriever stack: dense only, BM25 only, or hybrid?
- Target topN from retriever: 50, 100, or 200?
- Generator consumes topK: 5, 8, or 10?
- Label source priority: human > answer citation > LLM judge > user behavior?
- Which external HPO backend satisfies the V0 adapter contract?
- What HPO budget is allowed for scout-stage and finalist-stage searches?
- Latency budget for reranking path.
- Whether document chunks have stable version IDs today.
- Whether query taxonomy already exists or needs to be built.

## 23. Success Criteria

V0 is successful if:

- XGBoost reranker beats dense/sparse/RRF baselines on validation nDCG@10 and MRR@10.
- Golden query set does not regress.
- Failure regression set blocks known bad reranks.
- Feature extraction p95 latency stays within budget.
- At least one automatically discovered feature is promoted through ablation and slice gates.
- Debug output can explain why a document moved up or down.
