# Adaptive XGBoost Heuristic Learning Spec

Generated: 2026-06-26
Status: Draft
Scope: General-purpose adaptive XGBoost framework for ranking, classification, regression, and other supervised tabular learning tasks
Derived From: Q-D Reranker Spec, generalized beyond RAG reranking

## 1. One-Line Goal

Build an adaptive XGBoost training framework that learns from labeled examples, historical failures, feature experiments, and task-specific feedback to continuously improve tabular models across multiple task types.

```text
Data snapshot -> feature recipe -> XGBoost model -> evaluation gates -> failure memory -> feature discovery -> controlled promotion
```

The framework should support:

```text
ranking        -> query-document reranking, search ranking, recommendation ordering
binary class   -> fraud detection, churn prediction, risk classification
multi-class    -> ticket routing, intent classification, document category prediction
regression     -> pricing, demand forecasting, score prediction
count/rate     -> event count prediction, frequency prediction, conversion-rate modeling
survival/time  -> duration or time-to-event modeling when XGBoost objective support is appropriate
```

## 2. Core Principles

1. Evaluation snapshots are fixed.
   Experiments must compare candidate models on the same data snapshot, feature snapshot, and task split. Otherwise metric changes cannot be attributed to the model or feature change.

2. Train, validation, regression, and test sets are hard-isolated.
   Regression cases are gate tests, not training material.

3. Features must be available at prediction time.
   Post-outcome signals, future labels, analyst corrections, user actions after prediction, and downstream decisions are labels or diagnostics, never online model features.

4. Optimize task utility, not generic accuracy.
   Each task profile must define the business or product outcome it cares about. For ranking this may be evidence quality; for classification it may be false-negative cost; for regression it may be calibrated absolute error.

5. New features are experiments.
   Every candidate feature needs ablation, slice evaluation, latency/cost checks, leakage checks, and a promote/reject/quarantine decision.

6. Every fixed failure becomes a regression case.
   Historical failures should become durable gates so the system does not relearn the same mistakes.

7. Heuristics are first-class system state.
   The system should remember which feature ideas worked, which failed, where they worked, and under what task profile.

## 3. System Overview

```text
Online / batch prediction path
------------------------------
input entity or request
  -> task profile resolver
  -> data snapshot / online feature fetch
  -> feature extractor L0-L2
  -> XGBoost model
  -> optional post-processor / decision policy
  -> prediction output
  -> feedback / outcome / diagnostic logs

Offline learning path
---------------------
logs + labels + failure cases
  -> dataset builder
  -> train / validation / regression / test split
  -> feature extraction
  -> XGBoost training
  -> task-specific evaluation
  -> regression gate
  -> shadow or backtest
  -> A/B or staged rollout
```

## 4. Task Profiles

A task profile binds data semantics, target type, objective, metrics, gates, slices, and serving behavior.

```yaml
task_profile:
  name: fraud_binary_classification
  task_type: binary_classification
  entity_key: transaction_id
  label_field: is_fraud
  prediction_type: probability
  objective: binary:logistic
  primary_metric: pr_auc
  secondary_metrics:
    - roc_auc
    - logloss
    - recall_at_fixed_precision
  decision_policy:
    type: threshold
    threshold_source: validation_cost_curve
  critical_slices:
    - high_value_transaction
    - new_merchant
    - cross_border
```

### 4.1 Supported Task Types

| Task Type | Example | XGBoost Objective | Typical Metrics |
|---|---|---|---|
| `ranking` | Q-D reranking, search ranking | `rank:ndcg`, `rank:pairwise`, `rank:map` | nDCG@k, MAP@k, MRR@k, Recall@k |
| `binary_classification` | fraud, churn, risk | `binary:logistic`, `binary:hinge` | PR-AUC, ROC-AUC, logloss, F1, recall@precision |
| `multi_classification` | intent routing, category prediction | `multi:softprob`, `multi:softmax` | accuracy, macro-F1, logloss, per-class recall |
| `regression` | price, score, demand | `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror` | RMSE, MAE, MAPE, pinball loss |
| `count` | event counts | `count:poisson` | Poisson deviance, RMSE, calibration by bucket |
| `survival` | time-to-event | `survival:cox`, `survival:aft` | concordance, AFT negative log likelihood |

The framework should not hardcode one objective. It should select the objective and metrics from the task profile.

## 5. Non-Goals For V0

- Do not attempt to become a full AutoML platform.
- Do not replace domain-specific data collection, labeling, or product decision policy.
- Do not train on regression cases directly.
- Do not allow unstructured feature code to grow without a feature registry.
- Do not optimize only one global metric while ignoring slices, historical failures, and operational constraints.
- Do not use expensive LLM or human review in the online hot path unless the task profile explicitly allows it.

## 6. Data Model

### 6.1 LearningExample

One row represents one supervised training example. For ranking, multiple rows share a group key. For classification/regression, each row usually corresponds to one entity or event.

```json
{
  "example_id": "txn_001",
  "task_profile": "fraud_binary_classification",
  "entity_id": "transaction_001",
  "group_id": null,
  "event_timestamp": "2026-06-26T10:00:00Z",
  "feature_snapshot_id": "feat_2026_06_26_0001",
  "data_snapshot_id": "data_2026_06_26_0001",
  "label": 1,
  "label_type": "binary",
  "label_sources": ["chargeback", "human_review"],
  "label_confidence": 0.98,
  "slice_tags": ["high_value_transaction", "new_merchant"],
  "weight": 4.0,
  "split": "train"
}
```

Fields:

```text
example_id
  Stable row identifier.

task_profile
  Which task configuration this example belongs to.

entity_id
  The predicted object: transaction, user, document, query-doc pair, listing, product, etc.

group_id
  Ranking group or grouped prediction context. Null for ordinary row-wise classification/regression.

event_timestamp
  Time of the prediction event. Required for temporal splits and leakage checks.

feature_snapshot_id / data_snapshot_id
  Immutable references to the data and feature material used for reproducible experiments.

label / label_type / label_sources / label_confidence
  Target and provenance.

slice_tags
  Evaluation slices. Examples: query type, customer segment, geography, product category, source type.

weight
  Optional training weight for cost-sensitive learning or exposure correction.
```

### 6.2 PredictionContextSnapshot

This generalizes `CandidateSnapshot` from Q-D reranking.

For ranking tasks, it stores candidate sets. For classification/regression, it stores the immutable feature/data context used by a training or evaluation run.

```json
{
  "snapshot_id": "ctx_2026_06_26_0001",
  "task_profile": "search_ranking",
  "context_type": "ranking_candidates",
  "group_id": "query_001",
  "created_at": "2026-06-26T10:00:00Z",
  "source_config_id": "hybrid_retriever_v3",
  "entities": [
    {
      "entity_id": "doc_123::chunk_004",
      "entity_version": "v2026-06-20",
      "source_rank": 3,
      "source_score": 0.812
    }
  ]
}
```

Classification/regression context example:

```json
{
  "snapshot_id": "ctx_2026_06_26_1000",
  "task_profile": "churn_binary_classification",
  "context_type": "row_features",
  "entity_id": "user_123",
  "event_timestamp": "2026-06-26T10:00:00Z",
  "source_config_id": "feature_pipeline_v7",
  "feature_snapshot_id": "feat_2026_06_26_0001"
}
```

Rule: evaluation must reuse the same context snapshot for baseline and candidate model.

### 6.3 RegressionCase

Regression cases are historical failures used as gates. They must not enter the training set directly.

```json
{
  "case_id": "reg_001",
  "task_profile": "fraud_binary_classification",
  "entity_id": "transaction_987",
  "data_snapshot_id": "data_2026_06_26_0001",
  "expected_behavior": {
    "type": "classification",
    "positive_label": 1,
    "min_score": 0.85,
    "must_not_predict_label": 0
  },
  "failure_type": "false_negative_high_value",
  "slice_tags": ["high_value_transaction", "new_merchant"],
  "notes": "Prior model scored a confirmed fraud transaction below manual review threshold."
}
```

Ranking regression case example:

```json
{
  "case_id": "reg_rank_001",
  "task_profile": "qd_reranking",
  "group_id": "query_001",
  "must_include_entity_ids": ["doc_123::chunk_004"],
  "acceptable_entity_ids": ["doc_456::chunk_009"],
  "must_not_include_entity_ids": ["doc_old_2023::chunk_002"],
  "failure_type": "temporal_hard_negative"
}
```

Regression cases should express expected behavior in task-specific terms:

```text
ranking        -> must_include / must_not_include / rank threshold
classification -> expected class, score threshold, false-positive/false-negative guard
regression     -> acceptable prediction range or max absolute error
count/rate     -> acceptable bucket, calibration bucket, max relative error
```

### 6.4 FeatureRecipe

All generated features must be declared in a registry or DSL.

```yaml
name: merchant_velocity_24h
version: 1
description: Counts prior transactions for the same merchant in the last 24 hours.
task_profiles:
  - fraud_binary_classification
inputs:
  - entity_id
  - event_timestamp
  - transaction_history
expr: count_events(entity.merchant_id, window="24h", before=event_timestamp)
type: numeric
default_value: 0.0
cost_tier: L1
online_safe: true
leakage_risk: medium
expected_slices:
  - new_merchant
  - high_velocity_merchant
owner: risk_modeling
```

Required fields:

```text
name
version
description
task_profiles
inputs
expr or implementation reference
type
default_value
cost_tier
online_safe
leakage_risk
expected_slices
owner
```

## 7. Label Construction

Labels can come from multiple sources. Treat them as different confidence levels, not as equally clean truth.

Recommended source classes:

```text
human_label
  Highest confidence. Use for golden sets, regression cases, and audit samples.

verified_outcome
  High confidence. Examples: chargeback, cancellation, purchase, default, delivered package, resolved ticket.

downstream_decision
  Medium confidence. Examples: user clicked, agent escalated, document cited, reviewer selected category.

llm_judge_label
  Medium confidence. Useful for scalable supervision, but must be calibrated against human labels.

pseudo_label
  Low confidence. Use mainly for bootstrapping, weak supervision, and hard-negative mining.
```

Conflict handling:

```text
human_label overrides all other labels.
verified_outcome overrides behavioral proxies.
known regression expected behavior overrides weak labels.
LLM judge disagreement with verified labels sends the example to audit or lowers confidence.
User behavior on an exposed but wrong prediction does not make the prediction correct.
```

Each row should carry label metadata:

```json
{
  "label": 1,
  "label_type": "binary",
  "label_confidence": 0.98,
  "label_sources": ["chargeback", "human_review"],
  "label_rationale": "Confirmed chargeback and manual fraud review.",
  "exposure_position": null,
  "was_exploration_candidate": false
}
```

Important: label fields are never online model features.

## 8. Dataset Split Rules

Recommended datasets:

```text
train_set       -> model fitting and feature selection
validation_set  -> early stopping, hyperparameter selection, feature promotion
regression_set  -> historical failure gate only
test_set        -> cold holdout, low-frequency final evaluation
```

Hard rules:

- Same `entity_id` should not cross splits when entity history creates leakage risk.
- Same `group_id` cannot cross splits for ranking tasks.
- Same user/session/account should not cross splits when behavioral labels or entity history are used.
- Temporal tasks should split by event time, not random row shuffle.
- Same manually repaired failure family should not be split across train and regression.
- Same context snapshot must be reused inside one evaluation run.
- Regression cases are immutable gate examples unless explicitly retired.

Optional aging policy:

```text
Recent labeled data -> validation, shadow, and backtest emphasis.
Stable historical failures -> regression set.
Old broad logs -> train if labels and feature definitions are still valid.
```

### 8.1 Regression Set Lifecycle

Regression cases should be small, hard, and stable.

Creation triggers:

```text
production incident caused by model prediction
critical false positive or false negative
ranking must-include item dropped below topK
ranking must-not-include item entered topK
regression prediction outside acceptable range
manual investigation fixed a failure mode
high-value golden case regressed
```

Lifecycle:

```text
new_case
  -> active regression gate
  -> periodically reviewed for schema/data validity
  -> either retained, retired, or generalized
```

Do not move raw regression cases into training by default. If the set grows too large, generalize the pattern instead.

Example:

```text
Raw case:
  high-value fraudulent transaction was scored low because merchant velocity was missing

Generalized training generator:
  for high-value transactions, create hard positives with sparse merchant history and high velocity indicators
```

This teaches the pattern without training on the exact exam question.

## 9. Feature Taxonomy

Feature taxonomy should be task-agnostic at the framework level and task-specific at the recipe level.

### 9.1 L0 Source Features

Cheap features already available from upstream systems.

```text
source_score
source_rank
retriever_score
baseline_model_score
raw_numeric_field
raw_categorical_field
existing_rule_score
candidate_count_for_group
```

### 9.2 Entity Features

Describe the predicted object.

```text
entity_age
entity_type
entity_status
source_type
category
region
device_type
account_tenure
document_length
transaction_amount
```

### 9.3 Context Features

Describe the prediction context.

```text
event_hour
event_day_of_week
seasonality_bucket
request_channel
query_type
market_condition
traffic_source
current_intent
```

### 9.4 Interaction Features

Describe relationships between entities, context, and candidates.

```text
query_document_overlap
user_item_affinity
merchant_velocity
account_device_match
price_to_category_median_ratio
current_value_vs_historical_average
entity_context_compatibility
```

### 9.5 Temporal / Window Features

```text
count_1h
count_24h
count_7d
rolling_mean_7d
rolling_std_30d
days_since_last_event
trend_slope
seasonal_deviation
```

### 9.6 Evidence / Quality Features

These capture whether a candidate, entity, or observation contains the information needed for the task.

```text
has_required_evidence
entity_overlap_count
date_overlap_count
number_overlap_count
field_completeness
information_density
measurement_quality
source_reliability
freshness_score
```

### 9.7 Anti-Features

These downweight misleading or low-quality examples/candidates.

```text
duplicate_score
boilerplate_score
stale_version_flag
wrong_entity_flag
missing_required_field_flag
low_information_density_flag
outlier_flag
known_bad_source_flag
conflicting_signal_flag
```

### 9.8 Candidate-Context Features For Ranking

Only applicable when examples compete inside a group.

```text
score_gap_to_top
semantic_redundancy_with_higher_ranked_items
source_diversity_score
same_parent_as_higher_ranked_candidate
best_candidate_in_parent_flag
```

## 10. Initial V0 Feature Set

Start with 20 to 40 stable features per task profile.

Generic V0 feature categories:

```text
L0 upstream scores and ranks
raw numeric fields
raw categorical fields with safe encoding
entity age and status
context timestamp features
missing indicators
basic count/window features
basic interaction ratios
source reliability/freshness
duplicate or stale flags
```

Ranking-specific V0 example:

```text
dense_score
sparse_score
dense_rank
sparse_rank
rrf_score
exact_term_overlap_ratio
entity_overlap_count
date_overlap_count
duplicate_score
wrong_date_flag
```

Binary-classification V0 example:

```text
baseline_rule_score
amount
account_age_days
event_hour
merchant_country
device_country
country_mismatch_flag
merchant_velocity_24h
user_velocity_24h
days_since_last_transaction
missing_device_flag
```

Regression V0 example:

```text
baseline_prediction
entity_age
category
region
seasonality_bucket
rolling_mean_7d
rolling_mean_30d
trend_slope
price_to_category_median_ratio
missing_history_flag
```

Do not begin with hundreds of features. Add new features through the discovery pipeline.

## 11. Hard Example Types

Each task should define hard examples. They are the cases that force the model to learn the important distinction.

### 11.1 Ranking

```text
semantic_hard_negative
lexical_hard_negative
temporal_hard_negative
entity_hard_negative
duplicate_negative
stale_version_negative
```

### 11.2 Classification

```text
hard_positive
  Positive label that baseline model scores low.

hard_negative
  Negative label that baseline model scores high.

near_boundary_case
  Example close to the decision threshold.

rare_class_case
  Underrepresented class or segment.

costly_false_negative
  Positive example where missing it is expensive.

costly_false_positive
  Negative example where flagging it is expensive.
```

### 11.3 Regression

```text
large_residual_case
  Prediction error is large.

tail_value_case
  Target is in a high or low tail.

segment_bias_case
  A slice has systematic overprediction or underprediction.

drift_case
  Recent data differs from training distribution.

calibration_bucket_failure
  Predicted bucket does not match observed distribution.
```

Training should include random examples and tagged hard examples. Validation and regression should report metrics by hard example type.

## 12. Model Training

Training config is selected by task profile.

### 12.1 Generic Training Config

```yaml
model:
  type: xgboost
  task_profile: fraud_binary_classification
  objective: binary:logistic
  eval_metric:
    - aucpr
    - logloss
  tree_method: hist
  early_stopping_rounds: 100
  feature_set_version: features_v12
  training_snapshot_id: train_2026_06_26_0001
```

### 12.2 Ranking Config

```yaml
model:
  type: xgboost_ranker
  objective: rank:ndcg
  eval_metric:
    - ndcg@5
    - ndcg@10
  group_key: group_id
  tree_method: hist
  early_stopping_rounds: 100
```

### 12.3 Classification Config

```yaml
model:
  type: xgboost_classifier
  objective: binary:logistic
  eval_metric:
    - aucpr
    - logloss
  class_weighting:
    mode: scale_pos_weight
  threshold_selection:
    metric: recall_at_precision
    min_precision: 0.95
```

### 12.4 Regression Config

```yaml
model:
  type: xgboost_regressor
  objective: reg:squarederror
  eval_metric:
    - rmse
    - mae
  target_transform:
    type: log1p
    inverse: expm1
```

Important:

- Preserve feature names and recipe versions in the model artifact.
- Save model, task profile, feature config, training data snapshot ID, and evaluation report together.
- Ranking tasks must group rows by `group_id`.
- Classification tasks must record threshold policy separately from model score.
- Regression tasks must record target transforms and inverse transforms.
- Temporal tasks should validate on future data relative to training.

### 12.5 HPO Integration

The framework may use an external hyperparameter optimization backend for XGBoost parameter search. It should not implement search algorithms, trial pruning, or distributed scheduling itself. Its responsibility is to make parameter search reproducible, bounded, comparable, and correctly attributed.

Responsibility split:

```text
LLM / Feature Discovery Agent
  -> proposes FeatureRecipe candidates from failures, residuals, hard examples, and weak slices

Feature Engine
  -> computes features from the registry/DSL

AutoML / HPO Engine
  -> searches XGBoost parameters under a fixed budget

Evaluator
  -> runs global, slice, hard example, feature health, latency, and regression checks

Promotion Gate
  -> promote / reject / quarantine

Memory Store
  -> remembers feature decisions and useful parameter priors per task profile
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
  -> reports weak slices, hard examples, and task-specific cost metrics

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

Common XGBoost search dimensions:

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
scale_pos_weight
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

## 13. Evaluation

### 13.1 Ranking Metrics

```text
nDCG@5
nDCG@10
MAP@k
MRR@10
Recall@k for must_include_entity_ids
BadEntity@k for must_not_include_entity_ids
```

### 13.2 Classification Metrics

```text
PR-AUC
ROC-AUC
logloss
F1
precision@threshold
recall@threshold
recall_at_fixed_precision
precision_at_fixed_recall
false_positive_rate by slice
false_negative_rate by slice
calibration error
```

### 13.3 Multi-Class Metrics

```text
accuracy
macro_f1
micro_f1
per_class_precision
per_class_recall
confusion_matrix
top_k_accuracy
logloss
```

### 13.4 Regression Metrics

```text
RMSE
MAE
MAPE
SMAPE
median_absolute_error
pinball_loss
residual_bias_by_slice
prediction_interval_coverage
```

### 13.5 Feature Health Metrics

```text
feature_missing_rate
feature_default_rate
feature_latency_p50_ms
feature_latency_p95_ms
feature_drift_score
feature_cardinality
feature_outlier_rate
training_serving_skew
```

### 13.6 Slice Metrics

Every task profile must declare slices.

Examples:

```text
ranking        -> query_type, source_type, hard_negative_type
fraud          -> transaction_amount_bucket, merchant_age, country_pair, device_type
churn          -> plan_type, tenure_bucket, usage_level, region
regression     -> target_bucket, category, geography, seasonality_bucket
multi-class    -> class_label, source_channel, language, segment
```

## 14. Promotion Gates

Candidate model can be promoted only if it passes task-specific gates.

### 14.1 Generic Gate Template

```yaml
promotion_gate:
  global:
    primary_metric_min_delta: 0.002
    secondary_metric_max_drop: 0.000
  regression:
    critical_case_failures_max: 0
    golden_set_metric_drop_max: 0.000
  slices:
    max_allowed_primary_metric_drop: 0.010
    critical_slice_max_allowed_drop: 0.000
  latency:
    feature_extraction_p95_ms_max: 20
    model_inference_p95_ms_max: 5
  reliability:
    feature_missing_rate_max: 0.300
    training_serving_skew_max: 0.010
    online_safe_required: true
```

### 14.2 Ranking Gate Example

```yaml
promotion_gate:
  global:
    ndcg10_min_delta: 0.002
    mrr10_min_delta: 0.000
  regression:
    must_include_recall_at_5_drop_max: 0.000
    must_not_include_at_5_increase_max: 0.000
```

### 14.3 Classification Gate Example

```yaml
promotion_gate:
  global:
    pr_auc_min_delta: 0.003
    logloss_max_increase: 0.000
  operating_point:
    recall_at_precision_95_min_delta: 0.000
  regression:
    critical_false_negative_increase_max: 0
    critical_false_positive_increase_max: 0
  slices:
    max_allowed_recall_drop: 0.010
    critical_slice_recall_drop_max: 0.000
```

### 14.4 Regression Gate Example

```yaml
promotion_gate:
  global:
    mae_max_delta: -0.002
    rmse_max_delta: 0.000
  regression:
    max_abs_error_case_failures: 0
  slices:
    max_allowed_mae_increase: 0.010
    residual_bias_abs_max: 0.020
```

Interpretation:

- Global improvement is not enough.
- Regression failures block promotion.
- A feature that helps one slice but hurts a critical slice should become conditional, not globally applied.

## 15. Automatic Feature Discovery

### 15.1 Discovery Inputs

```text
failed cases
positive and negative examples
large residual cases
near-threshold cases
slice-level regressions
task taxonomy
upstream scores/ranks/rules
verified outcomes
LLM judge labels where allowed
user or operator feedback
human annotations
```

### 15.2 Discovery Loop

```text
1. Select failure cluster or underperforming slice.
2. Compare correctly handled examples against failures.
3. Identify discriminating difference.
4. Generate candidate FeatureRecipe in DSL.
5. Compute feature offline.
6. Run ablation:
   - baseline features
   - baseline + candidate feature
   - baseline + candidate feature group
7. Evaluate global, slice, regression, feature health, and latency.
8. Promote, reject, or quarantine.
9. Write decision to feature memory.
```

### 15.3 Feature Discovery + HPO Fairness

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
Use the same data snapshot, feature snapshot, split, metric, and regression gate.
Run HPO under the same budget for baseline and candidate feature sets.
Do not tune repeatedly against the regression set.
Do not let HPO optimize only one global metric while ignoring slices and hard examples.
```

### 15.4 LLM Role

LLM may propose hypotheses, not bypass validation.

Allowed:

```text
"This failure cluster needs a recency feature."
"The model misses high-value positives with sparse history."
"Try merchant_velocity_24h or missing_history_flag."
```

Not allowed:

```text
LLM online score as a feature for every prediction unless explicitly budgeted and gated.
Unstructured feature code with no recipe entry.
Feature using future outcomes or post-decision signals as model input.
```

### 15.5 Feature Candidate Record

```json
{
  "feature_name": "merchant_velocity_24h",
  "task_profile": "fraud_binary_classification",
  "hypothesis": "High-value fraud misses often have abnormal merchant velocity.",
  "source_failure_cases": ["reg_001", "reg_017"],
  "expected_slices": ["high_value_transaction", "new_merchant"],
  "ablation_result": {
    "global_pr_auc_delta": 0.004,
    "high_value_recall_delta": 0.021,
    "false_positive_rate_delta": 0.002,
    "regression_failures": 0,
    "latency_p95_ms": 1.4
  },
  "decision": "promote_conditional",
  "decision_reason": "Strong high-value recall gain with no regression failure."
}
```

## 16. Feature Memory

Maintain a feature memory table by task profile.

```text
promoted_features.jsonl
rejected_features.jsonl
quarantined_features.jsonl
feature_ablation_runs.jsonl
task_profile_feature_summary.jsonl
```

Rejected features should include a reason so the system does not retry the same weak idea.

Example:

```json
{
  "feature_name": "entity_age_raw",
  "task_profile": "churn_binary_classification",
  "decision": "rejected",
  "reason": "Improved global PR-AUC by 0.0003 but hurt new_user recall by 0.018.",
  "date": "2026-06-26"
}
```

## 17. Serving Flow

```text
1. Resolve task profile.
2. Fetch online-safe features or batch feature snapshot.
3. Apply FeatureRecipe versions used by the model.
4. Score with XGBoost model.
5. Apply optional post-processor:
   - ranking diversity
   - classification threshold
   - regression clipping/calibration
   - decision policy
6. Return prediction.
7. Log model version, feature hash, prediction, decision, later outcome, and diagnostics.
```

Online/batch log record:

```json
{
  "prediction_id": "pred_001",
  "task_profile": "fraud_binary_classification",
  "model_version": "fraud_xgb_2026_06_26_01",
  "feature_set_version": "features_v12",
  "data_snapshot_id": "data_live_2026_06_26",
  "entity_id": "transaction_001",
  "prediction": {
    "score": 0.932,
    "predicted_label": 1,
    "threshold": 0.81
  },
  "decision": "manual_review",
  "feature_hash": "abc123"
}
```

## 18. Shadow, Backtest, And A/B Rollout

Rollout stages:

```text
offline pass
  -> regression pass
  -> backtest or shadow traffic
  -> limited A/B or staged batch rollout
  -> full rollout
```

Shadow/backtest metrics:

```text
prediction_delta_distribution
decision_delta_distribution
slice_metric_delta
regression_case_delta
latency_delta
feature_missing_delta
calibration_delta
```

A/B or staged rollout metrics depend on task:

```text
ranking        -> click/citation/usefulness/support rate
classification -> precision/recall at operating point, manual review load, incident rate
regression     -> realized error, calibration, business loss, over/underprediction rate
```

## 19. Exploration

To avoid feedback loops, reserve controlled exploration when task risk allows it.

Examples:

```text
ranking
  Keep top positions conservative, explore lower positions with high-uncertainty candidates.

classification
  Sample near-threshold cases for human review to improve labels.

regression
  Audit high-uncertainty or tail predictions for later label quality checks.
```

Exploration candidates should be tagged so downstream labels do not overstate production confidence.

## 20. Explainability

Each prediction should be explainable enough for debugging.

Minimum debug output:

```json
{
  "entity_id": "transaction_001",
  "prediction": 0.932,
  "top_positive_features": [
    ["merchant_velocity_24h", 0.31],
    ["country_mismatch_flag", 0.24],
    ["amount", 0.18]
  ],
  "top_negative_features": [
    ["account_age_days", -0.09]
  ],
  "reason_template": "High score because merchant velocity, country mismatch, and amount are elevated."
}
```

Use SHAP or model contribution APIs offline. Online explanations can be approximate and template-based.

## 21. Suggested Repository Layout

```text
adaptive_xgb/
  configs/
    task_profiles/
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
    weak_supervision.py
    llm_judge.py
  training/
    build_dataset.py
    train_xgb.py
    evaluate.py
  regression/
    cases/
    run_regression_gate.py
  serving/
    predict.py
    postprocess.py
    explain.py
  experiments/
    ablation.py
    feature_discovery.py
  docs/
    ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md
    QD_RERANKER_SPEC.md
```

## 22. Minimal Implementation Plan

### Milestone 1: Task Profiles And Schemas

- Define `LearningExample`, `PredictionContextSnapshot`, `RegressionCase`, `FeatureRecipe`.
- Define at least one ranking profile and one classification or regression profile.
- Implement data snapshot references and split rules.
- Build baseline metrics for each profile.

### Milestone 2: Feature Extraction

- Implement V0 feature registry.
- Store feature matrix with feature set version.
- Add leakage checks and online-safe flags.
- Add feature health report.

### Milestone 3: XGBoost Training

- Train XGBoost model using task-profile objective.
- Support ranking groups when `task_type=ranking`.
- Support threshold policy for classification.
- Support target transform metadata for regression.
- Add an `HPOEngine` adapter with a pluggable external backend.
- Run tuned-parameter baselines before attributing gains to new features.
- Produce global and slice metrics.
- Add regression gate.

### Milestone 4: Feature Discovery

- Mine failed case clusters and underperforming slices.
- Generate `FeatureRecipe` candidates.
- Use scout-stage HPO for cheap filtering and finalist-stage HPO for shortlisted features.
- Compare A/B/C/D ablation cells: baseline features/params, tuned params, candidate feature, candidate feature plus tuned params.
- Run ablation and promotion gates.
- Maintain promoted/rejected/quarantined feature memory.

### Milestone 5: Serving Or Batch Scoring

- Add prediction endpoint or batch scoring job.
- Run shadow/backtest evaluation.
- Add explanation output.
- Start limited A/B or staged rollout.

## 23. AI Implementation Checklist

When an AI agent implements this, follow this order:

1. Create task profiles first.
2. Create schemas before model code.
3. Build dataset builder around fixed snapshots.
4. Implement V0 features through a registry, not scattered code.
5. Train model using objective and metrics from task profile.
6. Add an HPO adapter by calling an existing tool, not by implementing search algorithms.
7. Add validation and regression gates before automatic feature discovery.
8. Add feature discovery only after gates exist.
9. For new features, compare tuned baseline against tuned candidate before promotion.
10. Add shadow/backtest before A/B or full batch rollout.
11. Never train on regression cases directly.
12. Never use post-outcome signals as online model features.
13. Always report metrics by slice and hard example type.

## 24. Open Decisions

These should be decided before implementation:

- Which task profile is V0: ranking, binary classification, multi-class classification, or regression?
- What is the primary metric and operating point?
- What are the critical slices?
- Which external HPO backend satisfies the V0 adapter contract?
- What HPO budget is allowed for scout-stage and finalist-stage searches?
- What labels are trusted, weak, or only diagnostic?
- What data snapshot and feature snapshot system exists today?
- What is the latency or batch runtime budget?
- Are entity IDs and versions stable?
- Does the task require temporal splitting?
- How will regression cases be created, reviewed, and retired?

## 25. Success Criteria

V0 is successful if:

- At least one task profile trains a reproducible XGBoost model from fixed snapshots.
- Candidate model beats task-specific baselines on validation primary metric.
- Golden/regression cases do not regress.
- Feature extraction p95 latency or batch runtime stays within budget.
- Metrics are reported globally, by slice, and by hard example type.
- At least one automatically discovered feature is promoted through ablation and gates.
- Debug output can explain why a prediction, rank, or score changed.

## 26. Q-D Reranking As A Task Profile

The original Q-D Reranker is a specialization of this generic framework.

```yaml
task_profile:
  name: qd_reranking
  task_type: ranking
  entity_key: query_document_pair
  group_key: query_id
  objective: rank:ndcg
  primary_metric: ndcg@10
  secondary_metrics:
    - mrr@10
    - recall@5
    - bad_entity@5
  prediction_type: ranked_list
  critical_slices:
    - time_sensitive
    - numeric
    - troubleshooting
```

Q-D-specific mappings:

```text
QueryDocExample        -> LearningExample with group_id=query_id
CandidateSnapshot      -> PredictionContextSnapshot with context_type=ranking_candidates
must_include_doc_ids   -> ranking regression expected behavior
must_not_include_doc_ids -> ranking regression expected behavior
answer citation labels -> downstream_decision or weak label source
LLM support judge      -> llm_judge_label, never online feature
```
