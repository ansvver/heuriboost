# HeuriBoost RAG Skill V0 Design

## Boundary

V0 is a coding-agent skill plus runnable local CSV-based scripts/templates. The skill drives the workflow; the scripts provide deterministic execution and must be able to run the demo end to end.

The skill file should use a Codex-compatible `SKILL.md` structure. Other coding agents are not a first-class V0 installation target, but users can still run the Python scripts manually.

The intended repository layout is:

```text
skills/heuriboost-rag/
  SKILL.md
  templates/
    query_doc_examples.csv
    regression_cases.yaml
    feature_recipes.yaml
    promotion_gate.yaml
  scripts/
    inspect_rag_repo.py
    validate_dataset.py
    train_reranker.py
    eval_reranker.py
    build_fiqa_csv.py
  requirements.txt
  requirements-build.txt
examples/fiqa/
  query_doc_examples.csv   # committed (generated offline)
  regression_cases.yaml    # committed (hand-confirmed)
  DATA_CARD.md
  .cache/                  # gitignored build intermediates
```

The minimum runnable substrate is standalone scripts under the skill directory:

```text
validate_dataset.py
train_reranker.py
eval_reranker.py
```

V0 should not introduce a root Python package, stable public API, formal CLI, `pyproject.toml`, or publishing workflow. Users should run the scripts directly while the skill workflow is still being validated.

The root package/CLI can be introduced after the skill workflow is validated. If added later, it should expose only narrow commands that match the skill modes:

```text
heuriboost audit
heuriboost bootstrap
heuriboost experiment
```

## Data Contract

The first stable contract is a row-wise query-document CSV. Rows with the same `query_id` form one ranking group.

The model training path must preserve group boundaries and must not shuffle rows across groups when building XGBoost ranking data.

Labels use graded evidence support rather than binary relevance:

```text
3  directly supports the answer
2  partially supports the answer
1  related but weak evidence
0  irrelevant
-1 misleading hard negative
```

`doc_text` is supported as the default zero-dependency source. `doc_text_ref` may be accepted as an optional field but should not be required for the first demo.

## Workflow

### audit

Read-only project scan:

- detect likely retriever/eval/log files
- identify whether candidate snapshots already exist
- identify whether query-document labels exist
- report missing pieces without modifying the project

### bootstrap

Create a local HeuriBoost workspace:

- copy CSV/YAML templates
- create empty reports/models directories
- add a README snippet explaining the input contract
- optionally copy the FiQA demo CSV

### experiment

Run the local reranking loop:

- validate CSV columns and split values
- compute V0 lexical/entity/evidence features from query and document text
- train a real XGBoost LambdaMART model grouped by `query_id`
- compare against dense/sparse/RRF baselines when rank/score columns exist
- evaluate global metrics and hard-negative slices
- run regression gate from YAML cases
- write deterministic failure analysis lite from regression cases, rank movement,
  evidence-term hits, and V0 feature contrasts
- emit reports and model artifacts

The demo is not complete unless the scripts produce reports and a model artifact from `examples/fiqa/query_doc_examples.csv`. That CSV is generated offline by `build_fiqa_csv.py` and committed; the build script is not part of the demo's runtime path.

## Feature Approach

V0 should begin with a small feature set targeting FiQA-style (non-temporal)
hard negatives:

- dense rank/score when present
- sparse rank/score when present
- RRF score when both ranks are present
- exact term overlap ratio
- number overlap count
- entity overlap count
- important-term overlap
- low-information-density flag
- query/doc length features

`feature_recipes.yaml` should list these features and stay consistent with the
hardcoded feature set in `common.py`. A formal feature registry/DSL is deferred
beyond V0.

## Evaluation

Primary metrics:

- `nDCG@10`
- `MRR@10`

Secondary metrics:

- `Recall@5` for positive evidence labels
- `hard_negative@5` or equivalent count/rate of `-1` labels in top-k

Regression cases are gates, not training rows. The regression gate should fail if required documents drop out of the configured top-k or known bad documents appear inside the forbidden top-k.

Failure analysis lite is deterministic and rule-based. It may explain a case
using:

- regression-case `failure_type`
- required vs forbidden document rank movement
- expected evidence term hits
- selected feature contrasts such as `entity_overlap_count`, `number_overlap_count`,
  `important_term_overlap`, and `term_overlap_ratio`

It must not generate new feature recipes, run ablations, or promote/reject
features in V0.

## Dependencies

V0 scripts may use:

- Python 3.10+
- `xgboost`
- `pandas`
- `numpy`
- `scikit-learn`
- `pyyaml`

These dependencies should be listed in:

```text
skills/heuriboost-rag/requirements.txt
```

Use unpinned dependencies in V0 to reduce early maintenance overhead:

```text
xgboost
pandas
numpy
scikit-learn
pyyaml
```

Scripts should catch missing imports at startup and print an install command, for example:

```bash
python -m pip install -r skills/heuriboost-rag/requirements.txt
```

Do not silently fall back to a fake model. If `xgboost` is unavailable, training should stop with a clear dependency error.

The offline demo-build script `build_fiqa_csv.py` has heavier dependencies
(`rank-bm25`, `sentence-transformers`, `datasets`, and an LLM client) that the
demo runtime does not need. These are isolated in a separate
`skills/heuriboost-rag/requirements-build.txt` so that `git clone` + train/eval
stays lightweight. The build script downloads the FiQA corpus into a gitignored
cache (`examples/fiqa/.cache/`) and uses model weights from the standard
Hugging Face cache; neither the corpus, the weights, nor the build packages are
committed.

## Tradeoffs

CSV-first trades direct framework convenience for universal compatibility. This is the right V0 tradeoff because any RAG project can export CSV, while framework-specific adapters create dependency and version churn.

Agent-skill-first positioning trades library familiarity for a clearer differentiated story. This is the right V0 tradeoff because the product value is not only scoring documents; it is helping a coding agent build a repeatable failure-improvement loop inside a user's existing repo.

Requiring minimal labels avoids the weak-label construction problem in V0. The skill still provides templates for turning failures into labels and regression cases so the later automation path remains visible.

Runnable scripts add implementation cost, but they prevent the project from becoming only a prompt/template bundle. The first release should prove that a user can hand HeuriBoost a CSV and get a concrete reranking report.

Keeping scripts inside the skill directory avoids premature package/API design. This is intentionally less polished than a real CLI, but it keeps V0 focused on proving the workflow.

Using real `xgboost` keeps the demo honest. A simulated model would reduce installation friction, but it would weaken the core claim that HeuriBoost trains an explainable LambdaMART reranker.

A skill-local `requirements.txt` is enough for V0 because there is no formal package. Pinning can be introduced after the demo workflow stabilizes.

Python 3.10+ is the minimum runtime target for V0. Older Python versions are out of scope.

Codex-compatible skill format keeps the first installation path concrete. Broader agent packaging can be added after the local script workflow is proven.
