# HeuriBoost

RAG reranking that remembers its mistakes.

[中文 README](./README.zh-CN.md)

Your RAG system answers a personal-finance question with a passage about the
wrong situation.

The retriever did not completely fail. It found the right evidence, but it also
found a semantically similar hard negative — same financial topic, wrong
entity/situation — and ranked that misleading passage too high. The generator
then saw plausible-looking evidence that could not support the answer.

HeuriBoost turns that kind of failure into a reranking upgrade:

```text
query: "Can I deduct home-office expenses as a sole proprietor?"

dense retrieval:
  #1 fiqa_doc_corporate_office_lease   hard negative: same topic, wrong entity
  #2 fiqa_doc_standard_deduction       weak/irrelevant
  #3 fiqa_doc_home_office_deduction    direct evidence

HeuriBoost rerank:
  #1 fiqa_doc_home_office_deduction    direct evidence
  #2 fiqa_doc_simplified_method        partial evidence
  #4 fiqa_doc_corporate_office_lease   remembered hard negative
```

It also writes the mistake down as a regression gate, so the next reranker must
keep the misleading passage out of the protected top-k.

## The Loop

HeuriBoost is a CSV-first, Codex-compatible agent skill for failure-driven RAG
reranking. It turns labeled query-document examples and past retrieval failures
into a local XGBoost/LambdaMART reranker with regression gates and lightweight
case analysis.

V0 deliberately keeps the loop small:

```text
existing RAG system
  -> export query-document-label CSV
  -> run the HeuriBoost RAG skill
  -> train an explainable reranker
  -> compare against retriever baselines
  -> analyze known failures
  -> preserve failures as regression gates
```

## What V0 Does

- Validates a standard query-document CSV contract.
- Trains a real XGBoost ranking model grouped by `query_id`.
- Uses a fixed V0 feature set from retriever scores/ranks and query-document
  text signals: term overlap, number overlap, entity overlap, important-term
  overlap, low-information-density flag, and length features.
- Evaluates nDCG, MRR, recall, and hard-negative exposure.
- Produces ranking diffs, feature importance, regression gate results, and
  deterministic failure analysis lite.
- Ships as a Codex-compatible skill plus runnable local scripts.

## What V0 Does Not Do

V0 does not:

- replace your first-stage retriever
- label your data automatically
- require LangChain, LlamaIndex, or a vector database
- run online A/B tests
- provide a stable Python package or public API
- perform automatic feature discovery, ablation, promotion, or feature memory
- try to become a general AutoML platform

`failure_analysis.md` is deterministic lite analysis, not automatic feature
discovery. It summarizes regression-case metadata, rank movement, expected
evidence hits, and selected V0 feature contrasts.

## Repository Layout

```text
.
├── README.md
├── README.zh-CN.md
├── CODEBUDDY.md
├── docs/
│   └── specs/
│       ├── ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md
│       ├── ADAPTIVE_XGBOOST_HEURISTIC_SPEC_CN.html
│       ├── QD_RERANKER_SPEC.md
│       └── QD_RERANKER_SPEC_CN.html
├── examples/
│   └── fiqa/
│       ├── query_doc_examples.csv
│       ├── regression_cases.yaml
│       └── DATA_CARD.md
└── skills/
    └── heuriboost-rag/
        ├── SKILL.md
        ├── requirements.txt
        ├── requirements-build.txt
        ├── scripts/
        │   ├── common.py
        │   ├── inspect_rag_repo.py
        │   ├── validate_dataset.py
        │   ├── train_reranker.py
        │   ├── eval_reranker.py
        │   └── build_fiqa_csv.py
        └── templates/
            ├── query_doc_examples.csv
            ├── regression_cases.yaml
            ├── feature_recipes.yaml
            └── promotion_gate.yaml
```

There is no `pyproject.toml` in V0. Use the skill-local scripts directly.

## Quick Start

Install dependencies:

```bash
python -m pip install -r skills/heuriboost-rag/requirements.txt
```

On macOS, if `xgboost` cannot load OpenMP, install `libomp`:

```bash
brew install libomp
```

Validate the demo dataset:

```bash
python3 skills/heuriboost-rag/scripts/validate_dataset.py examples/fiqa/query_doc_examples.csv
```

Train the reranker:

```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output
```

Evaluate and run regression gates:

```bash
python3 skills/heuriboost-rag/scripts/eval_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --regression-cases examples/fiqa/regression_cases.yaml
```

Expected outputs:

```text
examples/fiqa/output/
├── models/
│   ├── reranker.json
│   └── reranker_metadata.json
├── reports/
│   ├── eval_report.md
│   ├── ranking_diff.csv
│   ├── failure_cases.md
│   ├── failure_analysis.md
│   ├── failure_analysis.json
│   └── feature_importance.json
└── regression_cases.yaml
```

The generated `output/` directory is ignored by git.

## Regenerating the demo dataset

The committed `examples/fiqa/query_doc_examples.csv` is generated offline from
BEIR/FiQA-2018 by `skills/heuriboost-rag/scripts/build_fiqa_csv.py`. To
regenerate it:

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
export OPENAI_API_KEY=sk-...
python skills/heuriboost-rag/scripts/build_fiqa_csv.py --output examples/fiqa/query_doc_examples.csv
```

This step needs network access (to download FiQA) and an LLM API key (to judge
labels), so it is run locally by a maintainer, not in CI. Its heavy build
dependencies, the downloaded FiQA corpus, and the dense-encoder weights are not
committed. See `examples/fiqa/DATA_CARD.md` for provenance.

## CSV Contract

Required columns:

```csv
query_id,query_text,doc_id,doc_text,label,split
```

Recommended V0 columns:

```csv
query_id,query_text,doc_id,chunk_id,doc_text,dense_rank,dense_score,sparse_rank,sparse_score,label,split
```

Label scale:

```text
3  directly supports the answer
2  partially supports the answer
1  related but weak evidence
0  irrelevant
-1 misleading hard negative
```

For XGBoost training, labels are mapped to non-negative ordered relevance:

```text
-1 -> 0
 0 -> 1
 1 -> 2
 2 -> 3
 3 -> 4
```

Evaluation keeps the original labels so hard negatives remain visible in reports
and regression gates.

## Regression Cases

Regression cases are gates, not training rows.

```yaml
cases:
  - case_id: fiqa_expense_deduction_wrong_topic
    query_id: fiqa_q_001
    query: "Can I deduct home-office expenses as a sole proprietor?"
    must_include_doc_ids:
      - fiqa_doc_home_office_deduction
    must_not_include_doc_ids:
      - fiqa_doc_corporate_office_lease
    top_k: 3
    failure_type: semantic_hard_negative
    expected_evidence:
      - "home office"
      - "deduction"
      - "sole proprietor"
```

If a required document drops out of top-k, or a forbidden document enters top-k,
`eval_reranker.py` fails the regression gate.

## Reports

`eval_report.md`
: Global metrics and regression gate status.

`ranking_diff.csv`
: Before/after rank movement using dense rank as the default baseline.

`failure_cases.md`
: Hard-negative exposure report for the top 3.

`failure_analysis.md`
: Deterministic regression-case analysis with reason summary, rank movement,
evidence hits, feature contrast, and suggested next actions.

`feature_importance.json`
: XGBoost gain-based feature importance normalized across the V0 feature list.

## Agent Skill

The Codex-compatible skill lives in:

```text
skills/heuriboost-rag/SKILL.md
```

It exposes three modes:

- `audit`: scan a RAG repo for retriever/eval/log/dataset signals
- `bootstrap`: copy templates and explain the CSV contract
- `experiment`: validate CSV, train, evaluate, and inspect reports

Other coding agents can still run the Python scripts manually, but V0 does not
provide a complete multi-agent installation experience.

## Current Status

Status: V0 prototype.

The demo uses a real slice of BEIR/FiQA-2018 (financial question answering),
where a passage on the same financial topic but the wrong entity/situation is
semantically similar to the query yet cannot support the answer. HeuriBoost
learns to lift the supporting passage and push the misleading hard negative
down. The committed CSV is generated offline (see "Regenerating the demo
dataset" and `examples/fiqa/DATA_CARD.md`).

Long-form design specs live in `docs/specs/`.
