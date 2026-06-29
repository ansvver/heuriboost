# HeuriBoost

RAG reranking that remembers its mistakes.

[中文 README](./README.zh-CN.md)

Your RAG system answers a 2024 Q3 question with a 2023 Q3 document.

The retriever did not completely fail. It found the right evidence, but it also
found a semantically similar hard negative and ranked that wrong-year document
too high. The generator then saw plausible-looking evidence that could not
support the answer.

HeuriBoost turns that kind of failure into a reranking upgrade:

```text
query: "What caused 2024 Q3 gross margin decline?"

dense retrieval:
  #1 doc_2023_q3_margin   hard negative: right topic, wrong year
  #2 doc_2024_q3_ops      partial evidence
  #3 doc_2024_q3_margin   direct evidence

HeuriBoost rerank:
  #1 doc_2024_q3_margin   direct evidence
  #2 doc_2024_q3_ops      partial evidence
  #4 doc_2023_q3_margin   remembered hard negative
```

It also writes the mistake down as a regression gate, so the next reranker must
keep the wrong-year document out of the protected top-k.

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
  text overlap.
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
│   └── financial_rag/
│       ├── query_doc_examples.csv
│       └── regression_cases.yaml
└── skills/
    └── heuriboost-rag/
        ├── SKILL.md
        ├── requirements.txt
        ├── scripts/
        │   ├── common.py
        │   ├── inspect_rag_repo.py
        │   ├── validate_dataset.py
        │   ├── train_reranker.py
        │   └── eval_reranker.py
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
python3 skills/heuriboost-rag/scripts/validate_dataset.py examples/financial_rag/query_doc_examples.csv
```

Train the reranker:

```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/financial_rag/query_doc_examples.csv --output-dir examples/financial_rag/output
```

Evaluate and run regression gates:

```bash
python3 skills/heuriboost-rag/scripts/eval_reranker.py examples/financial_rag/query_doc_examples.csv --output-dir examples/financial_rag/output --regression-cases examples/financial_rag/regression_cases.yaml
```

Expected outputs:

```text
examples/financial_rag/output/
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
  - case_id: q_val_margin_2024_q3_wrong_year
    query_id: q_val_margin_2024_q3
    query: "What caused 2024 Q3 gross margin decline?"
    must_include_doc_ids:
      - doc_2024_q3_margin
    must_not_include_doc_ids:
      - doc_2023_q3_margin
    top_k: 3
    failure_type: temporal_hard_negative
    expected_evidence:
      - "2024 Q3"
      - "gross margin"
      - "raw material costs"
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

The demo intentionally uses a small financial RAG scenario where a 2023 Q3
document is semantically similar to a 2024 Q3 question but should not support
the answer. HeuriBoost learns to lift the 2024 Q3 evidence and push the
wrong-year hard negative down.

Long-form design specs live in `docs/specs/`.
