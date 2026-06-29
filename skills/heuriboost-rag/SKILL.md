---
name: heuriboost-rag
description: Failure-driven RAG reranking workflow. Use when improving RAG retrieval from labeled query-document CSVs, hard negatives, or regression cases.
---

# HeuriBoost RAG Skill

Turn RAG retrieval failures into reranking upgrades.

This skill is CSV-first and script-backed. Use it to audit a RAG project,
bootstrap HeuriBoost templates, or run a local XGBoost/LambdaMART reranker
experiment from labeled query-document examples.

## Modes

### audit

Use audit mode when the user asks whether an existing RAG project is ready for
HeuriBoost.

1. Inspect the repo for retriever, reranker, eval, log, and dataset files.
2. Identify whether query-document candidates are available.
3. Identify whether labels or failure cases exist.
4. Report missing pieces without changing files.

Suggested command:

```bash
python skills/heuriboost-rag/scripts/inspect_rag_repo.py .
```

### bootstrap

Use bootstrap mode when the user wants to add HeuriBoost templates.

1. Create a project-local workspace or copy files from
   `skills/heuriboost-rag/templates/`.
2. Explain the CSV contract.
3. Ask the user to export labeled query-document examples.
4. Do not train until labels exist.

### experiment

Use experiment mode when the user has a CSV dataset.

1. Install dependencies if needed:

   ```bash
   python -m pip install -r skills/heuriboost-rag/requirements.txt
   ```

2. Validate the dataset:

   ```bash
   python skills/heuriboost-rag/scripts/validate_dataset.py path/to/query_doc_examples.csv
   ```

3. Train:

   ```bash
   python skills/heuriboost-rag/scripts/train_reranker.py path/to/query_doc_examples.csv --output-dir path/to/output
   ```

4. Evaluate:

   ```bash
   python skills/heuriboost-rag/scripts/eval_reranker.py path/to/query_doc_examples.csv --output-dir path/to/output --regression-cases path/to/regression_cases.yaml
   ```

5. Read `reports/eval_report.md`, `reports/ranking_diff.csv`,
   `reports/failure_cases.md`, and `reports/failure_analysis.md`.

## CSV Contract

Required columns:

```csv
query_id,query_text,doc_id,doc_text,label,split
```

Recommended columns:

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

Training maps labels to non-negative ordered relevance:
`-1 -> 0`, `0 -> 1`, `1 -> 2`, `2 -> 3`, `3 -> 4`. Evaluation keeps `-1` as a
hard-negative signal.

## Guardrails

- Do not use answer citations, human labels, clicks, or post-generation signals
  as online model features.
- Keep rows with the same `query_id` in the same split.
- Treat regression cases as gates, not training rows.
- Prefer CSV export over framework-specific adapters in V0.
- Do not add a formal Python package scaffold in V0.
- Do not describe V0 as automatic feature discovery. `failure_analysis.md` is a
  deterministic lite analysis, not a feature generation/promotion loop.
