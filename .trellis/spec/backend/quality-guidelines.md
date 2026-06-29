# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

HeuriBoost V0 backend code is a small script-backed workflow. Quality is judged
by whether a CSV-first RAG reranking demo runs end to end and preserves the
core contracts below.

---

## Forbidden Patterns

- Do not add a formal package scaffold (`pyproject.toml`, stable Python API, or
  publishing flow) for V0.
- Do not silently fall back to a fake model when `xgboost` is unavailable.
- Do not train directly on `regression_cases.yaml`; regression cases are gates.
- Do not describe failure analysis lite as automatic feature discovery.
- Do not use post-generation signals such as answer citations, clicks, or human
  labels as online model features.
- Do not allow the same `query_id` to cross dataset splits.

---

## Required Patterns

### Command Signatures

```bash
python skills/heuriboost-rag/scripts/validate_dataset.py <query_doc_examples.csv>
python skills/heuriboost-rag/scripts/train_reranker.py <query_doc_examples.csv> --output-dir <dir>
python skills/heuriboost-rag/scripts/eval_reranker.py <query_doc_examples.csv> --output-dir <dir> --regression-cases <regression_cases.yaml>
```

### CSV Contract

Required columns:

```text
query_id, query_text, doc_id, doc_text, label, split
```

Recommended columns:

```text
query_id, query_text, doc_id, chunk_id, doc_text, dense_rank, dense_score, sparse_rank, sparse_score, label, split
```

Valid labels are `-1`, `0`, `1`, `2`, and `3`.

Training maps labels to non-negative relevance:

```text
-1 -> 0
 0 -> 1
 1 -> 2
 2 -> 3
 3 -> 4
```

This keeps hard negatives below ordinary irrelevant documents during training.
Evaluation and reports must preserve the original label values.

### Failure Analysis Lite Contract

`eval_reranker.py` writes:

```text
reports/failure_analysis.md
reports/failure_analysis.json
```

This analysis is deterministic and rule-based. It can use regression-case
metadata, rank movement, evidence-term hits, and V0 feature contrasts. It must
not generate feature recipes, run ablations, or promote/reject features.

### Dependency Errors

Scripts must print a clear install hint when imports fail:

```bash
python -m pip install -r skills/heuriboost-rag/requirements.txt
```

If `xgboost` is installed but cannot load OpenMP on macOS, mention:

```bash
brew install libomp
```

---

## Testing Requirements

Run these commands after changing V0 scripts, templates, or demo data:

```bash
python3 -m py_compile skills/heuriboost-rag/scripts/*.py
python3 skills/heuriboost-rag/scripts/validate_dataset.py examples/financial_rag/query_doc_examples.csv
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/financial_rag/query_doc_examples.csv --output-dir examples/financial_rag/output
python3 skills/heuriboost-rag/scripts/eval_reranker.py examples/financial_rag/query_doc_examples.csv --output-dir examples/financial_rag/output --regression-cases examples/financial_rag/regression_cases.yaml
```

Expected demo behavior:

- `doc_2024_q3_margin` rises to rank 1 for `q_val_margin_2024_q3`.
- `doc_2023_q3_margin` falls out of top 3.
- `reports/eval_report.md` says the regression gate passed.
- `reports/failure_analysis.md` summarizes the temporal hard-negative reason.

---

## Code Review Checklist

- [ ] Required CSV columns and label values are validated at script entry.
- [ ] Ranking groups are built by `query_id`.
- [ ] `-1` hard negatives remain visible in reports and gates.
- [ ] Missing dependency errors are readable and actionable.
- [ ] Demo output stays ignored by git.
- [ ] README commands match the actual script signatures.
