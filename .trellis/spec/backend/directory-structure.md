# Directory Structure

> How backend code is organized in this project.

---

## Overview

HeuriBoost V0 is a skill-local Python runtime, not a formal Python package.
Runtime code lives under `skills/heuriboost-rag/` and is invoked as scripts.
Do not add a root `pyproject.toml`, package API, or publish workflow until the
skill workflow is validated.

---

## Directory Layout

```text
skills/heuriboost-rag/
  SKILL.md
  requirements.txt
  scripts/
    common.py
    inspect_rag_repo.py
    validate_dataset.py
    train_reranker.py
    eval_reranker.py
  templates/
    query_doc_examples.csv
    regression_cases.yaml
    feature_recipes.yaml
    promotion_gate.yaml

examples/financial_rag/
  query_doc_examples.csv
  regression_cases.yaml
```

---

## Module Organization

- Put shared CSV validation, feature extraction, ranking metrics, and output
  helpers in `skills/heuriboost-rag/scripts/common.py`.
- Keep command entrypoints thin:
  - `inspect_rag_repo.py` is read-only audit.
  - `validate_dataset.py` validates the CSV contract.
  - `train_reranker.py` trains the model and writes `models/`.
  - `eval_reranker.py` evaluates, runs regression gates, and writes `reports/`.
- Put user-copyable starter files under `skills/heuriboost-rag/templates/`.
- Put the runnable toy scenario under `examples/financial_rag/`.
- Generated demo outputs belong under `examples/financial_rag/output/` and are
  ignored by git.

---

## Naming Conventions

- Use snake_case for Python scripts and CSV/YAML template names.
- Use `query_doc_examples.csv` for the canonical CSV input name.
- Use `regression_cases.yaml` for failure gate definitions.
- Use `reranker.json` for the XGBoost model artifact to avoid ambiguous binary
  extension behavior.

---

## Examples

- `skills/heuriboost-rag/scripts/common.py`
- `skills/heuriboost-rag/scripts/train_reranker.py`
- `examples/financial_rag/query_doc_examples.csv`
