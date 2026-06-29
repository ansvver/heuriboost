# HeuriBoost RAG Skill V0 Plan

## Goal

Define the first public version of HeuriBoost as a CSV-first coding-agent skill for failure-driven RAG reranking.

HeuriBoost V0 should help a coding agent enter an existing RAG repository, standardize labeled query-document examples, train and evaluate an explainable XGBoost/LambdaMART reranker, and turn past retrieval failures into durable regression cases.

## Positioning

Primary project name:

- `HeuriBoost`

Primary V0 skill name:

- `heuriboost-rag`

Tagline:

- `RAG reranking that remembers its mistakes.`

Core story:

- HeuriBoost is an agent skill for turning RAG failures into reranking upgrades.
- It is not another generic reranker API. It is a failure-driven loop: inspect, standardize, train, evaluate, explain, and preserve known failures as gates.
- The first version should be framed as an agent skill first and a local reranker toolkit second.

## Confirmed Product Decisions

- V0 focuses on RAG query-document learning-to-rank, not generic classification/regression.
- V0 is CSV-first. It does not directly depend on LangChain, LlamaIndex, a vector database, or a specific retriever framework.
- V0 requires users to provide minimal labels.
- V0 must include templates that help users convert RAG failure cases into labels and regression cases.
- The main artifact should live under a coding-agent skill, with local scripts/templates as the execution substrate.
- The README/product narrative should emphasize agent workflow and failure memory, not XGBoost internals.
- V0 must include runnable minimum scripts, not only skill instructions and templates.
- V0 should keep runnable scripts inside `skills/heuriboost-rag/scripts/` and should not create a formal Python package scaffold yet.
- V0 should use real `xgboost` for the reranker rather than a simulated or placeholder model.
- V0 should include `skills/heuriboost-rag/requirements.txt` with unpinned runtime dependencies.
- V0 should document Python 3.10+ as the minimum runtime.
- V0 should author `skills/heuriboost-rag/SKILL.md` in a Codex-compatible skill format. Other agents can run the scripts manually, but V0 does not provide full multi-agent installation support.
- V0 should include deterministic failure analysis lite, not automatic feature discovery.
- V0 ships a single demo from a real public dataset (BEIR/FiQA-2018), not a hand-built toy demo. The committed CSV is generated offline by `build_fiqa_csv.py`; that script's heavy dependencies, the downloaded corpus, and model weights are not committed.
- V0 features target FiQA-style (non-temporal) hard negatives: dense/sparse/RRF, term overlap, number overlap, entity overlap, important-term overlap, low-information-density flag, and length features. Temporal year/quarter/wrong-year features are removed.

## Requirements

### R1. Golden Path

The V0 plan must support this first-user path:

```text
existing RAG system
  -> export query-document-label CSV
  -> install/use HeuriBoost RAG skill
  -> agent audits the repository and available data
  -> agent bootstraps a HeuriBoost workspace
  -> local scripts train an XGBoost/LambdaMART reranker
  -> evaluation compares against retriever baselines
  -> reports explain ranking changes and failures
  -> selected failures become regression cases
```

### R2. Skill Modes

The skill should be written as a Codex-compatible `SKILL.md` and expose three user-facing modes:

- `audit`: read-only scan of an existing RAG project to find retriever, reranker, logs, evals, and data shape.
- `bootstrap`: create a HeuriBoost workspace with CSV/YAML templates, demo data, and config placeholders.
- `experiment`: train/evaluate a reranker from CSV input and generate reports/regression artifacts.

### R3. Minimum CSV Contract

The required CSV schema must be small enough for a user to understand in 10 minutes:

```csv
query_id,query_text,doc_id,chunk_id,doc_text,dense_rank,dense_score,sparse_rank,sparse_score,label,split
```

Required fields:

- `query_id`
- `query_text`
- `doc_id`
- `doc_text`
- `label`
- `split`

Optional fields:

- `chunk_id`
- `dense_rank`
- `dense_score`
- `sparse_rank`
- `sparse_score`
- `doc_text_ref`
- additional safe feature columns

Label scale:

```text
3  directly supports the answer
2  partially supports the answer
1  related but weak evidence
0  irrelevant
-1 misleading hard negative
```

V0 must support self-contained `doc_text` for zero-dependency demos. It may also allow optional `doc_text_ref` for real projects.

### R4. Regression Case Template

V0 must provide a human-editable failure-to-regression template. Cases are
mined from observed FiQA rerank failures (a high-ranked candidate that does not
support the answer) and hand-confirmed by a human before becoming a gate:

```yaml
case_id: fiqa_expense_deduction_wrong_topic
query: "Can I deduct home-office expenses as a sole proprietor?"
must_include_doc_ids:
  - fiqa_doc_home_office_deduction
must_not_include_doc_ids:
  - fiqa_doc_corporate_office_lease
failure_type: semantic_hard_negative
expected_evidence:
  - "home office"
  - "deduction"
  - "sole proprietor"
notes: "Same financial topic but answers a different entity/situation; retriever ranked it high, but it cannot support the answer."
```

### R5. V0 Outputs

The experiment mode must be backed by runnable local scripts. At minimum, V0 should include:

- `validate_dataset.py`
- `train_reranker.py`
- `eval_reranker.py`

These scripts should be able to run the FiQA demo end to end without framework-specific adapters. A separate `build_fiqa_csv.py` (with its own `requirements-build.txt`) regenerates the demo CSV offline using BM25 + a dense encoder + RRF retrieval and an LLM judge for labels; it is not on the demo's runtime path.

V0 should not add a formal package/CLI scaffold. Do not add `pyproject.toml`, package versioning, publish workflow, or stable public Python API until the skill workflow and demo are validated.

The scripts should target Python 3.10+. They may depend on `xgboost`, `pandas`, `numpy`, `scikit-learn`, and `pyyaml`. These dependencies should be listed in `skills/heuriboost-rag/requirements.txt` without version pins. If dependencies are missing, scripts must fail with a clear install hint rather than a raw stack trace.

The experiment mode should produce these planned outputs:

- `reports/eval_report.md`
- `reports/ranking_diff.csv`
- `reports/failure_cases.md`
- `reports/failure_analysis.md`
- `reports/failure_analysis.json`
- `reports/feature_importance.json`
- `regression_cases.yaml`
- `models/reranker.json`

`failure_analysis.md` should summarize regression-case metadata, before/after rank movement, expected evidence hits, selected feature contrasts, and suggested next actions. It must not claim to perform automatic feature discovery, ablation, or promotion.

### R6. Demo

V0 must include a single `examples/fiqa/` demo built from a slice of BEIR/FiQA-2018
(~150 train / 40 validation / 40 test queries, top-20 candidates, doc_text ≤400
chars), with dense/sparse/RRF scores and 5-level labels produced offline by
`build_fiqa_csv.py`. The committed CSV is checked in; the build script's heavy
dependencies, downloaded corpus, and model weights are NOT committed. The demo
must show:

- a real financial-QA query from FiQA
- a semantically similar but unsupported hard negative the retriever ranked high
- the correct evidence document rising after reranking
- the learned reranker beating dense/sparse/RRF baselines on validation nDCG@10 / MRR@10
- a hand-confirmed regression case that prevents a known bad rerank from passing gates

Provenance (source, CC BY-SA 4.0 license, retriever config, LLM-judge model/prompt/date)
is recorded in `examples/fiqa/DATA_CARD.md`.

Note: the toy hand-built financial demo (`examples/financial_rag/`) and the
temporal `wrong_year` hard-negative are removed in this revision; FiQA is the
single V0 demo.

### R7. Non-goals

V0 explicitly does not:

- automatically label all user data
- replace the first-stage retriever
- include or require a vector database
- require LangChain or LlamaIndex
- promise no-label training
- run online A/B tests
- automatically discover, ablate, promote, or quarantine new features
- implement generic classification/regression workflows
- become a full AutoML platform
- provide a complete multi-agent installation experience

## Acceptance Criteria

- [ ] The public plan clearly positions HeuriBoost as an agent skill for failure-driven RAG reranking.
- [ ] The plan documents the CSV-first contract, required fields, optional fields, and label scale.
- [ ] The plan includes the three skill modes: `audit`, `bootstrap`, and `experiment`.
- [ ] The plan requires runnable local scripts for validation, training, and evaluation.
- [ ] The plan explicitly keeps V0 runtime scripts inside the skill directory rather than introducing a formal package scaffold.
- [ ] The plan requires real `xgboost` training and clear missing-dependency guidance.
- [ ] The plan includes a skill-local unpinned `requirements.txt` for runtime dependencies.
- [ ] The plan documents Python 3.10+ as the minimum runtime.
- [ ] The plan targets a Codex-compatible skill format and limits other-agent support to manual script usage.
- [ ] The plan includes a regression-case template for turning past RAG failures into gates.
- [ ] The plan includes deterministic failure analysis lite while keeping automatic feature discovery out of V0.
- [ ] The plan defines expected V0 outputs and the single FiQA demo (with offline `build_fiqa_csv.py` and data card).
- [ ] The plan records V0 non-goals so the first release does not expand into a generic ML platform.

## Open Questions

- None blocking.
