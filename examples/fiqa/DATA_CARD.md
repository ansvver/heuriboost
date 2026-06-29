# FiQA Demo Data Card

Provenance record for the committed HeuriBoost FiQA demo dataset
(`examples/fiqa/query_doc_examples.csv`).

## Source

- **Dataset**: BEIR / FiQA-2018 (financial-domain question answering).
- **Loaded via**: Hugging Face `datasets` (`BeIR/fiqa` corpus + queries,
  `BeIR/fiqa-qrels` for relevance), honoring FiQA's native
  train/validation/test split (`BeIR/fiqa-qrels` exposes the dev split as
  `validation`).

## License

- **License**: CC BY-SA 4.0.
- **Share-alike obligation**: the committed `query_doc_examples.csv` is a
  derivative of FiQA-2018 and is therefore distributed under the same CC BY-SA
  4.0 terms. Downstream redistribution must preserve attribution and the
  share-alike license.

## Retriever Configuration

Candidates and retriever scores are produced offline by
`skills/heuriboost-rag/scripts/build_fiqa_csv.py`. FiQA does not ship a
candidate set, so the build script runs retrieval itself:

- **Sparse**: BM25 via `rank_bm25` (`BM25Okapi`) over the sliced corpus.
- **Dense**: `sentence-transformers/all-MiniLM-L6-v2` (cosine on normalized
  embeddings), CPU at build time.
- **Fusion**: Reciprocal Rank Fusion (RRF) to choose the per-query candidate
  union.

Each `(query, doc)` row records `dense_rank`, `dense_score`, `sparse_rank`, and
`sparse_score`.

## Labels

Labels use the 5-level HeuriBoost scale `{3, 2, 1, 0, -1}`. qrel-positive docs
are always seeded as `3`. The remaining candidates are labeled by one of two
build modes (`--label-mode`):

**`llm` (default)** — a build-time LLM judge grades each non-positive candidate
on the full 5-level scale. Fill in the actual values used for the committed CSV:

- **Judge model**: `__________` (default: `deepseek-chat` via DeepSeek's
  OpenAI-compatible API at `https://api.deepseek.com`)
- **Judge prompt version/date**: `__________`
- **Generated on (date)**: `__________`

**`heuristic`** — zero-cost, deterministic, no LLM. A non-positive candidate
ranked highly by the dense retriever (`dense_rank <= --hard-negative-rank`,
default 5) is labeled `-1` (semantic hard negative); everything else is `0`.
Note: FiQA qrels are sparsely annotated, so some heuristic `-1`s may actually be
unlabeled-relevant passages. This is an acceptable demo approximation, NOT a
benchmark-grade label set. Record which mode produced the committed CSV:

- **Label mode used**: `__________` (`llm` or `heuristic`)
- **Generated on (date)**: `__________`

> Labels (LLM-judged or heuristic) may feed training and evaluation. They may
> NOT define regression-gate `must_not_include` cases and may NEVER be used as
> an online model feature. Regression cases
> (`examples/fiqa/regression_cases.yaml`) are hand-confirmed from trusted labels.

## Slice Caps

- **Train queries**: 150
- **Validation queries**: 40
- **Test queries**: 40
- **Candidates per query**: top-20 (RRF union)
- **doc_text**: truncated to ≤ 400 characters

## What is committed vs. not committed

Committed to the repo:

- `examples/fiqa/query_doc_examples.csv` — generated offline, then committed.
- `examples/fiqa/regression_cases.yaml` — hand-confirmed regression cases.
- `examples/fiqa/DATA_CARD.md` — this file.

NOT committed:

- The downloaded FiQA corpus / queries / qrels (cached under
  `examples/fiqa/.cache/`, gitignored, plus the standard Hugging Face cache).
- The `all-MiniLM-L6-v2` model weights (standard Hugging Face cache).
- The build-time Python packages
  (`skills/heuriboost-rag/requirements-build.txt`): `rank-bm25`,
  `sentence-transformers`, `datasets`, `openai`.

## Regenerating

Heuristic mode (no LLM, no API key):

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
python skills/heuriboost-rag/scripts/build_fiqa_csv.py \
  --label-mode heuristic --output examples/fiqa/query_doc_examples.csv
```

LLM mode (DeepSeek by default):

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
export DEEPSEEK_API_KEY=sk-...   # or OPENAI_API_KEY with --base-url ""
python skills/heuriboost-rag/scripts/build_fiqa_csv.py \
  --label-mode llm --output examples/fiqa/query_doc_examples.csv
```

Both modes need network access to download FiQA. Only `llm` mode needs an API
key. The build is run locally by a maintainer, never in CI.
