# FiQA Demo Data Card

Provenance record for the committed HeuriBoost FiQA demo dataset
(`examples/fiqa/query_doc_examples.csv`).

## Source

- **Dataset**: BEIR / FiQA-2018 (financial-domain question answering).
- **Loaded via**: Hugging Face `datasets` (`BeIR/fiqa` corpus + queries,
  `BeIR/fiqa-qrels` for relevance), honoring FiQA's native train/dev/test split
  (dev is mapped to `validation`).

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

## Labels (build-time LLM judge)

Labels use the 5-level HeuriBoost scale `{3, 2, 1, 0, -1}`. qrel-positive docs
are seeded as `3` before judging; remaining candidates are graded by a
build-time LLM judge.

Fill in the actual values used for the committed CSV:

- **Judge model**: `__________` (default placeholder: `gpt-4o-mini`)
- **Judge prompt version/date**: `__________`
- **Generated on (date)**: `__________`

> LLM-judged labels may feed training and evaluation. They may NOT define
> regression-gate `must_not_include` cases and may NEVER be used as an online
> model feature. Regression cases (`examples/fiqa/regression_cases.yaml`) are
> hand-confirmed from trusted labels.

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

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
export OPENAI_API_KEY=sk-...
python skills/heuriboost-rag/scripts/build_fiqa_csv.py --output examples/fiqa/query_doc_examples.csv
```

The build script needs network access (to download FiQA) and an LLM API key (to
judge labels). It is run locally by a maintainer and never in CI.
