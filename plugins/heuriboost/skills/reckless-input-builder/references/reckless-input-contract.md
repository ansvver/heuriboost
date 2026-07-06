# Reckless Input Contract

Use this reference when turning raw user materials into HeuriBoost reckless-mode
repair inputs. The goal is to create the two user-facing files consumed by:

```bash
python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/compile_cases.py" \
  --base-dataset <base_dataset.csv|jsonl|json> \
  --production-cases <production_cases.csv|jsonl|json> \
  --output-dir <dir> \
  [--strict] [--resplit]

python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/repair_reranker.py" \
  --base-dataset <base_dataset.csv|jsonl|json> \
  --production-cases <production_cases.csv|jsonl|json> \
  --output-dir <dir> \
  --reckless [--acceptance-level full|weak] [--reset-anchor]
```

## Contract Summary

- Users author only `base_dataset` and `production_cases`.
- `compile_cases.py` generates `.heuriboost/compiled/query_doc_examples.csv`,
  `.heuriboost/compiled/regression_cases.yaml`, `.heuriboost/compiled/case_sets/`,
  and `.heuriboost/compiled/production_cases.json`.
- `domain` is a hard boundary. Any `production_cases.domain` value must exist in
  `base_dataset.domain`. Missing domain compiles to `default`.
- Existing ids should be preserved. Missing ids are generated from stable hashes.
- Current production repair samples are intentionally allowed into training only
  through `repair_reranker.py --reckless`.
- Base test rows remain the global regression suite. Do not append production
  cases or historical gates to base test.

## File Formats

CSV is preferred for flat materials. JSON and JSONL are accepted when source
material is nested or already event-shaped.

For JSON/JSONL `base_dataset`, an object may include:

```json
{
  "domain": "tax",
  "query_id": "q_home_office",
  "query": "How do I deduct home office expenses?",
  "documents": [
    {"id": "doc_good", "text": "Home office deduction rules...", "relevance": "good"},
    {"id": "doc_bad", "text": "Corporate office lease rules...", "relevance": "bad"}
  ],
  "split": "train"
}
```

For JSON/JSONL `production_cases`, use `shown`:

```json
{
  "domain": "tax",
  "case_id": "prod_home_office",
  "query": "How do I deduct home office expenses?",
  "shown": [
    {"id": "doc_bad", "text": "Corporate office lease rules...", "verdict": "bad", "rank": 1},
    {"id": "doc_good", "text": "Home office deduction rules...", "verdict": "good", "rank": 5}
  ]
}
```

## base_dataset

Purpose: normal query-document evidence used for training, validation, and
base test metrics.

Minimal columns:

```text
query, text, relevance
```

Recommended columns:

```text
domain, query_id, query, doc_id, text, relevance, split, rank, score
```

Accepted aliases:

| Canonical | Accepted input names |
| --- | --- |
| `domain` | `domain` |
| `query_id` | `query_id`, `qid` |
| `query` | `query`, `query_text` |
| `doc_id` | `doc_id`, `id` |
| `text` | `text`, `doc_text`, `document`, `document_text` |
| `relevance` | `relevance`, `label`, `verdict` |
| `split` | `split` |
| `rank` | `rank` |
| `score` | `score` |
| `dense_rank` | `dense_rank` |
| `dense_score` | `dense_score` |
| `sparse_rank` | `sparse_rank` |
| `sparse_score` | `sparse_score` |

Accepted relevance values:

| Meaning | Values |
| --- | --- |
| Directly good | `3`, `good`, `positive` |
| Partial | `2`, `partial` |
| Weak | `1`, `weak` |
| Irrelevant | `0`, `irrelevant`, `negative` |
| Hard negative | `-1`, `bad`, `hard_negative`, `hard-negative` |

Rules:

- Every row needs non-empty query text, document text, and relevance.
- Relevance must be explicit. Do not infer relevance from rank alone.
- If `split` is present, every row must have `train`, `validation`, `test`,
  `valid`, or `val`. Use `--resplit` to replace existing splits.
- If `split` is absent, the compiler assigns deterministic query-group splits.
- Rows with the same compiled `query_id` stay in the same split.
- `rank` and `score` become dense rank/score if no dense-specific columns exist.

## production_cases

Purpose: current production failures or repair targets. These rows are the
deliberate reckless-mode exception that may enter training during repair.

Minimal columns:

```text
query, shown_doc_text, user_verdict
```

Recommended columns:

```text
domain, case_id, query, shown_doc_id, shown_doc_text, user_verdict, rank, score
```

Accepted aliases:

| Canonical | Accepted input names |
| --- | --- |
| `domain` | `domain` |
| `case_id` | `case_id` |
| `query` | `query`, `query_text` |
| `shown_doc_id` | `shown_doc_id`, `doc_id`, `id` |
| `shown_doc_text` | `shown_doc_text`, `doc_text`, `text`, `document_text` |
| `user_verdict` | `user_verdict`, `verdict` |
| `rank` | `rank` |
| `score` | `score` |

Accepted verdict values:

| Meaning | Values |
| --- | --- |
| Desired target | `good`, `positive`, `accepted` |
| Must suppress | `bad`, `negative`, `wrong` |
| Context only | `unknown` |

Rules:

- Every row needs query text, shown document text, and an explicit verdict.
- Rows with the same `case_id` form one production case and must share one
  domain and one query text.
- A document cannot be both `good` and `bad` in the same case.
- A file containing only `unknown` verdicts is invalid.
- `unknown` rows are context candidates only; they do not become repair labels.
- Full acceptance requires each actionable case to include at least one `good`
  target. A case with only `bad` evidence requires `--acceptance-level weak`.
- Weak bad-only repair can suppress bad documents but is not promotion eligible.

## Strict Sufficiency

When running `compile_cases.py --strict`, defaults are:

- Global test split: at least 10 query groups.
- Each touched production-case domain: at least 3 test query groups.
- Validation and test query groups: at least 2 candidate documents each.
- Train, validation, and test splits must each contain at least one positive
  label and one non-positive label.
- Full acceptance cases must include at least one good target.

If the user's material is too small for strict mode, still create the best
candidate files, run non-strict compile, and report exactly which strict
sufficiency requirement is missing.

## Material Mapping Rules

Use this mapping when raw materials come from logs, spreadsheets, tickets, or
manual notes:

| Raw material | Map to |
| --- | --- |
| Search request text | `query` |
| Retrieved chunk or document text | `text` or `shown_doc_text` |
| Retriever rank | `rank` |
| Retriever score | `score` |
| Human says "should have shown this" | `user_verdict=good` |
| Human says "wrong answer/source" | `user_verdict=bad` |
| Candidate was shown but not judged | `user_verdict=unknown` |
| Curated relevance label | `relevance` |
| Product surface, tenant, corpus, vertical | `domain` |
| Existing stable query/document/case ids | `query_id`, `doc_id`, `case_id` |

Do not use post-generation signals such as answer citations, clicks, or human
labels as online model features. They can be labels/verdicts in these files,
but they must not become feature columns.

## Output Checklist

Before handing back results:

1. Write the normalized `base_dataset` file.
2. Write the normalized `production_cases` file.
3. Run `compile_cases.py` with `--strict` when there appears to be enough data.
4. If strict compile fails for sufficiency only, run non-strict compile to catch
   schema errors and report the missing coverage separately.
5. Report file paths, row counts, touched domains, full/weak recommendation,
   warnings, and the exact next `repair_reranker.py --reckless` command.

Never silently downgrade full repair to weak repair. State the reason and the
acceptance-level command explicitly.
