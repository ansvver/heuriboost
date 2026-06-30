# HeuriBoost Reference

Operational reference for the HeuriBoost Q-D reranker. The [README](../README.md)
covers the story, concepts, and demo results; this file holds the contracts and
command details a maintainer needs day to day.

- [Feature registry](#feature-registry)
- [HPO](#hpo)
- [Ablation](#ablation)
- [Candidate discovery](#candidate-discovery)
- [CSV contract](#csv-contract)
- [Label scale](#label-scale)
- [Regression cases](#regression-cases)
- [Cross-round ledger](#cross-round-ledger)
- [Production-case repair](#production-case-repair)
- [Closing the loop: case_sets mining](#closing-the-loop-case_sets-mining)
- [Reports](#reports)
- [Agent skill](#agent-skill)
- [Regenerating the demo dataset](#regenerating-the-demo-dataset)

## Feature registry

Every feature is a declared `FeatureRecipe`, not scattered code. The source of
truth is `skills/heuriboost-rag/templates/feature_recipes.yaml`; the Python
implementation lives in `skills/heuriboost-rag/scripts/features/`
(`registry.py`, `primitives.py`, `recipes.py`).

Each recipe carries the spec §6.4 required fields:

| Field | Meaning |
|---|---|
| `name`, `version` | feature identifier + per-feature version |
| `description` | human-readable |
| `task_profiles` | profiles that use it (V0: `qd_reranker`) |
| `inputs` | input columns the feature reads (must be in `ALLOWED_INPUTS`) |
| `impl` | implementation reference (V0: `extract_all`, the shared fn) |
| `type`, `default_value` | `numeric` / 0.0 for all V0 |
| `cost_tier` | `L0`..`L3` |
| `online_safe` | must be true for the active profile |
| `leakage_risk` | `low`/`medium`/`high` |
| `expected_slices` | forward-looking; may be empty |
| `owner` | owning team |

`ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score, sparse_rank,
sparse_score}`. Any other column (notably `label`, `split`, `query_id`,
`doc_id`) is rejected at load time as a leakage / identifier vector.

Loading is eager: `import common` triggers `FeatureRegistry.validate()`, which
hard-fails (SystemExit) on a missing impl, a disallowed input, an
`online_safe: false` recipe, or an empty required field. This makes the
"FEATURE_NAMES must equal feature_recipes.yaml" contract a load-time check.

The trained model's `reranker_metadata.json` records `feature_set_name`,
`feature_set_version`, and a per-feature `feature_versions` dict.

## HPO

`scripts/run_hpo.py` searches XGBoost params via the `HPOEngine` adapter
(`scripts/hpo/`, Optuna backend). It is a **build/experiment** dependency
(`optuna` in `requirements-build.txt`), not a runtime one.

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
python3 skills/heuriboost-rag/scripts/run_hpo.py examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output --n-trials 20 --seed 42 [--timeout-sec 120]
```

Outputs land in `examples/fiqa/output/hpo/` (gitignored): `hpo_report.md`
(val + test nDCG@10 + val−test gap + trial table), `best_params.json`
(params + `best_iteration` + scores + feature_set attribution), `trials.json`
(full trial history).

Key contracts (see `.trellis/spec/backend/hpo-contracts.md`):

- **Anti-leak**: the HPO SEARCH sees only train+valid snapshots (case-blind +
  test-blind by signature). Post-hoc test eval is a single forward pass, not
  optimization.
- **Determinism**: `nthread=1` + `TPESampler(seed=...)` → same-seed runs produce
  byte-identical `trials.json`.
- **Reproducibility**: retrain with `best_params` + `num_boost_round =
  best_iteration + 1` to reproduce the HPO-best model exactly.
- **nDCG scale**: HPO scores use the SAME raw-label `ndcg_at_k` as the shipped
  baseline (0.853), so they are directly comparable.
- **Overfit caveat**: on the 40-query FiQA validation, HPO overfits (val > test
  by ~0.08, test may fall below the 0.83 baseline). The report surfaces this
  honestly via the val−test gap.

## Ablation

`scripts/run_ablation.py` runs a spec §15.3 A/B/C/D feature ablation: given a
candidate feature, it tests whether the candidate helps after fair HPO tuning.

```bash
python3 skills/heuriboost-rag/scripts/run_ablation.py examples/fiqa/query_doc_examples.csv \
  --candidate-recipe candidate_recipe.yaml \
  --candidate-impl candidate_impl.py:candidate \
  --output-dir examples/fiqa/output --n-trials 5 --seed 42 \
  --regression-cases examples/fiqa/regression_cases.yaml
```

A candidate = a recipe YAML (spec §6.4 fields, `inputs` ⊆ `ALLOWED_INPUTS`) +
an impl fn `(row) -> float` (`--candidate-impl pyfile:func`). The framework
wraps it onto the shipped `extract_all` without modifying the registry (a
probe).

The 4 cells (baseline±candidate × fixed/HPO params) use the same training
procedure; B and D use the same HPO budget + seed. Outputs land in
`examples/fiqa/output/ablation/` (gitignored): `ablation_report.md` (cell
table + deltas + recommendation) + `ablation_result.json`.

Deltas: B-A (param gain), C-A (feature-only), **D-B (candidate gain after
tuning — primary)**, D-C (tuning gain with candidate).

Recommendation (report only — promotion is always manual):
- `promote` iff D-B(val) > threshold (default 0.01) AND D-B(test) > 0 AND D gate cases pass.
- `reject` iff D-B(val) ≤ 0 OR D regresses a gate case.
- `quarantine` otherwise.

The dual val+test+gate check avoids cherry-picking HPO-overfit validation
noise. See `.trellis/spec/backend/ablation-contracts.md` for full contracts.

## Candidate discovery

`scripts/run_discover_candidates.py` reads the per-case `failure_analysis.md`
for PENDING regression cases + the existing feature set, calls an LLM ONCE
(DeepSeek, JSON mode) to propose N candidate features, validates them
statically, and writes candidate files ready for `run_ablation.py`.

```bash
export DEEPSEEK_API_KEY=sk-...
python3 skills/heuriboost-rag/scripts/run_discover_candidates.py \
  --out-dir examples/fiqa/output/discovery --n-candidates 5
```

Outputs land in `examples/fiqa/output/discovery/` (gitignored):
`candidates/<name>/{recipe.yaml, impl.py}` per valid candidate +
`candidates_report.md` (table + a "⚠ review impl.py before running ablation"
warning).

The LLM sees pending cases' Feature Contrast + Suggested Actions + existing
feature names + the primitives API + `ALLOWED_INPUTS` + the recipe schema —
NOT labels, NOT case rows, NOT the full `extract_all` source. Generated
`impl_code` is validated statically (`ast.parse` + `def candidate` + import
allowlist); it is NOT `importlib`-loaded during generation (untrusted). The
user reviews `impl.py`, then runs `run_ablation.py` on a candidate to test it.

Invalid candidates are dropped + warned (1 LLM call, no retry). See
`.trellis/spec/backend/discovery-contracts.md` for full contracts.

## CSV contract

Required columns:

```csv
query_id,query_text,doc_id,doc_text,label,split
```

Recommended columns (enable richer features):

```csv
query_id,query_text,doc_id,chunk_id,doc_text,dense_rank,dense_score,sparse_rank,sparse_score,label,split
```

Rows are grouped by `query_id`; never shuffle query-document pairs across
groups. `split` is one of `train` / `validation` / `test`.

## Label scale

| Label | Meaning |
|---:|---|
| `3` | directly supports the answer |
| `2` | partially supports the answer |
| `1` | related but weak evidence |
| `0` | irrelevant |
| `-1` | misleading hard negative |

For XGBoost training, labels are mapped to non-negative ordered relevance
(`-1→0, 0→1, 1→2, 2→3, 3→4`). Evaluation keeps the original labels so hard
negatives stay visible in reports and gates.

## Regression cases

Regression cases are exam questions, never training rows. Each case carries a
`status`:

| Status | Behavior |
|---|---|
| `gate` | attacked & frozen. A failure blocks (exit non-zero). |
| `pending` | a known gap to attack. Evaluated and reported, failure does NOT block. |
| `retired` | invalidated by drift. Not evaluated; kept for history. |

A missing `status` defaults to `gate` (backward compatible).

Optional per-case local checks:

- `require_rank` (int): the first `must_include` doc must reach rank <= this value.
- `min_ndcg10` (float): the per-query nDCG@10 must be >= this value.

A case **passes** iff all `must_include` are within `top_k` (and the first
reaches `require_rank` if set) AND no `must_not_include` is within `top_k` AND
`min_ndcg10` is satisfied if set.

```yaml
cases:
  - case_id: fiqa_expense_deduction_wrong_topic
    query_id: fiqa_q_001
    status: gate
    require_rank: 3
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

If a `gate` case fails, `eval_reranker.py` exits non-zero. `pending` failures
are reported but do not change the exit code.

`--reckless` is an explicit variant: `train_reranker.py --reckless` defaults
`--case-sets` to `examples/fiqa/case_sets` when omitted, and
`eval_reranker.py --reckless --split test` hard-fails unless the ledger anchor
exists, the test split exists, every referenced `source_case_id` still passes
its original regression rule, and test `nDCG@10` + `MRR@10` both beat the
anchor.

Empty `case_sets` inputs are allowed; in reckless mode that means there are no
source cases to re-accept, but the test anchor comparison still runs.

## Cross-round ledger

`regression_ledger.py` owns cross-round memory in a committed
`examples/fiqa/ledger.json` (version-controlled, NOT gitignored, NOT
auto-committed). Each evaluation round appends a snapshot (global metrics,
per-case pass/fail, and a comparison against the anchored baseline). The anchor
is a frozen snapshot's global metrics, manually refreshed when gains are
confirmed.

```bash
# After an eval round, set the anchor (manual, one-time or on confirmed gains):
python skills/heuriboost-rag/scripts/regression_ledger.py set-anchor --ledger examples/fiqa/ledger.json

# Print a progress summary (gate/pending counts, promotion candidates, baseline line):
python skills/heuriboost-rag/scripts/regression_ledger.py summary --ledger examples/fiqa/ledger.json

# Promote a pending case to gate (interactive confirmation, no auto-promotion):
python skills/heuriboost-rag/scripts/regression_ledger.py promote examples/fiqa/regression_cases.yaml <case_id> --ledger examples/fiqa/ledger.json
```

The anchored-baseline comparison is **reported, not auto-blocking** in the
default flow — promotion is always a manual decision. Use `--no-ledger` on
`eval_reranker.py` to skip ledger writes for ad-hoc eval. Reckless mode is
stricter and exits non-zero when test `nDCG@10` or `MRR@10` fail to beat the
anchor.

## Production-case repair

The user-facing reckless repair flow starts from two tables and compiles the
older internal artifacts automatically.

`base_dataset.csv` is the stable dataset for train, validation, and metric-level
test acceptance. Minimal columns:

```csv
query,text,relevance
```

Recommended columns:

```csv
domain,query_id,query,doc_id,text,relevance,split,rank,score
```

`production_cases.csv` is the online incident / feedback table. Minimal columns:

```csv
query,shown_doc_text,user_verdict
```

Recommended columns:

```csv
domain,case_id,query,shown_doc_id,shown_doc_text,user_verdict,rank,score
```

`domain` is optional and defaults to `default`, but once present it is a hard
boundary for synthetic ids, candidate completion, promoted repair memory,
historical gates, and touched-domain checks. If `split` is present in
`base_dataset`, the compiler respects it; if absent, it deterministically
auto-splits by query. A query with only one doc is a compile warning, but strict
repair still requires validation/test query groups to have at least two docs.

Label aliases for `base_dataset.relevance`:

| Alias | Internal label |
|---|---:|
| `good`, `positive` | `3` |
| `partial` | `2` |
| `weak` | `1` |
| `irrelevant`, `negative` | `0` |
| `bad`, `hard_negative` | `-1` |

`production_cases.user_verdict` is one of:

| Verdict | Behavior |
|---|---|
| `good` | positive repair sample and full-acceptance target |
| `bad` | hard-negative repair sample and suppression target |
| `unknown` | context only; not training or acceptance input |

Commands:

```bash
python3 skills/heuriboost-rag/scripts/compile_cases.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_full.csv \
  --output-dir examples/fiqa/output \
  --strict

python3 skills/heuriboost-rag/scripts/repair_reranker.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_full.csv \
  --output-dir examples/fiqa/output \
  --reckless

python3 skills/heuriboost-rag/scripts/promote_repair.py \
  --output-dir examples/fiqa/output
```

Generated audit artifacts land under `output/.heuriboost/compiled/`:
`query_doc_examples.csv`, `regression_cases.yaml`, `case_sets/`, and
`production_cases.json`. These are not user prerequisites.

Strict repair behavior:

- missing anchor initializes from a base-dataset-only baseline;
- existing anchor is reused unless `--reset-anchor` is explicit;
- one user-visible candidate model is written under `output/models/`;
- current production cases are added to repair training;
- base test remains the metric-level regression suite and is not silently
  extended with production cases;
- historical gates are self-contained case snapshots, not base-test rows.

Full acceptance is default. It requires at least one good doc in top-k, every
bad doc outside top-k, all historical gates passing, global base-test
`nDCG@10` and `MRR@10` both above anchor, and touched-domain metrics not below
their domain anchor. `--acceptance-level weak` allows bad-only suppression
checks, but the run is never promotion eligible.

`promote_repair.py` refuses failed or weak runs. A successful promotion refreshes
the repair anchor, freezes full production cases as historical gates, appends
promoted repair samples, and writes `output/.heuriboost/current_model.json`.
It does not mutate the user's input CSVs or deploy online.

## Closing the loop: case_sets mining

Pending cases are known gaps to attack. The textbook path is to mine
same-pattern training samples from the corpus, fold them into the train split,
and re-evaluate. The case itself stays an exam question — only mined samples
that are kept separate from the cases enter training.

The four-command closed loop (run by the maintainer; no auto-promotion):

```bash
# 1. Mine same-pattern samples for all pending cases (needs build deps)
python skills/heuriboost-rag/scripts/mine_case_sets.py \
  --dataset examples/fiqa/query_doc_examples.csv \
  --cases examples/fiqa/regression_cases.yaml \
  --out-dir examples/fiqa/case_sets

# 2. Retrain with mined samples folded into the train split
python skills/heuriboost-rag/scripts/train_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --case-sets examples/fiqa/case_sets \
  --regression-cases examples/fiqa/regression_cases.yaml

# 2b. Reckless variant: omit --case-sets to default to examples/fiqa/case_sets
python skills/heuriboost-rag/scripts/train_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --reckless

# 3. Eval + ledger (tags the round as having used case_sets)
python skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split validation \
  --regression-cases examples/fiqa/regression_cases.yaml \
  --case-sets-used

# 3b. Reckless acceptance: evaluate test and require anchor improvement
python skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split test \
  --reckless

# 4. (manual) If a pending case passed AND the baseline check is OK, promote it
python skills/heuriboost-rag/scripts/regression_ledger.py promote \
  examples/fiqa/regression_cases.yaml <case_id> --ledger examples/fiqa/ledger.json
```

**Mining rule** = intersection of three signals: semantic similarity to the
case's query (`all-MiniLM-L6-v2`, top-K), the same failure shape (hard negative
at `dense_rank <= --shape-rank`, positive at `dense_rank >= --shape-pos-gap`),
and the same `failure_type`.

**Isolation**: no mined `query_id` may equal any case's `query_id`, and no mined
`doc_id` may equal any case's `must_include`/`must_not_include` doc_id. A
defensive re-check runs again at training load time.

`sentence-transformers` is a build dependency (`requirements-build.txt`), not a
runtime dependency. Mining reuses `examples/fiqa/.cache/query_embeddings.npz`
when present.

> **Pipeline-validation caveat**: attack results under heuristic labels are
> pipeline-validation grade, not benchmark. They test whether the closed-loop
> mechanics work (mine → train → eval → promote), not whether the attack
> credibly moves a pending case. Credible attack quality waits for LLM-mode
> labels (`--label-mode llm` in `build_fiqa_csv.py`).

## Reports

`eval_reranker.py` writes to `examples/fiqa/output/reports/`:

| File | Contents |
|---|---|
| `eval_report.md` | Global metrics + regression gate status (Gates + Pending sections). |
| `ranking_diff.csv` | Before/after rank movement (dense rank as default baseline). |
| `failure_cases.md` | Hard-negative exposure report for the top 3. |
| `failure_analysis.md` | Deterministic regression-case analysis: reason summary, rank movement, evidence hits, feature contrast, suggested next actions. |
| `feature_importance.json` | XGBoost gain-based feature importance, normalized across the feature list. |

`failure_analysis.md` is deterministic lite analysis, not automatic feature
discovery.

## Agent skill

The Codex-compatible skill lives in `skills/heuriboost-rag/SKILL.md` and exposes
three modes:

- `audit` — scan a RAG repo for retriever/eval/log/dataset signals
- `bootstrap` — copy templates and explain the CSV contract
- `experiment` — validate CSV, train, evaluate, and inspect reports

Other coding agents can run the Python scripts manually.

## Regenerating the demo dataset

The committed `examples/fiqa/query_doc_examples.csv` is generated offline from
BEIR/FiQA-2018 by `build_fiqa_csv.py`. It runs BM25 + `all-MiniLM-L6-v2` + RRF
retrieval (FiQA ships no candidates), then labels with one of two modes.

Heuristic mode — zero-cost, deterministic, no LLM (this produced the committed CSV):

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
python skills/heuriboost-rag/scripts/build_fiqa_csv.py \
  --label-mode heuristic --output examples/fiqa/query_doc_examples.csv
```

LLM mode — full 5-level labels via an OpenAI-compatible judge (DeepSeek by default):

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
export DEEPSEEK_API_KEY=sk-...   # or OPENAI_API_KEY with --base-url ""
python skills/heuriboost-rag/scripts/build_fiqa_csv.py \
  --label-mode llm --output examples/fiqa/query_doc_examples.csv
```

Both modes need network access (to download FiQA); only LLM mode needs an API
key. The build runs locally, not in CI. Heavy build dependencies, the FiQA
corpus, and dense-encoder weights are not committed. See
`examples/fiqa/DATA_CARD.md` for provenance.
