# PRD: LLM Candidate Feature Generation (child of feature-discovery)

## Parent / dependencies

Child #3 of `.trellis/tasks/06-29-feature-discovery/`. Depends on child #2
(ablation framework, shipped `5e821e3`) — the candidates this task produces are
consumed by `run_ablation.py`. No code dependency on child #1 (HPO) directly.

**External blocker**: requires `DEEPSEEK_API_KEY` env var. The user pasted a
key in chat (leak — must be rotated after this session). Code reads from env,
never hardcodes.

## Goal

Build the candidate-generation half of the spec §15.2 discovery loop: read the
per-case `failure_analysis.md` (Feature Contrast + Suggested Actions) +
regression cases + the existing feature set, call an LLM ONCE to propose N
candidate FeatureRecipes (recipe YAML + impl Python fn), validate them, and
write candidate files ready for `run_ablation.py`.

## User value

- Closes the discovery loop: with #2 (ablation) done, this is the only missing
  piece. After #3, the flow is `#3 generates → run_ablation per candidate →
  human promotes`.
- Turns `failure_analysis.md` (already produced per eval) into concrete,
  testable feature candidates instead of passive prose.
- Cheap: ~1 LLM call (vs ~4.5k for LLM label generation).

## Confirmed facts (from inspection)

- Spec §15.4: LLM may propose hypotheses ("try merchant_velocity_24h"), NOT
  bypass validation. Unstructured feature code with no recipe entry is
  forbidden. Every generated candidate MUST become a FeatureRecipe + impl.
- Spec §15.2 step 4: "Generate candidate FeatureRecipe in DSL."
- `failure_analysis.md` (from `eval_reranker.py`) has per-case: Reason Summary,
  Rank Movement table, Feature Contrast (required vs forbidden delta per
  feature), Suggested Next Actions. ~4 pending cases.
- `build_fiqa_csv.py:90-163` `LLMJudge`: OpenAI-compatible client, lazy init,
  reads `DEEPSEEK_API_KEY` then `OPENAI_API_KEY`, defaults to `deepseek-chat`
  + `https://api.deepseek.com`. Reusable pattern.
- Candidate contract (from ablation-framework): recipe YAML (spec §6.4 fields,
  `inputs` ⊆ `ALLOWED_INPUTS`, `online_safe: true`, required fields) + impl fn
  `(row) -> float` loaded via `--candidate-impl pyfile:func`.
- `ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score,
  sparse_rank, sparse_score}`.
- `features.recipes.extract_all` + `feature_recipes.yaml` define the existing
  12 features (feed names+descriptions to LLM to avoid duplicate proposals).

## Requirements (draft — pending grilling)

R1. A `run_discover_candidates.py` CLI: `--failure-analysis <path>`,
    `--regression-cases <path>`, `--out-dir <dir>`, `--n-candidates N`
    (default 5), `--model deepseek-chat`, `--base-url https://api.deepseek.com`.
    Reads `DEEPSEEK_API_KEY` env var; clear error if missing.
R2. ONE LLM call (batch): feed pending cases' Feature Contrast + Suggested
    Actions + existing 12 feature names/descriptions + ALLOWED_INPUTS, ask LLM
    to propose N candidate FeatureRecipes as structured output.
R3. Parse LLM output into candidate (recipe YAML, impl Python) pairs.
R4. Validate each candidate: recipe has all spec §6.4 required fields, `inputs`
    ⊆ ALLOWED_INPUTS, `online_safe: true`, impl is a callable `(row) -> float`
    that loads cleanly. Drop invalid candidates with a clear warning.
R5. Write valid candidates to `<out-dir>/candidates/<name>/{recipe.yaml,
    impl.py}` — ready for `run_ablation.py --candidate-recipe ... --candidate-impl ...`.
R6. Do NOT execute generated impl code. Write a `candidates_report.md` listing
    proposals + validation status + a "review before running ablation" warning
    (generated Python is executed by run_ablation.py later; user must review).
R7. Deterministic given the same LLM output (no hidden state; the LLM call
    itself is non-deterministic but `--seed`/temperature can be set if the API
    supports it).

## Resolved decisions

- **LLM output structure**: DeepSeek JSON mode
  (`response_format={"type":"json_object"}`). One response:
  `{"candidates": [{"recipe": {...}, "impl_code": "def candidate(row):\\n ..."}]}`.
  `json.loads` parses; `impl_code` is a string field. Reliable + one call.
- **LLM context (full)**: pending cases' Feature Contrast + Suggested Actions
  + existing 12 feature name/description (avoid duplicates) + primitives API
  signatures (`tokenize/numbers/entities/numeric_value/rank_inverse`) +
  `ALLOWED_INPUTS` + recipe schema. ~2-3k tokens. NOT the full `extract_all`
  source (avoid bloat + style lock-in).
- **Invalid candidate handling**: drop + warn. 1 call total; low invalid rate
  with full context; user re-runs if too few valid. No retry loop.
- **Validation (static, safe)**: `ast.parse(impl_code)` + check `candidate`
  function defined + recipe field/inputs/online_safe checks. NO `importlib`
  load during generation — LLM-generated code is untrusted; execution deferred
  to `run_ablation.py` after human review.
- **Candidate count**: default 5 (`--n-candidates`).
- **Temperature**: 0.7 (creativity + stability). DeepSeek has no reliable seed;
  candidate generation is a creative step, non-determinism acceptable.
- **Client wrapper**: inline ~30-line `_LLMClient` (same pattern as
  `build_fiqa_csv.py::LLMJudge`); not refactored into a shared module to avoid
  touching `build_fiqa_csv.py` (blast radius).
- **Scope of THIS task**: generate + validate + write candidate files. Does NOT
  execute generated code, does NOT auto-run ablation, does NOT auto-promote.
  Closes the discovery loop when combined with `run_ablation.py` (already done)
  + human promotion.

## Out of scope (deferred)

- Discovery orchestration / batch ablation loop (child #4, deferred — manual
  loop suffices for V0 candidate volume).
- FeatureMemory (record promote/reject/quarantine institutionally).
- Scout-vs-finalist staging.
- Automatic promotion.
- LLM label generation (separate V1 step 3, ~4.5k calls).
- Shared LLM client module (unify with `build_fiqa_csv.py::LLMJudge` later).
