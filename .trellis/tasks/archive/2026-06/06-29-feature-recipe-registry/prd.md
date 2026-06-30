# PRD: Feature Recipe Registry DSL

## Goal

Replace the hardcoded `extract_features` function in
`skills/heuriboost-rag/scripts/common.py` with a declared `FeatureRecipe`
registry/DSL, per `docs/specs/ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md` §6.4 and
Milestone 2. This is the next item in the spec's implementation order (#4,
skipped in V0) and a prerequisite for automatic feature discovery, ablation,
and HPO.

## User value

- Unblocks future feature-discovery / ablation / HPO work (features become
  addressable, versioned, validated objects rather than scattered code).
- Makes the "FEATURE_NAMES must equal feature_recipes.yaml" contract
  machine-enforced instead of grep-enforced (`fiqa-demo-contracts.md:48`).
- Makes leakage / online-safety a gate at registration time, not a code review
  hope.

## Confirmed facts (from inspection)

- `common.py:34-47` defines `FEATURE_NAMES` (12 names); `common.py:207-256`
  `extract_features()` computes them in one monolithic function.
- `templates/feature_recipes.yaml` already declares all 12 features with
  `name/type/default_value/online_safe/leakage_risk/description` — but is a
  static template, NOT loaded or validated by code. Missing spec fields:
  `version, task_profiles, inputs, cost_tier, expected_slices, owner`.
- Spec §6.4 required fields: `name, version, description, task_profiles,
  inputs, expr OR implementation reference, type, default_value, cost_tier,
  online_safe, leakage_risk, expected_slices, owner`.
- Callers: `train_reranker.py:199,201,234` and `eval_reranker.py:232,563,564,592`.
  `build_fiqa_csv.py` does NOT call `extract_features` (it only builds raw CSV).
- Executable contract `fiqa-demo-contracts.md:48-66`: the three locations
  (FEATURE_NAMES, extract_features dict keys, feature_recipes.yaml) MUST be
  identical. Today this is a grep-time check; the registry will make it a
  load-time check.
- Contract `fiqa-demo-contracts.md:70-79`: every feature must be computable
  from `query_text + doc_text + retriever scores/ranks` alone.
- Spec Milestone 2: "Implement V0 feature registry. Store feature matrix with
  feature set version. Add leakage checks and online-safe flags. Add feature
  health report."
- All 12 current features are `online_safe: true, leakage_risk: low`.

## Requirements

R1. A `FeatureRecipe` registry where each feature is a declared object with
    all spec §6.4 required fields (metadata) plus an implementation reference.
R2. The shipped `feature_recipes.yaml` is the source of truth for metadata;
    a Python module registers one shared implementation function
    (`extract_all`) keyed by logical name. The registry validates that every
    recipe's `impl` field resolves to a registered impl name (enforces the
    contract at load time).
R3. `extract_features(df)` behavior is preserved bit-for-bit (same 12 features,
    same values, same canonical order) — verified by direct DataFrame identity
    on the demo CSV, not just metrics. Callers (`train_reranker.py`,
    `eval_reranker.py`) continue to work via re-exports; `FEATURE_NAMES` is
    derived from the registry.
R4. Leakage / online-safety validation: a feature's declared `inputs` must be
    in a tight allowlist (`query_text, doc_text, dense_rank, dense_score,
    sparse_rank, sparse_score` — the 6 columns V0 actually uses; `label`,
    `split`, `query_id`, `doc_id`, `chunk_id`, `doc_text_ref` are excluded as
    leakage/identifier vectors). A feature with `online_safe: false` is rejected
    for the Q-D reranker task profile. Registration fails loud (not a warning)
    — V0 stays green because all 12 features satisfy both.
R5. Feature-set version is recorded in the trained model artifact metadata
    (extend `train_reranker.py`'s `reranker_metadata.json` with
    `feature_set_name`, `feature_set_version`, and a per-feature
    `feature_versions` dict, all sourced from the YAML/registry).

## Acceptance criteria

- A1. `extract_features(df)` output is bit-for-bit identical to the pre-refactor output on the demo CSV, verified by `df.equals` on a snapshot taken before checkpoint 2. Train then produces identical metrics (nDCG@10 val ≈ 0.853).
- A2. `python3 skills/heuriboost-rag/scripts/eval_reranker.py ...` still passes all gate cases (2/2 gates green).
- A3. `reranker_metadata.json` now contains `feature_set_name`, `feature_set_version`, and `feature_versions`.
- A4. Removing a feature name from `feature_recipes.yaml` OR registering no impl for a recipe's `impl` field causes a loud load-time error (contract enforced).
- A5. Declaring a feature with `inputs: [label]` (a leakage vector) causes registration to fail with a clear message.
- A6. `python3 -m py_compile skills/heuriboost-rag/scripts/*.py skills/heuriboost-rag/scripts/features/*.py` passes.
- A7. `docs/REFERENCE.md` + `docs/REFERENCE.zh-CN.md` updated to describe the registry; README checklist item "Feature registry / recipe DSL" moves from not-yet to done.

## Resolved decisions

- **Mechanism**: YAML metadata (source of truth) + Python implementation
  reference, per spec §6.4 "expr OR implementation reference" (latter). Not a
  pure expr DSL — V0 features are regex/math, awkward in a generic interpreter.
- **Extraction shape (Option C)**: one shared function `recipes.extract_all(row)`
  holds the verbatim body of the current `extract_features`; `REGISTRY.extract`
  calls it. Guarantees bit-for-bit identity (A1) because it IS the same
  function. Per-feature dispatch deferred to the ablation task.
- **Load timing (Eager)**: `features/__init__.py` loads YAML + registers impl +
  validates at import; `common.py` imports `features` at top, so any
  `import common` (validate_dataset / mine_case_sets / train / eval) triggers
  validation once. Fail-fast, ~50ms, single validation entry. `build_fiqa_csv`
  and `inspect_rag_repo` do not import common → unaffected.
- **Scope**: full version — registry core + YAML field completion + hard-fail
  validation + feature_set version in model metadata. Deferred: feature health
  report, ablation subset / `--feature-recipes` CLI override, feature discovery,
  HPO, other task profiles, `pyproject.toml`.
- **Layout**: new `scripts/features/{__init__.py, registry.py, primitives.py,
  recipes.py}` subpackage (matches spec future layout). No `pyproject.toml`.
- **Backward compat**: `common.py` re-exports `FEATURE_NAMES`,
  `extract_features`, and primitive helpers, so `train_reranker.py` /
  `eval_reranker.py` import sites are unchanged. Primitive helpers
  (`tokenize/numbers/entities/rank_inverse/numeric_value`) are only used inside
  `common.py` (verified via grep) — safe to move. `eval_reranker.py` does NOT
  read `reranker_metadata.json` → new metadata fields are zero-risk for eval.
- **Task profile**: string `"qd_reranker"` as the active profile (no TaskProfile
  registry in this task).
- **Validation**: hard-fail (not warn) on online_safe=false, inputs outside the
  tight 6-item allowlist, impl not registered, empty required field.
  `expected_slices` may be empty (forward-looking). V0's 12 features all pass.
- **A1 verification (strong)**: direct DataFrame identity (`df.equals`) on the
  demo CSV against a pre-refactor snapshot, not just metrics.
- **Commit granularity (two checkpoints)**: CP1 adds `features/` + extended
  YAML (common.py untouched, demo on old path); CP2 refactors common.py + train
  metadata. Clean rollback point between additive and invasive changes.

## Out of scope (deferred)

- Feature health report (spec Milestone 2 stretch).
- Ablation subset selection / `--feature-recipes <path>` CLI override.
- Automatic feature discovery / promotion (`FeatureMemory`).
- HPO adapter; other task profiles; `pyproject.toml` scaffold.

## Technical notes

See `design.md` for the `Recipe` dataclass, `FeatureRegistry` API, YAML
structure, and `recipes.py` registration pattern. See `implement.md` for the
ordered checklist and validation commands.
