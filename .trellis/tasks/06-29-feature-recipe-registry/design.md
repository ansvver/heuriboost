# Design: Feature Recipe Registry DSL

## Architecture

New subpackage `skills/heuriboost-rag/scripts/features/`:

```text
scripts/features/
  __init__.py        # eager-loads YAML, registers impls, validates, re-exports
  registry.py        # Recipe dataclass, FeatureRegistry, validation, YAML loader
  primitives.py      # tokenize, numbers, entities, rank_inverse, numeric_value
  recipes.py         # extract_all(row) -> dict[str,float]  (verbatim current body)
```

`common.py` keeps **re-exports** so existing callers (`train_reranker.py`,
`eval_reranker.py`) change minimally:

```python
from features import REGISTRY, extract_features, FEATURE_NAMES
from features.primitives import tokenize, numbers, entities, rank_inverse, numeric_value
```

`rank_by_baseline` stays in `common.py` (uses `numeric_value` via the re-export).

## Data flow

```text
feature_recipes.yaml (source of truth for metadata)
        |
        v  eager load on `import common` (via features.__init__)
  FeatureRegistry
   - recipes: dict[name -> Recipe]              (metadata from YAML)
   - impls:   dict[impl_name -> callable(row) -> dict[str, float]]
              (V0 has ONE impl: "extract_all" -> recipes.extract_all)
   - validate(): impl resolves, inputs allowlist, online_safe, required fields
        |
        v  REGISTRY.extract(df)
  pd.DataFrame[FEATURE_NAMES]   (same 12 cols, same order, same values)
```

**Extraction shape (Option C, confirmed)**: a single shared function
`recipes.extract_all(row) -> dict[str, float]` holds the verbatim body of the
current `extract_features` per-row computation. `REGISTRY.extract(df)` iterates
rows, calls `extract_all`, and assembles a `pd.DataFrame` with `names()` cols.
This guarantees bit-for-bit identity (A1) because it IS the same function.
Per-feature dispatch is deferred to the ablation task.

**Load timing (Eager, confirmed)**: `features/__init__.py` loads the YAML,
imports `recipes` (which registers `extract_all`), and runs `validate()` at
import time. `common.py` imports `features` at top level, so any script that
`import common` (including `validate_dataset.py`, `mine_case_sets.py`,
`build_fiqa_csv.py`) triggers registry validation once. Fail-fast: a broken
YAML surfaces immediately, not at first train. Cost ~50ms, negligible.

## Recipe dataclass (registry.py)

```python
@dataclass(frozen=True)
class Recipe:
    name: str
    version: int
    description: str
    task_profiles: tuple[str, ...]
    inputs: tuple[str, ...]
    type: str               # "numeric" for all V0
    default_value: float
    cost_tier: str          # "L0".."L3"
    online_safe: bool
    leakage_risk: str       # "low"|"medium"|"high"
    expected_slices: tuple[str, ...]   # may be empty
    owner: str
    impl: str               # implementation reference, e.g. "extract_all"
```

`FeatureRegistry`:

- `register_impl(name, fn)` — register a shared impl function by logical name
  (e.g. `"extract_all"`). The fn signature is `(row) -> dict[str, float]`.
- `load_yaml(path)` — parse `feature_recipes.yaml` into `Recipe` objects.
- `validate()` — run at end of `__init__.py` load:
  1. every recipe's `impl` resolves to a registered impl name;
  2. every `inputs` value is in `ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score, sparse_rank, sparse_score}` (tight — no `chunk_id`/`doc_text_ref`/`label`/`split`/`query_id`/`doc_id`);
  3. `online_safe` is true for the active task profile (`qd_reranker`); false → raise;
  4. required fields non-empty, EXCEPT `expected_slices` which may be empty (forward-looking declaration).
- `names()` — canonical ordered list (YAML order).
- `extract(df)` — iterate rows, call the resolved impl, return `pd.DataFrame` with `names()` columns. Bit-for-bit identical output to current `extract_features`.
- `feature_set_name` / `feature_set_version` — from YAML top-level `feature_set.{name,version}`.
- `feature_versions()` — `{name: version}` dict for full spec-faithful metadata.

## feature_recipes.yaml changes

Extend each of the 12 features with the missing spec fields. Use a `defaults`
YAML anchor for the repeated fields:

```yaml
feature_set:
  name: heuriboost_rag_v0
  version: 1

defaults: &defaults
  task_profiles: [qd_reranker]
  type: numeric
  default_value: 0.0
  cost_tier: L0
  online_safe: true
  leakage_risk: low
  owner: heuriboost-rag
  expected_slices: []
  impl: extract_all

features:
  - name: dense_score
    version: 1
    description: "Dense retriever score when available."
    inputs: [dense_score]
    <<: *defaults
  ...
```

Each feature declares its own `name, version, description, inputs` (the
discriminating fields); the shared fields come from `&defaults`. All 12 point
to `impl: extract_all` (the single shared function).

## recipes.py

One registered function — the verbatim body of the current `extract_features`
per-row computation, returning a dict:

```python
from features.primitives import tokenize, numbers, entities, rank_inverse, numeric_value
import math

def extract_all(row) -> dict[str, float]:
    query_text = str(row["query_text"])
    doc_text = str(row["doc_text"])
    query_tokens = tokenize(query_text)
    doc_tokens = tokenize(doc_text)
    shared_tokens = query_tokens & doc_tokens
    # ... (verbatim from common.py:207-256) ...
    return {
        "dense_score": numeric_value(row, "dense_score"),
        "dense_rank_inverse": dense_rank_inv,
        # ... all 12 ...
    }
```

Registration in `__init__.py`:

```python
from features.registry import REGISTRY
from features.recipes import extract_all
REGISTRY.register_impl("extract_all", extract_all)
REGISTRY.load_yaml(DEFAULT_YAML_PATH)
REGISTRY.validate()
```

## train_reranker.py change

`reranker_metadata.json` gains three fields sourced from the registry:

```json
{
  "feature_names": [...],
  "feature_set_name": "heuriboost_rag_v0",
  "feature_set_version": 1,
  "feature_versions": {"dense_score": 1, ...}
}
```

## Compatibility & migration

- `common.py::extract_features` becomes `return REGISTRY.extract(df)`.
- `common.py::FEATURE_NAMES` becomes `REGISTRY.names()`.
- No caller changes required (re-exports preserve import sites).
- The `fiqa-demo-contracts.md` contract "FEATURE_NAMES must equal
  feature_recipes.yaml" is now enforced at `REGISTRY` load time (A4).
- The "features must be prediction-time computable" contract is enforced via
  the `ALLOWED_INPUTS` check (A5).

## Trade-offs

- **Option C (shared fn) vs per-feature dispatch**: chose shared fn for V0 —
  bit-for-bit identity is guaranteed (A1) and ablation subset is deferred. When
  ablation lands, add a per-feature dispatch path then.
- **Subpackage vs flat file**: subpackage matches spec's future layout
  (`features/{registry.py, primitives.py, extractors/}`). No `pyproject.toml`.
- **Eager vs lazy load**: eager (fail-fast, single validation entry, ~50ms).
- **Hard-fail validation vs warn**: hard-fail. V0's 12 features all pass.
- **Re-exports vs updating callers**: re-exports minimize blast radius.
- **`task_profiles: [qd_reranker]` as a string**: introduces the profile name
  without a TaskProfile registry (out of scope).
- **Tight `ALLOWED_INPUTS`**: only the 6 columns V0 actually uses; `chunk_id`/
  `doc_text_ref` omitted (no feature uses them). Tighter leakage control.
- **`feature_versions` dict in metadata**: spec-faithful ("preserve feature
  recipe versions"), cheap, included.

## Rollback

Pure refactor + additive metadata. Rollback = revert the commit. No data
migration, no artifact format change that breaks old models (old
`reranker_metadata.json` simply lacks the new fields).
