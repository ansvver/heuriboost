# Feature Recipe Registry Contracts

> Executable contracts for the `FeatureRecipe` registry (`scripts/features/`).
> Captured 2026-06-29 after the feature-recipe-registry task. The
> "FEATURE_NAMES must equal feature_recipes.yaml" contract lives in
> `fiqa-demo-contracts.md`; this file covers the registry module itself.

---

## Scenario: FeatureRecipe registry load + validation

### 1. Scope / Trigger

Any change to `scripts/features/`, `templates/feature_recipes.yaml`, or the
feature-extraction path. The registry is a cross-layer contract (YAML metadata
↔ Python impl ↔ model metadata), so code-spec depth is mandatory.

### 2. Signatures

```python
# scripts/features/registry.py
class FeatureRegistry:
    def register_impl(self, name: str, fn: Callable[[row], dict[str, float]]) -> None
    def load_yaml(self, path: str | Path) -> None
    def validate(self) -> None                       # hard-fail (SystemExit)
    def names(self) -> list[str]                      # canonical YAML order
    def extract(self, df) -> "pd.DataFrame"           # columns = names()
    @property
    def feature_set_name(self) -> str
    @property
    def feature_set_version(self) -> int
    def feature_versions(self) -> dict[str, int]

# scripts/features/recipes.py
def extract_all(row) -> dict[str, float]              # the ONE shared impl (Option C)
```

### 3. Contracts

**YAML (`templates/feature_recipes.yaml`)** — source of truth for metadata:

| Field | Required | Constraint |
|---|---|---|
| `feature_set.name` | yes | non-empty string |
| `feature_set.version` | yes | integer |
| `features[].name` | yes | non-empty, unique |
| `features[].version` | yes | integer |
| `features[].description` | yes | non-empty |
| `features[].task_profiles` | yes | list containing `qd_reranker` |
| `features[].inputs` | yes | every item in `ALLOWED_INPUTS` |
| `features[].impl` | yes | resolves to a registered impl name |
| `features[].type` | yes | one of `numeric`/`categorical`/`boolean` |
| `features[].default_value` | no | float (default 0.0) |
| `features[].cost_tier` | yes | one of `L0`/`L1`/`L2`/`L3` |
| `features[].online_safe` | yes | `true` for the active profile |
| `features[].leakage_risk` | yes | one of `low`/`medium`/`high` |
| `features[].expected_slices` | yes | list; MAY be empty |
| `features[].owner` | yes | non-empty |

**`ALLOWED_INPUTS`** = `{query_text, doc_text, dense_rank, dense_score,
sparse_rank, sparse_score}`. Anything else (`label`, `split`, `query_id`,
`doc_id`, `chunk_id`, `doc_text_ref`) is a leakage/identifier vector and is
rejected.

**Active task profile** = `qd_reranker` (string; no TaskProfile registry yet).

**Model metadata (`reranker_metadata.json`)** gains three fields:
`feature_set_name` (str), `feature_set_version` (int), `feature_versions`
(dict[str,int]).

### 4. Validation & Error Matrix

| Condition | Error |
|---|---|
| `feature_recipes.yaml` missing | `SystemExit: Feature recipes file not found: <path>` |
| no PyYAML installed | `SystemExit: PyYAML is required...` |
| `feature_set` missing/non-mapping | `SystemExit: ...must declare a top-level 'feature_set'` |
| `feature_set.name`/`version` missing | `SystemExit: feature_set.name and feature_set.version are required` |
| `features` missing/empty/non-list | `SystemExit: ...non-empty 'features' list` |
| duplicate feature name | `SystemExit: Duplicate feature name...` |
| required field missing/empty | `SystemExit: Feature recipe is missing required field '<field>'` |
| `version` not int | `SystemExit: Feature '<name>' version must be an integer` |
| active profile not in `task_profiles` | `SystemExit: ...active profile is 'qd_reranker'` |
| `cost_tier` invalid | `SystemExit: cost_tier must be one of...` |
| `leakage_risk` invalid | `SystemExit: leakage_risk must be one of...` |
| `type` invalid | `SystemExit: type must be one of...` |
| `impl` not registered | `SystemExit: ...declares impl='<x>' but no such implementation is registered` |
| input outside `ALLOWED_INPUTS` | `SystemExit: ...'<input>' is not in ALLOWED_INPUTS... (leakage)` |
| `online_safe: false` | `SystemExit: ...requires online-safe features` |
| duplicate impl registration | `SystemExit: impl '<name>' is already registered` |

### 5. Good / Base / Bad Cases

- **Good**: add a feature by (a) adding a recipe block to YAML with all
  required fields + `impl: extract_all`, (b) ensuring `extract_all` returns a
  dict key with that name. Loads clean.
- **Base**: bump an existing feature's `version` (e.g. 1→2) after changing its
  computation. Loads clean; `feature_versions` reflects the bump.
- **Bad**: add a recipe with `inputs: [label]`. Registry refuses to load;
  every `import common` script exits with the leakage message.

### 6. Tests Required

- A1 bit-for-bit: `pandas.testing.assert_frame_equal(new, old_snapshot)`
  dtype-strict on the demo CSV after any change to `extract_all`.
- Load test: `from features import REGISTRY; len(REGISTRY.names()) == 12;
  REGISTRY.feature_set_name == "heuriboost_rag_v0"`.
- Negative test: set `inputs: [label]` on one feature → `import common` raises
  SystemExit. Revert.
- Order test: `REGISTRY.names() == FEATURE_NAMES == list(reranker_metadata["feature_names"])`.
- py_compile: `python3 -m py_compile scripts/*.py scripts/features/*.py`.

### 7. Wrong vs Correct

#### Wrong — per-feature dispatch with re-tokenization

```python
@REGISTRY.recipe("term_overlap_ratio")
def term_overlap_ratio(row, ctx):
    q = tokenize(row["query_text"])   # re-tokenizes per feature
    d = tokenize(row["doc_text"])
    return len(q & d) / max(len(q), 1)
```

12 features each re-tokenizing → 7× redundant work, and A1 bit-for-bit identity
becomes a careful ctx-alignment exercise instead of a structural guarantee.

#### Correct — single shared `extract_all` (Option C)

```python
def extract_all(row) -> dict[str, float]:
    query_tokens = tokenize(row["query_text"])   # tokenize once
    doc_tokens = tokenize(row["doc_text"])
    shared = query_tokens & doc_tokens
    return {
        "dense_score": numeric_value(row, "dense_score"),
        "term_overlap_ratio": len(shared) / max(len(query_tokens), 1),
        # ... all 12, verbatim from the old extract_features
    }
```

One pass per row, tokens shared. A1 is guaranteed by construction (it IS the
old function body). Per-feature dispatch is deferred to the ablation task.

---

## Design Decision: eager load

**Context**: when should the registry load YAML + validate?

**Options**:
1. Eager — `import common` triggers load + validate.
2. Lazy — load on first `REGISTRY.extract()` / `names()` access.

**Decision**: Eager. A broken `feature_recipes.yaml` breaks the whole skill, so
failing fast on any `import common` (including `validate_dataset.py`,
`mine_case_sets.py`) is better than deferring the failure to train/eval. Cost
~50ms. `build_fiqa_csv.py` and `inspect_rag_repo.py` do not import `common`, so
they are unaffected.

---

## Convention: re-exports preserve import sites

`common.py` re-exports `FEATURE_NAMES`, `extract_features`, and the primitive
helpers (`tokenize/numbers/entities/numeric_value/rank_inverse`) so
`train_reranker.py` / `eval_reranker.py` import sites are unchanged. When a
future task migrates callers to `from features import REGISTRY` directly, these
re-exports can be removed.
