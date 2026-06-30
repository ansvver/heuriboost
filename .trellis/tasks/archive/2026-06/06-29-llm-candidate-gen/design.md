# Design: LLM Candidate Feature Generation

## Architecture

Single CLI `scripts/run_discover_candidates.py` (inline LLM client wrapper —
~30 lines, same pattern as `build_fiqa_csv.py::LLMJudge`; not refactored into a
shared module to avoid touching `build_fiqa_csv.py`).

Reuses:
- `features.registry.ALLOWED_INPUTS` + recipe field constants for validation.
- `features.recipes` primitives API (signatures fed to the LLM).
- `yaml` for reading `regression_cases.yaml` + `feature_recipes.yaml`.

## Data flow

```text
run_discover_candidates.py
  | read failure_analysis.md (per-case Feature Contrast + Suggested Actions)
  | read regression_cases.yaml (pending cases metadata)
  | read feature_recipes.yaml (existing 12 name+description, avoid duplicates)
  | build prompt (system + user: cases + existing features + primitives API
  |               + ALLOWED_INPUTS + recipe schema + "propose N as JSON")
  v
DeepSeek chat.completions.create(response_format={"type":"json_object"},
                                 temperature=0.7)
  | JSON response: {"candidates": [{"recipe": {...}, "impl_code": "def candidate(row):\n ..."}]}
  v
parse + validate each candidate:
  - recipe required fields present + non-empty (except expected_slices)
  - task_profiles ⊇ [qd_reranker]
  - inputs ⊆ ALLOWED_INPUTS
  - online_safe == true
  - impl_code: ast.parse OK + defines a `candidate` function (STATIC check;
    no importlib load — generated code is executed only later by
    run_ablation.py after human review)
  | drop invalid + warn (per A.丢弃+告警)
  v
write valid candidates to <out-dir>/candidates/<name>/{recipe.yaml, impl.py}
write <out-dir>/candidates_report.md (proposals + validation status + review warning)
```

## LLM client wrapper (inline)

```python
class _LLMClient:
    def __init__(self, model="deepseek-chat", base_url="https://api.deepseek.com"):
        self.model, self.base_url = model, base_url
        self._client = None
    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit("No API key. Export DEEPSEEK_API_KEY=sk-...")
            kwargs = {"api_key": key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client
    def chat_json(self, system, user, temperature=0.7) -> dict:
        resp = self._ensure().chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"},
            temperature=temperature,
        )
        return json.loads(resp.choices[0].message.content)
```

## Prompt shape

**System**: "You are a feature engineer for a RAG query-document reranker.
Propose candidate features that discriminate required docs from hard-negative
docs for the pending failure cases. Every candidate must be computable from
ALLOWED_INPUTS only. Output strict JSON."

**User** includes:
1. Pending cases: each case's `case_id`, `query`, `failure_type`,
   `must_include`/`must_not_include`, and the Feature Contrast +
   Suggested Actions from `failure_analysis.md`.
2. Existing 12 features (name + description) — "do not propose duplicates".
3. Primitives API: `tokenize(text)->set[str]`, `numbers(text)->set[str]`,
   `entities(text)->set[str]`, `numeric_value(row, col, default=0.0)->float`,
   `rank_inverse(row, col)->float`. "Use these; import as
   `from features.primitives import ...`."
4. `ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score,
   sparse_rank, sparse_score}`. "inputs MUST be a subset; label/split/ids are
   forbidden (leakage)."
5. Recipe schema: required fields (name, version, description, task_profiles,
   inputs, type, default_value, cost_tier, online_safe, leakage_risk,
   expected_slices, owner).
6. Output contract: `{"candidates": [{"recipe": {...}, "impl_code":
   "def candidate(row):\n    ..."}]}`. `impl_code` is a Python string defining
   a `candidate(row) -> float` function.
7. "Propose N candidates" (default 5).

## Validation (static, safe)

```python
import ast
def _validate_candidate(entry) -> tuple[bool, str]:
    recipe = entry.get("recipe", {})
    for f in REQUIRED_RECIPE_FIELDS:
        if not recipe.get(f):
            return False, f"missing/empty field: {f}"
    if "qd_reranker" not in recipe.get("task_profiles", []):
        return False, "task_profiles must include qd_reranker"
    bad = [i for i in recipe["inputs"] if i not in ALLOWED_INPUTS]
    if bad:
        return False, f"input {bad[0]!r} not in ALLOWED_INPUTS"
    if not recipe.get("online_safe"):
        return False, "online_safe must be true"
    code = entry.get("impl_code", "")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"impl_code syntax error: {e}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "candidate" for n in tree.body):
        return False, "impl_code must define a `candidate` function"
    return True, ""
```

NO `importlib` load during generation — generated code runs only later via
`run_ablation.py` after human review (safety: LLM-generated Python is untrusted).

## Output

`<out-dir>/candidates/<name>/recipe.yaml` + `impl.py` per valid candidate —
ready for:
```bash
python3 scripts/run_ablation.py examples/fiqa/query_doc_examples.csv \
  --candidate-recipe <out-dir>/candidates/<name>/recipe.yaml \
  --candidate-impl <out-dir>/candidates/<name>/impl.py:candidate ...
```

`<out-dir>/candidates_report.md`: table of proposed candidates (name, inputs,
description, validation status, drop reason if any) + a prominent
"⚠ Review generated impl.py before running ablation — it is executed by
run_ablation.py via importlib" warning.

## Anti-leak / safety

- The LLM never sees `label` values, `query_id`/`doc_id` identifiers, or
  regression case ROWS — only the case metadata (query text, failure_type,
  must_include/must_not_include doc IDs as identifiers, Feature Contrast
  deltas). Feature Contrast deltas are feature-value differences, not labels.
- Generated `impl_code` is treated as untrusted: static `ast.parse` validation
  only during generation; `importlib` execution deferred to `run_ablation.py`
  (user-reviewed).
- The framework does NOT execute generated code, does NOT auto-run ablation,
  does NOT auto-promote. Pure generation + validation + file writing.

## Trade-offs

- **Inline client vs shared module**: inline ~30-line wrapper (same pattern as
  `LLMJudge`). Avoids touching `build_fiqa_csv.py` (blast radius). Mild
  duplication; can unify later.
- **JSON mode vs fenced blocks**: JSON mode (DeepSeek supports
  `response_format={"type":"json_object"}`) — reliable `json.loads`, `impl_code`
  as a string field. Confirmed during grilling.
- **Drop+warn vs retry**: drop+warn. 1 call; low invalid rate with full context;
  user re-runs if too few valid. Confirmed.
- **Static ast validation vs importlib load**: static. Safety — LLM code is
  untrusted; execution deferred to user-reviewed `run_ablation.py`.
- **temperature=0.7**: creativity + stability. DeepSeek has no reliable seed;
  candidate generation is a creative step, non-determinism acceptable (not a
  metric).
- **Default 5 candidates**: enough for V0; configurable via `--n-candidates`.

## Rollback

Purely additive (new `scripts/run_discover_candidates.py`). No existing file's
behavior changes. Rollback = revert the commit. Generated candidates under
`output/discovery/` are gitignored (under output/).
