# LLM Candidate Discovery Contracts

> Executable contracts for `scripts/run_discover_candidates.py` (spec §15.2
> step 4, §15.4). Captured 2026-06-29 after the llm-candidate-gen task (child
> #3 of feature-discovery). Produces candidates consumed by `run_ablation.py`
> (child #2).

---

## Scenario: LLM candidate generation + static validation

### 1. Scope / Trigger

Any change to `scripts/run_discover_candidates.py` or the candidate contract.
Cross-layer (LLM ↔ recipe YAML ↔ impl Python ↔ ablation), so code-spec depth
is mandatory.

### 2. Signatures

```python
# scripts/run_discover_candidates.py
class _LLMClient:  # OpenAI-compatible (DeepSeek default), reads DEEPSEEK_API_KEY
    def chat_json(self, system, user, temperature=0.7) -> dict
def _load_pending_cases(path) -> list[dict]           # status == "pending"
def _extract_case_section(fa_md, case_id) -> str       # ## <case_id> section
def _build_prompt(fa_md, pending_cases, existing_features, n) -> (system, user)
def _validate_candidate(entry) -> (ok: bool, reason: str)  # STATIC, no importlib
```

CLI: `--failure-analysis --regression-cases --feature-recipes --out-dir
--n-candidates (default 5) --model deepseek-chat --base-url
https://api.deepseek.com --temperature 0.7`.

### 3. Contracts

**LLM call**: ONE batch call, DeepSeek JSON mode
(`response_format={"type":"json_object"}`). Response:
`{"candidates": [{"recipe": {...§6.4 fields...}, "impl_code": "def candidate(row):\\n ..."}]}`.

**LLM context (full, ~2-3k tokens)**: pending cases' Feature Contrast +
Suggested Actions (from `failure_analysis.md`) + existing 12 feature
name/description (avoid duplicates) + primitives API signatures +
`ALLOWED_INPUTS` + recipe schema. NOT the full `extract_all` source.

**Candidate validation (STATIC, safe)**:
- recipe required fields present + non-empty (except `expected_slices`).
- `task_profiles` ⊇ `["qd_reranker"]`.
- `inputs` ⊆ `ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score, sparse_rank, sparse_score}`.
- `online_safe == true`.
- `impl_code`: `ast.parse` succeeds + defines a `candidate` function.
- imports restricted to `{features.primitives, features, math, re}` (no
  `os`/`subprocess`/etc.).

**NO `importlib` load during generation** — LLM-generated Python is untrusted.
Execution is deferred to `run_ablation.py` after human review of `impl.py`.

**Invalid candidate handling**: drop + warn (1 LLM call total; no retry loop).

**Output**: `<out-dir>/candidates/<name>/{recipe.yaml, impl.py}` per valid
candidate + `<out-dir>/candidates_report.md` (table + "⚠ Review before running
ablation" warning).

### 4. Validation & Error Matrix

| Condition | Behavior |
|---|---|
| `DEEPSEEK_API_KEY`/`OPENAI_API_KEY` missing | `SystemExit: No API key found...` |
| `openai` not installed | `SystemExit: openai package required...` |
| `failure_analysis.md` missing | `SystemExit: ... Run eval_reranker.py first.` |
| no pending cases | `SystemExit: No pending regression cases — nothing to attack.` |
| LLM call fails | `SystemExit: LLM call failed: <exc>` |
| LLM response not JSON | `SystemExit: LLM response was not valid JSON...` |
| candidate recipe field missing | dropped, reason `missing/empty field: <f>` |
| candidate `inputs` ⊄ ALLOWED_INPUTS | dropped, reason `input '<x>' not in ALLOWED_INPUTS` |
| candidate `online_safe: false` | dropped |
| `impl_code` syntax error | dropped |
| `impl_code` no `candidate` fn | dropped |
| `impl_code` disallowed import | dropped |

### 5. Good / Base / Bad Cases

- **Good**: LLM proposes 5 candidates, all pass static validation, written to
  `candidates/<name>/`, each feeds into `run_ablation.py` end-to-end.
- **Base**: a candidate is dropped (e.g., used `inputs: [label]`) — reported in
  `candidates_report.md` with reason; other candidates still written.
- **Bad**: `importlib`-loading `impl_code` during generation to "test" it —
  executes untrusted LLM-generated code. Forbidden; static `ast` only.

### 6. Tests Required

- A1 `candidates/` dir has subdirs; `candidates_report.md` exists.
- A2 each candidate: `recipe.yaml` parses (all §6.4 fields, `inputs` ⊆
  ALLOWED_INPUTS, `online_safe: true`); `impl.py` `ast.parse`s + defines
  `candidate`; imports ⊆ allowlist.
- A3 `candidates_report.md` contains the "Review before running ablation" warning.
- A4 a generated candidate feeds into `run_ablation.py` end-to-end (4 cells +
  recommendation).
- A5 anti-leak: generated `impl.py` inputs ⊆ ALLOWED_INPUTS (no label/split/ids).

### 7. Wrong vs Correct

#### Wrong — importlib-load during generation to "test" the candidate

```python
spec = importlib.util.spec_from_file_location("c", impl_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # EXECUTES untrusted LLM code at generation time
mod.candidate(sample_row)
```

LLM-generated Python is untrusted. Executing it at generation time (before
human review) is a code-injection risk.

#### Correct — static ast validation; defer execution to user-reviewed run_ablation

```python
tree = ast.parse(code)
if not any(isinstance(n, ast.FunctionDef) and n.name == "candidate" for n in tree.body):
    return False, "impl_code must define a `candidate` function"
# + import allowlist check via ast.walk
# NO exec_module here. run_ablation.py loads it AFTER the user reviews impl.py.
```

Static checks validate structure without executing. The user inspects `impl.py`,
then runs `run_ablation.py` (which importlib-loads it) — execution happens only
after human review.

---

## Design Decision: ONE batch LLM call (JSON mode)

**Context**: candidate generation could be per-case (N calls) or batch (1 call).

**Decision**: ONE batch call with DeepSeek JSON mode. All pending cases' analysis
+ existing features + primitives API + schema in one prompt; LLM returns a JSON
array of candidates. Cheapest (~1 call vs ~4.5k for LLM label generation), and
JSON mode makes parsing reliable.

**Tradeoff**: if the LLM produces a malformed candidate, it's dropped (no
retry). With full context (primitives API + ALLOWED_INPUTS + schema), the
invalid rate is low; user re-runs if too few valid.

## Design Decision: LLM sees case metadata + Feature Contrast, NOT labels/rows

**Context**: the LLM needs failure signal to propose candidates, but must not
leak labels or case rows.

**Decision**: the LLM sees each pending case's `failure_analysis.md` section
(Feature Contrast = per-feature value deltas between required and forbidden
docs, Suggested Actions) + case metadata (query text, failure_type,
must_include/must_not_include doc IDs). It does NOT see raw `label` values,
`query_id`/`doc_id` rows, or the training data. Feature Contrast deltas are
feature-value differences (already computed), not labels — safe to expose.
