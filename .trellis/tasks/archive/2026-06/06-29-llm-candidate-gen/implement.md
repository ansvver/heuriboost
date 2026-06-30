# Implement: LLM Candidate Feature Generation

## Ordered checklist

Single checkpoint (additive — new `scripts/run_discover_candidates.py` only).

1. Create `scripts/run_discover_candidates.py`:
   - CLI: `--failure-analysis <path>` (default `examples/fiqa/output/reports/failure_analysis.md`), `--regression-cases <path>` (default `examples/fiqa/regression_cases.yaml`), `--feature-recipes <path>` (default shipped `templates/feature_recipes.yaml`), `--out-dir <dir>` (default `examples/fiqa/output/discovery`), `--n-candidates N` (default 5), `--model deepseek-chat`, `--base-url https://api.deepseek.com`, `--temperature 0.7`.
   - `_LLMClient` inline wrapper (OpenAI-compatible, reads `DEEPSEEK_API_KEY`/`OPENAI_API_KEY`, JSON mode).
   - `_build_prompt(failure_analysis_md, pending_cases, existing_features)` → (system, user).
   - `_parse_response(json_obj)` → list of {recipe, impl_code}.
   - `_validate_candidate(entry)` → (ok, reason) via `ast.parse` + field checks (no importlib).
   - Write valid candidates to `<out-dir>/candidates/<name>/{recipe.yaml, impl.py}`.
   - Write `<out-dir>/candidates_report.md` (table + review warning).
2. Update `.trellis/spec/backend/` with `discovery-contracts.md` (candidate contract, JSON mode, static validation, safety stance, drop+warn). Update `index.md`.
3. Update `docs/REFERENCE.md` + `docs/REFERENCE.zh-CN.md`: "Candidate discovery" subsection pointing to `run_discover_candidates.py`.
4. Update README checklist (both): note candidate generation is done (the "Automatic feature discovery + promotion" not-yet item narrows to "promotion/FeatureMemory + orchestration").
5. Update `CODEBUDDY.md` layout: add `run_discover_candidates.py`.

## Validation commands

```bash
# syntax
python3 -m py_compile skills/heuriboost-rag/scripts/run_discover_candidates.py

# requires failure_analysis.md to exist (run eval first if not)
ls examples/fiqa/output/reports/failure_analysis.md

# run with the rotated key (user sets env)
export DEEPSEEK_API_KEY=sk-...
python3 skills/heuriboost-rag/scripts/run_discover_candidates.py \
  --out-dir examples/fiqa/output/discovery --n-candidates 5

# A1: candidates dir has subdirs
ls examples/fiqa/output/discovery/candidates/

# A2: each candidate has recipe.yaml + impl.py; recipe parses; impl ast.parses + defines candidate
python3 -c "
import ast, yaml, pathlib
for d in pathlib.Path('examples/fiqa/output/discovery/candidates').iterdir():
    r = yaml.safe_load((d/'recipe.yaml').read_text())
    assert r.get('online_safe') is True
    assert all(i in {'query_text','doc_text','dense_rank','dense_score','sparse_rank','sparse_score'} for i in r['inputs'])
    tree = ast.parse((d/'impl.py').read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name=='candidate' for n in tree.body)
    print(d.name, 'OK', r['inputs'])
"

# A3: candidates_report.md exists + has review warning
grep -c 'review' examples/fiqa/output/discovery/candidates_report.md

# A4: a generated candidate can be fed into run_ablation.py (end-to-end, after human review)
# python3 scripts/run_ablation.py ... --candidate-recipe <candidate>/recipe.yaml --candidate-impl <candidate>/impl.py:candidate ...

# A5: anti-leak — no label/query_id/doc_id in generated impl.py inputs (recipe inputs ⊆ ALLOWED_INPUTS, checked in A2)
```

## Risky files / rollback points

- New `scripts/run_discover_candidates.py` — additive, low risk.
- No changes to `common.py`, `train_reranker.py`, `eval_reranker.py`, `hpo/`,
  `features/`, `run_ablation.py`, `build_fiqa_csv.py` — existing behavior untouched.

## Pre-`task.py start` checks

- [ ] `DEEPSEEK_API_KEY` env var read; clear error if missing.
- [ ] ONE LLM call (JSON mode) produces N candidates.
- [ ] Each valid candidate: recipe has all §6.4 fields, inputs ⊆ ALLOWED_INPUTS, online_safe true, impl ast.parses + defines `candidate`.
- [ ] Invalid candidates dropped + warned in report.
- [ ] No importlib load of generated code during generation (static ast only).
- [ ] candidates_report.md has "review before running ablation" warning.
- [ ] Generated candidate feeds into run_ablation.py end-to-end (manual spot-check).
- [ ] py_compile passes; REFERENCE/README/CODEBUDDY/spec updated.

## Note on key safety

The user pasted a DeepSeek key in chat (leak). Code reads `DEEPSEEK_API_KEY`
env var only — never hardcoded. The user must rotate this key after the session.
For verification, the main agent sets the env var in the shell (the key is
already in the transcript; no added leak) and runs the discovery + one
end-to-end ablation spot-check, then reminds the user to rotate.
