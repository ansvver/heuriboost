#!/usr/bin/env python3
"""LLM candidate feature generation (spec §15.2 step 4, §15.4).

Reads the per-case `failure_analysis.md` (Feature Contrast + Suggested Actions)
for PENDING regression cases + the existing feature set, calls an LLM ONCE
(DeepSeek, JSON mode) to propose up to five candidate FeatureRecipes, validates
them statically, and writes candidate files ready for `run_ablation.py`.

The LLM never sees label values or case rows. Generated `impl_code` is treated
as UNTRUSTED: only `ast.parse` + `def candidate` check during generation;
`importlib` execution is deferred to `run_ablation.py`. Pass `--auto-ablate`
to run ablation immediately for each valid candidate.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python3 scripts/run_discover_candidates.py \\
        --out-dir examples/fiqa/output/discovery --n-candidates 5
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from features.registry import ALLOWED_INPUTS

SKILL_DIR = Path(__file__).resolve().parent.parent
MAX_CANDIDATES_PER_DISCOVERY_ROUND = 5
MAX_DISCOVERY_PROMPT_CHARS = 24_000
REQUIRED_RECIPE_FIELDS = (
    "name", "version", "description", "task_profiles", "inputs",
    "type", "default_value", "cost_tier", "online_safe", "leakage_risk",
    "owner",
)
ACTIVE_TASK_PROFILE = "qd_reranker"

PRIMITIVES_API = """\
Available primitives (import as `from features.primitives import <name>`):
- tokenize(text) -> set[str]           lowercase alphanumeric tokens
- numbers(text) -> set[str]            numeric tokens (e.g. "401k", "3.5%")
- entities(text) -> set[str]           capitalized words / all-caps acronyms
- numeric_value(row, col, default=0.0) -> float   safe column read
- rank_inverse(row, col) -> float      1.0 / rank (0 if rank <= 0)"""

RECIPE_SCHEMA = """\
Recipe required fields (all non-empty except expected_slices):
  name, version (int), description, task_profiles (list incl. "qd_reranker"),
  inputs (list, subset of ALLOWED_INPUTS), type ("numeric"), default_value (float),
  cost_tier ("L0".."L3"), online_safe (true), leakage_risk ("low"|"medium"|"high"),
  expected_slices (list, may be empty), owner (str)"""


class _LLMClient:
    """OpenAI-compatible client (DeepSeek by default). Reads DEEPSEEK_API_KEY."""

    def __init__(self, model="deepseek-chat", base_url="https://api.deepseek.com"):
        self.model = model
        self.base_url = base_url
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "openai package required. Install with: python -m pip install -r "
                f"{SKILL_DIR / 'requirements-build.txt'}"
            ) from exc
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit(
                "No API key found. Export DEEPSEEK_API_KEY (or OPENAI_API_KEY) "
                "before running, e.g. export DEEPSEEK_API_KEY=sk-..."
            )
        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def chat_json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        client = self._ensure()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
        except Exception as exc:
            raise SystemExit(f"LLM call failed: {exc}") from exc
        try:
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"LLM response was not valid JSON: {exc}\n"
                f"Raw (truncated): {resp.choices[0].message.content[:500]}"
            ) from exc


def _load_pending_cases(path: str) -> list[dict]:
    """Load regression cases with status == 'pending'."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required.") from exc
    rp = Path(path)
    if not rp.exists():
        raise SystemExit(f"Regression cases file not found: {rp}")
    data = yaml.safe_load(rp.read_text()) or {}
    cases = data.get("cases", [])
    return [c for c in cases if c.get("status") == "pending"]


def _extract_case_section(failure_analysis_md: str, case_id: str) -> str:
    """Extract the `## <case_id>` section from failure_analysis.md."""
    pattern = re.compile(
        rf"(^##\s+{re.escape(case_id)}\s*$.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(failure_analysis_md)
    return m.group(1).strip() if m else ""


def _load_existing_features(path: str) -> list[dict]:
    """Load existing feature name+description from feature_recipes.yaml."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required.") from exc
    rp = Path(path)
    if not rp.exists():
        raise SystemExit(f"Feature recipes file not found: {rp}")
    data = yaml.safe_load(rp.read_text()) or {}
    return [
        {"name": f.get("name"), "description": f.get("description", "")}
        for f in data.get("features", [])
    ]


def _build_prompt(
    failure_analysis_md: str,
    pending_cases: list[dict],
    existing_features: list[dict],
    n_candidates: int,
) -> tuple[str, str]:
    system = (
        "You are a feature engineer for a RAG query-document reranker. Propose "
        "candidate features that discriminate required docs from hard-negative "
        "docs for the pending failure cases. Every candidate MUST be computable "
        "from ALLOWED_INPUTS only (no label/split/ids — those are leakage). "
        "Output STRICT JSON only."
    )
    sections = []
    for c in pending_cases:
        cid = c.get("case_id", "")
        section = _extract_case_section(failure_analysis_md, cid)
        if section:
            sections.append(f"### Case {cid}\n{section}")
        else:
            sections.append(
                f"### Case {cid}\n(query: {c.get('query','')}, "
                f"failure_type: {c.get('failure_type','')}, "
                f"must_include: {c.get('must_include_doc_ids', [])}, "
                f"must_not_include: {c.get('must_not_include_doc_ids', [])})"
            )
    cases_block = "\n\n".join(sections) if sections else "(no pending cases)"
    existing_block = "\n".join(
        f"- {f['name']}: {f['description']}" for f in existing_features
    )
    user = f"""Propose {n_candidates} candidate features for the pending failure cases below.

## Pending failure cases (from failure_analysis.md)
{cases_block}

## Existing features (DO NOT propose duplicates)
{existing_block}

## {PRIMITIVES_API}

## ALLOWED_INPUTS (inputs MUST be a subset)
{sorted(ALLOWED_INPUTS)}

## {RECIPE_SCHEMA}

## Output contract
Output a JSON object: {{"candidates": [{{"recipe": {{...all fields...}}, "impl_code": "def candidate(row):\\n    from features.primitives import ...\\n    return ..."}}]}}.
- `impl_code` is a Python source string defining a `candidate(row) -> float` function.
- The function receives a pandas Series `row` with columns from ALLOWED_INPUTS.
- Use the primitives above; do NOT import anything beyond stdlib + features.primitives.
- Propose {n_candidates} distinct candidates."""
    return system, user


def _validate_candidate(entry: dict) -> tuple[bool, str]:
    recipe = entry.get("recipe")
    if not isinstance(recipe, dict):
        return False, "recipe missing or not a mapping"
    for f in REQUIRED_RECIPE_FIELDS:
        v = recipe.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False, f"missing/empty field: {f}"
    if ACTIVE_TASK_PROFILE not in recipe.get("task_profiles", []):
        return False, "task_profiles must include qd_reranker"
    bad = [i for i in recipe["inputs"] if i not in ALLOWED_INPUTS]
    if bad:
        return False, f"input {bad[0]!r} not in ALLOWED_INPUTS (leakage/identifier)"
    if not recipe.get("online_safe"):
        return False, "online_safe must be true"
    code = entry.get("impl_code", "")
    if not isinstance(code, str) or not code.strip():
        return False, "impl_code missing/empty"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"impl_code syntax error: {e}"
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "candidate" for n in tree.body
    ):
        return False, "impl_code must define a `candidate` function"
    # reject imports beyond stdlib + features.primitives
    allowed_imports = {"features.primitives", "math", "re", "features"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_imports:
                    return False, f"disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in allowed_imports:
                return False, f"disallowed import: {node.module}"
    return True, ""


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "candidate"


def _write_auto_ablation_summary(out_dir: Path, rows: list[dict]) -> Path:
    report_path = out_dir / "auto_ablation_report.md"
    lines = ["# Auto-Ablation Report\n"]
    lines.append("| candidate | status | recommendation | result |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['status']} | {row.get('recommendation', 'n/a')} "
            f"| {row.get('result_path', row.get('error', 'n/a'))} |"
        )
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _run_auto_ablation(args, written: list[dict], cand_dir: Path, out_dir: Path) -> None:
    """Run run_ablation.py once per valid candidate."""
    script_path = Path(__file__).resolve().parent / "run_ablation.py"
    base_output_dir = (
        Path(args.ablation_output_dir)
        if args.ablation_output_dir
        else out_dir / "auto_ablation"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for candidate in written:
        name = candidate["name"]
        candidate_dir = cand_dir / name
        candidate_output_dir = base_output_dir / name
        result_path = candidate_output_dir / "ablation" / "ablation_result.json"
        cmd = [
            sys.executable,
            str(script_path),
            args.dataset,
            "--candidate-recipe",
            str(candidate_dir / "recipe.yaml"),
            "--candidate-impl",
            f"{candidate_dir / 'impl.py'}:candidate",
            "--output-dir",
            str(candidate_output_dir),
            "--n-trials",
            str(args.ablation_n_trials),
            "--seed",
            str(args.ablation_seed),
            "--promote-threshold",
            str(args.ablation_promote_threshold),
            "--regression-cases",
            args.regression_cases,
            "--split",
            args.ablation_split,
        ]
        print(f"\nAuto-ablation: {name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "name": name,
                    "status": "failed",
                    "error": f"exit {exc.returncode}",
                }
            )
            summary_path = _write_auto_ablation_summary(out_dir, rows)
            raise SystemExit(
                f"Auto-ablation failed for {name}. Summary: {summary_path}"
            ) from exc

        if not result_path.exists():
            rows.append(
                {
                    "name": name,
                    "status": "failed",
                    "error": f"missing result {result_path}",
                }
            )
            summary_path = _write_auto_ablation_summary(out_dir, rows)
            raise SystemExit(
                f"Auto-ablation for {name} did not write {result_path}. "
                f"Summary: {summary_path}"
            )
        result = json.loads(result_path.read_text())
        recommendation = result.get("recommendation", "unknown")
        rows.append(
            {
                "name": name,
                "status": "ok",
                "recommendation": recommendation,
                "result_path": str(result_path),
            }
        )

    summary_path = _write_auto_ablation_summary(out_dir, rows)
    print(f"Auto-ablation summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate features via LLM.")
    parser.add_argument(
        "--failure-analysis",
        default="examples/fiqa/output/reports/failure_analysis.md",
    )
    parser.add_argument("--regression-cases", default="examples/fiqa/regression_cases.yaml")
    parser.add_argument(
        "--feature-recipes",
        default=str(
            Path(__file__).resolve().parent.parent / "templates" / "feature_recipes.yaml"
        ),
    )
    parser.add_argument("--out-dir", default="examples/fiqa/output/discovery")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=MAX_CANDIDATES_PER_DISCOVERY_ROUND,
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--auto-ablate",
        action="store_true",
        help="Run run_ablation.py for each valid candidate after discovery.",
    )
    parser.add_argument(
        "--dataset",
        default="examples/fiqa/query_doc_examples.csv",
        help="Dataset passed to run_ablation.py when --auto-ablate is set.",
    )
    parser.add_argument(
        "--ablation-output-dir",
        default=None,
        help="Base output dir for auto-ablation; defaults to <out-dir>/auto_ablation.",
    )
    parser.add_argument("--ablation-n-trials", type=int, default=20)
    parser.add_argument("--ablation-seed", type=int, default=42)
    parser.add_argument("--ablation-promote-threshold", type=float, default=0.01)
    parser.add_argument("--ablation-split", default="validation")
    args = parser.parse_args()

    if not 1 <= args.n_candidates <= MAX_CANDIDATES_PER_DISCOVERY_ROUND:
        parser.error(
            "--n-candidates must be between 1 and "
            f"{MAX_CANDIDATES_PER_DISCOVERY_ROUND} for one discovery round"
        )

    fa_path = Path(args.failure_analysis)
    if not fa_path.exists():
        raise SystemExit(
            f"failure_analysis.md not found: {fa_path}. Run eval_reranker.py first."
        )
    failure_analysis_md = fa_path.read_text()

    pending_cases = _load_pending_cases(args.regression_cases)
    if not pending_cases:
        raise SystemExit("No pending regression cases — nothing to attack.")
    existing_features = _load_existing_features(args.feature_recipes)

    system, user = _build_prompt(
        failure_analysis_md, pending_cases, existing_features, args.n_candidates
    )
    prompt_chars = len(system) + len(user)
    if prompt_chars > MAX_DISCOVERY_PROMPT_CHARS:
        raise SystemExit(
            "Discovery prompt is too large "
            f"({prompt_chars} chars > {MAX_DISCOVERY_PROMPT_CHARS}). "
            "Do not auto-truncate pending cases; split the pending cases by "
            "failure theme before rerunning."
        )
    print(
        f"Generating up to {args.n_candidates} candidates for "
        f"{len(pending_cases)} pending case(s) via {args.model}..."
    )
    client = _LLMClient(model=args.model, base_url=args.base_url)
    result = client.chat_json(system, user, temperature=args.temperature)

    raw_candidates = result.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise SystemExit("LLM response has no 'candidates' list.")

    out_dir = Path(args.out_dir)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    written = []
    dropped = []
    seen_names = set()
    for entry in raw_candidates:
        if not isinstance(entry, dict):
            dropped.append({"name": "?", "reason": "entry not a mapping"})
            continue
        recipe = entry.get("recipe", {})
        name = _safe_name(recipe.get("name", ""))
        ok, reason = _validate_candidate(entry)
        if not ok:
            dropped.append({"name": name or "?", "reason": reason})
            continue
        if name in seen_names:
            dropped.append({"name": name, "reason": "duplicate name"})
            continue
        seen_names.add(name)

        sub = cand_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML required.") from exc
        (sub / "recipe.yaml").write_text(
            yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True)
        )
        (sub / "impl.py").write_text(entry["impl_code"])
        written.append(
            {
                "name": name,
                "inputs": recipe["inputs"],
                "description": recipe["description"],
            }
        )

    # Report
    lines = ["# Candidate Discovery Report\n"]
    lines.append(f"**Model**: {args.model}, temperature={args.temperature}\n")
    lines.append(
        f"**Requested**: {args.n_candidates}, **written**: {len(written)}, "
        f"**dropped**: {len(dropped)}\n"
    )
    lines.append(f"**Pending cases**: {[c.get('case_id') for c in pending_cases]}\n")
    lines.append("\n## Written candidates\n")
    if written:
        lines.append("| name | inputs | description |")
        lines.append("|---|---|---|")
        for c in written:
            lines.append(f"| `{c['name']}` | {c['inputs']} | {c['description']} |")
    else:
        lines.append("(none)")
    if dropped:
        lines.append("\n## Dropped candidates\n")
        lines.append("| name | reason |")
        lines.append("|---|---|")
        for d in dropped:
            lines.append(f"| `{d['name']}` | {d['reason']} |")
    if args.auto_ablate:
        lines.append("\n## ⚠ Auto-ablation requested\n")
        lines.append(
            "Generated `impl.py` files are executed by `run_ablation.py` via "
            "importlib because `--auto-ablate` was set. Promotion remains manual.\n"
        )
    else:
        lines.append("\n## ⚠ Review before running ablation\n")
        lines.append(
            "Generated `impl.py` files are executed by `run_ablation.py` via importlib. "
            "Inspect each `candidates/<name>/impl.py` before running:\n"
            "```\n"
            "python3 scripts/run_ablation.py examples/fiqa/query_doc_examples.csv \\\n"
            "  --candidate-recipe <out-dir>/candidates/<name>/recipe.yaml \\\n"
            "  --candidate-impl <out-dir>/candidates/<name>/impl.py:candidate ...\n"
            "```\n"
        )
    report_path = out_dir / "candidates_report.md"
    report_path.write_text("\n".join(lines) + "\n")

    print(f"\nWritten: {len(written)} candidates -> {cand_dir}")
    print(f"Dropped: {len(dropped)}")
    for w in written:
        print(f"  + {w['name']} (inputs={w['inputs']})")
    for d in dropped:
        print(f"  - {d['name']}: {d['reason']}")
    print(f"Report: {report_path}")
    if not written:
        raise SystemExit(
            f"No valid candidates were written. Review dropped candidates in {report_path}."
        )
    if args.auto_ablate:
        _run_auto_ablation(args, written, cand_dir, out_dir)


if __name__ == "__main__":
    main()
