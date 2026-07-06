#!/usr/bin/env python3
"""A/B/C/D ablation framework (spec §15.3).

Given a candidate feature (recipe YAML + impl fn), runs 4 cells under the SAME
data/feature snapshot/split/metric/gate:

  A = baseline features + baseline (fixed) params
  B = baseline features + HPO-tuned params
  C = baseline + candidate features + baseline params
  D = baseline + candidate features + HPO-tuned params

Deltas: B-A (param gain), C-A (feature-only gain), D-B (candidate gain after
tuning — primary), D-C (tuning gain with candidate).

Recommendation (report only — promotion is ALWAYS manual):
  promote     iff D-B(val) > threshold AND D-B(test) > 0 AND D gate cases pass
  reject      iff D-B(val) <= 0 OR D regresses a gate case
  quarantine  otherwise

Usage:
    python3 scripts/run_ablation.py examples/fiqa/query_doc_examples.csv \\
        --candidate-recipe candidate_recipe.yaml \\
        --candidate-impl candidate_impl.py:candidate \\
        --output-dir examples/fiqa/output --n-trials 5 --seed 42 \\
        --regression-cases examples/fiqa/regression_cases.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from common import (
    ensure_output_dirs,
    extract_features,
    group_sizes,
    load_dataset,
    rank_by_model,
    relevance_labels,
    require_dependencies,
    split_frame,
    validate_dataset_frame,
)
from eval_reranker import load_regression_cases, run_regression_cases
from features import REGISTRY
from features.recipes import extract_all
from features.registry import ALLOWED_INPUTS
from hpo import Budget, HPOEngine, Snapshot
from hpo.optuna_backend import _ndcg10_from_scores

# Shipped baseline params (train_reranker.py:210-219). Fixed for cells A/C.
BASELINE_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "eta": 0.08,
    "max_depth": 3,
    "min_child_weight": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "seed": 42,
    "nthread": 1,
}
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 20

REQUIRED_RECIPE_FIELDS = (
    "name", "version", "description", "task_profiles", "inputs",
    "type", "cost_tier", "online_safe", "leakage_risk", "owner",
)


def _load_impl(spec: str):
    if ":" not in spec:
        raise SystemExit(f"--candidate-impl must be 'pyfile:func', got: {spec}")
    pyfile, func_name = spec.split(":", 1)
    py_path = Path(pyfile)
    if not py_path.exists():
        raise SystemExit(f"Candidate impl file not found: {py_path}")
    loader = importlib.util.spec_from_file_location(f"_cand_{py_path.stem}", py_path)
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    if not hasattr(mod, func_name):
        raise SystemExit(f"Candidate impl {py_path} has no function '{func_name}'")
    return getattr(mod, func_name)


def _load_candidate_recipe(path: str) -> dict:
    require_dependencies("yaml")
    import yaml

    rp = Path(path)
    if not rp.exists():
        raise SystemExit(f"Candidate recipe not found: {rp}")
    entry = yaml.safe_load(rp.read_text())
    if not isinstance(entry, dict):
        raise SystemExit("Candidate recipe must be a mapping.")
    for field in REQUIRED_RECIPE_FIELDS:
        v = entry.get(field)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise SystemExit(f"Candidate recipe missing required field: {field}")
    if "qd_reranker" not in entry["task_profiles"]:
        raise SystemExit("Candidate recipe task_profiles must include 'qd_reranker'")
    bad = [i for i in entry["inputs"] if i not in ALLOWED_INPUTS]
    if bad:
        raise SystemExit(
            f"Candidate '{entry['name']}' inputs={entry['inputs']}; "
            f"'{bad[0]}' not in ALLOWED_INPUTS (leakage/identifier)."
        )
    if not entry["online_safe"]:
        raise SystemExit(f"Candidate '{entry['name']}' is online_safe=false; rejected.")
    return entry


def _make_extract_plus_df(candidate_fn, candidate_name):
    """Return a DataFrame-level extractor: df -> DataFrame[baseline + candidate]."""
    def extract_plus_df(df) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            out = extract_all(row)
            out[candidate_name] = float(candidate_fn(row))
            rows.append(out)
        return pd.DataFrame(rows, columns=list(out.keys()))
    return extract_plus_df


def _snapshot(df, extract_df_fn) -> Snapshot:
    X = extract_df_fn(df)
    return Snapshot(
        X=X,
        y=relevance_labels(df),
        raw_labels=[int(v) for v in df["label"].tolist()],
        groups=group_sizes(df),
    )


def _train_cell(params, train_snap, valid_snap):
    import xgboost as xgb

    dtrain = xgb.DMatrix(train_snap.X, label=train_snap.y)
    dtrain.set_group(train_snap.groups)
    dvalid = xgb.DMatrix(valid_snap.X, label=valid_snap.y)
    dvalid.set_group(valid_snap.groups)
    return xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )


def _score(model, snap) -> float:
    import xgboost as xgb

    d = xgb.DMatrix(snap.X, label=snap.y)
    d.set_group(snap.groups)
    best_iter = int(model.best_iteration)
    preds = model.predict(d, iteration_range=(0, best_iter + 1))
    return float(_ndcg10_from_scores(preds, snap.raw_labels, snap.groups))


def _gate(model, snap, df, cases) -> dict:
    import xgboost as xgb

    d = xgb.DMatrix(snap.X, label=snap.y)
    d.set_group(snap.groups)
    best_iter = int(model.best_iteration)
    scores = model.predict(d, iteration_range=(0, best_iter + 1))
    ranked = rank_by_model(df, scores)
    results = run_regression_cases(ranked, cases)
    gate_cases = [r for r in results if r.get("status", "gate") == "gate"]
    return {
        "gate_pass": sum(1 for r in gate_cases if r["passed"]),
        "gate_total": len(gate_cases),
        "all_gate_pass": all(r["passed"] for r in gate_cases) if gate_cases else True,
        "details": [
            {"case_id": r["case_id"], "status": r.get("status", "gate"), "passed": r["passed"]}
            for r in results
        ],
    }


def _eval_cell(model, valid_snap, test_snap, valid_df, cases) -> dict:
    return {
        "val_score": _score(model, valid_snap),
        "test_score": _score(model, test_snap) if test_snap is not None else None,
        "best_iteration": int(model.best_iteration),
        "gate": _gate(model, valid_snap, valid_df, cases),
    }


def main() -> None:
    require_dependencies("xgboost")
    parser = argparse.ArgumentParser(description="Run an A/B/C/D feature ablation.")
    parser.add_argument("dataset")
    parser.add_argument("--candidate-recipe", required=True)
    parser.add_argument("--candidate-impl", required=True, help="pyfile:func")
    parser.add_argument("--output-dir", default="examples/fiqa/output")
    parser.add_argument("--n-trials", type=int, default=20, help="HPO budget for B/D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--promote-threshold", type=float, default=0.01)
    parser.add_argument("--regression-cases", default="examples/fiqa/regression_cases.yaml")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    recipe = _load_candidate_recipe(args.candidate_recipe)
    candidate_fn = _load_impl(args.candidate_impl)
    candidate_name = recipe["name"]
    print(f"Candidate: {candidate_name} (inputs={recipe['inputs']})")

    df = load_dataset(args.dataset)
    validate_dataset_frame(df)
    train_df = split_frame(df, "train")
    valid_df = split_frame(df, args.split)
    has_test = "test" in set(df["split"].astype(str))
    test_df = split_frame(df, "test") if has_test else None
    cases = load_regression_cases(args.regression_cases)

    candidate_name = recipe["name"]
    extract_baseline = extract_features  # DataFrame -> DataFrame (12 cols)
    extract_plus_df = _make_extract_plus_df(candidate_fn, candidate_name)  # df -> 13 cols

    # Pre-build snapshots for baseline (12) and candidate (13) feature sets.
    snaps = {
        "baseline": {
            "train": _snapshot(train_df, extract_baseline),
            "valid": _snapshot(valid_df, extract_baseline),
            "test": _snapshot(test_df, extract_baseline) if has_test else None,
        },
        "candidate": {
            "train": _snapshot(train_df, extract_plus_df),
            "valid": _snapshot(valid_df, extract_plus_df),
            "test": _snapshot(test_df, extract_plus_df) if has_test else None,
        },
    }

    hpo = HPOEngine()
    budget = Budget(n_trials=args.n_trials)
    cells = {}

    # A: baseline + fixed params
    print("Cell A: baseline + fixed params")
    mA = _train_cell(BASELINE_PARAMS, snaps["baseline"]["train"], snaps["baseline"]["valid"])
    cells["A"] = _eval_cell(mA, snaps["baseline"]["valid"], snaps["baseline"]["test"], valid_df, cases)

    # B: baseline + HPO
    print(f"Cell B: baseline + HPO ({args.n_trials} trials, seed={args.seed})")
    rb = hpo.optimize(
        task_profile="qd_reranker",
        feature_set_name=REGISTRY.feature_set_name,
        feature_set_version=REGISTRY.feature_set_version,
        train_snapshot=snaps["baseline"]["train"],
        valid_snapshot=snaps["baseline"]["valid"],
        budget=budget,
        seed=args.seed,
    )
    mB = _train_cell({**BASELINE_PARAMS, **rb.best_params}, snaps["baseline"]["train"], snaps["baseline"]["valid"])
    cells["B"] = _eval_cell(mB, snaps["baseline"]["valid"], snaps["baseline"]["test"], valid_df, cases)

    # C: candidate + fixed params
    print("Cell C: candidate + fixed params")
    mC = _train_cell(BASELINE_PARAMS, snaps["candidate"]["train"], snaps["candidate"]["valid"])
    cells["C"] = _eval_cell(mC, snaps["candidate"]["valid"], snaps["candidate"]["test"], valid_df, cases)

    # D: candidate + HPO (same budget + seed as B)
    print(f"Cell D: candidate + HPO ({args.n_trials} trials, seed={args.seed})")
    rd = hpo.optimize(
        task_profile="qd_reranker",
        feature_set_name=f"{REGISTRY.feature_set_name}+{candidate_name}",
        feature_set_version=REGISTRY.feature_set_version,
        train_snapshot=snaps["candidate"]["train"],
        valid_snapshot=snaps["candidate"]["valid"],
        budget=budget,
        seed=args.seed,
    )
    mD = _train_cell({**BASELINE_PARAMS, **rd.best_params}, snaps["candidate"]["train"], snaps["candidate"]["valid"])
    cells["D"] = _eval_cell(mD, snaps["candidate"]["valid"], snaps["candidate"]["test"], valid_df, cases)

    # Deltas
    deltas = {}
    for label, (x, y) in [("B-A", ("B", "A")), ("C-A", ("C", "A")),
                          ("D-B", ("D", "B")), ("D-C", ("D", "C"))]:
        tx = cells[x]["test_score"]
        ty = cells[y]["test_score"]
        deltas[label] = {
            "val": cells[x]["val_score"] - cells[y]["val_score"],
            "test": (tx - ty) if (tx is not None and ty is not None) else None,
        }

    # Recommendation
    db_val = deltas["D-B"]["val"]
    db_test = deltas["D-B"]["test"]
    d_gate_ok = cells["D"]["gate"]["all_gate_pass"]
    if db_val <= 0 or not d_gate_ok:
        recommendation = "reject"
    elif db_val > args.promote_threshold and db_test is not None and db_test > 0:
        recommendation = "promote"
    else:
        recommendation = "quarantine"

    _write_outputs(args, recipe, cells, deltas, recommendation)


def _write_outputs(args, recipe, cells, deltas, recommendation):
    ensure_output_dirs(args.output_dir)
    abl_dir = Path(args.output_dir) / "ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "candidate": recipe,
        "hpo": {"n_trials": args.n_trials, "seed": args.seed},
        "promote_threshold": args.promote_threshold,
        "cells": cells,
        "deltas": deltas,
        "recommendation": recommendation,
    }
    (abl_dir / "ablation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    feat = {"A": "baseline 12", "B": "baseline 12", "C": "baseline+candidate 13", "D": "baseline+candidate 13"}
    par = {"A": "fixed", "B": "HPO", "C": "fixed", "D": "HPO"}
    lines = ["# Ablation Report\n"]
    lines.append(f"**Candidate**: `{recipe['name']}` (inputs={recipe['inputs']})\n")
    lines.append(f"**HPO**: {args.n_trials} trials, seed={args.seed}\n")
    lines.append(f"**Promote threshold**: D-B(val) > {args.promote_threshold}\n")
    lines.append("\n## Cells\n")
    lines.append("| Cell | Features | Params | val nDCG@10 | test nDCG@10 | gate | best_iter |")
    lines.append("|---|---|---|---:|---:|---|---:|")
    for c in "ABCD":
        cell = cells[c]
        ts = f"{cell['test_score']:.4f}" if cell["test_score"] is not None else "n/a"
        lines.append(
            f"| {c} | {feat[c]} | {par[c]} | {cell['val_score']:.4f} | {ts} | "
            f"{cell['gate']['gate_pass']}/{cell['gate']['gate_total']} | {cell['best_iteration']} |"
        )
    lines.append("\n## Deltas\n")
    lines.append("| Δ | val | test |")
    lines.append("|---|---:|---:|")
    for label, d in deltas.items():
        ts = f"{d['test']:+.4f}" if d["test"] is not None else "n/a"
        lines.append(f"| {label} | {d['val']:+.4f} | {ts} |")
    lines.append(f"\n## Recommendation: **{recommendation}**\n")
    lines.append(
        f"- D-B(val) = {deltas['D-B']['val']:+.4f}  (threshold {args.promote_threshold})\n"
        f"- D-B(test) = {deltas['D-B']['test']:+.4f}\n"
        f"- D gate: {cells['D']['gate']['gate_pass']}/{cells['D']['gate']['gate_total']} pass\n"
    )
    lines.append("\n> Promotion is a REPORT recommendation. Actual promotion is always manual.\n")
    (abl_dir / "ablation_report.md").write_text("\n".join(lines) + "\n")

    print(f"\nRecommendation: {recommendation}")
    dbtest = deltas["D-B"]["test"]
    print(f"D-B(val)={deltas['D-B']['val']:+.4f}  D-B(test)={dbtest:+.4f}" if dbtest is not None else f"D-B(val)={deltas['D-B']['val']:+.4f}")
    print(f"Reports: {abl_dir / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
