#!/usr/bin/env python3
"""HPO CLI: search XGBoost params on train+valid, then post-hoc evaluate the
best model on the test split.

The HPO SEARCH sees only train+valid snapshots (case-blind, test-blind).
Post-hoc test evaluation is a single forward pass, clearly labeled "honest test
estimate, not a search objective" — same behavior as eval_reranker.py.

Usage:
    python3 scripts/run_hpo.py examples/fiqa/query_doc_examples.csv \
        --output-dir examples/fiqa/output --n-trials 20 --seed 42 [--timeout-sec 120]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    ensure_output_dirs,
    load_dataset,
    require_dependencies,
    split_frame,
    validate_dataset_frame,
)
from features import REGISTRY
from hpo import Budget, HPOEngine
from hpo.optuna_backend import _ndcg10_from_scores
from ranking_snapshot import snapshot_from_frame


def main() -> None:
    require_dependencies("xgboost")
    import xgboost as xgb

    parser = argparse.ArgumentParser(description="Run HPO over XGBoost ranker params.")
    parser.add_argument("dataset", help="Path to the query_doc_examples CSV.")
    parser.add_argument("--output-dir", default="examples/fiqa/output")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--split", default="validation", help="Validation split name.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    validation = validate_dataset_frame(df)
    if not validation.splits.get("test"):
        print("WARNING: no 'test' split — post-hoc test evaluation will be skipped.")

    train_df = split_frame(df, "train")
    valid_df = split_frame(df, args.split)
    test_df = split_frame(df, "test") if "test" in set(df["split"].astype(str)) else None

    train_snap = snapshot_from_frame(train_df)
    valid_snap = snapshot_from_frame(valid_df)

    budget = Budget(n_trials=args.n_trials, timeout_sec=args.timeout_sec)
    engine = HPOEngine()
    print(
        f"Running HPO: {args.n_trials} trials (timeout={args.timeout_sec}s), "
        f"seed={args.seed}, feature_set={REGISTRY.feature_set_name} v{REGISTRY.feature_set_version}"
    )
    result = engine.optimize(
        task_profile="qd_reranker",
        feature_set_name=REGISTRY.feature_set_name,
        feature_set_version=REGISTRY.feature_set_version,
        train_snapshot=train_snap,
        valid_snapshot=valid_snap,
        budget=budget,
        seed=args.seed,
    )

    # Post-hoc honest test evaluation: retrain on train with best params +
    # best_iteration (no early stopping), predict on test.
    test_score = None
    if test_df is not None:
        best_params_full = dict(result.best_params)
        best_params_full.update(
            {
                "objective": result.objective,
                "eval_metric": result.eval_metric,
                "seed": 42,
                "nthread": 1,
            }
        )
        dtrain = xgb.DMatrix(train_snap.X, label=train_snap.y)
        dtrain.set_group(train_snap.groups)
        # best_iteration is 0-indexed; reproduce the HPO-best model by training
        # rounds 0..best_iteration, i.e. num_boost_round = best_iteration + 1.
        best_model = xgb.train(
            best_params_full, dtrain, num_boost_round=result.best_iteration + 1
        )
        test_snap = snapshot_from_frame(test_df)
        dtest = xgb.DMatrix(test_snap.X, label=test_snap.y)
        dtest.set_group(test_snap.groups)
        scores = best_model.predict(dtest)
        # Use the SAME raw-label ndcg@10 as the HPO objective + baseline (0.853).
        test_score = float(
            _ndcg10_from_scores(scores, test_snap.raw_labels, test_snap.groups)
        )

    # Write outputs.
    models_dir, _ = ensure_output_dirs(args.output_dir)
    hpo_dir = Path(args.output_dir) / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    best_params_payload = {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "best_iteration": result.best_iteration,
        "test_score": test_score,
        "feature_set_name": result.feature_set_name,
        "feature_set_version": result.feature_set_version,
        "objective": result.objective,
        "eval_metric": result.eval_metric,
        "early_stopping_rounds": result.early_stopping_rounds,
        "num_boost_round": result.num_boost_round,
        "seed": result.seed,
        "n_trials": result.n_trials,
        "timeout_sec": result.timeout_sec,
    }
    (hpo_dir / "best_params.json").write_text(
        json.dumps(best_params_payload, indent=2, ensure_ascii=False) + "\n"
    )

    trials_payload = {
        "feature_set_name": result.feature_set_name,
        "feature_set_version": result.feature_set_version,
        "objective": result.objective,
        "eval_metric": result.eval_metric,
        "seed": result.seed,
        "n_trials": result.n_trials,
        "timeout_sec": result.timeout_sec,
        "best_score": result.best_score,
        "best_iteration": result.best_iteration,
        "test_score": test_score,
        "trials": result.trials,
    }
    (hpo_dir / "trials.json").write_text(
        json.dumps(trials_payload, indent=2, ensure_ascii=False) + "\n"
    )

    # Human-readable report.
    overfit_gap = (result.best_score - test_score) if test_score is not None else None
    lines = []
    lines.append("# HPO Report\n")
    lines.append(f"**Feature set**: `{result.feature_set_name}` v{result.feature_set_version}\n")
    lines.append(f"**Objective**: `{result.objective}`, eval_metric=`{result.eval_metric}`\n")
    lines.append(f"**Budget**: {result.n_trials} trials (timeout={result.timeout_sec}s), seed={result.seed}\n")
    lines.append(f"**Fixed**: num_boost_round={result.num_boost_round}, early_stopping_rounds={result.early_stopping_rounds}, nthread=1\n")
    lines.append("\n## Best trial\n")
    lines.append(f"- **best_score (validation, search objective)**: {result.best_score:.4f}")
    lines.append(f"- **best_iteration**: {result.best_iteration}")
    if test_score is not None:
        lines.append(f"- **test_score (honest estimate, NOT a search objective)**: {test_score:.4f}")
        if overfit_gap is not None:
            lines.append(f"- **val−test gap**: {overfit_gap:+.4f}  (large positive ⇒ validation overfit)")
    else:
        lines.append("- **test_score**: skipped (no test split)")
    lines.append("\n### best_params\n```json")
    lines.append(json.dumps(result.best_params, indent=2))
    lines.append("```\n")
    complete = [t for t in result.trials if t["state"] == "complete"]
    failed = [t for t in result.trials if t["state"] == "failed"]
    lines.append(f"\n## Trials ({len(complete)} complete, {len(failed)} failed)\n")
    lines.append("| # | max_depth | eta | subsample | colsample_bytree | gamma | reg_lambda | min_child_weight | score | state |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for t in result.trials:
        p = t["params"]
        score_str = f"{t['score']:.4f}" if t["score"] == t["score"] else "nan"
        lines.append(
            f"| {t['number']} | {p.get('max_depth')} | {p.get('eta', 0):.4f} | "
            f"{p.get('subsample', 0):.3f} | {p.get('colsample_bytree', 0):.3f} | "
            f"{p.get('gamma', 0):.3f} | {p.get('reg_lambda', 0):.3f} | "
            f"{p.get('min_child_weight', 0):.3f} | {score_str} | {t['state']} |"
        )
    (hpo_dir / "hpo_report.md").write_text("\n".join(lines) + "\n")

    print(f"\nHPO complete: best_score (val)={result.best_score:.4f}, best_iteration={result.best_iteration}")
    if test_score is not None:
        print(f"Post-hoc test nDCG@10 (honest): {test_score:.4f}  (val−test gap={overfit_gap:+.4f})")
    print(f"Reports: {hpo_dir / 'hpo_report.md'}")


if __name__ == "__main__":
    main()
