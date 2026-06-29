#!/usr/bin/env python3
"""Train a HeuriBoost XGBoost LambdaMART reranker from CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    FEATURE_NAMES,
    ensure_output_dirs,
    extract_features,
    group_sizes,
    load_dataset,
    relevance_labels,
    require_dependencies,
    split_frame,
    validate_dataset_frame,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to query_doc_examples.csv")
    parser.add_argument(
        "--output-dir",
        default="heuriboost_output",
        help="Directory for models and reports",
    )
    parser.add_argument("--rounds", type=int, default=40, help="Boosting rounds")
    args = parser.parse_args()

    require_dependencies("numpy", "xgboost")
    import xgboost as xgb

    df = load_dataset(args.dataset)
    validate_dataset_frame(df)

    train_df = split_frame(df, "train")
    valid_df = split_frame(df, "validation")
    if train_df.empty:
        raise SystemExit("Training split is empty.")
    if valid_df.empty:
        raise SystemExit("Validation split is empty.")

    x_train = extract_features(train_df)
    y_train = relevance_labels(train_df)
    x_valid = extract_features(valid_df)
    y_valid = relevance_labels(valid_df)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_NAMES)
    dtrain.set_group(group_sizes(train_df))
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=FEATURE_NAMES)
    dvalid.set_group(group_sizes(valid_df))

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.08,
        "max_depth": 3,
        "min_child_weight": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        verbose_eval=False,
    )

    models_dir, _ = ensure_output_dirs(args.output_dir)
    model_path = models_dir / "reranker.json"
    model.save_model(model_path)

    metadata = {
        "dataset": str(Path(args.dataset)),
        "feature_names": FEATURE_NAMES,
        "params": params,
        "rounds": args.rounds,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "train_groups": int(train_df["query_id"].nunique()),
        "validation_groups": int(valid_df["query_id"].nunique()),
    }
    write_json(models_dir / "reranker_metadata.json", metadata)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {models_dir / 'reranker_metadata.json'}")


if __name__ == "__main__":
    main()
