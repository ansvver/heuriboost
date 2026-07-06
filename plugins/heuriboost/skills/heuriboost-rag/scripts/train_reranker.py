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
    load_case_sets,
    relevance_labels,
    require_dependencies,
    split_frame,
    validate_dataset_frame,
    write_json,
)
from features import REGISTRY


def load_case_denylist(cases_path: str) -> tuple[set[str], set[str]]:
    """Load the case query_id / doc_id denylist from regression_cases.yaml.

    This is the ONE narrow, documented exception to "train never reads cases":
    train reads the case IDS for B+C isolation, NEVER the case rows as
    training data. The case sets themselves (mined samples) are loaded
    separately via --case-sets and are physically distinct from the cases file.
    """
    require_dependencies("yaml")
    import yaml

    path = Path(cases_path)
    if not path.exists():
        raise SystemExit(f"Regression cases file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    cases = data.get("cases", [])
    case_query_ids: set[str] = set()
    case_doc_ids: set[str] = set()
    for case in cases:
        qid = case.get("query_id")
        if qid is not None:
            case_query_ids.add(str(qid))
        for key in ("must_include_doc_ids", "must_not_include_doc_ids"):
            for did in case.get(key, []) or []:
                case_doc_ids.add(str(did))
    return case_query_ids, case_doc_ids


def assert_case_set_isolation(case_set_df, case_query_ids: set[str], case_doc_ids: set[str]) -> None:
    """Defensive B+C re-check: assert no mined row leaks a case query_id or
    case doc_id. Fail loud with offending ids on any leak.

    B: no mined row's query_id may equal any case's query_id.
    C: no mined row's doc_id may equal any case's must_include/must_not_include doc_id.
    """
    if case_set_df.empty:
        return

    leaked_queries = set(case_set_df["query_id"].astype(str)) & case_query_ids
    if leaked_queries:
        raise SystemExit(
            f"ANTI-LEAK B check FAILED: case_set rows contain case query_ids: "
            f"{sorted(leaked_queries)}"
        )

    leaked_docs = set(case_set_df["doc_id"].astype(str)) & case_doc_ids
    if leaked_docs:
        raise SystemExit(
            f"ANTI-LEAK C check FAILED: case_set rows contain case doc_ids: "
            f"{sorted(leaked_docs)}"
        )


def _merge_train_frames(train_df, case_set_df):
    """Merge case_set rows into the train DataFrame, aligning columns to the
    train_df schema. Non-matching columns in case_set_df are dropped; missing
    columns are filled with defaults."""
    from common import load_pandas
    pd = load_pandas()

    # Align columns: keep only columns that exist in train_df.
    aligned = case_set_df.reindex(columns=train_df.columns)
    combined = pd.concat([train_df, aligned], ignore_index=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to query_doc_examples.csv")
    parser.add_argument(
        "--output-dir",
        default="heuriboost_output",
        help="Directory for models and reports",
    )
    parser.add_argument("--rounds", type=int, default=40, help="Boosting rounds")
    parser.add_argument(
        "--reckless",
        action="store_true",
        help=(
            "Train in reckless mode: fold case_sets directly into train and "
            "record reckless metadata."
        ),
    )
    parser.add_argument(
        "--case-sets",
        default=None,
        help=(
            "Path to a case_set CSV or a directory of case_set CSVs (mined "
            "training samples). When set, rows are merged into the TRAIN split "
            "only. A defensive B+C isolation re-check is run against the case "
            "denylist (requires --regression-cases). In reckless mode, this "
            "defaults to examples/fiqa/case_sets when omitted."
        ),
    )
    parser.add_argument(
        "--regression-cases",
        default=None,
        help=(
            "Path to regression_cases.yaml. ONLY used to load the case "
            "query_id/doc_id denylist for the B+C isolation re-check when "
            "--case-sets is set. Case ROWS never enter training. Without "
            "--case-sets, this flag is ignored. In reckless mode, this defaults "
            "to examples/fiqa/regression_cases.yaml when omitted."
        ),
    )
    args = parser.parse_args()

    if args.reckless:
        if not args.case_sets:
            args.case_sets = "examples/fiqa/case_sets"
        if not args.regression_cases:
            args.regression_cases = "examples/fiqa/regression_cases.yaml"

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

    # --- Optional: merge mined case_sets into the train split ---
    case_sets_used = False
    if args.case_sets:
        if not args.regression_cases:
            raise SystemExit(
                "--case-sets requires --regression-cases to load the case "
                "query_id/doc_id denylist for the B+C isolation re-check."
            )
        case_query_ids, case_doc_ids = load_case_denylist(args.regression_cases)
        case_set_df = load_case_sets(args.case_sets)
        if not case_set_df.empty:
            # Defensive B+C re-check BEFORE merging into train.
            assert_case_set_isolation(case_set_df, case_query_ids, case_doc_ids)
            # Force split="train" regardless of source; case_sets never enter
            # validation/test.
            case_set_df = case_set_df.copy()
            case_set_df["split"] = "train"
            # Align columns to the main train_df schema (case_sets have the
            # same schema minus source_case_id which was already dropped).
            train_df = _merge_train_frames(train_df, case_set_df)
            case_sets_used = True
            print(
                f"Merged {len(case_set_df)} case_set rows into train "
                f"(B+C isolation check passed)."
            )
        else:
            print("case_sets loaded but empty; no rows merged into train.")

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
        "feature_set_name": REGISTRY.feature_set_name,
        "feature_set_version": REGISTRY.feature_set_version,
        "feature_versions": REGISTRY.feature_versions(),
        "params": params,
        "rounds": args.rounds,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "train_groups": int(train_df["query_id"].nunique()),
        "validation_groups": int(valid_df["query_id"].nunique()),
        "case_sets_used": case_sets_used,
        "reckless_mode": bool(args.reckless),
    }
    write_json(models_dir / "reranker_metadata.json", metadata)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {models_dir / 'reranker_metadata.json'}")


if __name__ == "__main__":
    main()
