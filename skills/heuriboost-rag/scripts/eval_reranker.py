#!/usr/bin/env python3
"""Evaluate a HeuriBoost reranker and run regression gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    FEATURE_NAMES,
    copy_regression_cases,
    ensure_output_dirs,
    evaluate_ranked_frame,
    extract_features,
    hard_negative_at_k,
    load_dataset,
    rank_by_baseline,
    rank_by_model,
    require_dependencies,
    split_frame,
    validate_dataset_frame,
    write_json,
)


EVIDENCE_TERMS_MAX = 8


def load_regression_cases(path: str | None):
    if not path:
        return []
    require_dependencies("yaml")
    import yaml

    case_path = Path(path)
    if not case_path.exists():
        raise SystemExit(f"Regression cases file not found: {case_path}")
    data = yaml.safe_load(case_path.read_text()) or {}
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("regression_cases.yaml must contain a top-level cases list.")
    return cases


def run_regression_cases(ranked_df, cases) -> list[dict]:
    failures = []
    for case in cases:
        query_id = case.get("query_id")
        top_k = int(case.get("top_k", 5))
        if not query_id:
            failures.append(
                {
                    "case_id": case.get("case_id", "<missing>"),
                    "reason": "case is missing query_id",
                }
            )
            continue

        group = ranked_df[ranked_df["query_id"] == query_id].head(top_k)
        ranked_doc_ids = group["doc_id"].astype(str).tolist()
        missing_required = [
            doc_id
            for doc_id in case.get("must_include_doc_ids", [])
            if str(doc_id) not in ranked_doc_ids
        ]
        forbidden_present = [
            doc_id
            for doc_id in case.get("must_not_include_doc_ids", [])
            if str(doc_id) in ranked_doc_ids
        ]
        if missing_required or forbidden_present:
            failures.append(
                {
                    "case_id": case.get("case_id", query_id),
                    "query_id": query_id,
                    "top_k": top_k,
                    "missing_required": missing_required,
                    "forbidden_present": forbidden_present,
                    "ranked_doc_ids": ranked_doc_ids,
                }
            )
    return failures


def row_by_doc_id(ranked_df, query_id: str, doc_id: str):
    group = ranked_df[
        (ranked_df["query_id"].astype(str) == str(query_id))
        & (ranked_df["doc_id"].astype(str) == str(doc_id))
    ]
    if group.empty:
        return None
    return group.iloc[0]


def rank_lookup(ranked_df, query_id: str) -> dict[str, int]:
    ranks = {}
    group = ranked_df[ranked_df["query_id"].astype(str) == str(query_id)]
    for rank, (_, row) in enumerate(group.iterrows(), start=1):
        ranks[str(row["doc_id"])] = rank
    return ranks


def evidence_hits(row, expected_terms: list[str]) -> list[str]:
    if row is None:
        return []
    text = str(row["doc_text"]).lower()
    return [term for term in expected_terms if str(term).lower() in text]


def analyze_reason(case, positive_row, negative_row, positive_features, negative_features) -> list[str]:
    reasons = []
    failure_type = case.get("failure_type")
    if failure_type:
        reasons.append(f"case is labeled as `{failure_type}`")

    if negative_row is not None and int(negative_row["label"]) == -1:
        reasons.append("the forbidden document is labeled as a hard negative")

    if positive_row is not None and negative_row is not None:
        dense_positive = positive_row.get("dense_rank")
        dense_negative = negative_row.get("dense_rank")
        try:
            if float(dense_negative) < float(dense_positive):
                reasons.append(
                    "retriever ranked forbidden doc higher initially "
                    f"(dense {dense_negative} vs {dense_positive})"
                )
        except (TypeError, ValueError):
            pass

    if positive_features and negative_features:
        if negative_features.get("wrong_year_flag", 0.0) > positive_features.get(
            "wrong_year_flag", 0.0
        ):
            reasons.append("forbidden doc has a stronger wrong-year signal")
        if positive_features.get("year_overlap_count", 0.0) > negative_features.get(
            "year_overlap_count", 0.0
        ):
            reasons.append("required doc matches the query year better")
        if positive_features.get("quarter_overlap_count", 0.0) > negative_features.get(
            "quarter_overlap_count", 0.0
        ):
            reasons.append("required doc matches the query quarter better")
        if positive_features.get("term_overlap_ratio", 0.0) > negative_features.get(
            "term_overlap_ratio", 0.0
        ):
            reasons.append("required doc has stronger query term overlap")

    if not reasons:
        reasons.append("no simple rule-based reason was detected")
    return reasons


def suggest_next_actions(reasons: list[str], case) -> list[str]:
    actions = []
    joined = " ".join(reasons).lower()
    failure_type = str(case.get("failure_type", "")).lower()
    if "wrong-year" in joined or "year" in joined or "temporal" in failure_type:
        actions.append("keep or strengthen temporal match features such as year/quarter overlap")
    if "term overlap" in joined:
        actions.append("inspect lexical/evidence overlap features for this query slice")
    if "retriever ranked forbidden doc higher" in joined:
        actions.append("track this as a hard-negative regression case for retriever drift")
    if case.get("expected_evidence"):
        actions.append("verify required documents contain the expected evidence terms")
    if not actions:
        actions.append("review the query/doc pair manually and add a more specific failure_type")
    return actions


def build_case_analyses(eval_df, model_ranked, baseline_ranked, feature_frame, cases):
    if not cases:
        return []

    feature_lookup = {}
    for row_index, (_, row) in enumerate(eval_df.iterrows()):
        key = (str(row["query_id"]), str(row["doc_id"]))
        feature_lookup[key] = {
            name: float(feature_frame.iloc[row_index][name]) for name in FEATURE_NAMES
        }

    analyses = []
    for case in cases:
        query_id = str(case.get("query_id", ""))
        model_ranks = rank_lookup(model_ranked, query_id)
        baseline_ranks = (
            rank_lookup(baseline_ranked, query_id) if baseline_ranked is not None else {}
        )
        expected = [str(term) for term in case.get("expected_evidence", [])]

        required_docs = []
        for doc_id in case.get("must_include_doc_ids", []):
            doc_id = str(doc_id)
            row = row_by_doc_id(model_ranked, query_id, doc_id)
            required_docs.append(
                {
                    "doc_id": doc_id,
                    "label": None if row is None else int(row["label"]),
                    "baseline_rank": baseline_ranks.get(doc_id),
                    "heuriboost_rank": model_ranks.get(doc_id),
                    "evidence_hits": evidence_hits(row, expected)[:EVIDENCE_TERMS_MAX],
                    "features": feature_lookup.get((query_id, doc_id), {}),
                }
            )

        forbidden_docs = []
        for doc_id in case.get("must_not_include_doc_ids", []):
            doc_id = str(doc_id)
            row = row_by_doc_id(model_ranked, query_id, doc_id)
            forbidden_docs.append(
                {
                    "doc_id": doc_id,
                    "label": None if row is None else int(row["label"]),
                    "baseline_rank": baseline_ranks.get(doc_id),
                    "heuriboost_rank": model_ranks.get(doc_id),
                    "evidence_hits": evidence_hits(row, expected)[:EVIDENCE_TERMS_MAX],
                    "features": feature_lookup.get((query_id, doc_id), {}),
                }
            )

        positive = required_docs[0] if required_docs else {}
        negative = forbidden_docs[0] if forbidden_docs else {}
        positive_row = row_by_doc_id(model_ranked, query_id, positive.get("doc_id", ""))
        negative_row = row_by_doc_id(model_ranked, query_id, negative.get("doc_id", ""))
        reasons = analyze_reason(
            case,
            positive_row,
            negative_row,
            positive.get("features", {}),
            negative.get("features", {}),
        )

        analyses.append(
            {
                "case_id": case.get("case_id", query_id),
                "query_id": query_id,
                "query": case.get("query", ""),
                "failure_type": case.get("failure_type", ""),
                "top_k": int(case.get("top_k", 5)),
                "required_docs": required_docs,
                "forbidden_docs": forbidden_docs,
                "reason_summary": reasons,
                "suggested_next_actions": suggest_next_actions(reasons, case),
            }
        )
    return analyses


def ranking_diff_frame(df, model_ranked):
    pd = __import__("pandas")

    rows = []
    original = rank_by_baseline(df, "dense")
    if original is None:
        original = df.sort_values(["query_id", "doc_id"], kind="stable")

    original_positions = {}
    for query_id, group in original.groupby("query_id", sort=False):
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            original_positions[(query_id, str(row["doc_id"]))] = rank

    for query_id, group in model_ranked.groupby("query_id", sort=False):
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            key = (query_id, str(row["doc_id"]))
            before = original_positions.get(key)
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": row["doc_id"],
                    "label": int(row["label"]),
                    "dense_rank": row.get("dense_rank", ""),
                    "heuriboost_rank": rank,
                    "rank_delta": None if before is None else before - rank,
                    "heuriboost_score": float(row["_heuriboost_score"]),
                    "doc_text": row["doc_text"],
                }
            )
    return pd.DataFrame(rows)


def write_eval_report(path: Path, metrics: dict, regression_failures: list[dict]) -> None:
    lines = [
        "# HeuriBoost Evaluation Report",
        "",
        "## Metrics",
        "",
        "| Ranker | nDCG@10 | MRR@10 | Recall@5 | Hard Negative@3 | Queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            "| {name} | {ndcg:.4f} | {mrr:.4f} | {recall:.4f} | {hard:.4f} | {queries:.0f} |".format(
                name=name,
                ndcg=values.get("ndcg@10", 0.0),
                mrr=values.get("mrr@10", 0.0),
                recall=values.get("recall@5", 0.0),
                hard=values.get("hard_negative@3", 0.0),
                queries=values.get("query_count", 0.0),
            )
        )

    lines.extend(["", "## Regression Gate", ""])
    if regression_failures:
        lines.append(f"FAILED: {len(regression_failures)} case(s) failed.")
        for failure in regression_failures:
            lines.append("")
            lines.append(f"- `{failure.get('case_id')}`")
            if failure.get("missing_required"):
                lines.append(
                    f"  - Missing required docs: {', '.join(map(str, failure['missing_required']))}"
                )
            if failure.get("forbidden_present"):
                lines.append(
                    f"  - Forbidden docs in top-k: {', '.join(map(str, failure['forbidden_present']))}"
                )
    else:
        lines.append("PASSED: no regression failures.")

    path.write_text("\n".join(lines) + "\n")


def write_failure_cases(path: Path, ranked_df, top_k: int = 3) -> None:
    lines = ["# HeuriBoost Hard-Negative Exposure", ""]
    any_exposure = False
    for query_id, group in ranked_df.groupby("query_id", sort=False):
        labels = group["label"].astype(int).tolist()
        if hard_negative_at_k(labels, top_k) == 0:
            continue
        any_exposure = True
        lines.append(f"## {query_id}")
        lines.append("")
        top = group.head(top_k)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                f"- #{rank} `{row['doc_id']}` label={int(row['label'])} score={float(row['_heuriboost_score']):.4f}"
            )
        lines.append("")
    if not any_exposure:
        lines.append(f"No hard negatives appeared in top {top_k}.")
    path.write_text("\n".join(lines) + "\n")


def format_feature_delta(name: str, required_features: dict, forbidden_features: dict) -> str:
    required = float(required_features.get(name, 0.0))
    forbidden = float(forbidden_features.get(name, 0.0))
    return f"`{name}` required={required:.4f}, forbidden={forbidden:.4f}, delta={required - forbidden:+.4f}"


def write_failure_analysis(path: Path, analyses: list[dict]) -> None:
    lines = ["# HeuriBoost Failure Analysis Lite", ""]
    if not analyses:
        lines.append("No regression cases were provided.")
        path.write_text("\n".join(lines) + "\n")
        return

    for analysis in analyses:
        lines.append(f"## {analysis['case_id']}")
        lines.append("")
        if analysis.get("query"):
            lines.append(f"Query: {analysis['query']}")
            lines.append("")
        if analysis.get("failure_type"):
            lines.append(f"Failure type: `{analysis['failure_type']}`")
            lines.append("")

        lines.append("### Reason Summary")
        lines.append("")
        for reason in analysis["reason_summary"]:
            lines.append(f"- {reason}")
        lines.append("")

        lines.append("### Rank Movement")
        lines.append("")
        lines.append("| Role | Doc | Label | Baseline Rank | HeuriBoost Rank | Evidence Hits |")
        lines.append("|---|---|---:|---:|---:|---|")
        for role, docs in (
            ("required", analysis["required_docs"]),
            ("forbidden", analysis["forbidden_docs"]),
        ):
            for doc in docs:
                hits = ", ".join(doc["evidence_hits"]) if doc["evidence_hits"] else "-"
                lines.append(
                    f"| {role} | `{doc['doc_id']}` | {doc['label']} | "
                    f"{doc['baseline_rank']} | {doc['heuriboost_rank']} | {hits} |"
                )
        lines.append("")

        if analysis["required_docs"] and analysis["forbidden_docs"]:
            required_features = analysis["required_docs"][0].get("features", {})
            forbidden_features = analysis["forbidden_docs"][0].get("features", {})
            lines.append("### Feature Contrast")
            lines.append("")
            for feature in (
                "dense_score",
                "term_overlap_ratio",
                "year_overlap_count",
                "quarter_overlap_count",
                "wrong_year_flag",
            ):
                lines.append(f"- {format_feature_delta(feature, required_features, forbidden_features)}")
            lines.append("")

        lines.append("### Suggested Next Actions")
        lines.append("")
        for action in analysis["suggested_next_actions"]:
            lines.append(f"- {action}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to query_doc_examples.csv")
    parser.add_argument(
        "--output-dir",
        default="heuriboost_output",
        help="Directory containing model artifacts and receiving reports",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--regression-cases", help="Path to regression_cases.yaml")
    args = parser.parse_args()

    require_dependencies("xgboost", "pandas", "yaml")
    import xgboost as xgb

    df = load_dataset(args.dataset)
    validate_dataset_frame(df)
    eval_df = split_frame(df, args.split)
    if eval_df.empty:
        raise SystemExit(f"Split is empty: {args.split}")

    models_dir, reports_dir = ensure_output_dirs(args.output_dir)
    model_path = models_dir / "reranker.json"
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}. Run train_reranker.py before eval_reranker.py."
        )

    model = xgb.Booster()
    model.load_model(model_path)

    x_eval = extract_features(eval_df)
    dmatrix = xgb.DMatrix(x_eval, feature_names=FEATURE_NAMES)
    scores = model.predict(dmatrix)
    model_ranked = rank_by_model(eval_df, scores)
    baseline_ranked = rank_by_baseline(eval_df, "dense")

    metrics = {"heuriboost": evaluate_ranked_frame(model_ranked)}
    for baseline in ("dense", "sparse", "rrf"):
        ranked = rank_by_baseline(eval_df, baseline)
        if ranked is not None:
            metrics[baseline] = evaluate_ranked_frame(ranked)

    cases = load_regression_cases(args.regression_cases)
    regression_failures = run_regression_cases(model_ranked, cases)
    case_analyses = build_case_analyses(
        eval_df, model_ranked, baseline_ranked, x_eval, cases
    )

    write_eval_report(reports_dir / "eval_report.md", metrics, regression_failures)
    ranking_diff_frame(eval_df, model_ranked).to_csv(
        reports_dir / "ranking_diff.csv", index=False
    )
    write_failure_cases(reports_dir / "failure_cases.md", model_ranked)
    write_failure_analysis(reports_dir / "failure_analysis.md", case_analyses)
    write_json(reports_dir / "failure_analysis.json", case_analyses)

    # XGBoost omits unused features and may key by feature name. Normalize for report consumers.
    raw_scores = model.get_score(importance_type="gain")
    feature_importance = {
        name: float(raw_scores.get(name, 0.0)) for name in FEATURE_NAMES
    }
    write_json(reports_dir / "feature_importance.json", feature_importance)
    copy_regression_cases(args.regression_cases, args.output_dir)

    print(f"Saved report: {reports_dir / 'eval_report.md'}")
    print(f"Saved ranking diff: {reports_dir / 'ranking_diff.csv'}")
    print(f"Saved failure cases: {reports_dir / 'failure_cases.md'}")
    print(f"Saved failure analysis: {reports_dir / 'failure_analysis.md'}")
    print(f"Saved feature importance: {reports_dir / 'feature_importance.json'}")
    if regression_failures:
        raise SystemExit(f"Regression gate failed: {len(regression_failures)} case(s)")


if __name__ == "__main__":
    main()
