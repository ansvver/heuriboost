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
    ndcg_at_k,
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
    """Evaluate regression cases with status awareness.

    Skips ``retired`` cases entirely. For each non-retired case, computes the
    hit check (must_include in top_k, must_not_include out of top_k) plus the
    optional A checks (``require_rank``, ``min_ndcg10``).

    Returns a list of rich per-case result dicts:
    ``{case_id, status, passed, missing_required, forbidden_present,
    rank_of_required, query_ndcg10}``.
    """
    results = []
    for case in cases:
        status = case.get("status", "gate")
        if status == "retired":
            continue

        case_id = case.get("case_id", "<missing>")
        query_id = case.get("query_id")
        top_k = int(case.get("top_k", 5))

        if not query_id:
            results.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "passed": False,
                    "missing_required": [],
                    "forbidden_present": [],
                    "rank_of_required": None,
                    "query_ndcg10": 0.0,
                    "reason": "case is missing query_id",
                }
            )
            continue

        query_id_str = str(query_id)
        group = ranked_df[ranked_df["query_id"].astype(str) == query_id_str]

        # Full model-ranked doc-id list for this query (ranked_df is already
        # sorted by query_id then _heuriboost_score descending).
        ranked_doc_ids = group["doc_id"].astype(str).tolist()
        top_k_doc_ids = ranked_doc_ids[:top_k]

        must_include = [str(d) for d in case.get("must_include_doc_ids", [])]
        must_not_include = [str(d) for d in case.get("must_not_include_doc_ids", [])]

        missing_required = [d for d in must_include if d not in top_k_doc_ids]
        forbidden_present = [d for d in must_not_include if d in top_k_doc_ids]

        # Rank of the first must_include doc in the full model ranking.
        rank_of_required = None
        if must_include:
            for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                if doc_id in must_include:
                    rank_of_required = rank
                    break

        # Per-query nDCG@10 on model-ranked labels.
        labels = group["label"].astype(int).tolist()
        query_ndcg10 = ndcg_at_k(labels, 10)

        # Determine pass/fail.
        passed = not missing_required and not forbidden_present

        # A check: require_rank — first must_include doc must reach rank <= N.
        require_rank = case.get("require_rank")
        if require_rank is not None and rank_of_required is not None:
            if rank_of_required > int(require_rank):
                passed = False

        # A check: min_ndcg10 — per-query nDCG@10 floor.
        min_ndcg10 = case.get("min_ndcg10")
        if min_ndcg10 is not None:
            if query_ndcg10 < float(min_ndcg10):
                passed = False

        results.append(
            {
                "case_id": case_id,
                "status": status,
                "passed": passed,
                "missing_required": missing_required,
                "forbidden_present": forbidden_present,
                "rank_of_required": rank_of_required,
                "query_ndcg10": query_ndcg10,
            }
        )
    return results


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
        if negative_features.get("entity_overlap_count", 0.0) < positive_features.get(
            "entity_overlap_count", 0.0
        ):
            reasons.append("required doc shares more query entities")
        if positive_features.get("number_overlap_count", 0.0) > negative_features.get(
            "number_overlap_count", 0.0
        ):
            reasons.append("required doc matches more query numbers")
        if positive_features.get("important_term_overlap", 0.0) > negative_features.get(
            "important_term_overlap", 0.0
        ):
            reasons.append("required doc matches more important query terms")
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
    if "entity" in joined or "number" in joined or "important term" in joined:
        actions.append(
            "inspect entity/number/important-term overlap features for this slice"
        )
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


def write_eval_report(path: Path, metrics: dict, case_results: list[dict]) -> None:
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

    gate_results = [r for r in case_results if r["status"] == "gate"]
    pending_results = [r for r in case_results if r["status"] == "pending"]

    lines.extend(["", "## Regression Gate", ""])

    # --- Gates ---
    lines.append("### Gates")
    lines.append("")
    if gate_results:
        gate_failures = [r for r in gate_results if not r["passed"]]
        if gate_failures:
            lines.append(f"FAILED: {len(gate_failures)} gate case(s) failed.")
        else:
            lines.append("PASSED: all gate cases passed.")
        for result in gate_results:
            mark = "PASS" if result["passed"] else "FAIL"
            lines.append("")
            lines.append(f"- `{result['case_id']}`: {mark}")
            if not result["passed"]:
                if result.get("missing_required"):
                    lines.append(
                        f"  - Missing required docs: {', '.join(map(str, result['missing_required']))}"
                    )
                if result.get("forbidden_present"):
                    lines.append(
                        f"  - Forbidden docs in top-k: {', '.join(map(str, result['forbidden_present']))}"
                    )
                if result.get("rank_of_required") is not None:
                    lines.append(f"  - Rank of required doc: {result['rank_of_required']}")
    else:
        lines.append("(no gate cases)")

    # --- Pending ---
    lines.append("")
    lines.append("### Pending")
    lines.append("")
    if pending_results:
        pending_passed = [r for r in pending_results if r["passed"]]
        if pending_passed:
            lines.append(
                f"PROMOTION CANDIDATES: {len(pending_passed)} pending case(s) "
                f"passed this round."
            )
        else:
            lines.append("No pending cases passed this round.")
        for result in pending_results:
            mark = "PASS" if result["passed"] else "FAIL"
            promotion = " (promotion candidate)" if result["passed"] else ""
            lines.append("")
            lines.append(f"- `{result['case_id']}`: {mark}{promotion}")
            if not result["passed"]:
                if result.get("missing_required"):
                    lines.append(
                        f"  - Missing required docs: {', '.join(map(str, result['missing_required']))}"
                    )
                if result.get("forbidden_present"):
                    lines.append(
                        f"  - Forbidden docs in top-k: {', '.join(map(str, result['forbidden_present']))}"
                    )
                if result.get("rank_of_required") is not None:
                    lines.append(f"  - Rank of required doc: {result['rank_of_required']}")
    else:
        lines.append("(no pending cases)")

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
                "number_overlap_count",
                "entity_overlap_count",
                "important_term_overlap",
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
    parser.add_argument(
        "--ledger",
        default="examples/fiqa/ledger.json",
        help="Path to the committed cross-round ledger JSON (default: examples/fiqa/ledger.json)",
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        help="Skip writing to the cross-round ledger (useful for ad-hoc eval)",
    )
    parser.add_argument(
        "--case-sets-used",
        action="store_true",
        help="Tag this round in the ledger as having used mined case_sets for training",
    )
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
    case_results = run_regression_cases(model_ranked, cases)
    case_analyses = build_case_analyses(
        eval_df, model_ranked, baseline_ranked, x_eval, cases
    )

    write_eval_report(reports_dir / "eval_report.md", metrics, case_results)
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

    # --- Per-status summary ---
    gate_results = [r for r in case_results if r["status"] == "gate"]
    pending_results = [r for r in case_results if r["status"] == "pending"]
    gate_pass = sum(1 for r in gate_results if r["passed"])
    pending_pass = sum(1 for r in pending_results if r["passed"])
    promotion_candidates = [r["case_id"] for r in pending_results if r["passed"]]

    # --- Cross-round ledger (Phase 3) ---
    if not args.no_ledger:
        import regression_ledger

        heuriboost_metrics = metrics.get("heuriboost", {})
        global_metrics = {
            "ndcg@10": heuriboost_metrics.get("ndcg@10", 0.0),
            "mrr@10": heuriboost_metrics.get("mrr@10", 0.0),
        }
        round_snapshot = regression_ledger.record(
            global_metrics, case_results, args.split, args.ledger,
            case_sets_used=args.case_sets_used,
        )
        vs_anchor = round_snapshot.get("vs_anchor")
        if vs_anchor is None:
            print("B vs anchor: no anchor yet — run set-anchor to establish one")
        else:
            regressed_str = "REGRESSED" if vs_anchor["regressed"] else "ok"
            print(
                f"B vs anchor: nDCG@10 delta={vs_anchor['ndcg@10']:+.4f} "
                f"({regressed_str})"
            )

    print(f"Gates: {gate_pass}/{len(gate_results)} pass")
    print(f"Pending: {pending_pass}/{len(pending_results)} pass")
    if promotion_candidates:
        print(f"Promotion candidates: {', '.join(promotion_candidates)}")

    # Exit non-zero ONLY on gate failure. Pending failures never block.
    gate_failures = [r for r in gate_results if not r["passed"]]
    if gate_failures:
        raise SystemExit(f"Regression gate failed: {len(gate_failures)} gate case(s)")


if __name__ == "__main__":
    main()
