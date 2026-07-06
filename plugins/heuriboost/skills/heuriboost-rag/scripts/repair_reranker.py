#!/usr/bin/env python3
"""Run strict production-case repair for a HeuriBoost reranker."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from repair_cases import (
    DEFAULT_CASE_TOP_K,
    DEFAULT_MIN_DOCS_PER_QUERY,
    DEFAULT_MIN_DOMAIN_TEST_QUERIES,
    DEFAULT_MIN_GLOBAL_TEST_QUERIES,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    CompileOptions,
    build_anchor,
    compile_repair_inputs,
    evaluate_cases,
    evaluate_model_by_domain,
    evaluate_model_on_split,
    load_gates,
    merge_training_frames,
    metric_deltas,
    metrics_improved,
    metrics_not_regressed,
    read_ledger,
    train_model_from_frame,
    utc_now_id,
    write_json,
    write_ledger,
)


def _parse_split_ratios(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split ratios must be train,validation,test")
    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split ratios must be numeric") from exc
    if any(part < 0 for part in ratios):
        raise argparse.ArgumentTypeError("split ratios must be non-negative")
    if sum(ratios) <= 0:
        raise argparse.ArgumentTypeError("split ratios must sum to a positive value")
    return ratios  # type: ignore[return-value]


def _ensure_anchor(result, ledger: dict, reset_anchor: bool, keep_baseline: bool, rounds: int) -> tuple[dict, dict | None]:
    anchor = ledger.get("anchor")
    if anchor and not reset_anchor:
        return anchor, None

    if anchor and reset_anchor:
        print("Resetting existing anchor from base dataset baseline.")
    else:
        print("No anchor found; initializing anchor from base dataset baseline.")

    temp_dir = result.heuriboost_dir / "_baseline_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    baseline_model = train_model_from_frame(result.base_df, temp_dir, rounds=rounds)
    global_metrics, _ = evaluate_model_on_split(baseline_model, result.base_df, "test")
    domain_metrics = evaluate_model_by_domain(baseline_model, result.base_df, "test")
    round_id = f"anchor-{utc_now_id()}"
    new_anchor = build_anchor(round_id, global_metrics, domain_metrics, "auto")
    ledger["anchor"] = new_anchor
    write_ledger(result.heuriboost_dir / "ledger.json", ledger)
    write_json(
        result.heuriboost_dir / "anchor_baseline.json",
        {
            "round_id": round_id,
            "global": global_metrics,
            "domains": domain_metrics,
            "source": "auto-initialized from base_dataset",
        },
    )
    if not keep_baseline and temp_dir.exists():
        shutil.rmtree(temp_dir)
    return new_anchor, {
        "round_id": round_id,
        "global": global_metrics,
        "domains": domain_metrics,
    }


def _write_repair_report(path: Path, metadata: dict) -> None:
    lines = [
        "# Production Case Repair Report",
        "",
        "## Summary",
        "",
        f"- Run id: `{metadata['run_id']}`",
        f"- Acceptance level: `{metadata['acceptance_level']}`",
        f"- Promotion eligible: `{metadata['promotion_eligible']}`",
        f"- Global passed: `{metadata['global_passed']}`",
        f"- Domain passed: `{metadata['domain_passed']}`",
        f"- Current cases passed: `{metadata['cases_passed']}`",
        f"- Historical gates passed: `{metadata['gates_passed']}`",
        "",
        "## Global Metrics",
        "",
        "| Metric | Current | Anchor | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in metadata["global_deltas"].items():
        lines.append(
            f"| {metric} | {values['current']:.4f} | {values['anchor']:.4f} | {values['delta']:+.4f} |"
        )

    lines.extend(["", "## Touched Domains", ""])
    if metadata["domain_deltas"]:
        for domain, deltas in metadata["domain_deltas"].items():
            lines.append(f"### {domain}")
            lines.append("")
            lines.append("| Metric | Current | Anchor | Delta |")
            lines.append("|---|---:|---:|---:|")
            for metric, values in deltas.items():
                lines.append(
                    f"| {metric} | {values['current']:.4f} | {values['anchor']:.4f} | {values['delta']:+.4f} |"
                )
            lines.append("")
    else:
        lines.append("(none)")

    lines.extend(["", "## Current Production Cases", ""])
    for result in metadata["case_results"]:
        mark = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"- `{result['case_id']}` domain={result['domain']} level={result['acceptance_level']} {mark}"
        )

    lines.extend(["", "## Historical Gates", ""])
    if metadata["gate_results"]:
        for result in metadata["gate_results"]:
            mark = "PASS" if result["passed"] else "FAIL"
            lines.append(
                f"- `{result['case_id']}` domain={result['domain']} level={result['acceptance_level']} {mark}"
            )
    else:
        lines.append("(none)")

    lines.extend(["", "## Warnings", ""])
    if metadata.get("warnings"):
        for warning in metadata["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("(none)")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", required=True, help="User-facing base dataset CSV/JSONL")
    parser.add_argument("--production-cases", required=True, help="User-facing production cases CSV/JSONL")
    parser.add_argument("--output-dir", default="heuriboost_output", help="Output directory")
    parser.add_argument(
        "--reckless",
        action="store_true",
        help="Required flag for strict production repair",
    )
    parser.add_argument(
        "--acceptance-level",
        choices=["full", "weak"],
        default="full",
        help="Case acceptance level (default: full)",
    )
    parser.add_argument("--case-top-k", type=int, default=DEFAULT_CASE_TOP_K)
    parser.add_argument("--rounds", type=int, default=40, help="XGBoost boosting rounds")
    parser.add_argument("--resplit", action="store_true", help="Replace user-provided splits")
    parser.add_argument("--split-ratio", type=_parse_split_ratios, default=DEFAULT_SPLIT_RATIOS)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--min-global-test-queries", type=int, default=DEFAULT_MIN_GLOBAL_TEST_QUERIES)
    parser.add_argument("--min-domain-test-queries", type=int, default=DEFAULT_MIN_DOMAIN_TEST_QUERIES)
    parser.add_argument("--min-docs-per-query", type=int, default=DEFAULT_MIN_DOCS_PER_QUERY)
    parser.add_argument(
        "--reset-anchor",
        action="store_true",
        help="Overwrite an existing anchor with a fresh base-dataset baseline",
    )
    parser.add_argument(
        "--keep-baseline-artifacts",
        action="store_true",
        help="Keep temporary baseline model artifacts used for auto-anchor initialization",
    )
    args = parser.parse_args()

    if not args.reckless:
        raise SystemExit("repair_reranker.py requires --reckless.")
    if args.acceptance_level == "weak":
        print("WARNING: weak acceptance is not promotion eligible.")

    output_dir = Path(args.output_dir)
    options = CompileOptions(
        output_dir=output_dir,
        resplit=args.resplit,
        split_ratios=args.split_ratio,
        split_seed=args.split_seed,
        case_top_k=args.case_top_k,
        strict=True,
        acceptance_level=args.acceptance_level,
        min_global_test_queries=args.min_global_test_queries,
        min_domain_test_queries=args.min_domain_test_queries,
        min_docs_per_query=args.min_docs_per_query,
    )
    result = compile_repair_inputs(args.base_dataset, args.production_cases, options)
    ledger_path = result.heuriboost_dir / "ledger.json"
    ledger = read_ledger(ledger_path)
    anchor, anchor_baseline = _ensure_anchor(
        result,
        ledger,
        reset_anchor=args.reset_anchor,
        keep_baseline=args.keep_baseline_artifacts,
        rounds=args.rounds,
    )

    promoted_samples_path = result.heuriboost_dir / "promoted_repair_samples.csv"
    training_df = merge_training_frames(
        result.base_df,
        result.repair_samples_df,
        promoted_samples_path=promoted_samples_path,
    )
    model = train_model_from_frame(training_df, output_dir, rounds=args.rounds)

    global_metrics, _ = evaluate_model_on_split(model, result.base_df, "test")
    domain_metrics = evaluate_model_by_domain(model, result.base_df, "test")
    case_results = evaluate_cases(
        result.production_cases, model, acceptance_level=args.acceptance_level
    )
    gates = load_gates(result.heuriboost_dir / "gates.jsonl")
    gate_results = evaluate_cases(gates, model, acceptance_level="full")

    global_passed = metrics_improved(global_metrics, anchor.get("global", {}))
    global_deltas = metric_deltas(global_metrics, anchor.get("global", {}))

    domain_deltas = {}
    domain_passed = True
    anchor_domains = anchor.get("domains", {})
    for domain in result.touched_domains:
        current = domain_metrics.get(domain)
        domain_anchor = anchor_domains.get(domain)
        if current is None:
            raise SystemExit(f"Touched domain {domain!r} has no current test metrics.")
        if domain_anchor is None:
            raise SystemExit(f"Touched domain {domain!r} has no anchor metrics.")
        domain_deltas[domain] = metric_deltas(current, domain_anchor)
        if not metrics_not_regressed(current, domain_anchor):
            domain_passed = False

    cases_passed = all(result_item["passed"] for result_item in case_results)
    gates_passed = all(result_item["passed"] for result_item in gate_results)
    promotion_eligible = (
        args.acceptance_level == "full"
        and cases_passed
        and gates_passed
        and global_passed
        and domain_passed
    )
    run_id = utc_now_id()
    metadata = {
        "run_id": run_id,
        "repair_mode": "production_cases",
        "acceptance_level": args.acceptance_level,
        "promotion_eligible": promotion_eligible,
        "global_passed": global_passed,
        "domain_passed": domain_passed,
        "cases_passed": cases_passed,
        "gates_passed": gates_passed,
        "touched_domains": result.touched_domains,
        "global_metrics": global_metrics,
        "domain_metrics": domain_metrics,
        "global_deltas": global_deltas,
        "domain_deltas": domain_deltas,
        "case_results": case_results,
        "gate_results": gate_results,
        "production_cases": result.production_cases,
        "compiled_artifacts": {
            "base_dataset": str(result.base_dataset_path),
            "production_cases": str(result.production_cases_json_path),
            "regression_cases": str(result.regression_cases_path),
            "case_sets": str(result.case_sets_dir),
        },
        "anchor_baseline": anchor_baseline,
        "warnings": result.warnings,
    }

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "repair_report.md"
    metadata_path = result.heuriboost_dir / "repair_run.json"
    _write_repair_report(report_path, metadata)
    write_json(metadata_path, metadata)

    round_snapshot = {
        "round_id": run_id,
        "repair_mode": "production_cases",
        "acceptance_level": args.acceptance_level,
        "promotion_eligible": promotion_eligible,
        "global": {
            "ndcg@10": float(global_metrics.get("ndcg@10", 0.0)),
            "mrr@10": float(global_metrics.get("mrr@10", 0.0)),
        },
        "domains": domain_metrics,
        "touched_domains": result.touched_domains,
        "global_deltas": global_deltas,
        "domain_deltas": domain_deltas,
        "case_results": case_results,
        "gate_results": gate_results,
    }
    ledger = read_ledger(ledger_path)
    ledger.setdefault("rounds", []).append(round_snapshot)
    write_ledger(ledger_path, ledger)

    print(f"Saved repaired model: {output_dir / 'models' / 'reranker.json'}")
    print(f"Saved repair report: {report_path}")
    print(f"Saved repair metadata: {metadata_path}")
    print(f"Promotion eligible: {promotion_eligible}")

    failures = []
    if not cases_passed:
        failures.append("current production case acceptance failed")
    if not gates_passed:
        failures.append("historical gate acceptance failed")
    if not global_passed:
        failures.append("global test metrics did not improve over anchor")
    if not domain_passed:
        failures.append("touched domain metrics regressed versus anchor")
    if failures:
        raise SystemExit("Reckless repair failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
