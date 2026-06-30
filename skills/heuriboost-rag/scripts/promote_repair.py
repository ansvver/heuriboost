#!/usr/bin/env python3
"""Promote an eligible production-case repair run."""

from __future__ import annotations

import argparse
from pathlib import Path

from repair_cases import (
    append_gates,
    append_promoted_samples,
    build_anchor,
    gate_snapshot_from_case,
    load_compiled_production_cases,
    require_dependencies,
    read_ledger,
    utc_now_id,
    write_json,
    write_ledger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="heuriboost_output", help="Repair output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    heuriboost_dir = output_dir / ".heuriboost"
    repair_run_path = heuriboost_dir / "repair_run.json"
    if not repair_run_path.exists():
        raise SystemExit(f"Repair metadata not found: {repair_run_path}")

    import json

    require_dependencies("pandas")
    import pandas as pd

    try:
        repair_run = json.loads(repair_run_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Repair metadata is corrupt: {repair_run_path}") from exc

    if repair_run.get("acceptance_level") != "full":
        raise SystemExit("Only full acceptance repair runs can be promoted.")
    if not repair_run.get("promotion_eligible"):
        raise SystemExit("Repair run is not promotion eligible.")

    cases_path = Path(repair_run["compiled_artifacts"]["production_cases"])
    production_cases = load_compiled_production_cases(cases_path)
    gates = [
        gate_snapshot_from_case(case, repair_run["run_id"])
        for case in production_cases
        if case.get("good_doc_ids")
    ]
    append_gates(heuriboost_dir / "gates.jsonl", gates)

    case_sets_dir = Path(repair_run["compiled_artifacts"]["case_sets"])
    current_case_set = case_sets_dir / "current_production_cases.csv"
    if current_case_set.exists():
        repair_samples = pd.read_csv(current_case_set)
        append_promoted_samples(heuriboost_dir / "promoted_repair_samples.csv", repair_samples)

    ledger_path = heuriboost_dir / "ledger.json"
    ledger = read_ledger(ledger_path)
    anchor = build_anchor(
        repair_run["run_id"],
        repair_run["global_metrics"],
        repair_run["domain_metrics"],
        "promote",
    )
    ledger["anchor"] = anchor
    write_ledger(ledger_path, ledger)

    current_model = {
        "run_id": repair_run["run_id"],
        "model_path": str(output_dir / "models" / "reranker.json"),
        "metadata_path": str(output_dir / "models" / "reranker_metadata.json"),
        "promoted_at": utc_now_id(),
    }
    write_json(heuriboost_dir / "current_model.json", current_model)
    write_json(
        heuriboost_dir / "promotion.json",
        {
            "run_id": repair_run["run_id"],
            "promoted_at": current_model["promoted_at"],
            "gates_added": len(gates),
            "anchor": anchor,
            "current_model": current_model,
        },
    )

    print(f"Promoted repair run: {repair_run['run_id']}")
    print(f"Current model: {current_model['model_path']}")
    print(f"Gates stored: {heuriboost_dir / 'gates.jsonl'}")
    print(f"Anchor refreshed: {ledger_path}")


if __name__ == "__main__":
    main()
