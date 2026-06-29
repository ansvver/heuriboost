#!/usr/bin/env python3
"""Cross-round regression ledger for the HeuriBoost case state machine.

Owns the committed ledger (default ``examples/fiqa/ledger.json``), the B2
anchor, the global-vs-anchor (B) comparison, the progress summary, and the
manual promotion/anchor helpers.

This module does NOT auto-commit the ledger file; the maintainer commits it
alongside the round's other changes.

CLI subcommands: ``set-anchor``, ``summary``, ``promote``.
Importable functions: :func:`record`, :func:`set_anchor`, :func:`summary`,
:func:`promote`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import require_dependencies

DEFAULT_LEDGER_PATH = "examples/fiqa/ledger.json"
# Epsilon for the B regression check. With eps=0.0, any drop in nDCG@10 below
# the anchor counts as a regression (exact comparison).
EPS = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ledger(ledger_path: str | Path) -> dict:
    """Load the ledger JSON, returning a well-formed dict even if the file is
    missing or partially populated."""
    path = Path(ledger_path)
    if not path.exists():
        return {"anchor": None, "rounds": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Ledger JSON is corrupt: {path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Ledger JSON must be a JSON object: {path}")
    data.setdefault("anchor", None)
    data.setdefault("rounds", [])
    return data


def _save_ledger(ledger_path: str | Path, data: dict) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_summary(cases: list[dict]) -> dict:
    """Tally gate/pending/retired counts and promotion candidates from a
    round's case list."""
    gate = [c for c in cases if c.get("status") == "gate"]
    pending = [c for c in cases if c.get("status") == "pending"]
    retired = [c for c in cases if c.get("status") == "retired"]
    return {
        "gate_pass": sum(1 for c in gate if c.get("passed")),
        "gate_total": len(gate),
        "pending_pass": sum(1 for c in pending if c.get("passed")),
        "pending_total": len(pending),
        "retired_total": len(retired),
        "promotion_candidates": [c["case_id"] for c in pending if c.get("passed")],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record(
    global_metrics: dict,
    case_results: list[dict],
    split: str,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    case_sets_used: bool = False,
) -> dict:
    """Append a round snapshot to the ledger.

    Parameters
    ----------
    global_metrics : dict
        ``{"ndcg@10": float, "mrr@10": float}`` — the heuriboost row's global
        metrics for this round.
    case_results : list[dict]
        Per-case result dicts from ``eval_reranker.run_regression_cases``.
        Each must have at least ``case_id``, ``status``, ``passed``.
    split : str
        The dataset split that was evaluated (e.g. ``"validation"``).
    ledger_path : str | Path
        Path to the committed ledger JSON.
    case_sets_used : bool
        Whether this round's model was trained with mined case_sets folded
        into the train split. Stored in the round snapshot so the summary
        can surface it.

    Returns
    -------
    dict
        The round snapshot that was appended, including ``vs_anchor`` (or
        ``None`` if no anchor exists yet).
    """
    ledger = _load_ledger(ledger_path)
    round_id = _now_utc_iso()

    case_entries = [
        {
            "case_id": result.get("case_id"),
            "status": result.get("status", "gate"),
            "passed": bool(result.get("passed", False)),
        }
        for result in case_results
    ]

    vs_anchor = None
    anchor = ledger.get("anchor")
    if anchor is not None:
        anchor_ndcg = float(anchor.get("global", {}).get("ndcg@10", 0.0))
        round_ndcg = float(global_metrics.get("ndcg@10", 0.0))
        delta = round_ndcg - anchor_ndcg
        regressed = round_ndcg < anchor_ndcg - EPS
        vs_anchor = {"ndcg@10": delta, "regressed": regressed}

    round_snapshot = {
        "round_id": round_id,
        "split": split,
        "global": {
            "ndcg@10": float(global_metrics.get("ndcg@10", 0.0)),
            "mrr@10": float(global_metrics.get("mrr@10", 0.0)),
        },
        "cases": case_entries,
        "vs_anchor": vs_anchor,
        "case_sets_used": bool(case_sets_used),
    }

    ledger["rounds"].append(round_snapshot)
    _save_ledger(ledger_path, ledger)

    return round_snapshot


def set_anchor(
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    round_id: str | None = None,
) -> dict:
    """Freeze a round's global metrics as the B2 anchor.

    If *round_id* is ``None``, the latest round is used.

    Returns the anchor dict that was written.
    """
    ledger = _load_ledger(ledger_path)
    rounds = ledger.get("rounds", [])
    if not rounds:
        raise SystemExit(
            "No rounds in ledger. Run an evaluation first to record a round."
        )

    if round_id is None:
        target = rounds[-1]
    else:
        target = None
        for r in rounds:
            if r.get("round_id") == round_id:
                target = r
                break
        if target is None:
            available = ", ".join(r["round_id"] for r in rounds[-5:])
            raise SystemExit(
                f"Round not found: {round_id}. Recent rounds: {available}"
            )

    anchor = {
        "round_id": target["round_id"],
        "global": target["global"],
        "set_by": "manual",
    }
    ledger["anchor"] = anchor
    _save_ledger(ledger_path, ledger)

    return anchor


def summary(ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> None:
    """Print a progress summary from the latest round."""
    ledger = _load_ledger(ledger_path)
    rounds = ledger.get("rounds", [])
    if not rounds:
        print("No rounds recorded yet.")
        return

    latest = rounds[-1]
    cases = latest.get("cases", [])
    tally = _round_summary(cases)

    print(f"Round: {latest['round_id']} (split={latest.get('split', '?')})")
    print(
        f"Global: nDCG@10={latest['global'].get('ndcg@10', 0.0):.4f}, "
        f"MRR@10={latest['global'].get('mrr@10', 0.0):.4f}"
    )
    if latest.get("case_sets_used"):
        print(f"Round {latest['round_id']}: used case_sets (mined samples in train)")
    print(f"Gates: {tally['gate_pass']}/{tally['gate_total']} pass")
    print(f"Pending: {tally['pending_pass']}/{tally['pending_total']} pass")
    print(f"Retired: {tally['retired_total']}")

    if tally["promotion_candidates"]:
        print(
            f"Promotion candidates: {', '.join(tally['promotion_candidates'])}"
        )
    else:
        print("Promotion candidates: (none)")

    anchor = ledger.get("anchor")
    vs_anchor = latest.get("vs_anchor")
    if anchor is None:
        print("B vs anchor: no anchor yet — run set-anchor to establish one")
    elif vs_anchor is not None:
        regressed_str = "REGRESSED" if vs_anchor.get("regressed") else "ok"
        print(
            f"B vs anchor: nDCG@10 delta={vs_anchor['ndcg@10']:+.4f} "
            f"({regressed_str})"
        )
    else:
        print("B vs anchor: (not computed for this round)")


def promote(
    cases_file: str,
    case_id: str,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> None:
    """Print the latest hit+A+B evidence for a case and, on interactive
    confirmation, flip its status from ``pending`` to ``gate`` in the YAML.

    .. note::

       This function uses ``pyyaml`` for the YAML round-trip.  ``pyyaml`` does
       NOT preserve comments or formatting.  If ``ruamel.yaml`` is available in
       the future, prefer it to preserve comments.  For now, comment loss is
       accepted and the user is warned.

    No auto-promotion: the user must type ``yes`` at the prompt.
    """
    require_dependencies("yaml")
    import yaml

    ledger = _load_ledger(ledger_path)
    rounds = ledger.get("rounds", [])

    # Collect per-round evidence for this case from the ledger.
    case_history: list[dict] = []
    for r in rounds:
        for c in r.get("cases", []):
            if c.get("case_id") == case_id:
                case_history.append(
                    {
                        "round_id": r["round_id"],
                        "global": r["global"],
                        "case": c,
                        "vs_anchor": r.get("vs_anchor"),
                    }
                )

    if not case_history:
        raise SystemExit(f"Case not found in ledger: {case_id}")

    latest = case_history[-1]
    print(f"=== Promotion evidence for {case_id} ===")
    print(f"Latest round: {latest['round_id']}")
    print(f"  Case status: {latest['case'].get('status')}")
    print(f"  Case passed (hit + A): {latest['case'].get('passed')}")
    print(
        f"  Global nDCG@10: {latest['global'].get('ndcg@10', 0.0):.4f}, "
        f"MRR@10: {latest['global'].get('mrr@10', 0.0):.4f}"
    )
    vs = latest.get("vs_anchor")
    if vs is not None:
        regressed_str = "REGRESSED" if vs.get("regressed") else "ok"
        print(
            f"  B vs anchor: nDCG@10 delta={vs['ndcg@10']:+.4f} ({regressed_str})"
        )
    else:
        print("  B vs anchor: no anchor")

    hit_history = [h["case"].get("passed") for h in case_history]
    print(f"\nAll rounds — passed history: {hit_history}")

    # Check current status in the ledger.
    if latest["case"].get("status") != "pending":
        print(
            f"\nCase is currently '{latest['case'].get('status')}' in the "
            f"ledger, not 'pending'. Nothing to promote."
        )
        return

    if not latest["case"].get("passed"):
        print(
            "\nWARNING: Case did NOT pass the latest round. "
            "Promotion is not recommended."
        )

    # Load and inspect the YAML.
    cases_path = Path(cases_file)
    if not cases_path.exists():
        raise SystemExit(f"Cases file not found: {cases_path}")

    data = yaml.safe_load(cases_path.read_text()) or {}
    cases = data.get("cases", [])

    target_case = None
    for case in cases:
        if case.get("case_id") == case_id:
            target_case = case
            break

    if target_case is None:
        raise SystemExit(f"Case not found in YAML: {case_id}")

    current_status = target_case.get("status", "gate")
    if current_status != "pending":
        print(
            f"\nCase status in YAML is '{current_status}', not 'pending'. "
            f"Nothing to promote."
        )
        return

    # Ask for confirmation.
    print(f"\nPromote '{case_id}' from pending to gate in {cases_path}?")
    response = input("Type 'yes' to confirm: ").strip().lower()
    if response != "yes":
        print("Promotion cancelled.")
        return

    # Flip status.
    target_case["status"] = "gate"

    # Write back (pyyaml round-trip; comments WILL be lost).
    cases_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )
    print(f"Promoted '{case_id}' to gate in {cases_path}.")
    print(
        "NOTE: pyyaml round-trip was used; YAML comments and formatting may "
        "have been lost."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    # --ledger is accepted on every subcommand via a parent parser so that
    # ``regression_ledger.py set-anchor --ledger X`` works (the flag comes
    # AFTER the subcommand, matching the documented CLI).
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--ledger",
        default=DEFAULT_LEDGER_PATH,
        help=f"Path to ledger.json (default: {DEFAULT_LEDGER_PATH})",
    )

    sub.add_parser("set-anchor", parents=[parent], help="Freeze a round's global metrics as the anchor")
    sub.add_parser("summary", parents=[parent], help="Print a progress summary")

    sub_promote = sub.add_parser("promote", parents=[parent], help="Promote a pending case to gate")
    sub_promote.add_argument("cases_file", help="Path to regression_cases.yaml")
    sub_promote.add_argument("case_id", help="Case ID to promote")

    args = parser.parse_args()

    if args.command == "set-anchor":
        anchor = set_anchor(args.ledger, getattr(args, "round_id", None))
        print(
            f"Anchor set to round {anchor['round_id']}: "
            f"nDCG@10={anchor['global']['ndcg@10']:.4f}, "
            f"MRR@10={anchor['global']['mrr@10']:.4f}"
        )
    elif args.command == "summary":
        summary(args.ledger)
    elif args.command == "promote":
        promote(args.cases_file, args.case_id, args.ledger)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
