"""Package-owned compatibility CLI implementation for Reckless workflows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import getpass
from pathlib import Path
import sys
from typing import Callable, Sequence

from ..backends.legacy_runtime import resolve_legacy_runtime
from ..backends.xgboost_rag import CompileSettings, TrainingSettings
from ..reckless.errors import HeuriBoostError
from ..reckless.hashing import sha256_file
from ..reckless.policy import RecklessPolicy, load_policy
from .workspace import (
    LocalWorkspace,
    bootstrap_local_workspace,
    default_promotion_idempotency_key,
    open_local_workspace,
    policy_for_compile_settings,
)


LEGACY_CASE_TOP_K = 3
LEGACY_SPLIT_RATIOS = (0.7, 0.15, 0.15)
LEGACY_SPLIT_SEED = 42
LEGACY_MIN_GLOBAL_TEST_QUERIES = 10
LEGACY_MIN_DOMAIN_TEST_QUERIES = 3
LEGACY_MIN_DOCS_PER_QUERY = 2


def parse_split_ratios(value: str) -> tuple[float, float, float]:
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


def _effective_policy(path: str | None, settings: CompileSettings) -> RecklessPolicy:
    source = load_policy(Path(path)) if path else RecklessPolicy.default()
    return policy_for_compile_settings(source, settings)


def _compile_settings(args: argparse.Namespace) -> CompileSettings:
    return CompileSettings(
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


def _workspace_config_path(output_dir: Path) -> Path:
    return Path(output_dir).expanduser().resolve() / ".reckless" / "workspace.json"


def _configuration_mismatch(label: str) -> None:
    raise ValueError(
        "immutable workspace configuration differs: "
        f"{label}; use a new output directory for changed repair settings"
    )


def _assert_pinned_source_matches(
    requested_path: Path,
    pinned: object,
    label: str,
) -> None:
    if pinned is None:
        _configuration_mismatch(label)
    configured_path = getattr(pinned, "path", None)
    configured_hash = getattr(pinned, "content_hash", None)
    if not isinstance(configured_path, Path) or not isinstance(configured_hash, str):
        _configuration_mismatch(label)
    try:
        source = requested_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot inspect requested {label}: {requested_path}") from exc
    if source != configured_path.resolve() or sha256_file(source) != configured_hash:
        _configuration_mismatch(label)


def _assert_existing_workspace_matches(
    workspace: LocalWorkspace,
    args: argparse.Namespace,
) -> None:
    config = workspace.backend.config
    if config is None:  # pragma: no cover - local workspace construction guarantees this.
        raise ValueError("existing local workspace has no immutable backend configuration")
    settings = _compile_settings(args)
    if workspace.workspace_id != args.workspace_id:
        _configuration_mismatch("workspace ID")
    if workspace.policy.content_hash != _effective_policy(args.policy, settings).content_hash:
        _configuration_mismatch("policy")
    if config.compile_settings != settings:
        _configuration_mismatch("compile settings")
    if config.training_settings != TrainingSettings(rounds=args.rounds):
        _configuration_mismatch("training settings")

    runtime = resolve_legacy_runtime()
    expected_revision = args.code_revision or f"legacy-{runtime.code_manifest_hash[:12]}"
    if config.code_revision != expected_revision:
        _configuration_mismatch("code revision")
    recipe_path = Path(args.feature_recipes) if args.feature_recipes else runtime.feature_recipe_path
    _assert_pinned_source_matches(recipe_path, config.feature_recipes, "feature recipes")
    if args.historical_gates:
        _assert_pinned_source_matches(
            Path(args.historical_gates),
            config.historical_gates,
            "historical gates",
        )
    if args.anchor_ledger:
        _assert_pinned_source_matches(
            Path(args.anchor_ledger),
            config.anchor_ledger,
            "anchor ledger",
        )

    expected_promoted = Path(args.promoted_samples) if args.promoted_samples else None
    expected_include = bool(args.include_promoted_samples)
    if args.use_legacy_promoted_samples:
        legacy_samples = Path(args.output_dir) / ".heuriboost" / "promoted_repair_samples.csv"
        if legacy_samples.is_file():
            expected_promoted = legacy_samples
            expected_include = True
    if config.include_promoted_samples != expected_include:
        _configuration_mismatch("promoted sample inclusion")
    if expected_include:
        if expected_promoted is None:
            _configuration_mismatch("promoted samples")
        _assert_pinned_source_matches(
            expected_promoted,
            config.promoted_samples,
            "promoted samples",
        )


def _bootstrap_or_open_workspace(args: argparse.Namespace) -> LocalWorkspace:
    output_dir = Path(args.output_dir)
    if _workspace_config_path(output_dir).is_file():
        workspace = open_local_workspace(output_dir)
        _assert_existing_workspace_matches(workspace, args)
        return workspace

    settings = _compile_settings(args)
    runtime = resolve_legacy_runtime()
    legacy_dir = output_dir / ".heuriboost"
    historical_gates = Path(args.historical_gates) if args.historical_gates else legacy_dir / "gates.jsonl"
    anchor_ledger = Path(args.anchor_ledger) if args.anchor_ledger else legacy_dir / "ledger.json"
    if not historical_gates.is_file() or not anchor_ledger.is_file():
        raise ValueError(
            "new local workspace requires --historical-gates and --anchor-ledger "
            "(or verified files under output/.heuriboost)"
        )
    feature_recipes = (
        Path(args.feature_recipes)
        if args.feature_recipes
        else runtime.feature_recipe_path
    )
    promoted_samples = Path(args.promoted_samples) if args.promoted_samples else None
    include_promoted_samples = bool(args.include_promoted_samples)
    if promoted_samples is None and args.use_legacy_promoted_samples:
        legacy_samples = legacy_dir / "promoted_repair_samples.csv"
        if legacy_samples.is_file():
            promoted_samples = legacy_samples
            include_promoted_samples = True
    return bootstrap_local_workspace(
        output_dir,
        workspace_id=args.workspace_id,
        policy=_effective_policy(args.policy, settings),
        feature_recipes=feature_recipes,
        historical_gates=historical_gates,
        anchor_ledger=anchor_ledger,
        compile_settings=settings,
        training_settings=TrainingSettings(rounds=args.rounds),
        code_revision=args.code_revision or f"legacy-{runtime.code_manifest_hash[:12]}",
        promoted_samples=promoted_samples,
        include_promoted_samples=include_promoted_samples,
    )


def _run_workspace_repair(args: argparse.Namespace) -> int:
    workspace = _bootstrap_or_open_workspace(args)
    base = workspace.register_local_dataset("base", Path(args.base_dataset))
    cases = workspace.register_local_dataset("production_cases", Path(args.production_cases))
    run = workspace.run(
        workspace.create_repair_request(
            base,
            cases,
            requested_by=args.requested_by,
        )
    )
    print(f"Run id: {run.run_id}")
    print(f"Run state: {run.state}")
    if "report_evidence" in run.metadata:
        report = workspace.render_report(run.run_id)
        print(f"Saved repair report: {report.html_path}")
    print(f"Promotion eligible: {run.state == 'READY_FOR_PROMOTION'}")
    if run.state.startswith("BLOCKED_") and isinstance(run.error, Mapping):
        code = run.error.get("code")
        message = run.error.get("message")
        operator_action = run.error.get("operator_action")
        if isinstance(code, str) and isinstance(message, str):
            print(f"{code}: {message}", file=sys.stderr)
        if isinstance(operator_action, str) and operator_action:
            print(operator_action, file=sys.stderr)
    return 0 if run.state == "READY_FOR_PROMOTION" else 2


def _resume_workspace_repair(args: argparse.Namespace) -> int:
    workspace = open_local_workspace(Path(args.output_dir))
    run = workspace.resume(args.run_id)
    print(f"Run id: {run.run_id}")
    print(f"Run state: {run.state}")
    return 0 if run.state == "READY_FOR_PROMOTION" else 2


def _render_workspace_report(args: argparse.Namespace) -> int:
    workspace = open_local_workspace(Path(args.output_dir))
    report = workspace.render_report(args.run_id, locale=args.locale)
    print(f"Saved repair report: {report.html_path}")
    return 0


def _promote_workspace_run(args: argparse.Namespace) -> int:
    workspace = open_local_workspace(Path(args.output_dir))
    run_id = args.run_id or workspace.find_single_promotable_run()
    idempotency_key = args.idempotency_key or default_promotion_idempotency_key(run_id)
    receipt = workspace.promote(
        run_id,
        approved_by=args.approved_by,
        idempotency_key=idempotency_key,
    )
    print(f"Promoted repair run: {receipt.run_id}")
    print(f"Current model: {receipt.current_model}")
    print(f"Promotion receipt: {receipt.receipt_html_path}")
    print(f"Promotion idempotency key: {idempotency_key}")
    return 0


def _add_workspace_bootstrap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-id", default="local-reckless")
    parser.add_argument("--policy", help="Versioned Reckless policy YAML")
    parser.add_argument("--historical-gates", help="Immutable historical gates JSONL")
    parser.add_argument("--anchor-ledger", help="Immutable anchor ledger JSON")
    parser.add_argument("--feature-recipes", help="Trusted feature recipe YAML")
    parser.add_argument("--promoted-samples", help="Optional promoted samples CSV")
    parser.add_argument("--include-promoted-samples", action="store_true")
    parser.add_argument("--use-legacy-promoted-samples", action="store_true")
    parser.add_argument("--code-revision", help="Pinned HeuriBoost source revision")


def _add_compile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--acceptance-level", choices=["full", "weak"], default="full")
    parser.add_argument("--case-top-k", type=int, default=LEGACY_CASE_TOP_K)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--resplit", action="store_true")
    parser.add_argument("--split-ratio", type=parse_split_ratios, default=LEGACY_SPLIT_RATIOS)
    parser.add_argument("--split-seed", type=int, default=LEGACY_SPLIT_SEED)
    parser.add_argument(
        "--min-global-test-queries",
        type=int,
        default=LEGACY_MIN_GLOBAL_TEST_QUERIES,
    )
    parser.add_argument(
        "--min-domain-test-queries",
        type=int,
        default=LEGACY_MIN_DOMAIN_TEST_QUERIES,
    )
    parser.add_argument("--min-docs-per-query", type=int, default=LEGACY_MIN_DOCS_PER_QUERY)


def build_legacy_repair_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict production-case repair for a HeuriBoost reranker.")
    parser.add_argument("--base-dataset", required=True, help="User-facing base dataset CSV/JSONL")
    parser.add_argument("--production-cases", required=True, help="User-facing production cases CSV/JSONL")
    parser.add_argument("--output-dir", default="heuriboost_output", help="Output directory")
    parser.add_argument("--reckless", action="store_true", help="Required flag for strict production repair")
    _add_compile_arguments(parser)
    parser.add_argument("--reset-anchor", action="store_true")
    parser.add_argument("--keep-baseline-artifacts", action="store_true")
    _add_workspace_bootstrap_arguments(parser)
    parser.add_argument("--requested-by", default=getpass.getuser())
    return parser


def _print_cli_error(exc: BaseException) -> int:
    if isinstance(exc, HeuriBoostError):
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        if exc.operator_action:
            print(exc.operator_action, file=sys.stderr)
    else:
        print(str(exc), file=sys.stderr)
    return 2


def legacy_repair_main(argv: Sequence[str] | None = None) -> int:
    args = build_legacy_repair_parser().parse_args(argv)
    if not args.reckless:
        print("repair_reranker.py requires --reckless.", file=sys.stderr)
        return 2
    if args.reset_anchor:
        print("--reset-anchor is not supported by immutable Reckless releases.", file=sys.stderr)
        return 2
    if args.keep_baseline_artifacts:
        print("--keep-baseline-artifacts is not supported by immutable Reckless releases.", file=sys.stderr)
        return 2
    args.use_legacy_promoted_samples = True
    try:
        return _run_workspace_repair(args)
    except (HeuriBoostError, OSError, ValueError) as exc:
        return _print_cli_error(exc)


def build_legacy_promote_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote an eligible production-case repair run.")
    parser.add_argument("--output-dir", default="heuriboost_output", help="Repair output directory")
    parser.add_argument("--run-id", help="Ready run ID; inferred only when exactly one is ready")
    parser.add_argument("--approved-by", default=getpass.getuser())
    parser.add_argument("--idempotency-key")
    return parser


def legacy_promote_main(argv: Sequence[str] | None = None) -> int:
    args = build_legacy_promote_parser().parse_args(argv)
    try:
        return _promote_workspace_run(args)
    except (HeuriBoostError, OSError, ValueError) as exc:
        return _print_cli_error(exc)


def build_autopilot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and promote immutable HeuriBoost Reckless repairs.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a new Reckless repair")
    run.add_argument("--base-dataset", required=True)
    run.add_argument("--production-cases", required=True)
    run.add_argument("--output-dir", default="heuriboost_output")
    run.add_argument("--requested-by", default=getpass.getuser())
    _add_compile_arguments(run)
    _add_workspace_bootstrap_arguments(run)
    run.set_defaults(handler=_run_workspace_repair)

    resume = commands.add_parser("resume", help="Resume an interrupted repair")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--output-dir", default="heuriboost_output")
    resume.set_defaults(handler=_resume_workspace_repair)

    report = commands.add_parser("report", help="Render an immutable Pre Promote report")
    report.add_argument("--run-id", required=True)
    report.add_argument("--output-dir", default="heuriboost_output")
    report.add_argument("--locale", choices=["zh-CN", "en"], default="zh-CN")
    report.set_defaults(handler=_render_workspace_report)

    promote = commands.add_parser("promote", help="Approve and promote a ready repair")
    promote.add_argument("--run-id")
    promote.add_argument("--output-dir", default="heuriboost_output")
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--idempotency-key")
    promote.set_defaults(handler=_promote_workspace_run)
    return parser


def autopilot_main(argv: Sequence[str] | None = None) -> int:
    args = build_autopilot_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (HeuriBoostError, OSError, ValueError) as exc:
        return _print_cli_error(exc)


__all__ = [
    "autopilot_main",
    "build_autopilot_parser",
    "build_legacy_promote_parser",
    "build_legacy_repair_parser",
    "legacy_promote_main",
    "legacy_repair_main",
    "parse_split_ratios",
]
