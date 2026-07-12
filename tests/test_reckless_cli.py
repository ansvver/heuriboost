from __future__ import annotations

from contextlib import redirect_stderr
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from heuriboost_rag.backends.legacy_runtime import resolve_legacy_runtime
from heuriboost_rag.backends.xgboost_rag import CompileSettings, TrainingSettings
from heuriboost_rag.reckless.errors import PromotionConflictError
from heuriboost_rag.reckless.policy import RecklessPolicy


class LocalWorkspaceAdapterTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        base = root / "base.csv"
        base.write_text(
            "domain,query,text,relevance,split\n"
            "tax,question one,correct answer,good,train\n"
            "tax,question one,wrong answer,bad,train\n",
            encoding="utf-8",
        )
        cases = root / "cases.csv"
        cases.write_text(
            "domain,case_id,query,shown_doc_text,user_verdict,rank\n"
            "tax,case-1,question one,correct answer,good,2\n",
            encoding="utf-8",
        )
        gates = root / "gates.jsonl"
        gates.write_text(
            json.dumps(
                {
                    "gate_id": "gate-1",
                    "source_case_id": "case-1",
                    "domain": "tax",
                    "query": "question one",
                    "top_k": 3,
                    "acceptance_level": "full",
                    "source_run_id": "anchor-1",
                    "promoted_at": "2026-07-10T00:00:00Z",
                    "candidates": [],
                    "good_doc_ids": ["doc-1"],
                    "bad_doc_ids": [],
                    "context_doc_ids": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        anchor = root / "anchor.json"
        anchor.write_text(
            json.dumps(
                {
                    "anchor": {
                        "round_id": "anchor-1",
                        "global": {"ndcg@10": 0.1, "mrr@10": 0.1},
                        "domains": {
                            "tax": {"ndcg@10": 0.1, "mrr@10": 0.1}
                        },
                        "set_by": "test",
                    },
                    "rounds": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return base, cases, gates, anchor

    def _bootstrap(self, root: Path):
        from heuriboost_rag.adapters.workspace import bootstrap_local_workspace

        _, _, gates, anchor = self._write_inputs(root)
        runtime = resolve_legacy_runtime()
        return bootstrap_local_workspace(
            root / "output",
            workspace_id="cli-fixture",
            policy=RecklessPolicy.default(),
            feature_recipes=runtime.feature_recipe_path,
            historical_gates=gates,
            anchor_ledger=anchor,
            compile_settings=CompileSettings(
                min_global_test_queries=10,
                min_domain_test_queries=3,
            ),
            training_settings=TrainingSettings(rounds=2),
            code_revision="test-revision",
        )

    def test_registration_is_content_addressed_and_request_keeps_model_settings_out_of_run_options(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._bootstrap(root)
            base_path, cases_path, _, _ = self._write_inputs(root)

            base = workspace.register_local_dataset("base", base_path)
            same_base = workspace.register_local_dataset("base", base_path)
            cases = workspace.register_local_dataset("production_cases", cases_path)
            request = workspace.create_repair_request(
                base,
                cases,
                requested_by="tester",
            )

            self.assertEqual(base, same_base)
            self.assertTrue(base.dataset_id.startswith("base-"))
            self.assertNotEqual(base.schema_hash, base.content_hash)
            self.assertEqual(request.workspace_id, "cli-fixture")
            self.assertEqual(request.backend_name, "xgboost-rag")
            self.assertEqual(dict(request.run_options), {})
            self.assertEqual(
                workspace.backend.config.compile_settings.min_global_test_queries,
                10,
            )
            self.assertEqual(
                workspace.backend.config.compile_settings.min_domain_test_queries,
                3,
            )
            self.assertEqual(workspace.backend.config.training_settings.rounds, 2)

    def test_open_rehydrates_the_frozen_backend_configuration(self) -> None:
        from heuriboost_rag.adapters.workspace import open_local_workspace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = self._bootstrap(root)
            reopened = open_local_workspace(root / "output")

            self.assertEqual(reopened.workspace_id, created.workspace_id)
            self.assertEqual(
                reopened.policy.content_hash,
                created.policy.content_hash,
            )
            self.assertEqual(
                reopened.backend.execution_identity(),
                created.backend.execution_identity(),
            )

    def test_workspace_rejects_promotion_until_legacy_state_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._bootstrap(root)
            legacy = root / "output" / ".heuriboost"
            legacy.mkdir()
            (legacy / "ledger.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(PromotionConflictError, "migrat"):
                workspace.assert_promotion_allowed()

    def test_workspace_allows_promotion_after_legacy_state_has_a_durable_migration(self) -> None:
        from heuriboost_rag.reckless.migration import migrate_legacy_state

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._bootstrap(root)
            legacy = root / "output" / ".heuriboost"
            legacy.mkdir()
            model_dir = root / "output" / "models"
            model_dir.mkdir()
            model = model_dir / "reranker.json"
            model.write_text("legacy model", encoding="utf-8")
            metadata = model_dir / "reranker_metadata.json"
            metadata.write_text("{}", encoding="utf-8")
            (legacy / "ledger.json").write_text("{}", encoding="utf-8")
            (legacy / "gates.jsonl").write_text("{}\n", encoding="utf-8")
            (legacy / "promoted_repair_samples.csv").write_text(
                "query_id,doc_id,label\nlegacy,good,3\n",
                encoding="utf-8",
            )
            (legacy / "current_model.json").write_text(
                json.dumps(
                    {
                        "run_id": "legacy-run",
                        "model_path": str(model),
                        "metadata_path": str(metadata),
                    }
                ),
                encoding="utf-8",
            )

            migrate_legacy_state(legacy, workspace.promotion_stores.releases)
            workspace.assert_promotion_allowed()

    def test_legacy_compile_settings_are_reflected_in_the_effective_policy(self) -> None:
        from heuriboost_rag.adapters.workspace import policy_for_compile_settings

        effective = policy_for_compile_settings(
            RecklessPolicy.default(),
            CompileSettings(
                acceptance_level="weak",
                min_global_test_queries=10,
                min_domain_test_queries=3,
                min_docs_per_query=2,
            ),
        )

        self.assertEqual(effective.acceptance_level, "weak")
        self.assertEqual(effective.input.min_global_test_queries, 10)
        self.assertEqual(effective.input.min_domain_test_queries, 3)
        self.assertFalse(effective.promotion.allow_weak)
        self.assertNotEqual(effective.content_hash, RecklessPolicy.default().content_hash)

    def test_default_promotion_idempotency_key_is_stable_per_run(self) -> None:
        from heuriboost_rag.adapters.workspace import default_promotion_idempotency_key

        self.assertEqual(
            default_promotion_idempotency_key("run-123"),
            default_promotion_idempotency_key("run-123"),
        )
        self.assertNotEqual(
            default_promotion_idempotency_key("run-123"),
            default_promotion_idempotency_key("run-456"),
        )


class CliCompatibilityTests(unittest.TestCase):
    _scripts = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "heuriboost"
        / "skills"
        / "heuriboost-rag"
        / "scripts"
    )

    def _load_script(self, name: str):
        path = self._scripts / name
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_passing_cli_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        production_cases = root / "passing_cases.csv"
        production_cases.write_text(
            "domain,case_id,query,shown_doc_text,user_verdict,rank\n"
            "tax,prod-home-office,How do I deduct home office expenses?,Home office deduction rules for a sole proprietor require exclusive business use.,good,1\n"
            "tax,prod-home-office,How do I deduct home office expenses?,Corporate office lease accounting rules for a company headquarters.,bad,99\n",
            encoding="utf-8",
        )
        gates = root / "gates.jsonl"
        gates.write_text(
            json.dumps(
                {
                    "case_id": "historical-home-office",
                    "domain": "tax",
                    "query": "How do I deduct home office expenses?",
                    "top_k": 1,
                    "candidates": [
                        {
                            "doc_id": "good-doc",
                            "text": "Home office deduction rules for a sole proprietor require exclusive business use.",
                            "role": "good",
                        },
                        {
                            "doc_id": "context-1",
                            "text": "A credit card interest charge is not a business expense.",
                            "role": "context",
                        },
                        {
                            "doc_id": "context-2",
                            "text": "Bond duration measures interest rate sensitivity.",
                            "role": "context",
                        },
                        {
                            "doc_id": "context-3",
                            "text": "Insurance premiums are paid to maintain coverage.",
                            "role": "context",
                        },
                    ],
                    "good_doc_ids": ["good-doc"],
                    "bad_doc_ids": [],
                    "context_doc_ids": ["context-1", "context-2", "context-3"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        anchor = root / "anchor.json"
        anchor.write_text(
            json.dumps(
                {
                    "anchor": {
                        "round_id": "bootstrap-anchor",
                        "global": {"ndcg@10": 0.0, "mrr@10": 0.0},
                        "domains": {"tax": {"ndcg@10": 0.0, "mrr@10": 0.0}},
                        "set_by": "test",
                    },
                    "rounds": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return production_cases, gates, anchor

    def test_repair_reranker_parser_preserves_legacy_defaults(self) -> None:
        module = self._load_script("repair_reranker.py")
        args = module.build_parser().parse_args(
            [
                "--base-dataset",
                "base.csv",
                "--production-cases",
                "cases.csv",
                "--reckless",
            ]
        )

        self.assertEqual(args.output_dir, "heuriboost_output")
        self.assertEqual(args.acceptance_level, "full")
        self.assertEqual(args.case_top_k, 3)
        self.assertEqual(args.rounds, 40)
        self.assertEqual(args.split_ratio, (0.7, 0.15, 0.15))
        self.assertEqual(args.split_seed, 42)
        self.assertEqual(args.min_global_test_queries, 10)
        self.assertEqual(args.min_domain_test_queries, 3)
        self.assertEqual(args.min_docs_per_query, 2)

    def test_legacy_mutable_switches_are_rejected_before_input_processing(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(self._scripts / "repair_reranker.py"),
                "--base-dataset",
                "missing-base.csv",
                "--production-cases",
                "missing-cases.csv",
                "--reckless",
                "--reset-anchor",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reset-anchor", result.stderr.lower())

    def test_legacy_repair_uses_existing_promoted_samples_by_default(self) -> None:
        from heuriboost_rag.adapters import cli

        with mock.patch.object(cli, "_run_workspace_repair", return_value=0) as run:
            result = cli.legacy_repair_main(
                [
                    "--base-dataset",
                    "base.csv",
                    "--production-cases",
                    "cases.csv",
                    "--reckless",
                ]
            )

        self.assertEqual(result, 0)
        self.assertTrue(run.call_args.args[0].use_legacy_promoted_samples)

    def test_cli_prints_blocked_error_summary_and_operator_action(self) -> None:
        from heuriboost_rag.adapters import cli

        workspace = mock.Mock()
        workspace.register_local_dataset.side_effect = [mock.sentinel.base, mock.sentinel.cases]
        workspace.create_repair_request.return_value = mock.sentinel.request
        workspace.run.return_value = SimpleNamespace(
            run_id="run-blocked",
            state="BLOCKED_INPUT",
            metadata={},
            error={
                "code": "BLOCKED_INPUT",
                "message": "dataset labels are incomplete",
                "operator_action": "Provide verified labels and create a new run.",
            },
        )
        args = SimpleNamespace(
            base_dataset="base.csv",
            production_cases="cases.csv",
            requested_by="tester",
        )
        stderr = io.StringIO()

        with mock.patch.object(cli, "_bootstrap_or_open_workspace", return_value=workspace), redirect_stderr(stderr):
            result = cli._run_workspace_repair(args)

        self.assertEqual(result, 2)
        self.assertIn("BLOCKED_INPUT: dataset labels are incomplete", stderr.getvalue())
        self.assertIn("Provide verified labels", stderr.getvalue())

    def test_autopilot_exposes_the_run_resume_report_and_promote_commands(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(self._scripts / "reckless_autopilot.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("run", "resume", "report", "promote"):
            self.assertIn(command, result.stdout)

    def test_autopilot_run_and_legacy_promote_delegate_to_immutable_package_flow(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        base_dataset = repo / "examples" / "fiqa" / "repair" / "base_dataset_minimal.csv"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            production_cases, gates, anchor = self._write_passing_cli_inputs(root)
            output_dir = root / "output"
            run = subprocess.run(
                [
                    sys.executable,
                    str(self._scripts / "reckless_autopilot.py"),
                    "run",
                    "--base-dataset",
                    str(base_dataset),
                    "--production-cases",
                    str(production_cases),
                    "--output-dir",
                    str(output_dir),
                    "--historical-gates",
                    str(gates),
                    "--anchor-ledger",
                    str(anchor),
                    "--workspace-id",
                    "cli-integration",
                    "--rounds",
                    "2",
                    "--case-top-k",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("Run state: READY_FOR_PROMOTION", run.stdout)
            self.assertTrue(
                (output_dir / ".reckless" / "workspace.json").is_file()
            )

            promote = subprocess.run(
                [
                    sys.executable,
                    str(self._scripts / "promote_repair.py"),
                    "--output-dir",
                    str(output_dir),
                    "--approved-by",
                    "cli-tester",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo,
            )
            self.assertEqual(promote.returncode, 0, promote.stdout + promote.stderr)
            self.assertIn("Promoted repair run:", promote.stdout)
            self.assertTrue(
                (output_dir / ".reckless" / "current_model.json").is_file()
            )

            drifted = subprocess.run(
                [
                    sys.executable,
                    str(self._scripts / "reckless_autopilot.py"),
                    "run",
                    "--base-dataset",
                    str(base_dataset),
                    "--production-cases",
                    str(production_cases),
                    "--output-dir",
                    str(output_dir),
                    "--historical-gates",
                    str(gates),
                    "--anchor-ledger",
                    str(anchor),
                    "--workspace-id",
                    "cli-integration",
                    "--rounds",
                    "3",
                    "--case-top-k",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo,
            )
            self.assertNotEqual(drifted.returncode, 0)
            self.assertIn("immutable workspace configuration", drifted.stderr)

    def test_weak_acceptance_is_blocked_and_cannot_promote(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        base_dataset = repo / "examples" / "fiqa" / "repair" / "base_dataset_minimal.csv"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            production_cases, gates, anchor = self._write_passing_cli_inputs(root)
            output_dir = root / "output"
            run = subprocess.run(
                [
                    sys.executable,
                    str(self._scripts / "reckless_autopilot.py"),
                    "run",
                    "--base-dataset",
                    str(base_dataset),
                    "--production-cases",
                    str(production_cases),
                    "--output-dir",
                    str(output_dir),
                    "--historical-gates",
                    str(gates),
                    "--anchor-ledger",
                    str(anchor),
                    "--workspace-id",
                    "weak-cli-integration",
                    "--rounds",
                    "2",
                    "--case-top-k",
                    "1",
                    "--acceptance-level",
                    "weak",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo,
            )
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("Run state: BLOCKED_NOT_ELIGIBLE", run.stdout)

            promote = subprocess.run(
                [
                    sys.executable,
                    str(self._scripts / "promote_repair.py"),
                    "--output-dir",
                    str(output_dir),
                    "--approved-by",
                    "cli-tester",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo,
            )
            self.assertNotEqual(promote.returncode, 0, promote.stdout + promote.stderr)
            self.assertFalse(
                (output_dir / ".reckless" / "current_model.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
