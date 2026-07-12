from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from heuriboost_rag.reckless.contracts import (
    ActivationResult,
    ArtifactRef,
    PreparedActivation,
    PromotionApproval,
    ReleaseSnapshot,
    RepairRequest,
    RunRecord,
    TargetValidation,
)
from heuriboost_rag.reckless.errors import ArtifactIntegrityError
from heuriboost_rag.reckless.hashing import sha256_file
from heuriboost_rag.reckless.migration import migrate_legacy_state
from heuriboost_rag.reckless.release_store import FileReleaseStore


class LegacyStateMigrationTests(unittest.TestCase):
    def _legacy_state(self, root: Path) -> Path:
        legacy = root / ".heuriboost"
        legacy.mkdir()
        model_dir = root / "models"
        model_dir.mkdir()
        model = model_dir / "reranker.json"
        model.write_text('{"model":"legacy"}\n', encoding="utf-8")
        metadata = model_dir / "reranker_metadata.json"
        metadata.write_text('{"schema":"legacy"}\n', encoding="utf-8")
        (legacy / "ledger.json").write_text(
            json.dumps(
                {
                    "anchor": {
                        "round_id": "legacy-run",
                        "global": {"ndcg@10": 0.7, "mrr@10": 0.6},
                        "domains": {
                            "tax": {"ndcg@10": 0.7, "mrr@10": 0.6}
                        },
                        "set_by": "legacy",
                    },
                    "rounds": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (legacy / "gates.jsonl").write_text(
            json.dumps(
                {
                    "gate_id": "legacy-gate",
                    "query": "What is a wash sale?",
                    "good_doc_ids": ["good"],
                    "bad_doc_ids": [],
                    "candidates": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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
                    "promoted_at": "2026-07-10T00:00:00Z",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return legacy

    def test_migration_creates_one_immutable_bootstrap_release_and_preserves_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self._legacy_state(root)
            original = {
                path.name: path.read_bytes()
                for path in legacy.iterdir()
                if path.is_file()
            }
            store = FileReleaseStore(root / ".reckless")

            receipt = migrate_legacy_state(legacy, store)
            release_manifest = json.loads(
                (receipt.release_path / "release_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            artifact_hashes = {
                artifact["artifact_type"]: artifact["content_hash"]
                for artifact in release_manifest["artifacts"]
            }

            self.assertTrue(receipt.release_path.is_dir())
            self.assertTrue(store.legacy_migration_complete())
            self.assertEqual(store.read_current_model(), receipt.current_model)
            self.assertEqual(artifact_hashes["legacy-ledger"], sha256_file(legacy / "ledger.json"))
            self.assertEqual(artifact_hashes["legacy-gates"], sha256_file(legacy / "gates.jsonl"))
            self.assertEqual(
                artifact_hashes["legacy-promoted-samples"],
                sha256_file(legacy / "promoted_repair_samples.csv"),
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in legacy.iterdir() if path.is_file()},
                original,
            )
            self.assertEqual(migrate_legacy_state(legacy, store), receipt)

    def test_migration_refuses_changed_legacy_state_after_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self._legacy_state(root)
            store = FileReleaseStore(root / ".reckless")
            migrate_legacy_state(legacy, store)
            (legacy / "ledger.json").write_text(
                '{"anchor": null, "rounds": ["changed"]}',
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactIntegrityError):
                migrate_legacy_state(legacy, store)

    def test_migration_remains_complete_after_a_later_package_release_becomes_current(self) -> None:
        class LocalTarget:
            name = "test"

            def __init__(self, current: str | None) -> None:
                self.current = current

            def validate_target(self, expected_current: str | None) -> TargetValidation:
                return TargetValidation(
                    valid=self.current == expected_current,
                    current_model=self.current,
                    errors=(),
                )

            def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation:
                return PreparedActivation(
                    run_id=release.run_id,
                    pointer_payload={},
                    metadata={},
                )

            def activate(self, prepared: PreparedActivation) -> ActivationResult:
                self.current = prepared.run_id
                return ActivationResult(current_model=prepared.run_id, metadata={})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self._legacy_state(root)
            store = FileReleaseStore(root / ".reckless")
            bootstrap = migrate_legacy_state(legacy, store)
            model = root / "new-model.json"
            model.write_text('{"model":"new"}\n', encoding="utf-8")
            artifact = ArtifactRef(
                artifact_type="model",
                path=model,
                content_hash=sha256_file(model),
                size_bytes=model.stat().st_size,
            )
            run = RunRecord(
                run_id="run-new",
                state="READY_FOR_PROMOTION",
                version=1,
                request=RepairRequest(
                    workspace_id="workspace",
                    base_dataset_id="base",
                    production_cases_id="cases",
                    policy_version="1",
                    backend_name="test",
                    requested_by="tester",
                ),
                policy_hash="policy-hash",
                input_hash="input-hash",
            )
            approval = PromotionApproval(
                run_id=run.run_id,
                approved_by="tester",
                approved_at="2026-07-10T00:00:00+00:00",
                report_hash="report-hash",
                decision_hash="decision-hash",
                expected_current_model=bootstrap.current_model,
                idempotency_key="new-release",
            )
            receipt = store.promote(
                run,
                approval,
                LocalTarget(bootstrap.current_model),
                ReleaseSnapshot(
                    run_id=run.run_id,
                    artifacts=(artifact,),
                    manifest_hash="snapshot-hash",
                    previous_model=bootstrap.current_model,
                ),
                report_hash="report-hash",
                decision_hash="decision-hash",
                model_hash=artifact.content_hash,
                schema_hash=artifact.content_hash,
            )

            self.assertEqual(store.read_current_model(), receipt.current_model)
            self.assertTrue(store.legacy_migration_complete())

    def test_migration_cli_imports_an_output_directory_without_mutating_legacy_files(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = (
            repo
            / "plugins"
            / "heuriboost"
            / "skills"
            / "heuriboost-rag"
            / "scripts"
            / "migrate_reckless_state.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self._legacy_state(root)
            before = (legacy / "ledger.json").read_bytes()

            result = subprocess.run(
                [sys.executable, str(script), "--output-dir", str(root)],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Migrated legacy state:", result.stdout)
            self.assertTrue((root / ".reckless" / "current_model.json").is_file())
            self.assertEqual((legacy / "ledger.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
