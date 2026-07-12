from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from heuriboost_rag.reckless.contracts import (
    ActivationResult,
    PreparedActivation,
    PromotionApproval,
    ReleaseSnapshot,
    RepairRequest,
    TargetValidation,
)
from heuriboost_rag.reckless.errors import (
    ArtifactIntegrityError,
    PromotionConflictError,
)
from heuriboost_rag.reckless.promotion import (
    JsonPromotionRepository,
    PromotionStores,
    assert_ready_and_unchanged,
    promote_repair,
    rollback_release,
)
from heuriboost_rag.reckless.release_store import FileReleaseStore
from heuriboost_rag.reckless.report import render_run_pre_promote_report
from heuriboost_rag.reckless.state import RunState
from heuriboost_rag.reckless.storage import JsonRunRepository, LocalArtifactStore


class RecordingTarget:
    name = "recording-target"

    def __init__(self, current_model: str | None, *, fail_activate: bool = False) -> None:
        self.current_model = current_model
        self.fail_activate = fail_activate
        self.activations = 0
        self.rollbacks = 0

    def validate_target(self, expected_current: str | None) -> TargetValidation:
        valid = self.current_model == expected_current
        return TargetValidation(
            valid=valid,
            current_model=self.current_model,
            errors=() if valid else ("current model changed",),
        )

    def prepare_release(self, release):
        return PreparedActivation(
            run_id=release.run_id,
            pointer_payload={"target": self.name},
            metadata={"prepared": True},
        )

    def activate(self, prepared: PreparedActivation) -> ActivationResult:
        if self.fail_activate:
            raise RuntimeError("activation failed")
        self.activations += 1
        self.current_model = prepared.run_id
        return ActivationResult(
            current_model=prepared.run_id,
            metadata={"activated": True},
        )

    def rollback(self, receipt) -> ActivationResult:
        self.rollbacks += 1
        self.current_model = receipt.previous_model
        return ActivationResult(
            current_model=receipt.previous_model or "",
            metadata={"rolled_back": True},
        )


class PromotionFixture:
    def __init__(
        self,
        root: Path,
        *,
        current_model: str | None = "run-0",
        fail_before_pointer_swap: bool = False,
        final_training_stage: str = "TRAINED",
    ) -> None:
        self.root = root
        self.final_training_stage = final_training_stage
        self.artifacts = LocalArtifactStore(root)
        self.runs = JsonRunRepository(root)
        self.run = self._create_ready_run()
        self.release_store = FileReleaseStore(
            root,
            before_pointer_swap=(
                self._raise_before_pointer_swap if fail_before_pointer_swap else None
            ),
        )
        self.release_store.bootstrap_current_model(current_model)
        self.target = RecordingTarget(current_model)
        self.stores = PromotionStores(
            runs=self.runs,
            artifacts=self.artifacts,
            promotions=JsonPromotionRepository(root),
            releases=self.release_store,
        )
        report = render_run_pre_promote_report(self.run, self.artifacts)
        self.approval = PromotionApproval(
            run_id=self.run.run_id,
            approved_by="tester",
            approved_at="2026-07-10T00:00:00+00:00",
            report_hash=report.html_hash,
            decision_hash=str(report.manifest["decision_hash"]),
            expected_current_model=current_model,
            idempotency_key="approval-1",
        )

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def _raise_before_pointer_swap(self) -> None:
        raise RuntimeError("injected pointer swap failure")

    def _create_ready_run(self):
        request = RepairRequest(
            workspace_id="workspace",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
        )
        run = self.runs.create(request, "policy-hash", "input-hash")
        run = self.runs.transition(run.run_id, RunState.VALIDATING)
        run = self.runs.transition(run.run_id, RunState.COMPILED)
        run = self.runs.transition(run.run_id, RunState.TRAINING)
        run = self.runs.transition(run.run_id, RunState.TRAINED)
        run = self.runs.transition(run.run_id, RunState.EVALUATING)
        run = self.runs.transition(run.run_id, RunState.REPORTING)

        model = self.root / "candidate-model.json"
        schema = self.root / "candidate-schema.json"
        binding = self.root / "candidate-binding.json"
        model.write_text('{"model": true}', encoding="utf-8")
        schema.write_text('{"schema": true}', encoding="utf-8")
        binding.write_text('{"binding": true}', encoding="utf-8")
        trained = self.artifacts.complete_stage(
            run.run_id,
            "TRAINED",
            run.input_hash,
            {
                "xgboost-model": model,
                "xgboost-model-metadata": schema,
                "candidate-binding": binding,
            },
        )
        final_trained = trained
        if self.final_training_stage == "TRAINED_RETRY":
            retry_model = self.root / "candidate-model-retry.json"
            retry_schema = self.root / "candidate-schema-retry.json"
            retry_binding = self.root / "candidate-binding-retry.json"
            retry_model.write_text('{"model": "retry"}', encoding="utf-8")
            retry_schema.write_text('{"schema": "retry"}', encoding="utf-8")
            retry_binding.write_text('{"binding": "retry"}', encoding="utf-8")
            final_trained = self.artifacts.complete_stage(
                run.run_id,
                "TRAINED_RETRY",
                run.input_hash,
                {
                    "xgboost-model": retry_model,
                    "xgboost-model-metadata": retry_schema,
                    "candidate-binding": retry_binding,
                },
            )
        refs = {artifact.artifact_type: artifact for artifact in final_trained.artifacts}
        evidence = {
            "schema_version": 1,
            "run": {"run_id": run.run_id},
            "request": {
                "workspace_id": request.workspace_id,
                "base_dataset_id": request.base_dataset_id,
                "production_cases_id": request.production_cases_id,
                "policy_version": request.policy_version,
                "backend_name": request.backend_name,
                "requested_by": request.requested_by,
                "run_options": {},
            },
            "policy": {"version": 1, "content_hash": run.policy_hash},
            "input": {
                "input_hash": run.input_hash,
                "base_dataset_id": request.base_dataset_id,
                "production_cases_id": request.production_cases_id,
            },
            "outcome": {
                "state": RunState.READY_FOR_PROMOTION.value,
                "promotion_eligible": True,
                "acceptance_level": "full",
            },
            "datasets": {"base": {}, "production_cases": {}},
            "execution_identity": {
                "backend_version": "backend-v1",
                "feature_names": ["feature_a"],
                "feature_version": "features-v1",
                "code_commit": "revision",
                "training_params": {"rounds": 2},
                "random_seed": 7,
            },
            "validation": {"valid": True, "metadata": {}, "warnings": []},
            "compilation": {"metadata": {"train_rows": 4}},
            "training": {
                "metadata": {"rounds": 2},
                "model_ref": {
                    "artifact_type": "xgboost-model",
                    "content_hash": refs["xgboost-model"].content_hash,
                    "size_bytes": refs["xgboost-model"].size_bytes,
                },
            },
            "evaluation": {"global_metrics": {"ndcg@10": 1.0, "mrr@10": 1.0}},
            "decision": {
                "promotion_eligible": True,
                "acceptance_level": "full",
                "checks": [
                    {"check_id": "historical_gates", "passed": True},
                    {"check_id": "artifacts", "passed": True},
                ],
                "blockers": [],
                "warnings": [],
            },
            "warnings": [],
            "artifacts": [
                {
                    "stage": self.final_training_stage,
                    "artifact_type": artifact.artifact_type,
                    "path": artifact.path.as_posix(),
                    "content_hash": artifact.content_hash,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in final_trained.artifacts
            ],
            "completed_stage_manifests": [],
            "component_hashes": {"decision": "decision-hash"},
        }
        evidence_source = self.root / "report-evidence.json"
        evidence_source.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        reporting = self.artifacts.complete_stage(
            run.run_id,
            "REPORTING",
            run.input_hash,
            {"report-evidence": evidence_source},
        )
        evidence_ref = reporting.artifacts[0]
        return self.runs.transition(
            run.run_id,
            RunState.READY_FOR_PROMOTION,
            metadata={
                "report_evidence": {
                    "stage": "REPORTING",
                    "artifact_type": evidence_ref.artifact_type,
                    "path": evidence_ref.path.as_posix(),
                    "content_hash": evidence_ref.content_hash,
                    "size_bytes": evidence_ref.size_bytes,
                }
            },
        )

    def promote_concurrently(self, count: int):
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [
                executor.submit(
                    promote_repair,
                    self.run_id,
                    self.approval,
                    self.target,
                    self.stores,
                )
                for _ in range(count)
            ]
        return [future.result() for future in futures]


class RecklessPromotionTests(unittest.TestCase):
    def make_ready_fixture(
        self,
        *,
        current_model: str | None = "run-0",
        fail_before_pointer_swap: bool = False,
        final_training_stage: str = "TRAINED",
    ) -> PromotionFixture:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        return PromotionFixture(
            Path(self.temporary_directory.name),
            current_model=current_model,
            fail_before_pointer_swap=fail_before_pointer_swap,
            final_training_stage=final_training_stage,
        )

    def test_promotion_rejects_changed_report_hash(self):
        fixture = self.make_ready_fixture()
        approval = replace(fixture.approval, report_hash="changed")

        with self.assertRaises(ArtifactIntegrityError):
            promote_repair(fixture.run_id, approval, fixture.target, fixture.stores)

    def test_promotion_rejects_changed_current_model(self):
        fixture = self.make_ready_fixture(current_model="run-0")
        approval = replace(fixture.approval, expected_current_model="older-run")

        with self.assertRaises(PromotionConflictError):
            promote_repair(fixture.run_id, approval, fixture.target, fixture.stores)

    def test_duplicate_idempotency_key_returns_same_receipt(self):
        fixture = self.make_ready_fixture()

        first = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        second = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )

        self.assertEqual(first, second)
        self.assertEqual(fixture.target.activations, 1)
        self.assertEqual(fixture.release_store.read_current_model(), fixture.run_id)

    def test_release_manifest_references_published_artifacts_not_staging_paths(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        manifest = json.loads(
            (receipt.release_path / "release_manifest.json").read_text(encoding="utf-8")
        )

        for artifact in manifest["artifacts"]:
            path = Path(artifact["path"])
            self.assertTrue(path.is_relative_to(receipt.release_path))
            self.assertTrue(path.is_file())

    def test_promotion_uses_trained_retry_stage_when_report_points_to_retry_model(self):
        fixture = self.make_ready_fixture(final_training_stage="TRAINED_RETRY")

        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        manifest = json.loads(
            (receipt.release_path / "release_manifest.json").read_text(encoding="utf-8")
        )
        model_artifact = next(
            artifact
            for artifact in manifest["artifacts"]
            if artifact["artifact_type"] == "xgboost-model"
        )

        self.assertEqual(model_artifact["content_hash"], manifest["model_hash"])
        self.assertEqual(Path(model_artifact["path"]).read_text(encoding="utf-8"), '{"model": "retry"}')

    def test_release_receipt_remains_readable_without_the_idempotency_repository(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )

        stored = fixture.release_store.read_promotion_receipt(fixture.run_id)

        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored[0], receipt)
        self.assertEqual(stored[1], fixture.approval.idempotency_key)

    def test_release_receipt_rejects_a_tampered_release_manifest(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        manifest = receipt.release_path / "release_manifest.json"
        manifest.write_text("tampered", encoding="utf-8")

        with self.assertRaises(ArtifactIntegrityError):
            fixture.release_store.read_promotion_receipt(fixture.run_id)

    def test_idempotent_retry_rejects_a_tampered_durable_release(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        (receipt.release_path / "release_manifest.json").write_text(
            "tampered",
            encoding="utf-8",
        )

        with self.assertRaises(ArtifactIntegrityError):
            promote_repair(
                fixture.run_id,
                fixture.approval,
                fixture.target,
                fixture.stores,
            )

    def test_failure_before_pointer_swap_keeps_current_model(self):
        fixture = self.make_ready_fixture(fail_before_pointer_swap=True)

        with self.assertRaises(RuntimeError):
            promote_repair(
                fixture.run_id,
                fixture.approval,
                fixture.target,
                fixture.stores,
            )

        self.assertEqual(fixture.release_store.read_current_model(), "run-0")
        self.assertEqual(
            fixture.runs.get(fixture.run_id).state,
            RunState.PROMOTION_FAILED.value,
        )

    def test_failed_promotion_retries_the_already_activated_release(self):
        fixture = self.make_ready_fixture(fail_before_pointer_swap=True)
        with self.assertRaises(RuntimeError):
            promote_repair(
                fixture.run_id,
                fixture.approval,
                fixture.target,
                fixture.stores,
            )
        fixture.release_store._before_pointer_swap = None

        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )

        self.assertEqual(receipt.current_model, fixture.run_id)
        self.assertEqual(fixture.release_store.read_current_model(), fixture.run_id)
        self.assertEqual(fixture.target.activations, 1)
        self.assertEqual(
            fixture.runs.get(fixture.run_id).state,
            RunState.PROMOTED.value,
        )

    def test_durable_release_recovers_when_repository_write_was_interrupted(self):
        fixture = self.make_ready_fixture()
        evidence = assert_ready_and_unchanged(
            fixture.run,
            fixture.approval,
            fixture.artifacts,
            fixture.release_store,
        )
        promoting = fixture.runs.transition(fixture.run_id, RunState.PROMOTING)
        receipt = fixture.release_store.promote(
            promoting,
            fixture.approval,
            fixture.target,
            ReleaseSnapshot(
                run_id=evidence.snapshot.run_id,
                artifacts=evidence.snapshot.artifacts,
                manifest_hash=evidence.snapshot.manifest_hash,
                previous_model=fixture.approval.expected_current_model,
            ),
            report_hash=evidence.report_hash,
            decision_hash=evidence.decision_hash,
            model_hash=evidence.model_hash,
            schema_hash=evidence.schema_hash,
        )

        recovered = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )

        self.assertEqual(recovered, receipt)
        self.assertEqual(fixture.target.activations, 1)
        self.assertEqual(
            fixture.runs.get(fixture.run_id).state,
            RunState.PROMOTED.value,
        )

    def test_two_concurrent_promotions_create_one_release(self):
        fixture = self.make_ready_fixture()

        receipts = fixture.promote_concurrently(count=2)

        self.assertEqual(receipts[0], receipts[1])
        self.assertEqual(fixture.target.activations, 1)
        self.assertTrue((fixture.root / "releases" / fixture.run_id).is_dir())

    def test_rollback_restores_previous_model_and_writes_receipt(self):
        fixture = self.make_ready_fixture(current_model="run-0")
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )

        rollback = rollback_release(
            receipt,
            fixture.target,
            fixture.stores,
            approved_by="tester",
        )

        self.assertEqual(rollback.restored_model, "run-0")
        self.assertTrue(rollback.receipt_json_path.exists())
        self.assertTrue(rollback.receipt_html_path.exists())
        self.assertEqual(fixture.release_store.read_current_model(), "run-0")
        self.assertEqual(fixture.target.rollbacks, 1)

    def test_idempotent_retry_after_rollback_returns_the_original_receipt(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        rollback_release(
            receipt,
            fixture.target,
            fixture.stores,
            approved_by="tester",
        )

        repeated = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )

        self.assertEqual(repeated, receipt)
        self.assertEqual(fixture.target.activations, 1)
        self.assertEqual(fixture.release_store.read_current_model(), "run-0")

    def test_rollback_retries_a_target_already_restored_before_pointer_failure(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        fixture.release_store._before_pointer_swap = fixture._raise_before_pointer_swap
        with self.assertRaises(RuntimeError):
            rollback_release(
                receipt,
                fixture.target,
                fixture.stores,
                approved_by="tester",
            )
        fixture.release_store._before_pointer_swap = None

        rollback = rollback_release(
            receipt,
            fixture.target,
            fixture.stores,
            approved_by="tester",
        )

        self.assertEqual(rollback.restored_model, "run-0")
        self.assertEqual(fixture.release_store.read_current_model(), "run-0")
        self.assertEqual(fixture.target.rollbacks, 1)

    def test_rollback_rejects_a_forged_receipt_before_calling_the_target(self):
        fixture = self.make_ready_fixture()
        receipt = promote_repair(
            fixture.run_id,
            fixture.approval,
            fixture.target,
            fixture.stores,
        )
        forged = replace(receipt, previous_model="forged-model")

        with self.assertRaises(ArtifactIntegrityError):
            rollback_release(
                forged,
                fixture.target,
                fixture.stores,
                approved_by="tester",
            )

        self.assertEqual(fixture.target.rollbacks, 0)
        self.assertEqual(fixture.release_store.read_current_model(), fixture.run_id)


if __name__ == "__main__":
    unittest.main()
