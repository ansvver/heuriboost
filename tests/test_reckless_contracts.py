import copy
from dataclasses import fields, FrozenInstanceError
from enum import Enum
import json
import pickle
from pathlib import Path
import unittest

from heuriboost_rag.reckless import contracts as contracts_module
from heuriboost_rag.reckless.contracts import (
    ActivationResult,
    ArtifactRef,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    DatasetRef,
    Decision,
    EvaluationResult,
    GateCheck,
    PreparedActivation,
    PromotionApproval,
    PromotionReceipt,
    RepairRequest,
    ReleaseSnapshot,
    ReportArtifact,
    RollbackReceipt,
    RunContext,
    RunRecord,
    StageManifest,
    TargetValidation,
    ValidationResult,
)
from heuriboost_rag.reckless.errors import (
    ArtifactIntegrityError,
    EvaluationBlockedError,
    HeuriBoostError,
    InputBlockedError,
    NotEligibleError,
    PromotionConflictError,
)


class EvidenceKind(Enum):
    CURRENT = "current"
    ANCHOR = "anchor"


class MutableLeaf:
    def __init__(self) -> None:
        self.values = []


class RecklessContractTests(unittest.TestCase):
    def test_repair_request_is_frozen(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"rounds": 2},
        )
        with self.assertRaises(FrozenInstanceError):
            request.workspace_id = "changed"

    def test_approval_carries_stale_write_guards(self):
        approval = PromotionApproval(
            run_id="run-1",
            approved_by="tester",
            approved_at="2026-07-10T00:00:00Z",
            report_hash="report-hash",
            decision_hash="decision-hash",
            expected_current_model="run-0",
            idempotency_key="approval-1",
        )
        self.assertEqual(approval.expected_current_model, "run-0")

    def test_every_mapping_field_snapshots_source_and_is_read_only(self):
        artifact = ArtifactRef(
            artifact_type="model",
            path=Path("model.bin"),
            content_hash="artifact-hash",
            size_bytes=10,
        )
        dataset = DatasetRef(
            dataset_id="base-v1",
            role="base",
            path=Path("base.csv"),
            content_hash="dataset-hash",
            schema_hash="schema-hash",
        )
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
        )
        cases = []

        def add_case(label, field_name, value, factory):
            source = {"value": value}
            cases.append((label, factory(source), field_name, source, value))

        add_case(
            "RepairRequest.run_options",
            "run_options",
            2,
            lambda source: RepairRequest(
                workspace_id="prod-recog",
                base_dataset_id="base-v1",
                production_cases_id="cases-v2",
                policy_version="1",
                backend_name="fake",
                requested_by="tester",
                run_options=source,
            ),
        )
        add_case(
            "DatasetRef.metadata",
            "metadata",
            "base",
            lambda source: DatasetRef(
                dataset_id="base-v1",
                role="base",
                path=Path("base.csv"),
                content_hash="dataset-hash",
                schema_hash="schema-hash",
                metadata=source,
            ),
        )
        add_case(
            "ValidationResult.metadata",
            "metadata",
            True,
            lambda source: ValidationResult(valid=True, metadata=source),
        )
        add_case(
            "CompiledInputs.metadata",
            "metadata",
            "compiled",
            lambda source: CompiledInputs(artifacts=(artifact,), metadata=source),
        )
        add_case(
            "CandidateModel.metadata",
            "metadata",
            "candidate",
            lambda source: CandidateModel(
                model_path=Path("model.bin"),
                artifacts=(artifact,),
                metadata=source,
            ),
        )
        add_case(
            "PreparedActivation.pointer_payload",
            "pointer_payload",
            "run-1",
            lambda source: PreparedActivation(
                run_id="run-1",
                pointer_payload=source,
                metadata={},
            ),
        )
        add_case(
            "PreparedActivation.metadata",
            "metadata",
            "prepared",
            lambda source: PreparedActivation(
                run_id="run-1",
                pointer_payload={},
                metadata=source,
            ),
        )
        add_case(
            "ActivationResult.metadata",
            "metadata",
            "activated",
            lambda source: ActivationResult(
                current_model="run-1",
                metadata=source,
            ),
        )
        add_case(
            "EvaluationResult.global_metrics",
            "global_metrics",
            0.9,
            lambda source: EvaluationResult(
                acceptance_level="full",
                current_cases_passed=True,
                historical_gates_passed=True,
                global_metrics=source,
                anchor_metrics={},
                touched_domains={},
                artifacts_valid=True,
                details={},
            ),
        )
        add_case(
            "EvaluationResult.anchor_metrics",
            "anchor_metrics",
            0.8,
            lambda source: EvaluationResult(
                acceptance_level="full",
                current_cases_passed=True,
                historical_gates_passed=True,
                global_metrics={},
                anchor_metrics=source,
                touched_domains={},
                artifacts_valid=True,
                details={},
            ),
        )
        add_case(
            "EvaluationResult.touched_domains",
            "touched_domains",
            {"ndcg@10": 0.9},
            lambda source: EvaluationResult(
                acceptance_level="full",
                current_cases_passed=True,
                historical_gates_passed=True,
                global_metrics={},
                anchor_metrics={},
                touched_domains=source,
                artifacts_valid=True,
                details={},
            ),
        )
        add_case(
            "EvaluationResult.details",
            "details",
            {"passed": True},
            lambda source: EvaluationResult(
                acceptance_level="full",
                current_cases_passed=True,
                historical_gates_passed=True,
                global_metrics={},
                anchor_metrics={},
                touched_domains={},
                artifacts_valid=True,
                details=source,
            ),
        )
        add_case(
            "RunContext.datasets",
            "datasets",
            dataset,
            lambda source: RunContext(
                run_id="run-1",
                run_dir=Path("run-1"),
                datasets=source,
                options={},
            ),
        )
        add_case(
            "RunContext.options",
            "options",
            40,
            lambda source: RunContext(
                run_id="run-1",
                run_dir=Path("run-1"),
                datasets={},
                options=source,
            ),
        )
        add_case(
            "RunRecord.metadata",
            "metadata",
            "recorded",
            lambda source: RunRecord(
                run_id="run-1",
                state="RECEIVED",
                version=1,
                request=request,
                policy_hash="policy-hash",
                input_hash="input-hash",
                metadata=source,
            ),
        )
        add_case(
            "RunRecord.error",
            "error",
            {"code": "BLOCKED_INPUT"},
            lambda source: RunRecord(
                run_id="run-1",
                state="BLOCKED_INPUT",
                version=1,
                request=request,
                policy_hash="policy-hash",
                input_hash="input-hash",
                error=source,
            ),
        )
        add_case(
            "ReportArtifact.manifest",
            "manifest",
            "manifest-v1",
            lambda source: ReportArtifact(
                html_path=Path("report.html"),
                data_path=Path("report.json"),
                manifest_path=Path("manifest.json"),
                data_hash="data-hash",
                html_hash="html-hash",
                manifest=source,
            ),
        )

        for label, contract, field_name, source, original_value in cases:
            with self.subTest(field=label):
                frozen_mapping = getattr(contract, field_name)
                self.assertEqual(frozen_mapping["value"], original_value)
                source["value"] = "changed"
                self.assertEqual(frozen_mapping["value"], original_value)
                with self.assertRaises(TypeError):
                    frozen_mapping["new"] = "value"

    def test_nested_mapping_list_and_set_values_are_deeply_frozen(self):
        source = {
            "nested": {
                "items": [
                    {
                        "flags": {"anchor", "current"},
                    }
                ]
            }
        }
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options=source,
        )

        source["nested"]["items"][0]["flags"].add("source-change")
        source["nested"]["items"].append({"flags": set()})
        source["nested"]["added"] = True

        nested = request.run_options["nested"]
        self.assertEqual(len(nested["items"]), 1)
        self.assertEqual(nested["items"][0]["flags"], frozenset({"anchor", "current"}))
        self.assertNotIn("added", nested)
        with self.assertRaises(TypeError):
            nested["new"] = "value"
        with self.assertRaises(AttributeError):
            nested["items"].append("value")
        with self.assertRaises(TypeError):
            nested["items"][0]["new"] = "value"
        with self.assertRaises(AttributeError):
            nested["items"][0]["flags"].add("value")

    def test_default_mapping_fields_are_read_only(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
        )
        dataset = DatasetRef(
            dataset_id="base-v1",
            role="base",
            path=Path("base.csv"),
            content_hash="dataset-hash",
            schema_hash="schema-hash",
        )
        record = RunRecord(
            run_id="run-1",
            state="RECEIVED",
            version=1,
            request=request,
            policy_hash="policy-hash",
            input_hash="input-hash",
        )

        for mapping in (request.run_options, dataset.metadata, record.metadata):
            with self.assertRaises(TypeError):
                mapping["new"] = "value"
        self.assertIsNone(record.error)

    def test_concrete_errors_have_stable_codes_and_frozen_details(self):
        error_cases = (
            (InputBlockedError, "BLOCKED_INPUT"),
            (NotEligibleError, "BLOCKED_NOT_ELIGIBLE"),
            (EvaluationBlockedError, "BLOCKED_EVALUATION"),
            (PromotionConflictError, "PROMOTION_CONFLICT"),
            (ArtifactIntegrityError, "ARTIFACT_INTEGRITY"),
        )

        for error_type, expected_code in error_cases:
            with self.subTest(error=error_type.__name__):
                details = {"nested": {"items": ["first"]}}
                error = error_type(
                    "blocked",
                    stage="evaluation",
                    run_id="run-1",
                    retryable=True,
                    details=details,
                    operator_action="inspect report",
                )

                self.assertIsInstance(error, HeuriBoostError)
                self.assertEqual(str(error), "blocked")
                self.assertEqual(error.code, expected_code)
                self.assertEqual(error.message, "blocked")
                self.assertEqual(error.stage, "evaluation")
                self.assertEqual(error.run_id, "run-1")
                self.assertTrue(error.retryable)
                self.assertEqual(error.operator_action, "inspect report")

                details["nested"]["items"].append("source-change")
                details["new"] = "source-change"
                self.assertEqual(error.details["nested"]["items"], ("first",))
                self.assertNotIn("new", error.details)
                with self.assertRaises(TypeError):
                    error.details["new"] = "value"
                with self.assertRaises(TypeError):
                    error.details["nested"]["new"] = "value"
                with self.assertRaises(AttributeError):
                    error.details["nested"]["items"].append("value")

    def test_gate_check_evidence_is_recursively_snapshotted_and_frozen(self):
        observed_source = {
            "matches": [
                {
                    "path": Path("candidate.json"),
                    "kinds": {EvidenceKind.CURRENT},
                }
            ]
        }
        required_source = [{"minimum": 1}]
        check = GateCheck(
            check_id="current-cases",
            label="Current cases",
            passed=True,
            observed=observed_source,
            required=required_source,
            reason="all cases passed",
        )

        observed_source["matches"][0]["kinds"].add(EvidenceKind.ANCHOR)
        observed_source["matches"].append({"path": Path("changed.json")})
        required_source[0]["minimum"] = 2
        required_source.append({"minimum": 3})

        self.assertEqual(len(check.observed["matches"]), 1)
        self.assertEqual(
            check.observed["matches"][0]["kinds"],
            frozenset({EvidenceKind.CURRENT}),
        )
        self.assertEqual(check.required, ({"minimum": 1},))
        with self.assertRaises(TypeError):
            check.observed["new"] = "value"
        with self.assertRaises(AttributeError):
            check.observed["matches"].append("value")
        with self.assertRaises(TypeError):
            check.observed["matches"][0]["new"] = "value"
        with self.assertRaises(AttributeError):
            check.observed["matches"][0]["kinds"].add(EvidenceKind.ANCHOR)
        with self.assertRaises(AttributeError):
            check.required.append({"minimum": 2})
        with self.assertRaises(TypeError):
            check.required[0]["minimum"] = 2

    def test_unsupported_mutable_leaves_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "MutableLeaf"):
            RepairRequest(
                workspace_id="prod-recog",
                base_dataset_id="base-v1",
                production_cases_id="cases-v2",
                policy_version="1",
                backend_name="fake",
                requested_by="tester",
                run_options={"unsupported": MutableLeaf()},
            )

        with self.assertRaisesRegex(TypeError, "MutableLeaf"):
            GateCheck(
                check_id="unsupported",
                label="Unsupported evidence",
                passed=False,
                observed=MutableLeaf(),
                required=None,
                reason="unsupported evidence",
            )

    def test_contracts_support_deepcopy_and_pickle_round_trip(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={
                "paths": [Path("first.json"), Path("second.json")],
                "evidence": {EvidenceKind.CURRENT, EvidenceKind.ANCHOR},
                "nested": {"rounds": 2},
            },
        )

        copied = copy.deepcopy(request)
        restored = pickle.loads(pickle.dumps(request))

        self.assertEqual(copied, request)
        self.assertEqual(restored, request)
        with self.assertRaises(TypeError):
            restored.run_options["new"] = "value"
        with self.assertRaises(TypeError):
            restored.run_options["nested"]["rounds"] = 3

    def test_to_plain_data_is_deterministic_and_json_serializable(self):
        first = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={
                "path": Path("candidate.json"),
                "kind": EvidenceKind.CURRENT,
                "tags": {"zeta", "alpha"},
                "nested": [{"enabled": True}],
            },
        )
        second = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={
                "nested": [{"enabled": True}],
                "tags": {"alpha", "zeta"},
                "kind": EvidenceKind.CURRENT,
                "path": Path("candidate.json"),
            },
        )

        first_plain = contracts_module.to_plain_data(first)
        second_plain = contracts_module.to_plain_data(second)

        self.assertEqual(first_plain, second_plain)
        self.assertEqual(first_plain["run_options"]["path"], "candidate.json")
        self.assertEqual(first_plain["run_options"]["kind"], "current")
        self.assertEqual(first_plain["run_options"]["tags"], ["alpha", "zeta"])
        self.assertEqual(
            json.loads(json.dumps(first_plain, sort_keys=True)),
            first_plain,
        )
        with self.assertRaisesRegex(TypeError, "MutableLeaf"):
            contracts_module.to_plain_data(MutableLeaf())

    def test_every_public_contract_converts_to_json(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"rounds": 2},
        )
        approval = PromotionApproval(
            run_id="run-1",
            approved_by="tester",
            approved_at="2026-07-10T00:00:00Z",
            report_hash="report-hash",
            decision_hash="decision-hash",
            expected_current_model="run-0",
            idempotency_key="approval-1",
        )
        gate = GateCheck(
            check_id="gate-1",
            label="Gate",
            passed=True,
            observed={"path": Path("observed.json")},
            required={"kinds": {EvidenceKind.CURRENT}},
            reason="passed",
        )
        decision = Decision(
            promotion_eligible=True,
            acceptance_level="full",
            checks=(gate,),
            blockers=(),
            warnings=("review",),
        )
        dataset = DatasetRef(
            dataset_id="base-v1",
            role="base",
            path=Path("base.csv"),
            content_hash="dataset-hash",
            schema_hash="schema-hash",
            metadata={"kind": EvidenceKind.ANCHOR},
        )
        artifact = ArtifactRef(
            artifact_type="model",
            path=Path("model.bin"),
            content_hash="artifact-hash",
            size_bytes=10,
        )
        public_contracts = (
            request,
            approval,
            gate,
            decision,
            dataset,
            artifact,
            ValidationResult(valid=True, metadata={"rows": 10}),
            CompiledInputs(artifacts=(artifact,), metadata={"rows": 10}),
            CandidateModel(
                model_path=Path("model.bin"),
                artifacts=(artifact,),
                metadata={"rounds": 2},
            ),
            ArtifactVerification(valid=True, errors=()),
            TargetValidation(valid=True, current_model="run-0", errors=()),
            PreparedActivation(
                run_id="run-1",
                pointer_payload={"path": Path("release")},
                metadata={"ready": True},
            ),
            ActivationResult(current_model="run-1", metadata={"ready": True}),
            EvaluationResult(
                acceptance_level="full",
                current_cases_passed=True,
                historical_gates_passed=True,
                global_metrics={"ndcg@10": 0.9},
                anchor_metrics={"ndcg@10": 0.8},
                touched_domains={"prod": {"ndcg@10": 0.9}},
                artifacts_valid=True,
                details={"decision": decision},
            ),
            RunContext(
                run_id="run-1",
                run_dir=Path("run-1"),
                datasets={"base": dataset},
                options={"rounds": 2},
            ),
            RunRecord(
                run_id="run-1",
                state="READY_FOR_PROMOTION",
                version=1,
                request=request,
                policy_hash="policy-hash",
                input_hash="input-hash",
                metadata={"eligible": True},
                error=None,
            ),
            StageManifest(
                stage="evaluation",
                input_hash="input-hash",
                artifacts=(artifact,),
                started_at="2026-07-10T00:00:00Z",
                completed_at="2026-07-10T00:01:00Z",
            ),
            ReportArtifact(
                html_path=Path("report.html"),
                data_path=Path("report.json"),
                manifest_path=Path("manifest.json"),
                data_hash="data-hash",
                html_hash="html-hash",
                manifest={"artifacts": [Path("model.bin")]},
            ),
            ReleaseSnapshot(
                run_id="run-1",
                artifacts=(artifact,),
                manifest_hash="manifest-hash",
                previous_model="run-0",
            ),
            PromotionReceipt(
                run_id="run-1",
                release_path=Path("release/run-1"),
                promoted_at="2026-07-10T00:02:00Z",
                approved_by="tester",
                previous_model="run-0",
                current_model="run-1",
                release_manifest_hash="manifest-hash",
                receipt_json_path=Path("receipt.json"),
                receipt_html_path=Path("receipt.html"),
            ),
            RollbackReceipt(
                source_run_id="run-1",
                rolled_back_at="2026-07-10T00:03:00Z",
                approved_by="tester",
                previous_model="run-1",
                restored_model="run-0",
                receipt_json_path=Path("rollback.json"),
                receipt_html_path=Path("rollback.html"),
            ),
        )

        self.assertEqual(len(public_contracts), 21)
        for contract in public_contracts:
            with self.subTest(contract=type(contract).__name__):
                plain = contracts_module.to_plain_data(contract)
                self.assertEqual(
                    tuple(plain),
                    tuple(field.name for field in fields(contract)),
                )
                json.dumps(plain)

    def test_structured_error_converts_to_json(self):
        error = ArtifactIntegrityError(
            "artifact mismatch",
            stage="promotion",
            run_id="run-1",
            retryable=False,
            details={
                "path": Path("model.bin"),
                "kinds": {EvidenceKind.CURRENT, EvidenceKind.ANCHOR},
                "checks": [{"valid": False}],
            },
            operator_action="rebuild release",
        )

        plain = error.to_dict()

        self.assertEqual(
            plain,
            {
                "code": "ARTIFACT_INTEGRITY",
                "message": "artifact mismatch",
                "stage": "promotion",
                "run_id": "run-1",
                "retryable": False,
                "details": {
                    "checks": [{"valid": False}],
                    "kinds": ["anchor", "current"],
                    "path": "model.bin",
                },
                "operator_action": "rebuild release",
            },
        )
        self.assertEqual(json.loads(json.dumps(plain)), plain)

    def test_frozen_mapping_internal_storage_is_read_only_at_every_level(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"nested": {"rounds": 2}},
        )

        mappings = (
            ("top", request.run_options),
            ("nested", request.run_options["nested"]),
        )
        for label, mapping in mappings:
            with self.subTest(level=label):
                internal_name = (
                    "_FrozenMapping__data"
                    if hasattr(mapping, "_FrozenMapping__data")
                    else "_data"
                )
                internal = getattr(mapping, internal_name)
                with self.assertRaises(TypeError):
                    internal["new"] = "value"
                self.assertFalse(hasattr(mapping, "_data"))

    def test_frozen_mapping_blocks_slot_and_attribute_reassignment(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"rounds": 2},
        )
        mapping = request.run_options

        for attribute in ("_FrozenMapping__data", "_data", "other"):
            with self.subTest(attribute=attribute):
                with self.assertRaises(AttributeError):
                    setattr(mapping, attribute, {})
        with self.assertRaises(AttributeError):
            delattr(mapping, "_FrozenMapping__data")

    def test_all_errors_support_deepcopy_and_pickle_round_trip(self):
        error_cases = (
            HeuriBoostError(
                "CUSTOM_BLOCKED",
                "base blocked",
                stage="validation",
                run_id="run-base",
                retryable=True,
                details={"nested": {"paths": [Path("base.json")]}, "tags": {"b", "a"}},
                operator_action="inspect base error",
            ),
            InputBlockedError(
                "input blocked",
                stage="validation",
                run_id="run-input",
                details={"nested": {"paths": [Path("input.json")]}, "tags": {"b", "a"}},
                operator_action="fix input",
            ),
            NotEligibleError(
                "not eligible",
                stage="reporting",
                run_id="run-eligible",
                details={"nested": {"paths": [Path("eligible.json")]}, "tags": {"b", "a"}},
                operator_action="create a new run",
            ),
            EvaluationBlockedError(
                "evaluation blocked",
                stage="evaluation",
                run_id="run-evaluation",
                details={"nested": {"paths": [Path("evaluation.json")]}, "tags": {"b", "a"}},
                operator_action="inspect metrics",
            ),
            PromotionConflictError(
                "promotion conflict",
                stage="promotion",
                run_id="run-conflict",
                retryable=True,
                details={"nested": {"paths": [Path("conflict.json")]}, "tags": {"b", "a"}},
                operator_action="refresh current model",
            ),
            ArtifactIntegrityError(
                "artifact mismatch",
                stage="promotion",
                run_id="run-artifact",
                details={"nested": {"paths": [Path("artifact.json")]}, "tags": {"b", "a"}},
                operator_action="rebuild release",
            ),
        )

        for error in error_cases:
            with self.subTest(error=type(error).__name__):
                expected = error.to_dict()
                copied = copy.deepcopy(error)
                restored = pickle.loads(pickle.dumps(error))

                for round_tripped in (copied, restored):
                    self.assertIs(type(round_tripped), type(error))
                    self.assertEqual(round_tripped.to_dict(), expected)
                    self.assertEqual(str(round_tripped), str(error))
                    self.assertEqual(round_tripped.code, error.code)
                    self.assertEqual(round_tripped.message, error.message)
                    self.assertEqual(round_tripped.stage, error.stage)
                    self.assertEqual(round_tripped.run_id, error.run_id)
                    self.assertEqual(round_tripped.retryable, error.retryable)
                    self.assertEqual(round_tripped.operator_action, error.operator_action)
