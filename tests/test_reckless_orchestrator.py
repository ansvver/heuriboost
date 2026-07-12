from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from heuriboost_rag.reckless.contracts import (
    ArtifactRef,
    ArtifactVerification,
    CandidateModel,
    CompiledInputs,
    DatasetRef,
    EvaluationResult,
    RepairRequest,
    RunContext,
    SynthesizedFeatures,
    ValidationResult,
)
from heuriboost_rag.reckless.errors import ArtifactIntegrityError
from heuriboost_rag.reckless.hashing import ExecutionIdentity, build_run_fingerprint, sha256_file
from heuriboost_rag.reckless.orchestrator import (
    resume_reckless_repair,
    run_existing_reckless_repair,
    run_reckless_repair,
)
from heuriboost_rag.reckless.policy import RecklessPolicy
from heuriboost_rag.reckless.state import RunState
from heuriboost_rag.reckless.storage import (
    JsonDatasetRepository,
    JsonRunRepository,
    LocalArtifactStore,
    OrchestratorStores,
)


def _artifact(artifact_type: str, path: Path) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type,
        path=path,
        content_hash=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


class _DataFrameSentinel:
    pass


class _BoosterSentinel:
    pass


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        *,
        invalid_input: bool = False,
        weak_acceptance: bool = False,
        fail_evaluation: bool = False,
        fail_verification: bool = False,
        raise_compile: bool = False,
        interrupt_training_once: bool = False,
        omit_model_artifact: bool = False,
        backend_version: str = "test-v1",
    ) -> None:
        self.invalid_input = invalid_input
        self.weak_acceptance = weak_acceptance
        self.fail_evaluation = fail_evaluation
        self.fail_verification = fail_verification
        self.raise_compile = raise_compile
        self.interrupt_training_once = interrupt_training_once
        self.omit_model_artifact = omit_model_artifact
        self.backend_version = backend_version
        self.compile_calls = 0
        self.train_calls = 0
        self.train_input_paths: list[Path] = []
        self.evaluated_candidate: CandidateModel | None = None
        self.runtime_data_frame = _DataFrameSentinel()
        self.runtime_booster = _BoosterSentinel()

    def execution_identity(self) -> ExecutionIdentity:
        return ExecutionIdentity(
            backend_version=self.backend_version,
            feature_names=("feature_a", "feature_b"),
            feature_version="test-features-v1",
            code_commit="test-commit",
            training_params={"fixed_rounds": 2},
            random_seed=7,
        )

    def validate(
        self, request: RepairRequest, context: RunContext
    ) -> ValidationResult:
        if self.invalid_input:
            return ValidationResult(
                valid=False,
                metadata={"reason": "fixture rejected input"},
                warnings=("input warning",),
            )
        return ValidationResult(
            valid=True,
            metadata={"validated": True},
            warnings=("validation warning",),
        )

    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs:
        self.compile_calls += 1
        if self.raise_compile:
            raise RuntimeError("compile exploded")
        compiled_dir = context.run_dir / "backend-compiled"
        compiled_dir.mkdir(parents=True, exist_ok=True)
        payload = compiled_dir / "compiled-input.json"
        payload.write_text('{"compiled": true}', encoding="utf-8")
        binding = compiled_dir / "compiled-input-binding.json"
        binding.write_text('{"binding": "compiled"}', encoding="utf-8")
        return CompiledInputs(
            artifacts=(
                _artifact("compiled-input", payload),
                _artifact("compiled-input-binding", binding),
            ),
            metadata={"compiled": True, "touched_domains": ("medical",)},
        )

    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel:
        self.train_calls += 1
        self.train_input_paths.extend(artifact.path for artifact in inputs.artifacts)
        if self.interrupt_training_once:
            self.interrupt_training_once = False
            raise InterruptedError("intentional training interruption")
        model_dir = context.run_dir / "backend-trained"
        model_dir.mkdir(parents=True, exist_ok=True)
        model = model_dir / "reranker.json"
        model.write_text('{"model": true}', encoding="utf-8")
        binding = model_dir / "candidate-binding.json"
        binding.write_text('{"binding": "candidate"}', encoding="utf-8")
        artifacts = [*inputs.artifacts]
        if not self.omit_model_artifact:
            artifacts.append(_artifact("xgboost-model", model))
        artifacts.append(_artifact("candidate-binding", binding))
        return CandidateModel(
            model_path=model,
            artifacts=tuple(artifacts),
            metadata={"candidate": True, "touched_domains": ("medical",)},
        )

    def verify_artifacts(
        self, candidate: CandidateModel, context: RunContext
    ) -> ArtifactVerification:
        if self.fail_verification:
            return ArtifactVerification(valid=False, errors=("fixture verify failed",))
        return ArtifactVerification(valid=True, errors=())

    def evaluate(
        self, candidate: CandidateModel, context: RunContext) -> EvaluationResult:
        self.evaluated_candidate = candidate
        if self.fail_evaluation:
            return EvaluationResult(
                acceptance_level="full",
                current_cases_passed=False,
                historical_gates_passed=True,
                global_metrics={"ndcg@10": 0.8, "mrr@10": 0.7},
                anchor_metrics={"ndcg@10": 0.7, "mrr@10": 0.6},
                touched_domains={
                    "medical": {
                        "ndcg@10": 0.8,
                        "mrr@10": 0.7,
                        "anchor_ndcg@10": 0.7,
                        "anchor_mrr@10": 0.6,
                    }
                },
                artifacts_valid=True,
                details={"reason": "fixture failed evaluation"},
            )
        return EvaluationResult(
            acceptance_level="weak" if self.weak_acceptance else "full",
            current_cases_passed=True,
            historical_gates_passed=True,
            global_metrics={"ndcg@10": 0.8, "mrr@10": 0.7},
            anchor_metrics={"ndcg@10": 0.7, "mrr@10": 0.6},
            touched_domains={
                "medical": {
                    "ndcg@10": 0.8,
                    "mrr@10": 0.7,
                    "anchor_ndcg@10": 0.7,
                    "anchor_mrr@10": 0.6,
                }
            },
            artifacts_valid=True,
            details={"current_case_count": 1},
            warnings=("evaluation warning", "validation warning"),
        )


class FeatureSynthesisBackend(FakeBackend):
    def __init__(self, *, retry_passes: bool = True) -> None:
        super().__init__()
        self.retry_passes = retry_passes
        self.evaluate_calls = 0
        self.synthesize_calls = 0

    def evaluate(
        self, candidate: CandidateModel, context: RunContext
    ) -> EvaluationResult:
        self.evaluated_candidate = candidate
        self.evaluate_calls += 1
        if self.evaluate_calls == 1 or not self.retry_passes:
            return EvaluationResult(
                acceptance_level="full",
                current_cases_passed=False,
                historical_gates_passed=True,
                global_metrics={"ndcg@10": 0.8, "mrr@10": 0.7},
                anchor_metrics={"ndcg@10": 0.7, "mrr@10": 0.6},
                touched_domains={
                    "medical": {
                        "ndcg@10": 0.8,
                        "mrr@10": 0.7,
                        "anchor_ndcg@10": 0.7,
                        "anchor_mrr@10": 0.6,
                    }
                },
                artifacts_valid=True,
                details={"reason": "fixture failed before feature synthesis"},
            )
        return super().evaluate(candidate, context)

    def synthesize_features(
        self,
        compiled: CompiledInputs,
        candidate: CandidateModel,
        evaluation: EvaluationResult,
        decision,
        context: RunContext,
    ) -> SynthesizedFeatures:
        self.synthesize_calls += 1
        synth_dir = context.run_dir / "backend-synthesis"
        synth_dir.mkdir(parents=True, exist_ok=True)
        candidates = synth_dir / "llm-feature-candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "provider": "fake-llm",
                    "blockers": list(decision.blockers),
                    "feature_names": ["llm_case_match_score"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        enhanced = synth_dir / "compiled-input-with-llm-features.json"
        enhanced.write_text(
            json.dumps(
                {
                    "compiled": True,
                    "feature_synthesis": "llm_case_match_score",
                    "source_artifacts": [
                        artifact.artifact_type for artifact in compiled.artifacts
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        synthesized_artifact = _artifact("llm-feature-candidates", candidates)
        enhanced_artifact = _artifact("compiled-input", enhanced)
        return SynthesizedFeatures(
            artifacts=(synthesized_artifact,),
            compiled_inputs=CompiledInputs(
                artifacts=(enhanced_artifact, synthesized_artifact),
                metadata={
                    "compiled": True,
                    "synthesized_feature_names": ("llm_case_match_score",),
                },
            ),
            metadata={
                "provider": "fake-llm",
                "candidate_count": 1,
                "selected_feature_names": ("llm_case_match_score",),
            },
        )


class RecklessOrchestratorTests(unittest.TestCase):
    def _fixture(
        self, root: Path
    ) -> tuple[RepairRequest, OrchestratorStores, LocalArtifactStore]:
        base_path = root / "base.csv"
        base_path.write_text("query_id,label\nquery,3\n", encoding="utf-8")
        cases_path = root / "cases.jsonl"
        cases_path.write_text('{"case_id": "case-1"}\n', encoding="utf-8")
        datasets = JsonDatasetRepository(root)
        datasets.save(
            DatasetRef(
                dataset_id="base-v1",
                role="base",
                path=base_path,
                content_hash=sha256_file(base_path),
                schema_hash="base-schema-v1",
            )
        )
        datasets.save(
            DatasetRef(
                dataset_id="cases-v1",
                role="production_cases",
                path=cases_path,
                content_hash=sha256_file(cases_path),
                schema_hash="cases-schema-v1",
            )
        )
        artifacts = LocalArtifactStore(root)
        stores = OrchestratorStores(
            datasets=datasets,
            runs=JsonRunRepository(root),
            artifacts=artifacts,
        )
        request = RepairRequest(
            workspace_id="workspace",
            base_dataset_id="base-v1",
            production_cases_id="cases-v1",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"ignored": "by fake"},
        )
        return request, stores, artifacts

    def test_normal_run_seals_rebased_backend_artifacts_and_deterministic_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, stores, artifacts = self._fixture(root)
            backend = FakeBackend()

            run = run_reckless_repair(
                request,
                backend,
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(run.state, RunState.READY_FOR_PROMOTION.value)
            self.assertEqual(backend.compile_calls, 1)
            self.assertTrue(backend.train_input_paths)
            self.assertTrue(
                all(
                    path.is_relative_to(
                        artifacts.root / "runs" / run.run_id / "stages" / "COMPILED"
                    )
                    for path in backend.train_input_paths
                )
            )
            self.assertIsNotNone(backend.evaluated_candidate)
            candidate = backend.evaluated_candidate
            assert candidate is not None
            self.assertTrue(
                all(
                    path.is_relative_to(
                        artifacts.root / "runs" / run.run_id / "stages" / "TRAINED"
                    )
                    for path in (artifact.path for artifact in candidate.artifacts)
                )
            )
            model_ref = next(
                artifact
                for artifact in candidate.artifacts
                if artifact.artifact_type == "xgboost-model"
            )
            self.assertEqual(candidate.model_path, model_ref.path)

            compiled = artifacts.load_completed_stage(
                run.run_id,
                "COMPILED",
                run.input_hash,
            )
            trained = artifacts.load_completed_stage(
                run.run_id,
                "TRAINED",
                run.input_hash,
            )
            reported = artifacts.load_completed_stage(
                run.run_id,
                "REPORTING",
                run.input_hash,
            )
            self.assertIn("stage-result", {ref.artifact_type for ref in compiled.artifacts})
            self.assertIn("stage-result", {ref.artifact_type for ref in trained.artifacts})
            evidence_ref = next(
                ref for ref in reported.artifacts if ref.artifact_type == "report-evidence"
            )
            evidence = json.loads((artifacts.root / evidence_ref.path).read_text(encoding="utf-8"))
            self.assertEqual(evidence["outcome"]["state"], RunState.READY_FOR_PROMOTION.value)
            self.assertEqual(evidence["decision"]["blockers"], [])
            self.assertEqual(evidence["warnings"], ["evaluation warning", "validation warning"])
            self.assertEqual(
                [ref["artifact_type"] for ref in evidence["training"]["candidate_refs"]],
                [artifact.artifact_type for artifact in candidate.artifacts],
            )
            self.assertEqual(
                {ref.artifact_type for ref in reported.artifacts},
                {"report-evidence"},
            )
            encoded = json.dumps(evidence, sort_keys=True)
            self.assertNotIn("DataFrame", encoded)
            self.assertNotIn("Booster", encoded)
            self.assertNotIn("created_at", encoded)
            self.assertNotIn("timestamp", encoded)
            self.assertNotIn("report_evidence", evidence)
            self.assertIn("report_evidence", run.metadata)
            self.assertNotIn("decision", run.metadata)

    def test_invalid_input_becomes_blocked_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, _ = self._fixture(Path(tmp))
            backend = FakeBackend(invalid_input=True)

            run = run_reckless_repair(request, backend, stores, RecklessPolicy.default())

            self.assertEqual(run.state, RunState.BLOCKED_INPUT.value)
            self.assertEqual(run.error["code"], RunState.BLOCKED_INPUT.value)
            self.assertEqual(backend.compile_calls, 0)

    def test_existing_received_run_is_executed_without_creating_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, stores, _ = self._fixture(root)
            backend = FakeBackend()
            identity = backend.execution_identity()
            base = stores.datasets.get(request.base_dataset_id)
            cases = stores.datasets.get(request.production_cases_id)
            policy = RecklessPolicy.default()
            input_hash = build_run_fingerprint(request, policy, base, cases, identity)
            existing = stores.runs.create(request, policy.content_hash, input_hash)

            run = run_existing_reckless_repair(
                existing.run_id,
                backend,
                stores,
                policy,
            )

            self.assertEqual(run.run_id, existing.run_id)
            self.assertEqual(run.state, RunState.READY_FOR_PROMOTION.value)
            self.assertEqual(stores.runs.get(existing.run_id).run_id, existing.run_id)

    def test_fallback_model_is_restored_as_a_sealed_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, artifacts = self._fixture(Path(tmp))
            backend = FakeBackend(omit_model_artifact=True)

            run = run_reckless_repair(
                request,
                backend,
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(run.state, RunState.READY_FOR_PROMOTION.value)
            candidate = backend.evaluated_candidate
            assert candidate is not None
            model_ref = next(
                artifact
                for artifact in candidate.artifacts
                if artifact.artifact_type == "candidate-model"
            )
            self.assertEqual(candidate.model_path, model_ref.path)
            self.assertTrue(candidate.model_path.is_relative_to(artifacts.root))

    def test_weak_acceptance_becomes_blocked_not_eligible_after_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, artifacts = self._fixture(Path(tmp))

            run = run_reckless_repair(
                request,
                FakeBackend(weak_acceptance=True),
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(run.state, RunState.BLOCKED_NOT_ELIGIBLE.value)
            reported = artifacts.load_completed_stage(
                run.run_id,
                "REPORTING",
                run.input_hash,
            )
            evidence_ref = next(
                ref for ref in reported.artifacts if ref.artifact_type == "report-evidence"
            )
            evidence = json.loads((artifacts.root / evidence_ref.path).read_text(encoding="utf-8"))
            self.assertEqual(evidence["outcome"]["state"], RunState.BLOCKED_NOT_ELIGIBLE.value)

    def test_failed_evaluation_or_artifact_verification_becomes_blocked_evaluation(self):
        for backend in (FakeBackend(fail_evaluation=True), FakeBackend(fail_verification=True)):
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory() as tmp:
                    request, stores, _ = self._fixture(Path(tmp))

                    run = run_reckless_repair(
                        request,
                        backend,
                        stores,
                        RecklessPolicy.default(),
                    )

                    self.assertEqual(run.state, RunState.BLOCKED_EVALUATION.value)
                    self.assertEqual(
                        run.error["code"], RunState.BLOCKED_EVALUATION.value
                    )

    def test_feature_synthesis_retry_can_recover_a_gate_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, artifacts = self._fixture(Path(tmp))
            backend = FeatureSynthesisBackend(retry_passes=True)

            run = run_reckless_repair(
                request,
                backend,
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(run.state, RunState.READY_FOR_PROMOTION.value)
            self.assertEqual(backend.synthesize_calls, 1)
            self.assertEqual(backend.train_calls, 2)
            synthesis = artifacts.load_completed_stage(
                run.run_id,
                "FEATURE_SYNTHESIS",
                run.input_hash,
            )
            retry = artifacts.load_completed_stage(
                run.run_id,
                "TRAINED_RETRY",
                run.input_hash,
            )
            self.assertIn(
                "llm-feature-candidates",
                {artifact.artifact_type for artifact in synthesis.artifacts},
            )
            self.assertIn(
                "xgboost-model",
                {artifact.artifact_type for artifact in retry.artifacts},
            )
            reported = artifacts.load_completed_stage(
                run.run_id,
                "REPORTING",
                run.input_hash,
            )
            evidence_ref = next(
                ref for ref in reported.artifacts if ref.artifact_type == "report-evidence"
            )
            evidence = json.loads((artifacts.root / evidence_ref.path).read_text(encoding="utf-8"))
            self.assertEqual(
                evidence["feature_synthesis"]["metadata"]["selected_feature_names"],
                ["llm_case_match_score"],
            )
            self.assertEqual(
                [stage["stage"] for stage in evidence["completed_stage_manifests"]],
                ["COMPILED", "TRAINED", "FEATURE_SYNTHESIS", "TRAINED_RETRY"],
            )

    def test_feature_synthesis_retry_still_blocks_when_retry_fails_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, _ = self._fixture(Path(tmp))
            backend = FeatureSynthesisBackend(retry_passes=False)

            run = run_reckless_repair(
                request,
                backend,
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(run.state, RunState.BLOCKED_EVALUATION.value)
            self.assertEqual(backend.synthesize_calls, 1)
            self.assertEqual(backend.train_calls, 2)
            self.assertEqual(
                run.error["details"]["blockers"],
                ("current_production_cases",),
            )

    def test_unexpected_exception_becomes_failed_internal_with_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, _ = self._fixture(Path(tmp))

            run = run_reckless_repair(
                request,
                FakeBackend(raise_compile=True),
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(run.state, RunState.FAILED_INTERNAL.value)
            self.assertEqual(run.error["code"], RunState.FAILED_INTERNAL.value)
            self.assertEqual(run.error["stage"], "COMPILED")
            self.assertEqual(run.error["details"]["exception_type"], "RuntimeError")

    def test_interrupted_training_resumes_from_verified_compiled_snapshots_without_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, stores, artifacts = self._fixture(root)
            backend = FakeBackend(interrupt_training_once=True)

            interrupted = run_reckless_repair(
                request,
                backend,
                stores,
                RecklessPolicy.default(),
            )
            self.assertEqual(interrupted.state, RunState.INTERRUPTED.value)
            self.assertEqual(backend.compile_calls, 1)

            resumed = resume_reckless_repair(
                interrupted.run_id,
                backend,
                stores,
                RecklessPolicy.default(),
            )

            self.assertEqual(resumed.state, RunState.READY_FOR_PROMOTION.value)
            self.assertEqual(backend.compile_calls, 1)
            self.assertEqual(backend.train_calls, 2)
            self.assertTrue(
                all(
                    path.is_relative_to(
                        artifacts.root
                        / "runs"
                        / interrupted.run_id
                        / "stages"
                        / "COMPILED"
                    )
                    for path in backend.train_input_paths
                )
            )

    def test_corrupt_or_mismatched_compiled_snapshot_leaves_interrupted_run_unchanged(self):
        for mismatch in ("tamper", "fingerprint"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request, stores, artifacts = self._fixture(root)
                backend = FakeBackend(interrupt_training_once=True)
                interrupted = run_reckless_repair(
                    request,
                    backend,
                    stores,
                    RecklessPolicy.default(),
                )
                self.assertEqual(interrupted.state, RunState.INTERRUPTED.value)
                if mismatch == "tamper":
                    compiled = artifacts.load_completed_stage(
                        interrupted.run_id,
                        "COMPILED",
                        interrupted.input_hash,
                    )
                    source = next(
                        ref
                        for ref in compiled.artifacts
                        if ref.artifact_type == "compiled-input"
                    )
                    snapshot = artifacts.root / source.path
                    snapshot.chmod(0o600)
                    snapshot.write_text('{"compiled": false}', encoding="utf-8")
                    resumed_backend = backend
                else:
                    resumed_backend = FakeBackend(backend_version="changed-runtime")

                with self.assertRaises(ArtifactIntegrityError):
                    resume_reckless_repair(
                        interrupted.run_id,
                        resumed_backend,
                        stores,
                        RecklessPolicy.default(),
                    )

                self.assertEqual(
                    stores.runs.get(interrupted.run_id).state,
                    RunState.INTERRUPTED.value,
                )

    def test_blocked_runs_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, stores, _ = self._fixture(Path(tmp))
            blocked = run_reckless_repair(
                request,
                FakeBackend(invalid_input=True),
                stores,
                RecklessPolicy.default(),
            )

            with self.assertRaisesRegex(ValueError, "only INTERRUPTED runs can resume"):
                resume_reckless_repair(
                    blocked.run_id,
                    FakeBackend(),
                    stores,
                    RecklessPolicy.default(),
                )


if __name__ == "__main__":
    unittest.main()
