from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from heuriboost_rag.backends import xgboost_rag
from heuriboost_rag.backends.base import RepairBackend
from heuriboost_rag.backends.legacy_runtime import (
    LegacyRegistryMetadata,
    LegacyRuntime,
)
from heuriboost_rag.backends.xgboost_rag import (
    CompileSettings,
    PinnedFile,
    TrainingSettings,
    XGBoostRagBackend,
    XGBoostRagConfig,
    _artifact_set_hash,
)
from heuriboost_rag.reckless.contracts import (
    CandidateModel,
    CompiledInputs,
    DatasetRef,
    RepairRequest,
    RunContext,
    to_plain_data,
)
from heuriboost_rag.reckless.hashing import (
    ExecutionIdentityProvider,
    build_run_fingerprint,
    sha256_file,
)
from heuriboost_rag.reckless.policy import RecklessPolicy
from heuriboost_rag.reckless.storage import LocalArtifactStore


class _DataFrameSentinel:
    pass


class _BoosterSentinel:
    pass


class _CompileOptions:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


_FAKE_FIXED_PARAMS = {
    "colsample_bytree": 0.9,
    "eta": 0.08,
    "eval_metric": "ndcg@10",
    "max_depth": 3,
    "min_child_weight": 0.1,
    "objective": "rank:ndcg",
    "seed": 42,
    "subsample": 0.9,
}


class _FakeRuntimeState:
    def __init__(self, feature_recipe_path: Path) -> None:
        self.registry = SimpleNamespace(
            feature_names=("dense_score", "term_overlap"),
            feature_version="heuriboost-rag-v0:1",
            feature_set_name="heuriboost_rag_v0",
        )
        self.feature_recipe_path = feature_recipe_path
        self.compile_options_type = _CompileOptions
        self.calls: list[tuple[str, object]] = []

    def compile_repair_inputs(
        self,
        base_dataset: Path,
        production_cases: Path,
        options: _CompileOptions,
    ) -> SimpleNamespace:
        self.calls.append(("compile", options))
        compiled = Path(options.output_dir) / ".heuriboost" / "compiled"
        case_sets = compiled / "case_sets"
        case_sets.mkdir(parents=True)
        base_copy = compiled / "query_doc_examples.csv"
        base_copy.write_bytes(base_dataset.read_bytes())
        regression_cases = compiled / "regression_cases.yaml"
        regression_cases.write_text("cases: []\n", encoding="utf-8")
        production_json = compiled / "production_cases.json"
        production_json.write_text("[]", encoding="utf-8")
        case_set = case_sets / "current_production_cases.csv"
        case_set.write_text("query_id,label\ncase,3\n", encoding="utf-8")
        report = compiled / "compile_report.md"
        report.write_text("compiled\n", encoding="utf-8")
        return SimpleNamespace(
            base_dataset_path=base_copy,
            regression_cases_path=regression_cases,
            case_sets_dir=case_sets,
            production_cases_json_path=production_json,
            compile_report_path=report,
            base_df=_DataFrameSentinel(),
            repair_samples_df=_DataFrameSentinel(),
            production_cases=[{"case_id": "current-case", "domain": "medical"}],
            touched_domains=["medical"],
            warnings=["fake compiler warning"],
        )

    def load_dataset(self, path: Path) -> _DataFrameSentinel:
        self.calls.append(("load_dataset", path))
        return _DataFrameSentinel()

    def load_compiled_production_cases(self, path: Path) -> list[dict[str, object]]:
        self.calls.append(("load_cases", path))
        return [{"case_id": "current-case", "domain": "medical"}]

    def merge_training_frames(
        self,
        base: _DataFrameSentinel,
        repair_samples: _DataFrameSentinel,
        promoted_samples_path: Path | None = None,
    ) -> _DataFrameSentinel:
        self.calls.append(("merge", promoted_samples_path))
        return _DataFrameSentinel()

    def train_model_from_frame(
        self,
        frame: _DataFrameSentinel,
        output_dir: Path,
        rounds: int,
    ) -> _BoosterSentinel:
        self.calls.append(("train", rounds))
        models = Path(output_dir) / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "reranker.json").write_text("model", encoding="utf-8")
        (models / "reranker_metadata.json").write_text(
            json.dumps(
                {
                    "rounds": rounds,
                    "feature_names": ["dense_score", "term_overlap"],
                    "feature_set_name": "heuriboost_rag_v0",
                    "feature_set_version": 1,
                    "feature_versions": {
                        "dense_score": 1,
                        "term_overlap": 1,
                    },
                    "params": _FAKE_FIXED_PARAMS,
                }
            ),
            encoding="utf-8",
        )
        return _BoosterSentinel()

    def load_model(self, path: Path) -> _BoosterSentinel:
        self.calls.append(("load_model", path))
        if getattr(self, "fail_model_load", False):
            raise ValueError("model cannot be loaded")
        return _BoosterSentinel()

    def evaluate_model_on_split(
        self,
        model: _BoosterSentinel,
        frame: _DataFrameSentinel,
        split: str,
    ) -> tuple[dict[str, float], object]:
        self.calls.append(("evaluate_split", split))
        # The raw-label legacy scorer treats label 0 as MRR-irrelevant.
        return (
            {
                "ndcg@5": 0.79,
                "ndcg@10": 0.81,
                "mrr@5": 0.0,
                "mrr@10": 0.0,
                "recall@10": 0.92,
                "hard_negative_rate@10": 0.25,
                "query_count": 7,
            },
            _DataFrameSentinel(),
        )

    def evaluate_model_by_domain(
        self,
        model: _BoosterSentinel,
        frame: _DataFrameSentinel,
        split: str,
    ) -> dict[str, dict[str, float]]:
        self.calls.append(("evaluate_domain", split))
        return {
            "medical": {
                "ndcg@5": 0.8,
                "ndcg@10": 0.82,
                "mrr@5": 0.0,
                "mrr@10": 0.0,
                "recall@10": 0.93,
                "hard_negative_rate@10": 0.2,
                "query_count": 3,
            }
        }

    def evaluate_cases(
        self,
        cases: list[dict[str, object]],
        model: _BoosterSentinel,
        acceptance_level: str = "full",
    ) -> list[dict[str, object]]:
        self.calls.append(("evaluate_cases", acceptance_level))
        return [
            {
                "case_id": "case-1",
                "domain": "medical",
                "passed": True,
                "good_ranks": {"good-doc": 2},
                "bad_ranks": {"bad-doc": None},
                "top_k": 3,
            }
        ]

    def load_gates(self, path: Path) -> list[dict[str, object]]:
        self.calls.append(("load_gates", path))
        return [{"gate_id": "gate-1", "domain": "medical"}]


class XGBoostRagBackendTests(unittest.TestCase):
    def _request(self, **run_options: object) -> RepairRequest:
        return RepairRequest(
            workspace_id="workspace",
            base_dataset_id="base",
            production_cases_id="cases",
            policy_version="1",
            backend_name="xgboost-rag",
            requested_by="tester",
            run_options=run_options,
        )

    def _dataset(self, dataset_id: str, role: str, path: Path) -> DatasetRef:
        return DatasetRef(
            dataset_id=dataset_id,
            role=role,
            path=path,
            content_hash=sha256_file(path),
            schema_hash=f"{role}-schema-v1",
        )

    def _context(self, root: Path, base: Path, cases: Path) -> RunContext:
        return RunContext(
            run_id="run-1",
            run_dir=root / "run-1",
            datasets={
                "base": self._dataset("base", "base", base),
                "production_cases": self._dataset("cases", "production_cases", cases),
            },
            options={"rounds": 999},
        )

    def _pinned(self, path: Path) -> PinnedFile:
        return PinnedFile(path=path, content_hash=sha256_file(path))

    def _configured_backend(
        self,
        root: Path,
        *,
        legacy_code_manifest_hash: str | None = None,
        captured_recipe_hash: str | None = None,
    ) -> tuple[XGBoostRagBackend, _FakeRuntimeState]:
        recipes = root / "feature_recipes.yaml"
        recipes.write_text("feature recipes", encoding="utf-8")
        scripts_dir = root / "legacy-scripts"
        scripts_dir.mkdir()
        source_file = scripts_dir / "repair_cases.py"
        source_file.write_text("legacy code v1\n", encoding="utf-8")
        gates = root / "gates.jsonl"
        gates.write_text('{"gate_id":"gate-1"}\n', encoding="utf-8")
        ledger = root / "ledger.json"
        ledger.write_text(
            json.dumps(
                {
                    "anchor": {
                        "global": {"ndcg@10": 0.8, "mrr@10": 0.0},
                        "domains": {
                            "medical": {"ndcg@10": 0.8, "mrr@10": 0.0}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        state = _FakeRuntimeState(recipes)
        runtime = LegacyRuntime(
            scripts_dir=scripts_dir,
            registry=LegacyRegistryMetadata(
                feature_names=("dense_score", "term_overlap"),
                feature_set_name="heuriboost_rag_v0",
                feature_set_version=1,
                feature_versions=(("dense_score", 1), ("term_overlap", 1)),
                feature_recipe_path=recipes,
            ),
            compile_options_type=_CompileOptions,
            compile_repair_inputs=state.compile_repair_inputs,
            load_dataset=state.load_dataset,
            load_compiled_production_cases=state.load_compiled_production_cases,
            merge_training_frames=state.merge_training_frames,
            train_model_from_frame=state.train_model_from_frame,
            load_model=state.load_model,
            evaluate_model_on_split=state.evaluate_model_on_split,
            evaluate_model_by_domain=state.evaluate_model_by_domain,
            evaluate_cases=state.evaluate_cases,
            load_gates=state.load_gates,
            code_manifest_files=(source_file,),
            fixed_training_params=tuple(sorted(_FAKE_FIXED_PARAMS.items())),
            feature_recipe_hash_at_load=captured_recipe_hash,
        )
        config = XGBoostRagConfig(
            code_revision="deadbeef",
            legacy_code_manifest_hash=(
                runtime.code_manifest_hash
                if legacy_code_manifest_hash is None
                else legacy_code_manifest_hash
            ),
            feature_recipes=self._pinned(recipes),
            historical_gates=self._pinned(gates),
            anchor_ledger=self._pinned(ledger),
            compile_settings=CompileSettings(
                strict=True,
                min_global_test_queries=50,
                min_domain_test_queries=10,
            ),
            training_settings=TrainingSettings(rounds=7, random_seed=42),
        )
        return XGBoostRagBackend(config, runtime=runtime), state

    def test_zero_arg_backend_conforms_but_blocks_execution(self) -> None:
        backend = XGBoostRagBackend()
        self.assertIsInstance(backend, RepairBackend)
        self.assertIsInstance(backend, ExecutionIdentityProvider)
        self.assertEqual(backend.name, "xgboost-rag")
        self.assertEqual(
            backend.execution_identity(),
            XGBoostRagBackend().execution_identity(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            result = backend.validate(self._request(), self._context(root, base, cases))

        self.assertFalse(result.valid)
        self.assertIn("unconfigured", result.metadata["reason"])

    def test_config_is_immutable_and_all_effective_inputs_change_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            identity = backend.execution_identity()
            self.assertIn("runtime_config_hash", identity.training_params)
            self.assertEqual(
                identity.training_params["expected_legacy_code_manifest_hash"],
                backend.config.legacy_code_manifest_hash,
            )
            self.assertEqual(
                identity.training_params["legacy_code_manifest_hash"],
                backend.runtime.code_manifest_hash,
            )
            self.assertEqual(
                identity.training_params["pinned_input_hashes"]["historical_gates"],
                backend.config.historical_gates.content_hash,
            )
            with self.assertRaises(FrozenInstanceError):
                backend.config.code_revision = "changed"

            baseline = build_run_fingerprint(
                self._request(rounds=1),
                RecklessPolicy.default(),
                context.datasets["base"],
                context.datasets["production_cases"],
                identity,
            )
            changed_code = XGBoostRagBackend(
                replace(backend.config, code_revision="cafebabe"),
                runtime=backend.runtime,
            ).execution_identity()
            changed_input = XGBoostRagBackend(
                replace(
                    backend.config,
                    historical_gates=replace(
                        backend.config.historical_gates,
                        content_hash="f" * 64,
                    ),
                ),
                runtime=backend.runtime,
            ).execution_identity()
            promoted = root / "promoted.csv"
            promoted.write_text("query_id,label\nold,3\n", encoding="utf-8")
            changed_identities = (
                changed_code,
                changed_input,
                XGBoostRagBackend(
                    replace(
                        backend.config,
                        feature_recipes=replace(
                            backend.config.feature_recipes,
                            content_hash="e" * 64,
                        ),
                    ),
                    runtime=backend.runtime,
                ).execution_identity(),
                XGBoostRagBackend(
                    replace(
                        backend.config,
                        anchor_ledger=replace(
                            backend.config.anchor_ledger,
                            content_hash="d" * 64,
                        ),
                    ),
                    runtime=backend.runtime,
                ).execution_identity(),
                XGBoostRagBackend(
                    replace(
                        backend.config,
                        compile_settings=replace(
                            backend.config.compile_settings,
                            case_top_k=4,
                        ),
                    ),
                    runtime=backend.runtime,
                ).execution_identity(),
                XGBoostRagBackend(
                    replace(
                        backend.config,
                        training_settings=replace(
                            backend.config.training_settings,
                            rounds=8,
                        ),
                    ),
                    runtime=backend.runtime,
                ).execution_identity(),
                XGBoostRagBackend(
                    replace(
                        backend.config,
                        promoted_samples=self._pinned(promoted),
                        include_promoted_samples=True,
                    ),
                    runtime=backend.runtime,
                ).execution_identity(),
                XGBoostRagBackend(
                    replace(
                        backend.config,
                        legacy_code_manifest_hash="a" * 64,
                    ),
                    runtime=backend.runtime,
                ).execution_identity(),
                XGBoostRagBackend(
                    backend.config,
                    runtime=replace(
                        backend.runtime,
                        fixed_training_params=tuple(
                            sorted(
                                {
                                    **backend.runtime.fixed_training_params_mapping,
                                    "max_depth": 4,
                                }.items()
                            )
                        ),
                    ),
                ).execution_identity(),
            )
            self.assertEqual(
                baseline,
                build_run_fingerprint(
                    self._request(rounds=999, attacker_model_path="/tmp/model"),
                    RecklessPolicy.default(),
                    context.datasets["base"],
                    context.datasets["production_cases"],
                    identity,
                ),
            )
            for changed_identity in changed_identities:
                self.assertNotEqual(
                    baseline,
                    build_run_fingerprint(
                        self._request(),
                        RecklessPolicy.default(),
                        context.datasets["base"],
                        context.datasets["production_cases"],
                        changed_identity,
                    ),
                )

    def test_runtime_versions_change_execution_identity_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            baseline_versions = {
                "python": "3.10.14",
                "xgboost": "2.1.0",
                "pandas": "2.2.2",
                "numpy": "1.26.4",
                "pyyaml": "6.0.2",
            }
            changed_versions = {**baseline_versions, "xgboost": "2.1.1"}
            with mock.patch.object(
                xgboost_rag,
                "_runtime_dependency_versions",
                return_value=baseline_versions,
                create=True,
            ):
                baseline = build_run_fingerprint(
                    self._request(),
                    RecklessPolicy.default(),
                    context.datasets["base"],
                    context.datasets["production_cases"],
                    backend.execution_identity(),
                )
            with mock.patch.object(
                xgboost_rag,
                "_runtime_dependency_versions",
                return_value=changed_versions,
                create=True,
            ):
                changed = build_run_fingerprint(
                    self._request(),
                    RecklessPolicy.default(),
                    context.datasets["base"],
                    context.datasets["production_cases"],
                    backend.execution_identity(),
                )

        self.assertNotEqual(baseline, changed)

    def test_validate_rejects_unpinned_runtime_and_dataset_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)

            invalid_dataset_context = RunContext(
                run_id=context.run_id,
                run_dir=context.run_dir,
                datasets={
                    "base": replace(
                        context.datasets["base"], content_hash="0" * 64
                    ),
                    "production_cases": context.datasets["production_cases"],
                },
                options={},
            )
            self.assertFalse(
                backend.validate(self._request(), invalid_dataset_context).valid
            )

            other_recipes = root / "other_feature_recipes.yaml"
            other_recipes.write_text("different recipes", encoding="utf-8")
            wrong_source_backend = XGBoostRagBackend(
                replace(backend.config, feature_recipes=self._pinned(other_recipes)),
                runtime=backend.runtime,
            )
            self.assertFalse(
                wrong_source_backend.validate(self._request(), context).valid
            )

            weak_compile_backend = XGBoostRagBackend(
                replace(
                    backend.config,
                    compile_settings=replace(
                        backend.config.compile_settings,
                        strict=False,
                    ),
                ),
                runtime=backend.runtime,
            )
            self.assertFalse(weak_compile_backend.validate(self._request(), context).valid)

            backend.config.historical_gates.path.write_text("", encoding="utf-8")
            self.assertFalse(backend.validate(self._request(), context).valid)

    def test_validate_rejects_captured_recipe_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(
                root,
                captured_recipe_hash="0" * 64,
            )
            context = self._context(root, base, cases)

            validation = backend.validate(self._request(), context)

        self.assertFalse(validation.valid)
        self.assertTrue(
            any(
                "captured feature recipe" in error
                for error in validation.metadata["errors"]
            )
        )

    def test_validate_rejects_wrong_dataset_roles_before_creating_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, state = self._configured_backend(root)
            for dataset_key in ("base", "production_cases"):
                with self.subTest(dataset_key=dataset_key):
                    state.calls.clear()
                    context = self._context(root, base, cases)
                    datasets = dict(context.datasets)
                    datasets[dataset_key] = replace(
                        datasets[dataset_key], role=f"wrong-{dataset_key}"
                    )
                    wrong_role_context = RunContext(
                        run_id=context.run_id,
                        run_dir=context.run_dir,
                        datasets=datasets,
                        options=context.options,
                    )

                    validation = backend.validate(self._request(), wrong_role_context)

                    self.assertFalse(validation.valid)
                    self.assertTrue(
                        any(
                            f"{dataset_key} DatasetRef role" in error
                            for error in validation.metadata["errors"]
                        )
                    )
                    self.assertFalse(wrong_role_context.run_dir.exists())
                    self.assertFalse(
                        any(call[0] == "compile" for call in state.calls)
                    )
                    with self.assertRaisesRegex(ValueError, "DatasetRef role"):
                        backend.compile(self._request(), wrong_role_context)
                    self.assertFalse(wrong_role_context.run_dir.exists())
                    self.assertFalse(
                        any(call[0] == "compile" for call in state.calls)
                    )


    def test_startup_pinned_legacy_code_blocks_before_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(
                root,
                legacy_code_manifest_hash="0" * 64,
            )
            context = self._context(root, base, cases)

            validation = backend.validate(self._request(), context)

            self.assertFalse(validation.valid)
            self.assertTrue(
                any(
                    "legacy_code_manifest_hash" in error
                    for error in validation.metadata["errors"]
                )
            )
            with self.assertRaisesRegex(ValueError, "legacy_code_manifest_hash"):
                backend.compile(self._request(), context)

    def test_source_change_after_startup_blocks_before_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            backend.runtime.code_manifest_files[0].write_text(
                "legacy code v2\n",
                encoding="utf-8",
            )

            validation = backend.validate(self._request(), context)

            self.assertFalse(validation.valid)
            with self.assertRaisesRegex(ValueError, "legacy_code_manifest_hash"):
                backend.compile(self._request(), context)

    def test_code_manifest_drift_blocks_evaluate_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            backend.runtime.code_manifest_files[0].write_text(
                "legacy code v2\n",
                encoding="utf-8",
            )

            evaluation = backend.evaluate(candidate, context)
            verification = backend.verify_artifacts(candidate, context)

        self.assertFalse(evaluation.artifacts_valid)
        self.assertIn("legacy_code_manifest_hash", evaluation.details["artifact_errors"][0])
        self.assertFalse(verification.valid)
        self.assertIn("legacy_code_manifest_hash", verification.errors[0])

    def test_backend_runtime_and_config_cannot_be_reassigned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _ = self._configured_backend(root)
            with self.assertRaises(AttributeError):
                backend.config = backend.config
            with self.assertRaises(AttributeError):
                backend._runtime = backend.runtime
            with self.assertRaises(TypeError):
                XGBoostRagBackend(backend.config, runtime=object())

    def test_train_rejects_external_artifacts_and_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            outside = root / "outside.csv"
            outside.write_text("outside", encoding="utf-8")
            external = replace(
                next(
                    artifact
                    for artifact in compiled.artifacts
                    if artifact.artifact_type == "compiled-base-dataset"
                ),
                path=outside,
                content_hash=sha256_file(outside),
                size_bytes=outside.stat().st_size,
            )
            external_inputs = replace(
                compiled,
                artifacts=tuple(
                    external
                    if artifact.artifact_type == external.artifact_type
                    else artifact
                    for artifact in compiled.artifacts
                ),
            )
            with self.assertRaisesRegex(ValueError, "outside context.run_dir"):
                backend.train(external_inputs, context)

            bad_identity_inputs = replace(
                compiled,
                metadata={
                    **dict(compiled.metadata),
                    "runtime_config_hash": "0" * 64,
                },
            )
            with self.assertRaisesRegex(ValueError, "runtime_config_hash"):
                backend.train(bad_identity_inputs, context)

    def test_train_rejects_recomputed_pinned_snapshot_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            gates = next(
                artifact
                for artifact in compiled.artifacts
                if artifact.artifact_type == "historical-gates-snapshot"
            )
            gates.path.write_text('{"gate_id":"tampered"}\n', encoding="utf-8")
            tampered_inputs = replace(
                compiled,
                artifacts=tuple(
                    replace(
                        artifact,
                        content_hash=sha256_file(gates.path),
                        size_bytes=gates.path.stat().st_size,
                    )
                    if artifact.artifact_type == gates.artifact_type
                    else artifact
                    for artifact in compiled.artifacts
                ),
            )

            with self.assertRaisesRegex(ValueError, "historical_gates"):
                backend.train(tampered_inputs, context)

    def test_train_rejects_recomputed_context_dataset_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            snapshot = next(
                artifact
                for artifact in compiled.artifacts
                if artifact.artifact_type == "base-dataset-snapshot"
            )
            snapshot.path.write_text("query_id,label\ntampered,3\n", encoding="utf-8")
            tampered_inputs = replace(
                compiled,
                artifacts=tuple(
                    replace(
                        artifact,
                        content_hash=sha256_file(snapshot.path),
                        size_bytes=snapshot.path.stat().st_size,
                    )
                    if artifact.artifact_type == snapshot.artifact_type
                    else artifact
                    for artifact in compiled.artifacts
                ),
            )

            with self.assertRaisesRegex(ValueError, "base dataset snapshot"):
                backend.train(tampered_inputs, context)

    def test_train_rejects_recomputed_compiled_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            report = next(
                artifact
                for artifact in compiled.artifacts
                if artifact.artifact_type == "compiled-report"
            )
            report.path.write_text("tampered report\n", encoding="utf-8")
            tampered_inputs = replace(
                compiled,
                artifacts=tuple(
                    replace(
                        artifact,
                        content_hash=sha256_file(report.path),
                        size_bytes=report.path.stat().st_size,
                    )
                    if artifact.artifact_type == report.artifact_type
                    else artifact
                    for artifact in compiled.artifacts
                ),
            )

            with self.assertRaisesRegex(ValueError, "compiled_artifact_set_hash"):
                backend.train(tampered_inputs, context)

    def test_train_rejects_compiled_semantic_metadata_changed_without_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            tampered_inputs = replace(
                compiled,
                metadata={
                    **dict(compiled.metadata),
                    "touched_domains": ("unreviewed-domain",),
                },
            )

            with self.assertRaisesRegex(ValueError, "compiled.*binding|binding.*compiled"):
                backend.train(tampered_inputs, context)

    def test_verify_rejects_candidate_touched_domains_changed_without_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            tampered_candidate = replace(
                candidate,
                metadata={**dict(candidate.metadata), "touched_domains": ()},
            )

            verification = backend.verify_artifacts(tampered_candidate, context)

        self.assertFalse(verification.valid)
        self.assertTrue(any("candidate binding" in error for error in verification.errors))

    def test_verify_rejects_recomputed_model_reference_and_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            candidate.model_path.write_text("replacement model", encoding="utf-8")
            rewritten_artifacts = tuple(
                replace(
                    artifact,
                    content_hash=sha256_file(candidate.model_path),
                    size_bytes=candidate.model_path.stat().st_size,
                )
                if artifact.artifact_type == "xgboost-model"
                else artifact
                for artifact in candidate.artifacts
            )
            tampered_candidate = replace(
                candidate,
                artifacts=rewritten_artifacts,
                metadata={
                    **dict(candidate.metadata),
                    "candidate_artifact_set_hash": _artifact_set_hash(
                        rewritten_artifacts
                    ),
                },
            )

            verification = backend.verify_artifacts(tampered_candidate, context)

        self.assertFalse(verification.valid)
        self.assertTrue(any("candidate binding" in error for error in verification.errors))

    def test_train_rejects_unexpected_in_run_artifact_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            extra = replace(
                compiled.artifacts[0],
                artifact_type="unexpected-in-run-artifact",
            )
            tampered_inputs = replace(
                compiled,
                artifacts=(*compiled.artifacts, extra),
            )

            with self.assertRaisesRegex(ValueError, "unexpected artifact type"):
                backend.train(tampered_inputs, context)

    def test_train_rejects_missing_evaluation_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            missing_anchor = replace(
                compiled,
                artifacts=tuple(
                    artifact
                    for artifact in compiled.artifacts
                    if artifact.artifact_type != "anchor-ledger-snapshot"
                ),
            )

            with self.assertRaisesRegex(ValueError, "anchor-ledger-snapshot"):
                backend.train(missing_anchor, context)

    def test_evaluate_rejects_cross_config_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            other = XGBoostRagBackend(
                replace(backend.config, code_revision="other-revision"),
                runtime=backend.runtime,
            )

            result = other.evaluate(candidate, context)

        self.assertFalse(result.artifacts_valid)
        self.assertIn("runtime_config_hash", result.details["artifact_errors"][0])

    def test_verify_rejects_semantic_metadata_size_and_loader_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, state = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            metadata_artifact = next(
                artifact
                for artifact in candidate.artifacts
                if artifact.artifact_type == "xgboost-model-metadata"
            )
            original_metadata = metadata_artifact.path.read_bytes()
            metadata = json.loads(original_metadata.decode("utf-8"))
            metadata["feature_names"] = ["term_overlap", "dense_score"]
            metadata_artifact.path.write_text(json.dumps(metadata), encoding="utf-8")
            semantic_candidate = replace(
                candidate,
                artifacts=tuple(
                    replace(
                        artifact,
                        content_hash=sha256_file(metadata_artifact.path),
                        size_bytes=metadata_artifact.path.stat().st_size,
                    )
                    if artifact.artifact_type == metadata_artifact.artifact_type
                    else artifact
                    for artifact in candidate.artifacts
                ),
            )
            self.assertFalse(backend.verify_artifacts(semantic_candidate, context).valid)

            sized_candidate = replace(
                candidate,
                artifacts=tuple(
                    replace(artifact, size_bytes=artifact.size_bytes + 1)
                    if artifact.artifact_type == "xgboost-model"
                    else artifact
                    for artifact in candidate.artifacts
                ),
            )
            self.assertFalse(backend.verify_artifacts(sized_candidate, context).valid)

            metadata_artifact.path.write_bytes(original_metadata)
            state.fail_model_load = True
            loader_verification = backend.verify_artifacts(candidate, context)
            self.assertFalse(loader_verification.valid)
            self.assertTrue(
                any(
                    "candidate model cannot be loaded" in error
                    for error in loader_verification.errors
                )
            )

    def test_train_rejects_feature_recipe_and_legacy_code_drift_after_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            backend.config.feature_recipes.path.write_text("changed recipes", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature_recipes"):
                backend.train(compiled, context)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            backend.runtime.code_manifest_files[0].write_text("legacy code v2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy_code_manifest_hash"):
                backend.train(compiled, context)

    def test_compile_train_evaluate_use_snapshots_and_raw_legacy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, runtime = self._configured_backend(root)
            context = self._context(root, base, cases)
            request = self._request(rounds=999, code_revision="attacker-controlled")

            self.assertTrue(backend.validate(request, context).valid)
            compiled = backend.compile(request, context)
            json.dumps(to_plain_data(compiled), allow_nan=False, sort_keys=True)
            self.assertEqual(compiled.metadata["compile_settings"]["min_global_test_queries"], 50)
            self.assertFalse(
                any(
                    isinstance(value, _DataFrameSentinel)
                    for value in compiled.metadata.values()
                )
            )

            candidate = backend.train(compiled, context)
            json.dumps(to_plain_data(candidate), allow_nan=False, sort_keys=True)
            self.assertIn(("train", 7), runtime.calls)
            self.assertIn(("merge", None), runtime.calls)
            self.assertEqual(
                {
                    "compiled-base-dataset",
                    "compiled-production-cases",
                    "historical-gates-snapshot",
                    "anchor-ledger-snapshot",
                },
                {
                    artifact.artifact_type
                    for artifact in candidate.artifacts
                }
                & {
                    "compiled-base-dataset",
                    "compiled-production-cases",
                    "historical-gates-snapshot",
                    "anchor-ledger-snapshot",
                },
            )
            self.assertFalse((context.run_dir / ".heuriboost" / "current_model.json").exists())
            self.assertFalse(any(path.name == "current_model.json" for path in context.run_dir.rglob("*")))

            evaluation = backend.evaluate(candidate, context)
            self.assertEqual(evaluation.global_metrics["mrr@10"], 0.0)
            self.assertEqual(evaluation.touched_domains["medical"]["mrr@10"], 0.0)
            self.assertTrue(evaluation.current_cases_passed)
            self.assertTrue(evaluation.historical_gates_passed)
            self.assertTrue(evaluation.artifacts_valid)
            self.assertIn(("evaluate_split", "test"), runtime.calls)
            self.assertIn(("evaluate_domain", "test"), runtime.calls)
            self.assertIn(("evaluate_cases", "full"), runtime.calls)

    def test_compiled_artifacts_can_be_snapshotted_by_local_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            store_root = root / "artifact-store"
            store_root.mkdir()
            store = LocalArtifactStore(store_root)
            context = RunContext(
                run_id="run-1",
                run_dir=store.run_dir("run-1"),
                datasets={
                    "base": self._dataset("base", "base", base),
                    "production_cases": self._dataset(
                        "cases", "production_cases", cases
                    ),
                },
                options={},
            )

            compiled = backend.compile(self._request(), context)
            manifest = store.complete_stage(
                context.run_id,
                "compile",
                "input-hash",
                {artifact.artifact_type: artifact.path for artifact in compiled.artifacts},
            )
            resumed = CompiledInputs(
                artifacts=tuple(
                    replace(artifact, path=store.root / artifact.path)
                    for artifact in manifest.artifacts
                ),
                metadata=compiled.metadata,
            )
            candidate = backend.train(resumed, context)
            trained_manifest = store.complete_stage(
                context.run_id,
                "TRAINED",
                "trained-input-hash",
                {
                    artifact.artifact_type: artifact.path
                    for artifact in candidate.artifacts
                },
            )
            restored_artifacts = tuple(
                replace(artifact, path=store.root / artifact.path)
                for artifact in trained_manifest.artifacts
            )
            restored_model = next(
                artifact
                for artifact in restored_artifacts
                if artifact.artifact_type == "xgboost-model"
            )
            restored_candidate = CandidateModel(
                model_path=restored_model.path,
                artifacts=restored_artifacts,
                metadata=candidate.metadata,
            )
            verification = backend.verify_artifacts(restored_candidate, context)

        self.assertEqual(
            {artifact.artifact_type for artifact in manifest.artifacts},
            {artifact.artifact_type for artifact in compiled.artifacts},
        )
        self.assertEqual(
            {
                (artifact.artifact_type, artifact.content_hash, artifact.size_bytes)
                for artifact in manifest.artifacts
            },
            {
                (artifact.artifact_type, artifact.content_hash, artifact.size_bytes)
                for artifact in compiled.artifacts
            },
        )
        self.assertTrue(
            all(
                not any(part.startswith(".") for part in artifact.path.parts)
                for artifact in compiled.artifacts
            )
        )
        compiled_outputs = tuple(
            artifact
            for artifact in compiled.artifacts
            if artifact.artifact_type
            in {
                "compiled-base-dataset",
                "compiled-regression-cases",
                "compiled-production-cases",
                "compiled-current-case-set",
                "compiled-report",
            }
        )
        self.assertEqual(5, len(compiled_outputs))
        self.assertTrue(
            all(
                artifact.path.parent == context.run_dir / "compiled-artifacts"
                for artifact in compiled_outputs
            )
        )
        self.assertEqual(
            {artifact.artifact_type for artifact in trained_manifest.artifacts},
            {artifact.artifact_type for artifact in candidate.artifacts},
        )
        self.assertEqual(restored_candidate.model_path, restored_model.path)
        self.assertNotEqual(restored_candidate.model_path, candidate.model_path)
        self.assertTrue(verification.valid)

    def test_evaluate_preserves_all_finite_legacy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            evaluation = backend.evaluate(candidate, context)

        self.assertEqual(evaluation.global_metrics["ndcg@5"], 0.79)
        self.assertEqual(evaluation.global_metrics["recall@10"], 0.92)
        self.assertEqual(evaluation.global_metrics["query_count"], 7.0)
        self.assertEqual(
            evaluation.details["raw_domain_metrics"]["medical"][
                "hard_negative_rate@10"
            ],
            0.2,
        )

    def test_verify_requires_context_for_configured_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)

            with self.assertRaises(TypeError):
                backend.verify_artifacts(candidate)

    def test_verify_rejects_candidate_artifact_outside_context_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            candidate = backend.train(backend.compile(self._request(), context), context)
            outside = root / "outside-model.json"
            outside.write_text("outside", encoding="utf-8")
            escaped_candidate = replace(
                candidate,
                artifacts=tuple(
                    replace(
                        artifact,
                        path=outside,
                        content_hash=sha256_file(outside),
                        size_bytes=outside.stat().st_size,
                    )
                    if artifact.artifact_type == "xgboost-model"
                    else artifact
                    for artifact in candidate.artifacts
                ),
                model_path=outside,
            )

            verification = backend.verify_artifacts(escaped_candidate, context)

        self.assertFalse(verification.valid)
        self.assertTrue(
            any("outside context.run_dir" in error for error in verification.errors)
        )

    def test_verify_artifacts_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.csv"
            cases = root / "cases.jsonl"
            base.write_text("query_id,label\nq,3\n", encoding="utf-8")
            cases.write_text('{"case_id":"case"}\n', encoding="utf-8")
            backend, _ = self._configured_backend(root)
            context = self._context(root, base, cases)
            compiled = backend.compile(self._request(), context)
            candidate = backend.train(compiled, context)

            self.assertTrue(backend.verify_artifacts(candidate, context).valid)
            candidate.model_path.write_text("tampered", encoding="utf-8")
            verification = backend.verify_artifacts(candidate, context)

        self.assertFalse(verification.valid)
        self.assertTrue(any("hash mismatch" in error for error in verification.errors))


if __name__ == "__main__":
    unittest.main()
