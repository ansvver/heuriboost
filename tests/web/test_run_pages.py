from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.reckless.contracts import RepairRequest
from heuriboost_rag.reckless.errors import InputBlockedError
from heuriboost_rag.reckless.state import RunState
from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class RunPageTests(unittest.TestCase):
    def test_runs_page_lists_recent_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            run = app.state.store.runs.create(
                RepairRequest("workspace", "base", "cases", "1", "test", "tester"),
                "policy",
                "input",
            )
            with TestClient(app) as client:
                response = client.get("/runs")

            self.assertEqual(response.status_code, 200)
            self.assertIn(run.run_id, response.text)
            self.assertIn("RECEIVED", response.text)

    def test_page_renders_all_stages_and_status_without_relying_on_color(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            run = app.state.store.runs.create(
                RepairRequest("workspace", "base", "cases", "1", "test", "tester"),
                "policy",
                "input",
            )
            app.state.store.runs.transition(
                run.run_id,
                RunState.VALIDATING,
                metadata={
                    "validation": {
                        "valid": True,
                        "warnings": [],
                        "metadata": {"feature_set_name": "prod_recog_xgb"},
                    }
                },
            )
            app.state.store.runs.transition(
                run.run_id,
                RunState.COMPILED,
                metadata={
                    "compiled_stage": {
                        "stage": "COMPILED",
                        "artifact_set_hash": "compiled-hash",
                        "artifacts": [
                            {
                                "artifact_type": "compiled-training-csv",
                                "path": "runs/run-1/stages/COMPILED/artifacts/training.csv",
                                "size_bytes": 123,
                            }
                        ],
                    }
                },
            )
            app.state.store.runs.transition(run.run_id, RunState.TRAINING)
            app.state.store.runs.transition(
                run.run_id,
                RunState.TRAINED,
                metadata={
                    "trained_stage": {
                        "stage": "TRAINED",
                        "artifact_set_hash": "trained-hash",
                        "artifacts": [
                            {"artifact_type": "xgboost-model", "path": "runs/run-1/model.ubj", "size_bytes": 456}
                        ],
                    }
                },
            )
            app.state.store.runs.transition(run.run_id, RunState.EVALUATING)
            app.state.store.runs.transition(
                run.run_id,
                RunState.SYNTHESIZING_FEATURES,
                metadata={
                    "feature_synthesis_stage": {
                        "stage": "FEATURE_SYNTHESIS",
                        "features_added": ["llm_heuriboost_case_memory_score"],
                        "artifact_set_hash": "synthesis-hash",
                        "artifacts": [
                            {
                                "artifact_type": "feature-synthesis-plan",
                                "path": "runs/run-1/feature-synthesis.json",
                                "size_bytes": 789,
                            }
                        ],
                    }
                },
            )
            app.state.store.runs.transition(run.run_id, RunState.TRAINING)
            app.state.store.runs.transition(
                run.run_id,
                RunState.TRAINED,
                metadata={
                    "trained_retry_stage": {
                        "stage": "TRAINED_RETRY",
                        "artifact_set_hash": "retry-hash",
                        "artifacts": [
                            {
                                "artifact_type": "xgboost-model-metadata",
                                "path": "runs/run-1/retry-metadata.json",
                                "size_bytes": 321,
                            }
                        ],
                    }
                },
            )
            app.state.store.runs.transition(run.run_id, RunState.EVALUATING)
            app.state.store.runs.transition(run.run_id, RunState.REPORTING)
            app.state.store.runs.transition(
                run.run_id,
                RunState.READY_FOR_PROMOTION,
                metadata={
                    "report_evidence": {
                        "stage": "REPORTING",
                        "artifact_type": "report-evidence",
                        "path": "runs/run-1/report-evidence.json",
                        "content_hash": "report-hash",
                    }
                },
            )
            app.state.store.runs.transition(run.run_id, RunState.PROMOTING)
            promoted = app.state.store.runs.transition(
                run.run_id,
                RunState.PROMOTED,
                metadata={
                    "promotion": {
                        "current_model": run.run_id,
                        "release_manifest_hash": "release-hash",
                    }
                },
            )
            with TestClient(app) as client:
                response = client.get(f"/runs/{promoted.run_id}")

            self.assertEqual(response.status_code, 200)
            self.assertIn("PROMOTED", response.text)
            for stage in ("校验", "编译", "训练", "评测", "特征合成", "重训", "报告", "批准", "Promote"):
                self.assertIn(stage, response.text)
            for anchor in (
                "stage-validation",
                "stage-compile",
                "stage-training",
                "stage-evaluation",
                "stage-feature-synthesis",
                "stage-training-retry",
                "stage-report",
                "stage-approval",
                "stage-promote",
            ):
                self.assertIn(f'href="#{anchor}"', response.text)
                self.assertIn(f'id="{anchor}"', response.text)
            self.assertIn("feature-synthesis-plan", response.text)
            self.assertIn("llm_heuriboost_case_memory_score", response.text)
            self.assertIn("xgboost-model-metadata", response.text)
            self.assertIn(f"/runs/{promoted.run_id}/report", response.text)

    def test_run_detail_extracts_llm_feature_names_from_stage_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            run = app.state.store.runs.create(
                RepairRequest("workspace", "base", "cases", "1", "test", "tester"),
                "policy",
                "input",
            )
            artifact_path = (
                Path(tmp)
                / "artifacts"
                / "runs"
                / run.run_id
                / "stages"
                / "FEATURE_SYNTHESIS"
                / "artifacts"
                / "feature-synthesis-specs-test.snapshot"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                '{"features":[{"name":"llm_heuriboost_case_memory_score","entries":[]}]}',
                encoding="utf-8",
            )
            app.state.store.runs.transition(run.run_id, RunState.VALIDATING)
            app.state.store.runs.transition(run.run_id, RunState.COMPILED)
            app.state.store.runs.transition(run.run_id, RunState.TRAINING)
            app.state.store.runs.transition(run.run_id, RunState.TRAINED)
            app.state.store.runs.transition(run.run_id, RunState.EVALUATING)
            synthesized = app.state.store.runs.transition(
                run.run_id,
                RunState.SYNTHESIZING_FEATURES,
                metadata={
                    "feature_synthesis_stage": {
                        "stage": "FEATURE_SYNTHESIS",
                        "artifacts": [
                            {
                                "artifact_type": "feature-synthesis-specs",
                                "path": f"runs/{run.run_id}/stages/FEATURE_SYNTHESIS/artifacts/feature-synthesis-specs-test.snapshot",
                                "size_bytes": artifact_path.stat().st_size,
                            }
                        ],
                    }
                },
            )

            with TestClient(app) as client:
                response = client.get(f"/runs/{synthesized.run_id}")

            self.assertEqual(response.status_code, 200)
            self.assertIn("llm_heuriboost_case_memory_score", response.text)

    def test_blocked_run_page_shows_error_summary_and_operator_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WebConfig.for_test(Path(tmp)))
            run = app.state.store.runs.create(
                RepairRequest("workspace", "base", "cases", "1", "test", "tester"),
                "policy",
                "input",
            )
            validating = app.state.store.runs.transition(run.run_id, RunState.VALIDATING)
            blocked = app.state.store.runs.fail(
                validating.run_id,
                RunState.BLOCKED_INPUT,
                InputBlockedError(
                    "backend validation rejected the repair input",
                    stage="VALIDATING",
                    details={"metadata": {"errors": ("base test split is empty",)}},
                    operator_action="Correct the input data and create a new run.",
                ),
            )
            with TestClient(app) as client:
                response = client.get(f"/runs/{blocked.run_id}")

            self.assertEqual(response.status_code, 200)
            self.assertIn("base test split is empty", response.text)
            self.assertIn("Correct the input data", response.text)


if __name__ == "__main__":
    unittest.main()
