from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest

from heuriboost_rag.reckless.errors import ArtifactIntegrityError
from heuriboost_rag.reckless.contracts import RepairRequest, RunRecord
from heuriboost_rag.reckless.report import (
    build_localized_view,
    build_report_data,
    render_pre_promote_report,
    render_run_pre_promote_report,
)
from heuriboost_rag.reckless.storage import LocalArtifactStore


def sample_report_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"run_id": "run-1", "state": "READY_FOR_PROMOTION"},
        "decision": {
            "status": "READY_FOR_PROMOTION",
            "promotion_eligible": True,
            "blockers": [],
        },
        "data_lineage": {
            "base_dataset_id": "base-v1",
            "production_cases_id": "cases-v2",
        },
        "operator_summary": {
            "headline": "本次运行可晋级",
            "reason": "硬门禁全部通过，候选模型可以进入人工批准 Promote。",
            "promotion_eligible": True,
        },
        "data_overview": {
            "datasets": [
                {"role": "base", "dataset_id": "base-v1", "rows": 4, "purpose": "训练和验证基础排序能力"},
                {"role": "production_cases", "dataset_id": "cases-v2", "rows": 2, "purpose": "复核当前线上坏例"},
            ],
            "split": {"train_rows": 4, "validation_rows": 2, "test_rows": 2},
        },
        "process_timeline": {
            "kind": "gantt",
            "steps": [
                {"id": "validate", "status": "passed", "title": "校验"},
                {"id": "compile", "status": "completed", "title": "编译"},
                {"id": "train", "status": "completed", "title": "训练"},
                {"id": "feature_synthesis", "status": "used", "title": "LLM 特征挖掘"},
                {"id": "retry_train", "status": "completed", "title": "带新特征重训"},
                {"id": "report", "status": "completed", "title": "报告"},
            ],
        },
        "metric_summary": {
            "global_metrics": {"ndcg@10": 1.0, "mrr@10": 1.0},
            "training": {"objective": "rank:ndcg", "rounds": 2, "eval_metric": "ndcg@10", "seed": 7},
            "gate_summary": {"total": 1, "passed": 1, "failed": 0},
        },
        "deposits": [
            {"kind": "model", "title": "候选模型", "content_hash": "model-hash"},
            {"kind": "report", "title": "晋级前报告", "content_hash": "decision-hash"},
        ],
        "validation": {"passed": True, "warnings": []},
        "compilation": {"train_rows": 4, "validation_rows": 2, "test_rows": 2},
        "training": {
            "objective": "rank:ndcg",
            "rounds": 2,
            "model_ref": {
                "artifact_type": "model",
                "content_hash": "model-hash",
                "size_bytes": 12,
            },
        },
        "evaluation": {"ndcg@10": 1.0, "mrr@10": 1.0},
        "features": {
            "total_count": 2,
            "base_feature_names": ["feature_a"],
            "llm_feature_names": ["llm_heuriboost_case_memory_score"],
            "synthesized_features": [
                {
                    "name": "llm_heuriboost_case_memory_score",
                    "kind": "case_memory_score",
                    "description": "Case-memory feature generated from reviewed production outcomes.",
                    "entry_count": 3,
                    "positive_entries": 1,
                    "negative_entries": 1,
                    "neutral_entries": 1,
                }
            ],
        },
        "explainability": {
            "model": {"objective": "rank:ndcg", "rounds": 2, "eval_metric": "ndcg@10", "seed": 7},
            "feature_groups": [
                {"group": "LLM synthesized", "count": 1, "features": ["llm_heuriboost_case_memory_score"]}
            ],
            "notes": ["Feature importance was not sealed; explanations are derived from sealed feature metadata."],
        },
        "gate_checks": [{"check_id": "historical_gates", "passed": True}],
        "warnings": [],
        "artifacts": [
            {
                "artifact_type": "model",
                "content_hash": "model-hash",
                "size_bytes": 12,
            }
        ],
        "reproducibility": {
            "policy_hash": "policy-hash",
            "code_revision": "revision",
            "decision_hash": "decision-hash",
        },
    }


def sample_sealed_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"run_id": "run-1"},
        "request": {
            "workspace_id": "workspace",
            "base_dataset_id": "base-v1",
            "production_cases_id": "cases-v2",
            "policy_version": "1",
            "backend_name": "fake",
            "requested_by": "tester",
            "run_options": {},
        },
        "policy": {"version": 1, "content_hash": "policy-hash"},
        "input": {
            "input_hash": "input-hash",
            "base_dataset_id": "base-v1",
            "production_cases_id": "cases-v2",
        },
        "outcome": {
            "state": "READY_FOR_PROMOTION",
            "promotion_eligible": True,
            "acceptance_level": "full",
        },
        "datasets": {
            "base": {
                "dataset_id": "base-v1",
                "role": "base",
                "content_hash": "base-hash",
                "metadata": {
                    "rows": 4,
                    "columns": ["query_id", "query", "doc_id", "text", "relevance", "split"],
                },
            },
            "production_cases": {
                "dataset_id": "cases-v2",
                "role": "production_cases",
                "content_hash": "cases-hash",
                "metadata": {
                    "rows": 2,
                    "columns": ["case_id", "query", "shown_doc_id", "shown_doc_text", "user_verdict"],
                },
            },
        },
        "execution_identity": {
            "backend_version": "backend-v1",
            "feature_names": ["feature_a"],
            "feature_version": "features-v1",
            "code_commit": "revision",
            "training_params": {"rounds": 2},
            "random_seed": 7,
        },
        "validation": {"valid": True, "metadata": {}, "warnings": []},
        "compilation": {
            "artifact_refs": [
                {
                    "artifact_type": "compiled-training-csv",
                    "content_hash": "compiled-train-hash",
                    "size_bytes": 123,
                }
            ],
            "metadata": {
                "base_rows": 4,
                "production_case_rows": 2,
                "train_rows": 4,
                "validation_rows": 2,
                "test_rows": 2,
                "llm_feature_specs": [
                    {
                        "name": "llm_heuriboost_case_memory_score",
                        "kind": "case_memory_score",
                        "description": "Case-memory feature generated from reviewed production outcomes.",
                        "entries": [
                            {"score": 1.0},
                            {"score": -1.0},
                            {"score": 0.0},
                        ],
                    }
                ],
            }
        },
        "training": {
            "metadata": {
                "rounds": 2,
                "feature_names": ["feature_a", "llm_heuriboost_case_memory_score"],
                "llm_feature_names": ["llm_heuriboost_case_memory_score"],
                "params": {"objective": "rank:ndcg", "eval_metric": "ndcg@10", "seed": 7},
                "dataset_summary": {"rows": 10, "query_groups": 2},
            },
            "model_ref": {
                "artifact_type": "xgboost-model",
                "content_hash": "model-hash",
                "size_bytes": 12,
            },
        },
        "evaluation": {"global_metrics": {"ndcg@10": 1.0, "mrr@10": 1.0}},
        "feature_synthesis": {
            "attempted": True,
            "initial_decision": {
                "promotion_eligible": False,
                "blockers": ["current_production_cases"],
            },
            "initial_evaluation": {"global_metrics": {"ndcg@10": 0.75, "mrr@10": 0.5}},
            "metadata": {
                "provider": "llm",
                "llm_feature_names": ["llm_heuriboost_case_memory_score"],
                "initial_blockers": ["current_production_cases"],
            },
            "stage": {
                "stage": "FEATURE_SYNTHESIS",
                "artifacts": [
                    {
                        "artifact_type": "feature-synthesis-plan",
                        "content_hash": "feature-plan-hash",
                        "size_bytes": 45,
                    }
                ],
            },
        },
        "decision": {
            "promotion_eligible": True,
            "acceptance_level": "full",
            "checks": [{"check_id": "historical_gates", "passed": True}],
            "blockers": [],
            "warnings": [],
        },
        "warnings": [],
        "artifacts": [
            {
                "stage": "COMPILED",
                "artifact_type": "compiled-training-csv",
                "content_hash": "compiled-train-hash",
                "size_bytes": 123,
            },
            {
                "stage": "TRAINED",
                "artifact_type": "xgboost-model",
                "content_hash": "model-hash",
                "size_bytes": 12,
            }
        ],
        "completed_stage_manifests": [
            {"stage": "COMPILED", "artifacts": []},
            {"stage": "TRAINED", "artifacts": []},
            {"stage": "FEATURE_SYNTHESIS", "artifacts": []},
            {"stage": "TRAINED_RETRY", "artifacts": []},
            {"stage": "REPORTING", "artifacts": []},
        ],
        "component_hashes": {"decision": "decision-hash"},
    }


class JsonScriptParser(HTMLParser):
    def __init__(self, script_id: str) -> None:
        super().__init__()
        self.script_id = script_id
        self.capture = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.capture = tag == "script" and attributes.get("id") == self.script_id

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.capture = False


def extract_json_script(html: str, script_id: str) -> object:
    parser = JsonScriptParser(script_id)
    parser.feed(html)
    return json.loads("".join(parser.parts))


class PrePromoteReportTests(unittest.TestCase):
    def test_report_contains_decision_training_lineage_and_machine_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_pre_promote_report(
                sample_report_data(),
                Path(tmp),
                locale="zh-CN",
            )
            html = result.html_path.read_text(encoding="utf-8")
            data_bytes = result.data_path.read_bytes()
            embedded = extract_json_script(html, "heuriboost-pre-promote-data")

            self.assertEqual(
                embedded,
                json.loads(data_bytes.decode("utf-8")),
            )
            self.assertEqual(embedded["decision"]["status"], "READY_FOR_PROMOTION")
            self.assertIn("training", embedded)
            self.assertIn("data_lineage", embedded)
            self.assertIn("gate_checks", embedded)
            self.assertEqual(result.data_hash, hashlib.sha256(data_bytes).hexdigest())
            self.assertEqual(result.data_hash, result.manifest["data_hash"])
            self.assertEqual(result.manifest["decision_hash"], "decision-hash")
            self.assertEqual(result.manifest["model_hash"], "model-hash")
            self.assertEqual(result.manifest["policy_hash"], "policy-hash")

    def test_locale_switch_does_not_change_numeric_data(self):
        zh = build_localized_view(sample_report_data(), "zh-CN")
        en = build_localized_view(sample_report_data(), "en")

        self.assertEqual(zh["raw_data"], en["raw_data"])
        self.assertEqual(zh["labels"]["title"], "晋级前报告")
        self.assertEqual(en["labels"]["title"], "Pre Promote Report")

    def test_build_report_data_uses_only_sealed_evidence(self):
        report_data = build_report_data(sample_sealed_evidence())

        self.assertEqual(report_data["run"]["run_id"], "run-1")
        self.assertEqual(report_data["run"]["state"], "READY_FOR_PROMOTION")
        self.assertEqual(report_data["decision"]["status"], "READY_FOR_PROMOTION")
        self.assertEqual(report_data["training"]["model_ref"]["content_hash"], "model-hash")
        self.assertEqual(report_data["reproducibility"]["policy_hash"], "policy-hash")
        self.assertEqual(report_data["features"]["total_count"], 2)
        self.assertEqual(report_data["features"]["llm_feature_names"], ["llm_heuriboost_case_memory_score"])
        self.assertEqual(report_data["features"]["synthesized_features"][0]["positive_entries"], 1)
        self.assertEqual(report_data["features"]["synthesized_features"][0]["negative_entries"], 1)
        feature_details = {item["name"]: item for item in report_data["features"]["feature_details"]}
        self.assertEqual(feature_details["feature_a"]["source"], "built-in")
        self.assertIn("模型输入特征", feature_details["feature_a"]["description"])
        self.assertEqual(feature_details["llm_heuriboost_case_memory_score"]["source"], "LLM synthesized")
        self.assertIn("Case-memory feature", feature_details["llm_heuriboost_case_memory_score"]["description"])
        self.assertEqual(feature_details["llm_heuriboost_case_memory_score"]["entry_count"], 3)
        self.assertEqual(report_data["explainability"]["model"]["objective"], "rank:ndcg")
        self.assertEqual(report_data["explainability"]["feature_groups"][-1]["group"], "LLM synthesized")
        self.assertIn("可晋级", report_data["operator_summary"]["headline"])
        self.assertIn("硬门禁", report_data["operator_summary"]["reason"])
        self.assertEqual(
            [dataset["role"] for dataset in report_data["data_overview"]["datasets"]],
            ["base", "production_cases"],
        )
        self.assertEqual(report_data["data_overview"]["datasets"][0]["rows"], 4)
        self.assertEqual(report_data["data_overview"]["split"]["train_rows"], 4)
        timeline_ids = [step["id"] for step in report_data["process_timeline"]["steps"]]
        self.assertIn("validate", timeline_ids)
        self.assertIn("feature_synthesis", timeline_ids)
        self.assertIn("retry_train", timeline_ids)
        self.assertEqual(report_data["metric_summary"]["global_metrics"]["ndcg@10"], 1.0)
        self.assertEqual(report_data["metric_summary"]["comparison"][0]["before"], 0.75)
        self.assertEqual(report_data["metric_summary"]["comparison"][0]["after"], 1.0)
        self.assertEqual(report_data["metric_summary"]["gate_summary"], {"total": 1, "passed": 1, "failed": 0})
        self.assertEqual(report_data["metric_summary"]["training"]["rounds"], 2)
        deposit_kinds = {deposit["kind"] for deposit in report_data["deposits"]}
        self.assertIn("model", deposit_kinds)
        self.assertIn("compiled_dataset", deposit_kinds)
        self.assertIn("feature_synthesis", deposit_kinds)
        self.assertIn("report", deposit_kinds)

    def test_missing_required_data_is_rejected_before_any_file_is_written(self):
        invalid = sample_report_data()
        del invalid["training"]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaises(ArtifactIntegrityError):
                render_pre_promote_report(invalid, output_dir)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_identical_render_is_idempotent_and_changed_render_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = render_pre_promote_report(sample_report_data(), output_dir)
            second = render_pre_promote_report(sample_report_data(), output_dir)
            changed = deepcopy(sample_report_data())
            changed["training"]["rounds"] = 3

            self.assertEqual(first, second)
            with self.assertRaises(ArtifactIntegrityError):
                render_pre_promote_report(changed, output_dir)

    def test_render_from_run_requires_verified_sealed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            source = root / "report-evidence.json"
            source.write_text(
                json.dumps(sample_sealed_evidence(), sort_keys=True),
                encoding="utf-8",
            )
            manifest = store.complete_stage(
                "run-1",
                "REPORTING",
                "input-hash",
                {"report-evidence": source},
            )
            evidence_ref = manifest.artifacts[0]
            run = RunRecord(
                run_id="run-1",
                state="READY_FOR_PROMOTION",
                version=1,
                request=RepairRequest(
                    workspace_id="workspace",
                    base_dataset_id="base-v1",
                    production_cases_id="cases-v2",
                    policy_version="1",
                    backend_name="fake",
                    requested_by="tester",
                ),
                policy_hash="policy-hash",
                input_hash="input-hash",
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

            report = render_run_pre_promote_report(run, store)
            self.assertTrue(report.html_path.is_file())
            snapshot = store.root / evidence_ref.path
            snapshot.chmod(0o600)
            snapshot.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                render_run_pre_promote_report(run, store)

    def test_render_from_run_rejects_evidence_for_a_different_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            evidence = sample_sealed_evidence()
            evidence["run"]["run_id"] = "other-run"
            source = root / "report-evidence.json"
            source.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
            evidence_ref = store.complete_stage(
                "run-1",
                "REPORTING",
                "input-hash",
                {"report-evidence": source},
            ).artifacts[0]
            run = RunRecord(
                run_id="run-1",
                state="READY_FOR_PROMOTION",
                version=1,
                request=RepairRequest(
                    workspace_id="workspace",
                    base_dataset_id="base-v1",
                    production_cases_id="cases-v2",
                    policy_version="1",
                    backend_name="fake",
                    requested_by="tester",
                ),
                policy_hash="policy-hash",
                input_hash="input-hash",
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

            with self.assertRaises(ArtifactIntegrityError):
                render_run_pre_promote_report(run, store)


if __name__ == "__main__":
    unittest.main()
