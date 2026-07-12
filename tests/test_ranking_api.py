import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from heuriboost_rag.backends import ranking
from heuriboost_rag.backends.ranking import (
    evaluate_xgboost_ranker,
    predict_xgboost_ranker,
    train_xgboost_ranker,
)


LEGACY_RANKING_API_PATH = (
    Path(__file__).resolve().parent.parent
    / "plugins"
    / "heuriboost"
    / "skills"
    / "heuriboost-rag"
    / "scripts"
    / "ranking_api.py"
)


def _load_legacy_ranking_api():
    spec = importlib.util.spec_from_file_location("legacy_ranking_api", LEGACY_RANKING_API_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load legacy ranking API: {LEGACY_RANKING_API_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RankingApiTests(unittest.TestCase):
    def test_train_and_evaluate_feature_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"

            result = train_xgboost_ranker(
                train_features=[[3.0, 1.0], [0.0, 0.0], [2.5, 1.0], [0.2, 0.0]],
                train_labels=[3, 0, 3, 0],
                train_groups=[2, 2],
                validation_features=[[2.0, 1.0], [0.1, 0.0]],
                validation_labels=[2, -1],
                validation_groups=[2],
                output_dir=output_dir,
                feature_names=["score", "name_hit"],
                metadata={"feature_set_name": "unit_test_features", "rounds": 99},
                rounds=2,
            )

            model_path = Path(result["model_path"])
            metadata_path = Path(result["metadata_path"])
            self.assertTrue(model_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(model_path.relative_to(output_dir).as_posix(), "models/reranker.json")
            self.assertEqual(
                metadata_path.relative_to(output_dir).as_posix(),
                "models/reranker_metadata.json",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["feature_set_name"], "unit_test_features")
            self.assertEqual(metadata["feature_names"], ["score", "name_hit"])
            self.assertEqual(metadata["rounds"], 2)
            self.assertEqual(metadata["train_rows"], 4)
            self.assertEqual(metadata["validation_rows"], 2)
            self.assertEqual(metadata["train_groups"], 2)
            self.assertEqual(metadata["validation_groups"], 1)
            self.assertFalse(metadata["labels_are_mapped"])

            metrics = evaluate_xgboost_ranker(
                features=[[2.0, 1.0], [0.1, 0.0]],
                labels=[2, -1],
                groups=[2],
                model_path=model_path,
                split="validation",
                query_group_order=["q_validation"],
            )

            self.assertEqual(metrics["rows"], 2)
            self.assertEqual(metrics["query_groups"], 1)
            self.assertEqual(metrics["split"], "validation")
            self.assertEqual(metrics["query_group_order"], ["q_validation"])
            self.assertEqual(metrics["group_sizes"], [2])
            self.assertIn("ndcg@10", metrics)
            self.assertIn("mrr@10", metrics)

    def test_predict_xgboost_ranker_uses_metadata_feature_names_and_preserves_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            features = [[3.0, 1.0], [0.0, 0.0], [2.5, 1.0], [0.2, 0.0]]
            result = train_xgboost_ranker(
                train_features=features,
                train_labels=[3, 0, 3, 0],
                train_groups=[2, 2],
                validation_features=[[2.0, 1.0], [0.1, 0.0]],
                validation_labels=[2, -1],
                validation_groups=[2],
                output_dir=output_dir,
                feature_names=["score", "name_hit"],
                rounds=2,
            )

            model_path = Path(result["model_path"])
            predictions = predict_xgboost_ranker(features=features, model_path=model_path)
            one_row_predictions = [
                predict_xgboost_ranker(features=[row], model_path=model_path)[0] for row in features
            ]

            self.assertEqual(len(predictions), len(features))
            self.assertTrue(all(isinstance(score, float) and math.isfinite(score) for score in predictions))
            for prediction, one_row_prediction in zip(predictions, one_row_predictions):
                self.assertAlmostEqual(prediction, one_row_prediction, places=7)

    def test_mrr_at_10_is_group_aware_for_mapped_relevance_labels(self):
        mapped_labels = ranking.map_relevance_labels([3, -1, 2, -1, -1, 1])

        score = ranking._mrr_at_10(
            labels=mapped_labels,
            groups=[3, 3],
            predictions=[0.2, 0.9, 0.1, 0.9, 0.5, 0.1],
        )

        self.assertAlmostEqual(score, (1 / 2 + 1 / 3) / 2)

    def test_legacy_script_reexports_package_functions(self):
        legacy_ranking_api = _load_legacy_ranking_api()

        self.assertIs(legacy_ranking_api.train_xgboost_ranker, train_xgboost_ranker)
        self.assertIs(legacy_ranking_api.evaluate_xgboost_ranker, evaluate_xgboost_ranker)
        self.assertIs(legacy_ranking_api.predict_xgboost_ranker, predict_xgboost_ranker)
        self.assertEqual(
            legacy_ranking_api.__all__,
            [
                "train_xgboost_ranker",
                "evaluate_xgboost_ranker",
                "predict_xgboost_ranker",
            ],
        )


if __name__ == "__main__":
    unittest.main()
