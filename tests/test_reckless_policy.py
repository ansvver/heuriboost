from dataclasses import FrozenInstanceError
import json
import tempfile
from pathlib import Path
import unittest

import yaml

from heuriboost_rag.reckless.contracts import EvaluationResult, to_plain_data
from heuriboost_rag.reckless.policy import (
    EvaluationPolicy,
    InputPolicy,
    PromotionPolicy,
    RecklessPolicy,
    evaluate_promotion_eligibility,
    load_policy,
)


APPROVED_POLICY_TEMPLATE = """version: 1
acceptance_level: full
input:
  min_global_test_queries: 50
  min_domain_test_queries: 10
  min_docs_per_query: 2
  require_authoritative_labels: true
evaluation:
  require_all_current_cases: true
  require_all_historical_gates: true
  require_global_ndcg_improvement: true
  require_global_mrr_improvement: true
  allow_touched_domain_regression: false
promotion:
  allow_weak: false
  require_explicit_human_approval: true
  allow_anchor_reset: false
  allow_gate_retirement: false
"""


class RecklessPolicyTests(unittest.TestCase):
    def load_text(self, text: str) -> RecklessPolicy:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.yml"
            path.write_text(text, encoding="utf-8")
            return load_policy(path)

    def test_default_policy_forbids_weak_promotion(self):
        policy = RecklessPolicy.default()
        self.assertEqual(policy.acceptance_level, "full")
        self.assertFalse(policy.promotion.allow_weak)
        self.assertTrue(policy.promotion.require_explicit_human_approval)

    def test_policy_hash_is_stable_for_equivalent_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.yml"
            second = Path(tmp) / "second.yml"
            first.write_text("version: 1\nacceptance_level: full\n", encoding="utf-8")
            second.write_text("acceptance_level: full\nversion: 1\n", encoding="utf-8")
            self.assertEqual(load_policy(first).content_hash, load_policy(second).content_hash)

    def test_default_policy_has_exact_documented_fields(self):
        policy = RecklessPolicy.default()
        self.assertEqual(policy.version, 1)
        self.assertEqual(
            policy.input,
            InputPolicy(
                min_global_test_queries=50,
                min_domain_test_queries=10,
                min_docs_per_query=2,
                require_authoritative_labels=True,
            ),
        )
        self.assertEqual(
            policy.evaluation,
            EvaluationPolicy(
                require_all_current_cases=True,
                require_all_historical_gates=True,
                require_global_ndcg_improvement=True,
                require_global_mrr_improvement=True,
                allow_touched_domain_regression=False,
            ),
        )
        self.assertEqual(
            policy.promotion,
            PromotionPolicy(
                allow_weak=False,
                require_explicit_human_approval=True,
                allow_anchor_reset=False,
                allow_gate_retirement=False,
            ),
        )

    def test_policy_dataclasses_are_frozen(self):
        instances_and_fields = (
            (InputPolicy(), "min_global_test_queries"),
            (EvaluationPolicy(), "require_all_current_cases"),
            (PromotionPolicy(), "allow_weak"),
            (RecklessPolicy.default(), "acceptance_level"),
        )
        for instance, field_name in instances_and_fields:
            with self.subTest(type=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, None)

    def test_unknown_top_level_and_nested_keys_are_rejected(self):
        invalid_policies = (
            "unknown: true\n",
            "input:\n  unknown: true\n",
            "evaluation:\n  unknown: true\n",
            "promotion:\n  unknown: true\n",
        )
        for text in invalid_policies:
            with self.subTest(policy=text):
                with self.assertRaises(ValueError):
                    self.load_text(text)

    def test_duplicate_top_level_and_nested_keys_are_rejected(self):
        invalid_policies = (
            "version: 1\nversion: 1\n",
            (
                "input:\n"
                "  min_global_test_queries: 50\n"
                "  min_global_test_queries: 60\n"
            ),
        )
        for text in invalid_policies:
            with self.subTest(policy=text):
                with self.assertRaises(yaml.YAMLError):
                    self.load_text(text)

    def test_only_lowercase_canonical_yaml_booleans_are_accepted(self):
        invalid_scalars = (
            "yes",
            "no",
            "on",
            "off",
            "Yes",
            "NO",
            "TRUE",
            "False",
            "!!bool yes",
            "!!bool TRUE",
        )
        for scalar in invalid_scalars:
            with self.subTest(scalar=scalar):
                with self.assertRaises(TypeError):
                    self.load_text(f"promotion:\n  allow_weak: {scalar}\n")

        self.assertTrue(
            self.load_text("promotion:\n  allow_weak: true\n").promotion.allow_weak
        )
        self.assertFalse(
            self.load_text("promotion:\n  allow_weak: false\n").promotion.allow_weak
        )

    def test_only_canonical_base_ten_yaml_integers_are_accepted(self):
        invalid_scalars = (
            "1:20",
            "050",
            "0x32",
            "+50",
            "5_0",
            "!!int 1:20",
            "!!int 0x32",
        )
        for scalar in invalid_scalars:
            with self.subTest(scalar=scalar):
                with self.assertRaises(TypeError):
                    self.load_text(
                        "input:\n"
                        f"  min_global_test_queries: {scalar}\n"
                    )

        self.assertEqual(
            self.load_text(
                "input:\n  min_global_test_queries: 50\n"
            ).input.min_global_test_queries,
            50,
        )

    def test_strict_policy_loader_does_not_mutate_safe_loader_resolvers(self):
        self.load_text(APPROVED_POLICY_TEMPLATE)
        self.assertIs(yaml.safe_load("yes"), True)
        self.assertEqual(yaml.safe_load("1:20"), 80)

    def test_policy_and_sections_must_be_mappings(self):
        invalid_policies = (
            "- version\n- 1\n",
            "input: []\n",
            "evaluation: full\n",
            "promotion: false\n",
        )
        for text in invalid_policies:
            with self.subTest(policy=text):
                with self.assertRaises(TypeError):
                    self.load_text(text)

    def test_invalid_acceptance_level_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_text("acceptance_level: partial\n")
        with self.assertRaises(TypeError):
            self.load_text("acceptance_level: 1\n")

    def test_unsupported_version_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_text("version: 2\n")

    def test_policy_scalar_types_are_strictly_validated(self):
        invalid_policies = (
            'version: "1"\n',
            'input:\n  min_global_test_queries: "50"\n',
            "input:\n  min_domain_test_queries: true\n",
            "input:\n  min_docs_per_query: 2.5\n",
            "input:\n  require_authoritative_labels: 1\n",
            'evaluation:\n  require_all_current_cases: "true"\n',
            "evaluation:\n  require_all_historical_gates: 1\n",
            "evaluation:\n  require_global_ndcg_improvement: null\n",
            "evaluation:\n  require_global_mrr_improvement: 0\n",
            'evaluation:\n  allow_touched_domain_regression: "false"\n',
            "promotion:\n  allow_weak: 0\n",
            'promotion:\n  require_explicit_human_approval: "true"\n',
            "promotion:\n  allow_anchor_reset: null\n",
            "promotion:\n  allow_gate_retirement: 1\n",
        )
        for text in invalid_policies:
            with self.subTest(policy=text):
                with self.assertRaises(TypeError):
                    self.load_text(text)

    def test_query_and_document_minimums_must_be_positive(self):
        invalid_policies = (
            "input:\n  min_global_test_queries: 0\n",
            "input:\n  min_domain_test_queries: 0\n",
            "input:\n  min_docs_per_query: 0\n",
        )
        for text in invalid_policies:
            with self.subTest(policy=text):
                with self.assertRaises(ValueError):
                    self.load_text(text)

    def test_policy_hash_expands_defaults_before_canonicalization(self):
        implicit = self.load_text("promotion:\n  allow_weak: true\n")
        explicit = self.load_text(
            APPROVED_POLICY_TEMPLATE.replace("allow_weak: false", "allow_weak: true")
        )
        self.assertEqual(implicit.content_hash, explicit.content_hash)

    def test_approved_policy_template_is_exact_and_loads_as_default(self):
        template = (
            Path(__file__).resolve().parents[1]
            / "plugins/heuriboost/skills/heuriboost-rag/templates/reckless_policy.yml"
        )
        self.assertEqual(
            template.read_text(encoding="utf-8"),
            APPROVED_POLICY_TEMPLATE,
        )
        self.assertEqual(
            load_policy(template).content_hash,
            RecklessPolicy.default().content_hash,
        )


class PromotionEligibilityTests(unittest.TestCase):
    def evaluation(self, **overrides: object) -> EvaluationResult:
        values: dict[str, object] = {
            "acceptance_level": "full",
            "current_cases_passed": True,
            "historical_gates_passed": True,
            "global_metrics": {"ndcg@10": 0.81, "mrr@10": 0.71},
            "anchor_metrics": {"ndcg@10": 0.80, "mrr@10": 0.70},
            "touched_domains": {
                "medical": {
                    "ndcg@10": 0.80,
                    "anchor_ndcg@10": 0.80,
                    "mrr@10": 0.72,
                    "anchor_mrr@10": 0.70,
                }
            },
            "artifacts_valid": True,
            "details": {},
            "warnings": (),
        }
        values.update(overrides)
        return EvaluationResult(**values)

    def check(self, decision, check_id: str):
        return next(check for check in decision.checks if check.check_id == check_id)

    def assert_strict_json_safe(self, value: object) -> None:
        json.dumps(to_plain_data(value), allow_nan=False, sort_keys=True)

    def test_decision_always_emits_exactly_seven_named_checks(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(),
        )
        self.assertEqual(
            tuple(check.check_id for check in decision.checks),
            (
                "current_production_cases",
                "historical_gates",
                "global_ndcg_at_10_improvement",
                "global_mrr_at_10_improvement",
                "touched_domain_non_regression",
                "artifact_integrity",
                "acceptance_level",
            ),
        )
        self.assertEqual(len({check.check_id for check in decision.checks}), 7)
        self.assertTrue(all(check.reason for check in decision.checks))

    def test_full_acceptance_is_eligible_when_every_gate_passes(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(),
        )
        self.assertTrue(decision.promotion_eligible)
        self.assertEqual(decision.acceptance_level, "full")
        self.assertEqual(decision.blockers, ())
        self.assertTrue(all(check.passed for check in decision.checks))

    def test_weak_acceptance_is_blocked_by_default(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(acceptance_level="weak"),
        )
        self.assertFalse(decision.promotion_eligible)
        self.assertEqual(decision.blockers, ("acceptance_level",))

    def test_weak_acceptance_remains_blocked_when_allow_weak_is_true(self):
        policy = RecklessPolicy(
            promotion=PromotionPolicy(allow_weak=True),
        )
        decision = evaluate_promotion_eligibility(
            policy,
            self.evaluation(acceptance_level="weak"),
        )
        self.assertFalse(decision.promotion_eligible)
        self.assertEqual(decision.blockers, ("acceptance_level",))

    def test_weak_policy_is_blocked_even_when_evaluation_is_full(self):
        policy = RecklessPolicy(
            acceptance_level="weak",
            promotion=PromotionPolicy(allow_weak=True),
        )
        decision = evaluate_promotion_eligibility(
            policy,
            self.evaluation(acceptance_level="full"),
        )
        self.assertFalse(decision.promotion_eligible)
        self.assertEqual(decision.blockers, ("acceptance_level",))

    def test_current_and_historical_failures_are_hard_blockers(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                current_cases_passed=False,
                historical_gates_passed=False,
            ),
        )
        self.assertEqual(
            decision.blockers,
            ("current_production_cases", "historical_gates"),
        )

    def test_each_global_metric_requires_strict_improvement(self):
        cases = (
            (
                "ndcg@10",
                {"ndcg@10": 0.80, "mrr@10": 0.71},
                "global_ndcg_at_10_improvement",
            ),
            (
                "mrr@10",
                {"ndcg@10": 0.81, "mrr@10": 0.70},
                "global_mrr_at_10_improvement",
            ),
        )
        for metric, global_metrics, blocker in cases:
            with self.subTest(metric=metric):
                decision = evaluate_promotion_eligibility(
                    RecklessPolicy.default(),
                    self.evaluation(global_metrics=global_metrics),
                )
                self.assertFalse(decision.promotion_eligible)
                self.assertEqual(decision.blockers, (blocker,))

    def test_missing_required_global_metrics_are_blockers(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(global_metrics={}, anchor_metrics={}),
        )
        self.assertEqual(
            decision.blockers,
            (
                "global_ndcg_at_10_improvement",
                "global_mrr_at_10_improvement",
            ),
        )

    def test_invalid_global_metric_values_fail_without_exceptions(self):
        valid_global = {"ndcg@10": 0.81, "mrr@10": 0.71}
        valid_anchor = {"ndcg@10": 0.80, "mrr@10": 0.70}
        cases = (
            (
                "missing candidate",
                {"mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "missing",
            ),
            (
                "missing anchor",
                valid_global,
                {"mrr@10": 0.70},
                "anchor",
                "missing",
            ),
            (
                "nonnumeric candidate",
                {"ndcg@10": "bad", "mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "invalid_type",
            ),
            (
                "boolean candidate",
                {"ndcg@10": True, "mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "invalid_type",
            ),
            (
                "null anchor",
                valid_global,
                {"ndcg@10": None, "mrr@10": 0.70},
                "anchor",
                "invalid_type",
            ),
            (
                "NaN candidate",
                {"ndcg@10": float("nan"), "mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "non_finite",
            ),
            (
                "Infinity anchor",
                valid_global,
                {"ndcg@10": float("inf"), "mrr@10": 0.70},
                "anchor",
                "non_finite",
            ),
            (
                "negative Infinity candidate",
                {"ndcg@10": float("-inf"), "mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "non_finite",
            ),
            (
                "huge integer candidate",
                {"ndcg@10": 10**10000, "mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "out_of_range",
            ),
            (
                "negative candidate",
                {"ndcg@10": -0.01, "mrr@10": 0.71},
                valid_anchor,
                "candidate",
                "out_of_range",
            ),
            (
                "anchor above one",
                valid_global,
                {"ndcg@10": 1.01, "mrr@10": 0.70},
                "anchor",
                "out_of_range",
            ),
        )
        for name, global_metrics, anchor_metrics, invalid_side, status in cases:
            with self.subTest(name=name):
                decision = evaluate_promotion_eligibility(
                    RecklessPolicy.default(),
                    self.evaluation(
                        global_metrics=global_metrics,
                        anchor_metrics=anchor_metrics,
                    ),
                )
                self.assertIn(
                    "global_ndcg_at_10_improvement",
                    decision.blockers,
                )
                check = self.check(
                    decision,
                    "global_ndcg_at_10_improvement",
                )
                self.assertEqual(check.observed[invalid_side]["status"], status)
                self.assertFalse(check.observed[invalid_side]["valid"])
                self.assertEqual(
                    check.observed["comparison"]["status"],
                    "not_comparable",
                )
                self.assertFalse(check.observed["comparison"]["passed"])
                self.assert_strict_json_safe(decision)

    def test_any_touched_domain_metric_regression_is_a_blocker(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                touched_domains={
                    "medical": {
                        "ndcg@10": 0.80,
                        "anchor_ndcg@10": 0.80,
                        "mrr@10": 0.69,
                        "anchor_mrr@10": 0.70,
                    },
                    "travel": {
                        "ndcg@10": 0.75,
                        "anchor_ndcg@10": 0.74,
                    },
                }
            ),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))

    def test_empty_touched_domains_are_blocked(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(touched_domains={}),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        self.assertFalse(check.observed["valid"])
        self.assertEqual(check.observed["status"], "empty")
        self.assertEqual(check.observed["domains"], {})
        self.assert_strict_json_safe(decision)

    def test_blank_touched_domain_name_is_blocked(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                touched_domains={
                    "   ": {
                        "ndcg@10": 0.80,
                        "anchor_ndcg@10": 0.80,
                    }
                }
            ),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        domain = check.observed["domains"]["   "]
        self.assertFalse(domain["name_valid"])
        self.assertEqual(domain["status"], "invalid")
        self.assert_strict_json_safe(decision)

    def test_empty_touched_domain_metrics_are_blocked(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(touched_domains={"medical": {}}),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        domain = check.observed["domains"]["medical"]
        self.assertFalse(domain["valid"])
        self.assertEqual(domain["status"], "empty_metrics")
        self.assertEqual(domain["metrics"], {})
        self.assert_strict_json_safe(decision)

    def test_blank_touched_metric_name_is_blocked(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                touched_domains={
                    "medical": {
                        "   ": 0.80,
                        "anchor_   ": 0.80,
                    }
                }
            ),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        metric = check.observed["domains"]["medical"]["metrics"]["   "]
        self.assertFalse(metric["name_valid"])
        self.assertFalse(metric["valid"])
        self.assert_strict_json_safe(decision)

    def test_missing_touched_domain_anchor_is_a_blocker(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(touched_domains={"medical": {"ndcg@10": 0.80}}),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        metric = check.observed["domains"]["medical"]["metrics"]["ndcg@10"]
        self.assertEqual(metric["pairing_status"], "missing_anchor")
        self.assertEqual(metric["comparison"]["status"], "not_comparable")
        self.assert_strict_json_safe(decision)

    def test_orphan_touched_domain_anchor_is_a_blocker(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                touched_domains={
                    "medical": {"anchor_ndcg@10": 0.80},
                }
            ),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        metric = check.observed["domains"]["medical"]["metrics"]["ndcg@10"]
        self.assertEqual(metric["pairing_status"], "orphan_anchor")
        self.assertEqual(metric["comparison"]["status"], "not_comparable")
        self.assert_strict_json_safe(decision)

    def test_mismatched_touched_candidate_and_anchor_sets_are_blocked(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                touched_domains={
                    "medical": {
                        "ndcg@10": 0.80,
                        "anchor_mrr@10": 0.70,
                    }
                }
            ),
        )
        self.assertEqual(decision.blockers, ("touched_domain_non_regression",))
        check = self.check(decision, "touched_domain_non_regression")
        metrics = check.observed["domains"]["medical"]["metrics"]
        self.assertEqual(metrics["ndcg@10"]["pairing_status"], "missing_anchor")
        self.assertEqual(metrics["mrr@10"]["pairing_status"], "orphan_anchor")
        self.assert_strict_json_safe(decision)

    def test_invalid_touched_metric_values_fail_without_exceptions(self):
        cases = (
            ("nonnumeric", "bad", 0.80, "candidate", "invalid_type"),
            ("boolean", True, 0.80, "candidate", "invalid_type"),
            ("NaN", float("nan"), 0.80, "candidate", "non_finite"),
            ("Infinity", 0.80, float("inf"), "anchor", "non_finite"),
            ("huge integer", 10**10000, 0.80, "candidate", "out_of_range"),
            ("negative", -0.01, 0.80, "candidate", "out_of_range"),
            ("above one", 0.80, 1.01, "anchor", "out_of_range"),
        )
        for name, candidate, anchor, invalid_side, status in cases:
            with self.subTest(name=name):
                decision = evaluate_promotion_eligibility(
                    RecklessPolicy.default(),
                    self.evaluation(
                        touched_domains={
                            "medical": {
                                "ndcg@10": candidate,
                                "anchor_ndcg@10": anchor,
                            }
                        }
                    ),
                )
                self.assertEqual(
                    decision.blockers,
                    ("touched_domain_non_regression",),
                )
                check = self.check(decision, "touched_domain_non_regression")
                metric = check.observed["domains"]["medical"]["metrics"][
                    "ndcg@10"
                ]
                self.assertEqual(metric[invalid_side]["status"], status)
                self.assertFalse(metric[invalid_side]["valid"])
                self.assertEqual(
                    metric["comparison"]["status"],
                    "not_comparable",
                )
                self.assert_strict_json_safe(decision)

    def test_invalid_artifacts_are_a_blocker(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(artifacts_valid=False),
        )
        self.assertEqual(decision.blockers, ("artifact_integrity",))

    def test_disabled_policy_gates_still_emit_passing_checks(self):
        policy = RecklessPolicy(
            evaluation=EvaluationPolicy(
                require_all_current_cases=False,
                require_all_historical_gates=False,
                require_global_ndcg_improvement=False,
                require_global_mrr_improvement=False,
                allow_touched_domain_regression=True,
            )
        )
        decision = evaluate_promotion_eligibility(
            policy,
            self.evaluation(
                current_cases_passed=False,
                historical_gates_passed=False,
                global_metrics={},
                anchor_metrics={},
                touched_domains={
                    "medical": {
                        "ndcg@10": 0.70,
                        "anchor_ndcg@10": 0.80,
                    },
                    "broken": {
                        "ndcg@10": 0.70,
                    }
                },
            ),
        )
        self.assertTrue(decision.promotion_eligible)
        self.assertEqual(len(decision.checks), 7)
        self.assertTrue(all(check.passed for check in decision.checks))

        for check_id in ("current_production_cases", "historical_gates"):
            check = self.check(decision, check_id)
            self.assertFalse(check.required["enforced"])
            self.assertFalse(check.observed)
            self.assertIn("not enforced", check.reason)

        for check_id in (
            "global_ndcg_at_10_improvement",
            "global_mrr_at_10_improvement",
        ):
            check = self.check(decision, check_id)
            self.assertFalse(check.required["enforced"])
            self.assertEqual(check.observed["candidate"]["status"], "missing")
            self.assertEqual(check.observed["anchor"]["status"], "missing")
            self.assertEqual(
                check.observed["comparison"]["status"],
                "not_comparable",
            )
            self.assertIn("not enforced", check.reason)

        touched = self.check(decision, "touched_domain_non_regression")
        self.assertFalse(touched.required["enforced"])
        self.assertFalse(touched.observed["valid"])
        self.assertFalse(touched.observed["non_regressing"])
        regression = touched.observed["domains"]["medical"]["metrics"][
            "ndcg@10"
        ]
        self.assertTrue(regression["valid"])
        self.assertEqual(regression["comparison"]["status"], "failed")
        self.assertFalse(regression["comparison"]["passed"])
        malformed = touched.observed["domains"]["broken"]["metrics"][
            "ndcg@10"
        ]
        self.assertFalse(malformed["valid"])
        self.assertEqual(malformed["pairing_status"], "missing_anchor")
        self.assertEqual(
            malformed["comparison"]["status"],
            "not_comparable",
        )
        self.assertIn("not enforced", touched.reason)
        self.assert_strict_json_safe(decision)

    def test_evaluation_warnings_are_preserved(self):
        warnings = ("small domain sample", "candidate metadata warning")
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(warnings=warnings),
        )
        self.assertEqual(decision.warnings, warnings)

    def test_full_decision_with_invalid_metrics_is_strict_json_safe(self):
        decision = evaluate_promotion_eligibility(
            RecklessPolicy.default(),
            self.evaluation(
                global_metrics={"ndcg@10": float("nan"), "mrr@10": 0.71},
                anchor_metrics={"ndcg@10": 0.80, "mrr@10": float("inf")},
                touched_domains={
                    "medical": {
                        "ndcg@10": float("-inf"),
                        "anchor_ndcg@10": 10**10000,
                    }
                },
            ),
        )
        plain = to_plain_data(decision)
        encoded = json.dumps(plain, allow_nan=False, sort_keys=True)
        self.assertIn('"promotion_eligible": false', encoded)
        self.assertIn('"status": "non_finite"', encoded)
        self.assertIn('"status": "out_of_range"', encoded)
