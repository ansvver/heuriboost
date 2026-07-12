import unittest

from heuriboost_rag.reckless.state import ALLOWED_TRANSITIONS, RunState, assert_transition


LEGAL_EDGES = (
    (RunState.RECEIVED, RunState.VALIDATING),
    (RunState.RECEIVED, RunState.CANCELLED),
    (RunState.VALIDATING, RunState.COMPILED),
    (RunState.VALIDATING, RunState.BLOCKED_INPUT),
    (RunState.VALIDATING, RunState.FAILED_INTERNAL),
    (RunState.COMPILED, RunState.TRAINING),
    (RunState.COMPILED, RunState.CANCELLED),
    (RunState.TRAINING, RunState.TRAINED),
    (RunState.TRAINING, RunState.INTERRUPTED),
    (RunState.TRAINING, RunState.CANCELLED),
    (RunState.TRAINING, RunState.FAILED_INTERNAL),
    (RunState.INTERRUPTED, RunState.TRAINING),
    (RunState.TRAINED, RunState.EVALUATING),
    (RunState.EVALUATING, RunState.SYNTHESIZING_FEATURES),
    (RunState.EVALUATING, RunState.REPORTING),
    (RunState.EVALUATING, RunState.BLOCKED_EVALUATION),
    (RunState.EVALUATING, RunState.BLOCKED_NOT_ELIGIBLE),
    (RunState.EVALUATING, RunState.FAILED_INTERNAL),
    (RunState.SYNTHESIZING_FEATURES, RunState.TRAINING),
    (RunState.SYNTHESIZING_FEATURES, RunState.BLOCKED_EVALUATION),
    (RunState.SYNTHESIZING_FEATURES, RunState.FAILED_INTERNAL),
    (RunState.REPORTING, RunState.READY_FOR_PROMOTION),
    (RunState.REPORTING, RunState.BLOCKED_NOT_ELIGIBLE),
    (RunState.REPORTING, RunState.FAILED_INTERNAL),
    (RunState.READY_FOR_PROMOTION, RunState.PROMOTING),
    (RunState.PROMOTING, RunState.PROMOTED),
    (RunState.PROMOTING, RunState.PROMOTION_FAILED),
    (RunState.PROMOTION_FAILED, RunState.PROMOTING),
)

TERMINAL_STATES = frozenset(
    {
        RunState.PROMOTED,
        RunState.BLOCKED_INPUT,
        RunState.BLOCKED_NOT_ELIGIBLE,
        RunState.BLOCKED_EVALUATION,
        RunState.CANCELLED,
        RunState.FAILED_INTERNAL,
    }
)


class RecklessStateTests(unittest.TestCase):
    def test_normal_transition_is_allowed(self):
        assert_transition(RunState.REPORTING, RunState.READY_FOR_PROMOTION)

    def test_blocked_run_cannot_be_reopened_in_place(self):
        with self.assertRaises(ValueError):
            assert_transition(RunState.BLOCKED_INPUT, RunState.VALIDATING)

    def test_failed_promotion_can_retry(self):
        assert_transition(RunState.PROMOTION_FAILED, RunState.PROMOTING)

    def test_every_legal_transition_is_allowed_for_enums_and_strings(self):
        for current, target in LEGAL_EDGES:
            with self.subTest(current=current, target=target, representation="enum"):
                assert_transition(current, target)
            with self.subTest(current=current, target=target, representation="string"):
                assert_transition(current.value, target.value)

    def test_transition_table_exactly_matches_the_approved_policy(self):
        expected = {}
        for current, target in LEGAL_EDGES:
            expected.setdefault(current, set()).add(target)
        expected = {current: frozenset(targets) for current, targets in expected.items()}

        self.assertEqual(ALLOWED_TRANSITIONS, expected)

    def test_transition_policy_cannot_be_mutated(self):
        original_targets = ALLOWED_TRANSITIONS[RunState.RECEIVED]
        try:
            with self.assertRaises(TypeError):
                ALLOWED_TRANSITIONS[RunState.RECEIVED] = frozenset()
        finally:
            if ALLOWED_TRANSITIONS[RunState.RECEIVED] != original_targets:
                ALLOWED_TRANSITIONS[RunState.RECEIVED] = original_targets

        received_targets = ALLOWED_TRANSITIONS[RunState.RECEIVED]
        self.assertIsInstance(received_targets, frozenset)
        try:
            with self.assertRaises(AttributeError):
                received_targets.add(RunState.FAILED_INTERNAL)
        finally:
            if hasattr(received_targets, "discard"):
                received_targets.discard(RunState.FAILED_INTERNAL)

    def test_all_terminal_states_reject_every_target(self):
        self.assertEqual(set(RunState) - set(ALLOWED_TRANSITIONS), TERMINAL_STATES)
        for current in TERMINAL_STATES:
            for target in RunState:
                with self.subTest(current=current, target=target):
                    with self.assertRaises(ValueError):
                        assert_transition(current, target)

    def test_every_forbidden_enum_transition_is_rejected(self):
        legal_edges = set(LEGAL_EDGES)
        for current in RunState:
            for target in RunState:
                if (current, target) in legal_edges:
                    continue
                with self.subTest(current=current, target=target):
                    with self.assertRaises(ValueError):
                        assert_transition(current, target)

    def test_mixed_enum_and_persisted_string_states_are_supported(self):
        assert_transition("REPORTING", RunState.READY_FOR_PROMOTION)
        assert_transition(RunState.PROMOTION_FAILED, "PROMOTING")

    def test_invalid_state_strings_and_types_raise_value_error(self):
        invalid_states = ("", "reporting", "UNKNOWN", None, 1, object())
        for invalid in invalid_states:
            with self.subTest(invalid=invalid, position="current"):
                with self.assertRaises(ValueError):
                    assert_transition(invalid, RunState.VALIDATING)
            with self.subTest(invalid=invalid, position="target"):
                with self.assertRaises(ValueError):
                    assert_transition(RunState.RECEIVED, invalid)
