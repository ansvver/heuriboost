from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


class RunState(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    COMPILED = "COMPILED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    EVALUATING = "EVALUATING"
    SYNTHESIZING_FEATURES = "SYNTHESIZING_FEATURES"
    REPORTING = "REPORTING"
    READY_FOR_PROMOTION = "READY_FOR_PROMOTION"
    PROMOTING = "PROMOTING"
    PROMOTED = "PROMOTED"
    BLOCKED_INPUT = "BLOCKED_INPUT"
    BLOCKED_NOT_ELIGIBLE = "BLOCKED_NOT_ELIGIBLE"
    BLOCKED_EVALUATION = "BLOCKED_EVALUATION"
    PROMOTION_FAILED = "PROMOTION_FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"
    FAILED_INTERNAL = "FAILED_INTERNAL"


ALLOWED_TRANSITIONS: Final[Mapping[RunState, frozenset[RunState]]] = MappingProxyType(
    {
        RunState.RECEIVED: frozenset({RunState.VALIDATING, RunState.CANCELLED}),
        RunState.VALIDATING: frozenset(
            {
                RunState.COMPILED,
                RunState.BLOCKED_INPUT,
                RunState.FAILED_INTERNAL,
            }
        ),
        RunState.COMPILED: frozenset({RunState.TRAINING, RunState.CANCELLED}),
        RunState.TRAINING: frozenset(
            {
                RunState.TRAINED,
                RunState.INTERRUPTED,
                RunState.CANCELLED,
                RunState.FAILED_INTERNAL,
            }
        ),
        RunState.INTERRUPTED: frozenset({RunState.TRAINING}),
        RunState.TRAINED: frozenset({RunState.EVALUATING}),
        RunState.EVALUATING: frozenset(
            {
                RunState.SYNTHESIZING_FEATURES,
                RunState.REPORTING,
                RunState.BLOCKED_EVALUATION,
                RunState.BLOCKED_NOT_ELIGIBLE,
                RunState.FAILED_INTERNAL,
            }
        ),
        RunState.SYNTHESIZING_FEATURES: frozenset(
            {
                RunState.TRAINING,
                RunState.BLOCKED_EVALUATION,
                RunState.FAILED_INTERNAL,
            }
        ),
        RunState.REPORTING: frozenset(
            {
                RunState.READY_FOR_PROMOTION,
                RunState.BLOCKED_NOT_ELIGIBLE,
                RunState.FAILED_INTERNAL,
            }
        ),
        RunState.READY_FOR_PROMOTION: frozenset({RunState.PROMOTING}),
        RunState.PROMOTING: frozenset(
            {RunState.PROMOTED, RunState.PROMOTION_FAILED}
        ),
        RunState.PROMOTION_FAILED: frozenset({RunState.PROMOTING}),
    }
)


def _normalize_state(value: RunState | str) -> RunState:
    if isinstance(value, RunState):
        return value
    if isinstance(value, str):
        try:
            return RunState(value)
        except ValueError:
            raise ValueError(f"Unknown run state: {value!r}") from None
    raise ValueError(
        f"Run state must be RunState or str, got {type(value).__name__}"
    )


def assert_transition(current: RunState | str, target: RunState | str) -> None:
    current_state = _normalize_state(current)
    target_state = _normalize_state(target)
    if target_state not in ALLOWED_TRANSITIONS.get(current_state, frozenset()):
        raise ValueError(
            f"Invalid run-state transition: {current_state.value} -> {target_state.value}"
        )
