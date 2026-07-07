#!/usr/bin/env python3
"""HPO engine adapter (spec §12.5).

Re-exports `HPOEngine`, `TrialResult`, `Snapshot`, `Budget`. The HPO SEARCH
sees only train+valid snapshots; it is case-blind and test-blind by
construction. Post-hoc test evaluation lives in `run_hpo.py`, not here.

This package does NOT import `features` — HPO receives feature_set metadata
as arguments, keeping it decoupled from the registry.
"""

from hpo.engine import Budget, HPOEngine, TrialResult
from ranking_snapshot import Snapshot

__all__ = ["HPOEngine", "TrialResult", "Snapshot", "Budget"]
