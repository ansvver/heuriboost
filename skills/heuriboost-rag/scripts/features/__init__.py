#!/usr/bin/env python3
"""HeuriBoost feature recipe registry.

Importing this package eagerly:
  1. loads `templates/feature_recipes.yaml` (source of truth for metadata),
  2. registers the shared `extract_all` implementation,
  3. runs `validate()` (impl binding, inputs allowlist, online-safety, required
     fields).

Any failure raises `SystemExit` with an actionable message — fail-fast, single
validation entry. `common.py` imports this package at top level, so any script
that `import common` triggers validation once.

Re-exports: `REGISTRY`, `Recipe`, `extract_features`.
"""

from __future__ import annotations

from pathlib import Path

from features.registry import FeatureRegistry, Recipe
from features.recipes import extract_all

# Resolve the shipped feature_recipes.yaml relative to this file:
#   scripts/features/__init__.py -> scripts/features -> scripts -> heuriboost-rag
#   -> templates/feature_recipes.yaml
_DEFAULT_YAML = (
    Path(__file__).resolve().parent.parent.parent
    / "templates"
    / "feature_recipes.yaml"
)

REGISTRY = FeatureRegistry()
REGISTRY.register_impl("extract_all", extract_all)
REGISTRY.load_yaml(_DEFAULT_YAML)
REGISTRY.validate()


def extract_features(df):
    """Backward-compatible shim. Delegates to `REGISTRY.extract`."""
    return REGISTRY.extract(df)


__all__ = ["REGISTRY", "Recipe", "extract_features"]
