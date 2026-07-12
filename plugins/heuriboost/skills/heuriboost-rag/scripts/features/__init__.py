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

def _feature_recipe_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "legacy_templates/feature_recipes.yaml",
        "templates/feature_recipes.yaml",
    ):
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"Feature recipe source is missing below {root}")


_trusted_recipe_path = globals().get("_HEURIBOOST_TRUSTED_RECIPE_PATH")
_trusted_recipe_bytes = globals().get("_HEURIBOOST_TRUSTED_RECIPE_BYTES")
if _trusted_recipe_path is not None or _trusted_recipe_bytes is not None:
    if not isinstance(_trusted_recipe_path, (str, Path)) or not isinstance(
        _trusted_recipe_bytes, bytes
    ):
        raise SystemExit("Trusted feature recipe injection is invalid")
    FEATURE_RECIPE_PATH = Path(_trusted_recipe_path)
else:
    FEATURE_RECIPE_PATH = _feature_recipe_path()

REGISTRY = FeatureRegistry()
REGISTRY.register_impl("extract_all", extract_all)
if _trusted_recipe_bytes is not None:
    REGISTRY.load_yaml_bytes(_trusted_recipe_bytes, source=FEATURE_RECIPE_PATH)
else:
    REGISTRY.load_yaml(FEATURE_RECIPE_PATH)
REGISTRY.validate()


def extract_features(df):
    """Backward-compatible shim. Delegates to `REGISTRY.extract`."""
    return REGISTRY.extract(df)


__all__ = ["REGISTRY", "Recipe", "extract_features"]
