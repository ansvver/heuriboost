#!/usr/bin/env python3
"""FeatureRecipe registry: declared metadata + validated implementation binding.

Per `docs/specs/ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md` §6.4, every feature must be
declared in a registry with required fields (name, version, description,
task_profiles, inputs, implementation reference, type, default_value, cost_tier,
online_safe, leakage_risk, expected_slices, owner). The registry validates, at
load time, that:

  1. every recipe's `impl` resolves to a registered implementation name;
  2. every declared `inputs` value is in `ALLOWED_INPUTS` (leakage control);
  3. `online_safe` is true for the active task profile;
  4. all spec-required fields are non-empty (except `expected_slices`, which is
     a forward-looking declaration and may be empty).

This makes the "FEATURE_NAMES must equal feature_recipes.yaml" contract
(`fiqa-demo-contracts.md`) a load-time check instead of a grep-time check.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# The 6 input columns V0 features may legitimately read. Anything else (label,
# split, query_id, doc_id, chunk_id, doc_text_ref, ...) is a leakage or
# identifier vector and is rejected at validation time.
ALLOWED_INPUTS = {
    "query_text",
    "doc_text",
    "dense_rank",
    "dense_score",
    "sparse_rank",
    "sparse_score",
}

ACTIVE_TASK_PROFILE = "qd_reranker"

REQUIRED_NONEMPTY_FIELDS = (
    "name",
    "version",
    "description",
    "task_profiles",
    "inputs",
    "type",
    "cost_tier",
    "online_safe",
    "leakage_risk",
    "owner",
    "impl",
)

_VALID_COST_TIERS = {"L0", "L1", "L2", "L3"}
_VALID_LEAKAGE = {"low", "medium", "high"}
_VALID_TYPES = {"numeric", "categorical", "boolean"}
SKILL_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Recipe:
    name: str
    version: int
    description: str
    task_profiles: tuple[str, ...]
    inputs: tuple[str, ...]
    type: str
    default_value: float
    cost_tier: str
    online_safe: bool
    leakage_risk: str
    expected_slices: tuple[str, ...]
    owner: str
    impl: str


class FeatureRegistry:
    """Holds recipe metadata + registered implementation functions.

    V0 uses a single shared impl (`extract_all`) that computes every feature in
    one pass per row; the registry dispatches to it. Per-feature dispatch is
    deferred to the ablation task.
    """

    def __init__(self) -> None:
        self._recipes: dict[str, Recipe] = {}
        self._impls: dict[str, Callable] = {}
        self._feature_set_name: str | None = None
        self._feature_set_version: int | None = None

    # -- registration ------------------------------------------------------

    def register_impl(self, name: str, fn: Callable) -> None:
        """Register a shared implementation function by logical name.

        The fn signature is `(row) -> dict[str, float]` returning all feature
        values for one row.
        """
        if name in self._impls:
            raise SystemExit(
                f"FeatureRegistry: impl '{name}' is already registered"
            )
        self._impls[name] = fn

    # -- YAML loading ------------------------------------------------------

    def load_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required to load feature_recipes.yaml. "
                "Install with: python -m pip install -r "
                f"{SKILL_DIR / 'requirements.txt'}"
            ) from exc

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise SystemExit(f"Feature recipes file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise SystemExit(f"Feature recipes file must be a mapping: {yaml_path}")

        feature_set = data.get("feature_set")
        if not isinstance(feature_set, dict):
            raise SystemExit(
                f"feature_recipes.yaml must declare a top-level 'feature_set' "
                f"mapping with name+version: {yaml_path}"
            )
        self._feature_set_name = feature_set.get("name")
        self._feature_set_version = feature_set.get("version")
        if not self._feature_set_name or self._feature_set_version is None:
            raise SystemExit(
                "feature_set.name and feature_set.version are required"
            )

        features = data.get("features")
        if not isinstance(features, list) or not features:
            raise SystemExit(
                "feature_recipes.yaml must declare a non-empty 'features' list"
            )

        self._recipes = {}
        for entry in features:
            recipe = self._build_recipe(entry)
            if recipe.name in self._recipes:
                raise SystemExit(
                    f"Duplicate feature name in feature_recipes.yaml: {recipe.name}"
                )
            self._recipes[recipe.name] = recipe

    def _build_recipe(self, entry: dict) -> Recipe:
        if not isinstance(entry, dict):
            raise SystemExit("Each feature entry must be a mapping")
        # Required-field presence + non-empty (expected_slices may be empty).
        for field in REQUIRED_NONEMPTY_FIELDS:
            value = entry.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise SystemExit(
                    f"Feature recipe is missing required field '{field}': {entry}"
                )

        try:
            version = int(entry["version"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Feature '{entry.get('name')}' version must be an integer"
            ) from exc

        task_profiles = tuple(entry["task_profiles"])
        inputs = tuple(entry["inputs"])
        if ACTIVE_TASK_PROFILE not in task_profiles:
            raise SystemExit(
                f"Feature '{entry['name']}' declares task_profiles={task_profiles} "
                f"but the active profile is '{ACTIVE_TASK_PROFILE}'"
            )

        cost_tier = entry["cost_tier"]
        if cost_tier not in _VALID_COST_TIERS:
            raise SystemExit(
                f"Feature '{entry['name']}' cost_tier must be one of "
                f"{sorted(_VALID_COST_TIERS)}, got: {cost_tier}"
            )

        leakage_risk = entry["leakage_risk"]
        if leakage_risk not in _VALID_LEAKAGE:
            raise SystemExit(
                f"Feature '{entry['name']}' leakage_risk must be one of "
                f"{sorted(_VALID_LEAKAGE)}, got: {leakage_risk}"
            )

        ftype = entry["type"]
        if ftype not in _VALID_TYPES:
            raise SystemExit(
                f"Feature '{entry['name']}' type must be one of "
                f"{sorted(_VALID_TYPES)}, got: {ftype}"
            )

        return Recipe(
            name=entry["name"],
            version=version,
            description=entry["description"],
            task_profiles=task_profiles,
            inputs=inputs,
            type=ftype,
            default_value=float(entry.get("default_value", 0.0)),
            cost_tier=cost_tier,
            online_safe=bool(entry["online_safe"]),
            leakage_risk=leakage_risk,
            expected_slices=tuple(entry.get("expected_slices") or ()),
            owner=entry["owner"],
            impl=entry["impl"],
        )

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Run all load-time checks. Hard-fail on any violation."""
        if not self._recipes:
            raise SystemExit("FeatureRegistry: no recipes loaded")

        for recipe in self._recipes.values():
            # impl resolves to a registered implementation function
            if recipe.impl not in self._impls:
                raise SystemExit(
                    f"Feature '{recipe.name}' declares impl='{recipe.impl}' "
                    f"but no such implementation is registered "
                    f"(known: {sorted(self._impls)})"
                )
            # inputs allowlist (leakage control)
            bad_inputs = [i for i in recipe.inputs if i not in ALLOWED_INPUTS]
            if bad_inputs:
                raise SystemExit(
                    f"Feature '{recipe.name}' declares inputs={list(recipe.inputs)}; "
                    f"'{bad_inputs[0]}' is not in ALLOWED_INPUTS "
                    f"(allowed: {sorted(ALLOWED_INPUTS)}). Post-outcome / identifier "
                    f"columns cannot be model features (leakage)."
                )
            # online-safety for the active task profile
            if not recipe.online_safe:
                raise SystemExit(
                    f"Feature '{recipe.name}' is online_safe=false; "
                    f"the '{ACTIVE_TASK_PROFILE}' profile requires online-safe features."
                )

    # -- accessors ---------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._recipes.keys())

    @property
    def feature_set_name(self) -> str:
        if self._feature_set_name is None:
            raise SystemExit("FeatureRegistry: feature_set not loaded")
        return self._feature_set_name

    @property
    def feature_set_version(self) -> int:
        if self._feature_set_version is None:
            raise SystemExit("FeatureRegistry: feature_set not loaded")
        return self._feature_set_version

    def feature_versions(self) -> dict[str, int]:
        return {name: recipe.version for name, recipe in self._recipes.items()}

    # -- extraction --------------------------------------------------------

    def extract(self, df):
        """Compute features for every row. Bit-for-bit identical to the
        pre-refactor `extract_features` because the impl is the same body."""
        import pandas as pd

        impl = self._impls.get("extract_all")
        if impl is None:
            raise SystemExit(
                "FeatureRegistry: no 'extract_all' impl registered"
            )

        rows = [impl(row) for _, row in df.iterrows()]
        return pd.DataFrame(rows, columns=self.names())
