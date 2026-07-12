from __future__ import annotations

import builtins
from contextlib import contextmanager
import hashlib
import importlib
import importlib.abc
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock
import zipfile

from heuriboost_rag.backends import legacy_runtime
from heuriboost_rag.backends.legacy_runtime import (
    LegacyRuntimeResolutionError,
    resolve_legacy_runtime,
)


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "heuriboost"
    / "skills"
    / "heuriboost-rag"
)


class _PoisoningLegacyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Pretends trusted legacy names came from trusted paths.

    This models a hostile meta-path hook, not an ordinary sys.path shadow.  The
    old resolver accepted these modules because it imported normally and then
    trusted module metadata supplied by the loader.
    """

    _NAMES = frozenset(
        {
            "common",
            "features",
            "features.primitives",
            "features.recipes",
            "features.registry",
            "ranking_snapshot",
            "repair_cases",
        }
    )

    def __init__(self, trusted: Path, marker: Path) -> None:
        self._trusted = trusted
        self._marker = marker

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> ModuleSpec | None:
        if fullname not in self._NAMES:
            return None
        return ModuleSpec(fullname, self, is_package=fullname == "features")

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module_name = module.__name__
        relative = {
            "common": "common.py",
            "features": "features/__init__.py",
            "features.primitives": "features/primitives.py",
            "features.recipes": "features/recipes.py",
            "features.registry": "features/registry.py",
            "ranking_snapshot": "ranking_snapshot.py",
            "repair_cases": "repair_cases.py",
        }[module_name]
        module.__file__ = str(self._trusted / relative)
        if module_name == "features":
            module.__path__ = [str(self._trusted / "features")]
            source = (
                "from pathlib import Path\n"
                "class Registry:\n"
                "    feature_set_name = 'poisoned'\n"
                "    feature_set_version = 1\n"
                "    def names(self):\n"
                "        return ['score']\n"
                "    def feature_versions(self):\n"
                "        return {'score': 1}\n"
                "REGISTRY = Registry()\n"
                f"FEATURE_RECIPE_PATH = Path({str(self._trusted.parent / 'templates' / 'feature_recipes.yaml')!r})\n"
                "def extract_features(frame):\n"
                "    return frame\n"
            )
        elif module_name == "common":
            source = "def load_dataset(path):\n    return path\n"
        elif module_name == "repair_cases":
            source = (
                "from pathlib import Path\n"
                f"Path({str(self._marker)!r}).write_text('poisoned', encoding='utf-8')\n"
                "class CompileOptions:\n    pass\n"
                "FIXED_TRAINING_PARAMS = {'objective': 'rank:ndcg', 'seed': 42}\n"
                "def compile_repair_inputs(*args):\n    return None\n"
                "def load_compiled_production_cases(*args):\n    return []\n"
                "def merge_training_frames(*args, **kwargs):\n    return None\n"
                "def train_model_from_frame(*args, **kwargs):\n    return None\n"
                "def load_model(*args):\n    return None\n"
                "def evaluate_model_on_split(*args):\n    return {}, None\n"
                "def evaluate_model_by_domain(*args):\n    return {}\n"
                "def evaluate_cases(*args, **kwargs):\n    return []\n"
                "def load_gates(*args):\n    return []\n"
            )
        else:
            source = ""
        exec(compile(source, module.__file__, "exec"), module.__dict__)


class _PoisoningYamlFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Supplies forged recipe data when a legacy source imports PyYAML."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> ModuleSpec | None:
        if fullname == "yaml":
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.safe_load = lambda source: {
            "feature_set": {"name": "forged_features", "version": 999},
            "features": [
                {
                    "name": "forged_feature",
                    "version": 1,
                    "description": "forged",
                    "task_profiles": ["qd_reranker"],
                    "inputs": ["query_text"],
                    "type": "numeric",
                    "default_value": 0.0,
                    "cost_tier": "L0",
                    "online_safe": True,
                    "leakage_risk": "low",
                    "expected_slices": [],
                    "owner": "attacker",
                    "impl": "extract_all",
                }
            ],
        }


def _manifest_from_bytes(scripts: Path, contents: dict[Path, bytes]) -> str:
    digest = hashlib.sha256()
    root = scripts.resolve().parent
    for path in legacy_runtime._legacy_file_paths(scripts):
        resolved = path.resolve()
        digest.update(resolved.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents[resolved])
        digest.update(b"\0")
    return digest.hexdigest()


def _write_minimal_runtime_tree(
    root: Path,
    *,
    scripts_name: str = "scripts",
    templates_name: str = "templates",
) -> Path:
    scripts = root / scripts_name
    features = scripts / "features"
    templates = root / templates_name
    features.mkdir(parents=True)
    templates.mkdir()
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    (templates / "__init__.py").write_text("", encoding="utf-8")
    (templates / "feature_recipes.yaml").write_text("feature_set: test\n", encoding="utf-8")
    (scripts / "common.py").write_text(
        "def load_dataset(path):\n    return path\n",
        encoding="utf-8",
    )
    (scripts / "ranking_snapshot.py").write_text("", encoding="utf-8")
    (features / "__init__.py").write_text(
        "from pathlib import Path\n"
        "class Registry:\n"
        "    feature_set_name = 'test_features'\n"
        "    feature_set_version = 1\n"
        "    def names(self):\n"
        "        return ['score']\n"
        "    def feature_versions(self):\n"
        "        return {'score': 1}\n"
        "REGISTRY = Registry()\n"
        "FEATURE_RECIPE_PATH = next(\n"
        "    path for path in (\n"
        "        Path(__file__).resolve().parents[2] / 'legacy_templates' / 'feature_recipes.yaml',\n"
        "        Path(__file__).resolve().parents[2] / 'templates' / 'feature_recipes.yaml',\n"
        "    ) if path.is_file()\n"
        ")\n"
        "def extract_features(frame):\n"
        "    return frame\n",
        encoding="utf-8",
    )
    for name in ("primitives.py", "recipes.py", "registry.py"):
        (features / name).write_text("", encoding="utf-8")
    (scripts / "repair_cases.py").write_text(
        "class CompileOptions:\n    pass\n"
        "FIXED_TRAINING_PARAMS = {'objective': 'rank:ndcg', 'seed': 42}\n"
        "def compile_repair_inputs(*args):\n    return None\n"
        "def load_compiled_production_cases(*args):\n    return []\n"
        "def merge_training_frames(*args, **kwargs):\n    return None\n"
        "def train_model_from_frame(*args, **kwargs):\n    return None\n"
        "def load_model(*args):\n    return None\n"
        "def evaluate_model_on_split(*args):\n    return {}, None\n"
        "def evaluate_model_by_domain(*args):\n    return {}\n"
        "def evaluate_cases(*args, **kwargs):\n    return []\n"
        "def load_gates(*args):\n    return []\n",
        encoding="utf-8",
    )
    return scripts


@contextmanager
def _without_legacy_modules():
    names = (
        "common",
        "features",
        "features.primitives",
        "features.recipes",
        "features.registry",
        "ranking_snapshot",
        "repair_cases",
    )
    original = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(original)


class LegacyRuntimeResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_sys_path = list(sys.path)
        self._legacy_modules = {
            name: sys.modules.pop(name)
            for name in (
                "common",
                "features",
                "features.primitives",
                "features.recipes",
                "features.registry",
                "ranking_snapshot",
                "repair_cases",
            )
            if name in sys.modules
        }

    def tearDown(self) -> None:
        for name in (
            "common",
            "features",
            "features.primitives",
            "features.recipes",
            "features.registry",
            "ranking_snapshot",
            "repair_cases",
        ):
            sys.modules.pop(name, None)
        sys.modules.update(self._legacy_modules)
        sys.path[:] = self._original_sys_path

    def _resolve_from(self, trusted: Path):
        with mock.patch.object(
            legacy_runtime,
            "_trusted_scripts_dir",
            return_value=trusted,
            create=True,
        ):
            return resolve_legacy_runtime()

    def test_rejects_caller_controlled_scripts_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            with _without_legacy_modules():
                with self.assertRaises(TypeError):
                    resolve_legacy_runtime(trusted)

    def test_evicts_preloaded_sibling_module_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = _write_minimal_runtime_tree(root / "trusted")
            outside = root / "outside"
            outside.mkdir()
            foreign_common = ModuleType("common")
            foreign_common.__file__ = str(outside / "common.py")
            foreign_common.load_dataset = lambda path: "foreign"
            with _without_legacy_modules(), mock.patch.dict(
                sys.modules, {"common": foreign_common}
            ):
                runtime = self._resolve_from(trusted)

            self.assertIsNot(runtime.load_dataset, foreign_common.load_dataset)

    def test_uses_physical_installed_root_when_private_child_is_forged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_root = root / "site-packages" / "heuriboost_rag"
            trusted = _write_minimal_runtime_tree(
                package_root,
                scripts_name="legacy_scripts",
                templates_name="legacy_templates",
            )
            runtime_file = package_root / "backends" / "legacy_runtime.py"
            runtime_file.parent.mkdir(parents=True)
            runtime_file.write_text("", encoding="utf-8")
            forged = ModuleType("heuriboost_rag.legacy_scripts")
            forged.__file__ = str(root / "outside" / "__init__.py")
            with _without_legacy_modules(), mock.patch.object(
                legacy_runtime,
                "__file__",
                str(runtime_file),
            ), mock.patch.dict(sys.modules, {"heuriboost_rag.legacy_scripts": forged}):
                runtime = resolve_legacy_runtime()

            self.assertEqual(runtime.scripts_dir, trusted.resolve())

    def test_evicts_in_root_preload_with_the_wrong_expected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            forged_common = ModuleType("common")
            forged_common.__file__ = str(trusted / "repair_cases.py")
            forged_common.load_dataset = lambda path: path
            with _without_legacy_modules(), mock.patch.dict(
                sys.modules, {"common": forged_common}
            ):
                runtime = self._resolve_from(trusted)
                self.assertEqual(
                    Path(sys.modules["common"].__file__).resolve(),
                    (trusted / "common.py").resolve(),
                )
                self.assertIsNot(runtime.load_dataset, forged_common.load_dataset)

    def test_evicts_every_cached_features_child_before_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            stale_child = ModuleType("features.unlisted_child")
            with _without_legacy_modules(), mock.patch.dict(
                sys.modules, {"features.unlisted_child": stale_child}
            ):
                self._resolve_from(trusted)
                self.assertNotIn("features.unlisted_child", sys.modules)

    def test_ignores_meta_path_poisoning_even_when_loader_claims_trusted_origins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = _write_minimal_runtime_tree(root / "trusted")
            marker = root / "poisoned"
            finder = _PoisoningLegacyFinder(trusted, marker)
            with _without_legacy_modules():
                sys.meta_path.insert(0, finder)
                try:
                    runtime = self._resolve_from(trusted)
                finally:
                    sys.meta_path.remove(finder)

            self.assertEqual(runtime.registry.feature_set_name, "test_features")
            self.assertFalse(marker.exists())

    def test_ignores_builtins_import_poisoning_for_legacy_dependencies(self) -> None:
        class PoisonedRegistry:
            feature_set_name = "poisoned_features"
            feature_set_version = 1

            def register_impl(self, name, fn) -> None:
                pass

            def load_yaml(self, path) -> None:
                pass

            def validate(self) -> None:
                pass

            def names(self):
                return ["poisoned"]

            def feature_versions(self):
                return {"poisoned": 1}

        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            (trusted / "features" / "registry.py").write_text(
                "class FeatureRegistry:\n"
                "    feature_set_name = 'test_features'\n"
                "    feature_set_version = 1\n"
                "    def register_impl(self, name, fn):\n"
                "        pass\n"
                "    def load_yaml(self, path):\n"
                "        pass\n"
                "    def validate(self):\n"
                "        pass\n"
                "    def names(self):\n"
                "        return ['score']\n"
                "    def feature_versions(self):\n"
                "        return {'score': 1}\n"
                "Recipe = object\n",
                encoding="utf-8",
            )
            (trusted / "features" / "recipes.py").write_text(
                "def extract_all(frame):\n    return frame\n",
                encoding="utf-8",
            )
            (trusted / "features" / "__init__.py").write_text(
                "from pathlib import Path\n"
                "from features.registry import FeatureRegistry, Recipe\n"
                "from features.recipes import extract_all\n"
                "FEATURE_RECIPE_PATH = Path(__file__).resolve().parents[2] / 'templates' / 'feature_recipes.yaml'\n"
                "REGISTRY = FeatureRegistry()\n"
                "REGISTRY.register_impl('extract_all', extract_all)\n"
                "REGISTRY.load_yaml(FEATURE_RECIPE_PATH)\n"
                "REGISTRY.validate()\n"
                "def extract_features(frame):\n    return frame\n",
                encoding="utf-8",
            )
            fake_registry = ModuleType("features.registry")
            fake_registry.FeatureRegistry = PoisonedRegistry
            fake_registry.Recipe = object
            fake_recipes = ModuleType("features.recipes")
            fake_recipes.extract_all = lambda frame: frame
            original_import = builtins.__import__

            def poisoned_import(
                name: str,
                globals: object = None,
                locals: object = None,
                fromlist: object = (),
                level: int = 0,
            ):
                if level == 0 and name == "features.registry":
                    return fake_registry
                if level == 0 and name == "features.recipes":
                    return fake_recipes
                return original_import(name, globals, locals, fromlist, level)

            with _without_legacy_modules(), mock.patch.object(
                builtins,
                "__import__",
                new=poisoned_import,
            ):
                runtime = self._resolve_from(trusted)

        self.assertEqual(runtime.registry.feature_set_name, "test_features")

    def test_ignores_meta_path_poisoned_yaml_parser(self) -> None:
        finder = _PoisoningYamlFinder()
        original_yaml = sys.modules.pop("yaml", None)
        try:
            with _without_legacy_modules():
                sys.meta_path.insert(0, finder)
                try:
                    runtime = self._resolve_from(PACKAGE_ROOT / "scripts")
                finally:
                    sys.meta_path.remove(finder)
        finally:
            sys.modules.pop("yaml", None)
            if original_yaml is not None:
                sys.modules["yaml"] = original_yaml

        self.assertEqual(runtime.registry.feature_set_name, "heuriboost_rag_v0")

    def test_rejects_symlinked_feature_recipe_before_source_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = _write_minimal_runtime_tree(root)
            recipe = root / "templates" / "feature_recipes.yaml"
            outside = root / "outside-feature-recipes.yaml"
            outside.write_text("feature_set: outside\n", encoding="utf-8")
            recipe.unlink()
            recipe.symlink_to(outside)

            with _without_legacy_modules(), self.assertRaisesRegex(
                LegacyRuntimeResolutionError,
                "symlink",
            ):
                self._resolve_from(trusted)

    def test_captures_recipe_bytes_before_feature_initialization_source_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = _write_minimal_runtime_tree(root)
            recipe = root / "templates" / "feature_recipes.yaml"
            original_recipe = b"feature_set: original\n"
            recipe.write_bytes(original_recipe)
            (trusted / "features" / "__init__.py").write_text(
                "from pathlib import Path\n"
                "_recipe_path = globals().get('_HEURIBOOST_TRUSTED_RECIPE_PATH')\n"
                "_recipe_bytes = globals().get('_HEURIBOOST_TRUSTED_RECIPE_BYTES')\n"
                "FEATURE_RECIPE_PATH = Path(_recipe_path) if _recipe_path else (\n"
                "    Path(__file__).resolve().parents[2] / 'templates' / 'feature_recipes.yaml'\n"
                ")\n"
                "if _recipe_bytes is None:\n"
                "    _recipe_bytes = FEATURE_RECIPE_PATH.read_bytes()\n"
                "class Registry:\n"
                "    feature_set_name = 'original' if b'original' in _recipe_bytes else 'swapped'\n"
                "    feature_set_version = 1\n"
                "    def names(self):\n"
                "        return ['score']\n"
                "    def feature_versions(self):\n"
                "        return {'score': 1}\n"
                "REGISTRY = Registry()\n"
                "def extract_features(frame):\n"
                "    return frame\n",
                encoding="utf-8",
            )
            original_read_bytes = Path.read_bytes
            swapped = False

            def swap_after_recipe_read(path: Path) -> bytes:
                nonlocal swapped
                source = original_read_bytes(path)
                if path.resolve() == recipe.resolve() and not swapped:
                    recipe.write_bytes(b"feature_set: swapped\n")
                    swapped = True
                return source

            with _without_legacy_modules(), mock.patch.object(
                Path,
                "read_bytes",
                new=swap_after_recipe_read,
            ):
                runtime = self._resolve_from(trusted)

        self.assertTrue(swapped)
        self.assertEqual(
            runtime.feature_recipe_hash,
            hashlib.sha256(original_recipe).hexdigest(),
        )
        self.assertEqual(runtime.registry.feature_set_name, "original")

    def test_manifest_hash_uses_the_exact_bytes_executed_before_a_source_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            repair_cases = trusted / "repair_cases.py"
            repair_cases.write_text(
                repair_cases.read_text(encoding="utf-8").replace(
                    "def compile_repair_inputs(*args):\n    return None",
                    "def compile_repair_inputs(*args):\n    return 'original'",
                ),
                encoding="utf-8",
            )
            source_bytes = {
                path.resolve(): path.read_bytes()
                for path in legacy_runtime._legacy_file_paths(trusted)
            }
            expected_hash = _manifest_from_bytes(trusted, source_bytes)
            original_read_bytes = Path.read_bytes
            swapped = False

            def swap_after_exact_read(path: Path) -> bytes:
                nonlocal swapped
                source = original_read_bytes(path)
                if path.resolve() == repair_cases.resolve() and "repair_cases" not in sys.modules:
                    repair_cases.write_text(
                        "def compile_repair_inputs(*args):\n    return 'swapped'\n",
                        encoding="utf-8",
                    )
                    swapped = True
                return source

            with _without_legacy_modules(), mock.patch.object(
                Path,
                "read_bytes",
                new=swap_after_exact_read,
            ):
                runtime = self._resolve_from(trusted)

            self.assertTrue(swapped)
            self.assertEqual(runtime.code_manifest_hash, expected_hash)
            self.assertEqual(runtime.compile_repair_inputs(), "original")

    def test_ignores_malicious_sys_path_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trusted = _write_minimal_runtime_tree(root / "trusted")
            malicious = root / "malicious"
            malicious.mkdir()
            marker = root / "malicious-imported"
            (malicious / "repair_cases.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
                encoding="utf-8",
            )
            path = [str(malicious), str(trusted), *sys.path]
            original_path = list(path)
            with _without_legacy_modules(), mock.patch.object(sys, "path", path):
                runtime = self._resolve_from(trusted)
                self.assertEqual(runtime.scripts_dir, trusted.resolve())
                self.assertEqual(sys.path, original_path)
                self.assertFalse(marker.exists())

    def test_does_not_write_legacy_bytecode_during_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            with _without_legacy_modules():
                runtime = self._resolve_from(trusted)

            self.assertEqual(runtime.scripts_dir, trusted.resolve())
            self.assertEqual(list(trusted.rglob("*.pyc")), [])

    def test_rejects_non_callable_required_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            (trusted / "repair_cases.py").write_text(
                (trusted / "repair_cases.py")
                .read_text(encoding="utf-8")
                .replace("def load_gates(*args):\n    return []", "load_gates = 3"),
                encoding="utf-8",
            )
            with _without_legacy_modules():
                with self.assertRaisesRegex(
                    LegacyRuntimeResolutionError,
                    "required callable",
                ):
                    self._resolve_from(trusted)

    def test_post_execution_validation_failure_evicts_every_legacy_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            features_init = trusted / "features" / "__init__.py"
            features_init.write_text(
                features_init.read_text(encoding="utf-8").replace(
                    "return ['score']",
                    "return []",
                ),
                encoding="utf-8",
            )
            with _without_legacy_modules():
                with self.assertRaisesRegex(
                    LegacyRuntimeResolutionError,
                    "registry is empty",
                ):
                    self._resolve_from(trusted)
                self.assertFalse(
                    any(
                        name in {"common", "features", "ranking_snapshot", "repair_cases"}
                        or name.startswith("features.")
                        for name in sys.modules
                    )
                )

    def test_rejects_required_callable_reexported_from_another_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted = _write_minimal_runtime_tree(Path(tmp))
            (trusted / "repair_cases.py").write_text(
                (trusted / "repair_cases.py")
                .read_text(encoding="utf-8")
                .replace(
                    "def load_gates(*args):\n    return []",
                    "from common import load_dataset as load_gates",
                ),
                encoding="utf-8",
            )
            with _without_legacy_modules():
                with self.assertRaisesRegex(
                    LegacyRuntimeResolutionError,
                    "unexpected origin",
                ):
                    self._resolve_from(trusted)

    def test_feature_loader_ignores_forged_private_templates_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp)
            (outside / "feature_recipes.yaml").write_text(
                "feature_set: forged\n",
                encoding="utf-8",
            )
            forged = ModuleType("heuriboost_rag.legacy_templates")
            forged.__file__ = str(outside / "__init__.py")
            forged.__path__ = [str(outside)]
            forged.__spec__ = ModuleSpec(
                "heuriboost_rag.legacy_templates",
                loader=None,
                is_package=True,
            )
            forged.__spec__.submodule_search_locations = [str(outside)]
            scripts = PACKAGE_ROOT / "scripts"
            with _without_legacy_modules(), mock.patch.object(
                sys,
                "path",
                [str(scripts), *sys.path],
            ), mock.patch.dict(
                sys.modules,
                {"heuriboost_rag.legacy_templates": forged},
            ):
                features = importlib.import_module("features")

            self.assertEqual(
                Path(features.FEATURE_RECIPE_PATH).resolve(),
                (PACKAGE_ROOT / "templates" / "feature_recipes.yaml").resolve(),
            )


class LegacyWheelTests(unittest.TestCase):
    def test_wheel_contains_private_legacy_runtime_sources_and_feature_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            shutil.copytree(
                PACKAGE_ROOT,
                source,
                ignore=shutil.ignore_patterns(
                    ".venv",
                    "__pycache__",
                    "*.egg-info",
                    "build",
                    "dist",
                ),
            )
            wheel_dir = root / "wheel"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            wheel = next(wheel_dir.glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())

        self.assertTrue(
            {
                "heuriboost_rag/legacy_scripts/__init__.py",
                "heuriboost_rag/legacy_scripts/common.py",
                "heuriboost_rag/legacy_scripts/repair_cases.py",
                "heuriboost_rag/legacy_scripts/ranking_snapshot.py",
                "heuriboost_rag/legacy_scripts/features/__init__.py",
                "heuriboost_rag/legacy_scripts/hpo/__init__.py",
                "heuriboost_rag/legacy_scripts/hpo/engine.py",
                "heuriboost_rag/legacy_templates/feature_recipes.yaml",
                "heuriboost_rag/legacy_templates/reckless_policy.yml",
            }.issubset(names)
        )

    def test_installed_wheel_runs_hpo_help_and_contains_policy_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            shutil.copytree(
                PACKAGE_ROOT,
                source,
                ignore=shutil.ignore_patterns(
                    ".venv",
                    "__pycache__",
                    "*.egg-info",
                    "build",
                    "dist",
                ),
            )
            wheel_dir = root / "wheel"
            build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            install_root = root / "installed"
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-compile",
                    "--target",
                    str(install_root),
                    str(next(wheel_dir.glob("*.whl"))),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(install_root)
            help_result = subprocess.run(
                [
                    sys.executable,
                    str(
                        install_root
                        / "heuriboost_rag"
                        / "legacy_scripts"
                        / "run_hpo.py"
                    ),
                    "--help",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                cwd=root,
            )

            policy_template = (
                install_root
                / "heuriboost_rag"
                / "legacy_templates"
                / "reckless_policy.yml"
            )
            policy_template_exists = policy_template.is_file()

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("Run HPO", help_result.stdout)
        self.assertTrue(policy_template_exists)

    def test_installed_wheel_resolver_writes_no_private_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            shutil.copytree(
                PACKAGE_ROOT,
                source,
                ignore=shutil.ignore_patterns(
                    ".venv",
                    "__pycache__",
                    "*.egg-info",
                    "build",
                    "dist",
                ),
            )
            wheel_dir = root / "wheel"
            build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            install_root = root / "installed"
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-compile",
                    "--target",
                    str(install_root),
                    str(next(wheel_dir.glob("*.whl"))),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            environment = dict(os.environ)
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment["PYTHONPATH"] = str(install_root)
            environment["HEURIBOOST_INSTALLED_ROOT"] = str(install_root)
            check = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path\n"
                    "import os\n"
                    "from heuriboost_rag.backends.legacy_runtime import resolve_legacy_runtime\n"
                    "runtime = resolve_legacy_runtime()\n"
                    "assert Path(runtime.scripts_dir).resolve() == (Path(os.environ['HEURIBOOST_INSTALLED_ROOT']) / 'heuriboost_rag' / 'legacy_scripts').resolve()\n"
                    "assert not list(Path(runtime.scripts_dir).rglob('__pycache__'))\n",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                cwd=root,
            )

        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
