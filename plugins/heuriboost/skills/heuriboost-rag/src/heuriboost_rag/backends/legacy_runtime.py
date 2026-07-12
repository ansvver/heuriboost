"""Trusted resolver for the bundled legacy HeuriBoost repair scripts."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Callable

try:
    import yaml as _TRUSTED_YAML
except ImportError:
    _TRUSTED_YAML = None


class LegacyRuntimeResolutionError(RuntimeError):
    """Raised when the bundled legacy runtime cannot be initialized safely."""


_INITIALIZATION_LOCK = threading.RLock()
_TRUSTED_IMPORT = builtins.__import__
_TRUSTED_BUILTINS = dict(builtins.__dict__)
_LEGACY_MODULE_NAMES = (
    "common",
    "features",
    "features.primitives",
    "features.recipes",
    "features.registry",
    "ranking_snapshot",
    "repair_cases",
)
_EXECUTED_LEGACY_SOURCES = (
    ("features.primitives", "features/primitives.py", False),
    ("features.registry", "features/registry.py", False),
    ("features.recipes", "features/recipes.py", False),
    ("features", "features/__init__.py", True),
    ("common", "common.py", False),
    ("ranking_snapshot", "ranking_snapshot.py", False),
    ("repair_cases", "repair_cases.py", False),
)
_REQUIRED_REPAIR_CALLABLES = (
    "compile_repair_inputs",
    "load_compiled_production_cases",
    "merge_training_frames",
    "train_model_from_frame",
    "load_model",
    "evaluate_model_on_split",
    "evaluate_model_by_domain",
    "evaluate_cases",
    "load_gates",
)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _trusted_scripts_dir() -> Path:
    """Locate the physical legacy root from this resolver's own location."""

    runtime_path = Path(__file__).resolve()
    package_root = runtime_path.parents[1]
    candidates = [package_root / "legacy_scripts"]
    if package_root.parent.name == "src":
        candidates.append(package_root.parent.parent / "scripts")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise LegacyRuntimeResolutionError(
        f"trusted legacy scripts directory does not exist; checked: {rendered}"
    )


def _evict_legacy_modules() -> None:
    """Remove every legacy cache entry before creating fresh exact modules."""

    for module_name in tuple(sys.modules):
        if (
            module_name in {"common", "ranking_snapshot", "repair_cases", "features"}
            or module_name.startswith("features.")
        ):
            sys.modules.pop(module_name, None)


@dataclass(frozen=True)
class _ExactSource:
    module_name: str
    relative_path: str
    path: Path
    source_bytes: bytes
    is_package: bool


@dataclass(frozen=True)
class _ExactRecipe:
    path: Path
    source_bytes: bytes

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()


def _legacy_file_paths(scripts_dir: Path) -> tuple[Path, ...]:
    """Return exactly the source files executed by the direct resolver."""

    return tuple(
        scripts_dir / relative_path
        for _, relative_path, _ in _EXECUTED_LEGACY_SOURCES
    )


def _read_exact_sources(scripts_dir: Path) -> tuple[_ExactSource, ...]:
    """Read all executable legacy sources exactly once before any execution."""

    trusted_dir = scripts_dir.resolve()
    sources: list[_ExactSource] = []
    for module_name, relative_path, is_package in _EXECUTED_LEGACY_SOURCES:
        path = (trusted_dir / relative_path).resolve()
        if not path.is_file() or not _is_within(path, trusted_dir):
            raise LegacyRuntimeResolutionError(
                f"required legacy code file is unavailable: {path}"
            )
        try:
            source_bytes = path.read_bytes()
        except OSError as exc:
            raise LegacyRuntimeResolutionError(
                f"cannot read required legacy code file {path}: {exc}"
            ) from exc
        sources.append(
            _ExactSource(
                module_name=module_name,
                relative_path=relative_path,
                path=path,
                source_bytes=source_bytes,
                is_package=is_package,
            )
        )
    return tuple(sources)


def _manifest_hash(files: tuple[Path, ...], scripts_dir: Path) -> str:
    """Hash legacy files for manually constructed runtime test doubles."""

    digest = hashlib.sha256()
    manifest_root = scripts_dir.resolve().parent
    for path in files:
        resolved = path.resolve()
        if not resolved.is_file() or not _is_within(resolved, manifest_root):
            raise LegacyRuntimeResolutionError(
                f"required legacy code file is unavailable: {resolved}"
            )
        digest.update(resolved.relative_to(manifest_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            source_bytes = resolved.read_bytes()
        except OSError as exc:
            raise LegacyRuntimeResolutionError(
                f"cannot read required legacy code file {resolved}: {exc}"
            ) from exc
        digest.update(source_bytes)
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_hash_from_exact_sources(
    sources: tuple[_ExactSource, ...], scripts_dir: Path
) -> str:
    """Hash the exact bytes that will be executed, never a later file read."""

    digest = hashlib.sha256()
    manifest_root = scripts_dir.resolve().parent
    for source in sources:
        digest.update(
            source.path.resolve()
            .relative_to(manifest_root)
            .as_posix()
            .encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(source.source_bytes)
        digest.update(b"\0")
    return digest.hexdigest()


def _feature_recipe_path(scripts_dir: Path) -> Path:
    root = scripts_dir.resolve().parent
    for relative in (
        "legacy_templates/feature_recipes.yaml",
        "templates/feature_recipes.yaml",
    ):
        candidate = root / relative
        if candidate.is_symlink():
            raise LegacyRuntimeResolutionError(
                f"legacy feature recipe symlink is not allowed: {candidate}"
            )
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if not _is_within(resolved, root):
            raise LegacyRuntimeResolutionError(
                f"legacy feature recipe is outside the trusted root: {candidate}"
            )
        if not resolved.is_file():
            raise LegacyRuntimeResolutionError(
                f"legacy feature recipe is not a regular file: {candidate}"
            )
        return resolved
    raise LegacyRuntimeResolutionError(
        f"legacy feature recipe source is missing below {root}"
    )


def _read_exact_recipe(scripts_dir: Path) -> _ExactRecipe:
    """Read the trusted feature recipe once before feature code executes."""

    recipe_path = _feature_recipe_path(scripts_dir)
    try:
        source_bytes = recipe_path.read_bytes()
    except OSError as exc:
        raise LegacyRuntimeResolutionError(
            f"cannot read trusted legacy feature recipe {recipe_path}: {exc}"
        ) from exc
    return _ExactRecipe(path=recipe_path, source_bytes=source_bytes)


def _safe_legacy_import(
    exact_modules: Mapping[str, ModuleType],
) -> Callable[..., object]:
    """Resolve legacy names from the exact fresh module set only."""

    def trusted_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if level == 0 and name == "yaml":
            if _TRUSTED_YAML is None:
                raise ImportError("PyYAML is unavailable in the trusted runtime")
            return _TRUSTED_YAML
        if level == 0 and name in _LEGACY_MODULE_NAMES:
            module = exact_modules.get(name)
            if module is None or sys.modules.get(name) is not module:
                raise ImportError(f"trusted legacy module binding is unavailable: {name}")
            if fromlist:
                return module
            root_name = name.partition(".")[0]
            root_module = exact_modules.get(root_name)
            if root_module is None or sys.modules.get(root_name) is not root_module:
                raise ImportError(
                    f"trusted legacy package binding is unavailable: {root_name}"
                )
            return root_module
        if level == 0 and (name == "features" or name.startswith("features.")):
            raise ImportError(f"unknown trusted legacy module: {name}")
        return _TRUSTED_IMPORT(name, globals, locals, fromlist, level)

    return trusted_import


def _fresh_module(
    source: _ExactSource,
    trusted_import: Callable[..., object],
) -> ModuleType:
    module = ModuleType(source.module_name)
    # The legacy source itself legitimately uses __file__ to locate its adjacent
    # recipe.  This value is supplied from the resolver's physical source path;
    # it is never read back as evidence about what code was executed.
    module.__file__ = str(source.path)
    module.__package__ = (
        source.module_name if source.is_package else source.module_name.rpartition(".")[0]
    )
    module_builtins = dict(_TRUSTED_BUILTINS)
    module_builtins["__import__"] = trusted_import
    module.__dict__["__builtins__"] = module_builtins
    if source.is_package:
        module.__path__ = [str(source.path.parent)]
    return module


def _register_child_module(module: ModuleType) -> None:
    parent_name, _, child_name = module.__name__.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if isinstance(parent, ModuleType):
            setattr(parent, child_name, module)


def _execute_exact_sources(
    sources: tuple[_ExactSource, ...],
    recipe: _ExactRecipe,
) -> dict[str, ModuleType]:
    """Execute a fixed dependency order without invoking import resolution."""

    sources_by_name = {source.module_name: source for source in sources}
    feature_source = sources_by_name.get("features")
    if feature_source is None:
        raise LegacyRuntimeResolutionError("required features package source is missing")

    modules: dict[str, ModuleType] = {}
    trusted_import = _safe_legacy_import(modules)
    feature_package = _fresh_module(feature_source, trusted_import)
    feature_package.__dict__["_HEURIBOOST_TRUSTED_RECIPE_PATH"] = recipe.path
    feature_package.__dict__["_HEURIBOOST_TRUSTED_RECIPE_BYTES"] = recipe.source_bytes
    sys.modules["features"] = feature_package
    modules["features"] = feature_package
    try:
        for source in sources:
            if source.module_name == "features":
                module = feature_package
            else:
                module = _fresh_module(source, trusted_import)
                sys.modules[source.module_name] = module
                _register_child_module(module)
                modules[source.module_name] = module
            code = compile(source.source_bytes, str(source.path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
    except (Exception, SystemExit) as exc:
        _evict_legacy_modules()
        raise LegacyRuntimeResolutionError(
            f"failed to initialize trusted legacy runtime: {exc}"
        ) from exc
    return modules


def _required_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    value = module.__dict__.get(name)
    if not callable(value):
        raise LegacyRuntimeResolutionError(
            f"legacy runtime required callable is invalid: {name}"
        )
    # A direct definition has the exact fresh module globals used for execution.
    # This does not trust mutable __module__ metadata supplied by an importer.
    if getattr(value, "__globals__", None) is not module.__dict__:
        raise LegacyRuntimeResolutionError(
            f"legacy runtime required callable has unexpected origin: {name}"
        )
    return value


@dataclass(frozen=True)
class LegacyRegistryMetadata:
    feature_names: tuple[str, ...]
    feature_set_name: str
    feature_set_version: int
    feature_versions: tuple[tuple[str, int], ...]
    feature_recipe_path: Path

    @property
    def feature_version(self) -> str:
        return f"{self.feature_set_name}:{self.feature_set_version}"

    @property
    def feature_versions_mapping(self) -> dict[str, int]:
        return dict(self.feature_versions)


@dataclass(frozen=True)
class LegacyRuntime:
    """The exact legacy operations required by :class:`XGBoostRagBackend`."""

    scripts_dir: Path
    registry: LegacyRegistryMetadata
    compile_options_type: type
    compile_repair_inputs: Callable[..., Any]
    load_dataset: Callable[..., Any]
    load_compiled_production_cases: Callable[..., Any]
    merge_training_frames: Callable[..., Any]
    train_model_from_frame: Callable[..., Any]
    load_model: Callable[..., Any]
    evaluate_model_on_split: Callable[..., Any]
    evaluate_model_by_domain: Callable[..., Any]
    evaluate_cases: Callable[..., Any]
    load_gates: Callable[..., Any]
    code_manifest_files: tuple[Path, ...]
    fixed_training_params: tuple[tuple[str, object], ...]
    code_manifest_hash_at_load: str | None = None
    feature_recipe_hash_at_load: str | None = None

    @property
    def feature_recipe_path(self) -> Path:
        return self.registry.feature_recipe_path

    @property
    def code_manifest_hash(self) -> str:
        if self.code_manifest_hash_at_load is not None:
            return self.code_manifest_hash_at_load
        return _manifest_hash(self.code_manifest_files, self.scripts_dir)

    @property
    def feature_recipe_hash(self) -> str:
        if self.feature_recipe_hash_at_load is not None:
            return self.feature_recipe_hash_at_load
        try:
            source_bytes = self.feature_recipe_path.read_bytes()
        except OSError as exc:
            raise LegacyRuntimeResolutionError(
                f"cannot read legacy feature recipe {self.feature_recipe_path}: {exc}"
            ) from exc
        return hashlib.sha256(source_bytes).hexdigest()

    @property
    def fixed_training_params_mapping(self) -> dict[str, object]:
        return dict(self.fixed_training_params)


def resolve_legacy_runtime() -> LegacyRuntime:
    """Bind the legacy runtime from exact trusted source bytes only.

    The legacy scripts retain their historical top-level imports (``common`` and
    ``features``).  Fresh modules are registered under those names before their
    dependants execute, so their imports resolve from this exact in-memory set
    rather than from ``sys.path`` or ``sys.meta_path``.
    """

    with _INITIALIZATION_LOCK:
        original_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            trusted_dir = _trusted_scripts_dir().resolve()
            if not trusted_dir.is_dir():
                raise LegacyRuntimeResolutionError(
                    f"trusted legacy scripts directory does not exist: {trusted_dir}"
                )
            sources = _read_exact_sources(trusted_dir)
            recipe = _read_exact_recipe(trusted_dir)
            manifest_hash = _manifest_hash_from_exact_sources(sources, trusted_dir)
            _evict_legacy_modules()
            modules = _execute_exact_sources(sources, recipe)
        except (LegacyRuntimeResolutionError, OSError, SystemExit) as exc:
            raise LegacyRuntimeResolutionError(
                f"failed to initialize trusted legacy runtime: {exc}"
            ) from exc
        finally:
            sys.dont_write_bytecode = original_dont_write_bytecode

        try:
            repair_cases = modules["repair_cases"]
            common = modules["common"]
            features = modules["features"]
            compile_options_type = repair_cases.__dict__.get("CompileOptions")
            if not isinstance(compile_options_type, type):
                raise LegacyRuntimeResolutionError("legacy runtime CompileOptions must be a type")
            registry = features.__dict__.get("REGISTRY")
            if registry is None or not callable(getattr(registry, "names", None)):
                raise LegacyRuntimeResolutionError("legacy feature registry is missing")
            feature_names = tuple(str(name) for name in registry.names())
            if not feature_names:
                raise LegacyRuntimeResolutionError("legacy feature registry is empty")
            feature_set_name = getattr(registry, "feature_set_name", None)
            feature_set_version = getattr(registry, "feature_set_version", None)
            feature_versions = getattr(registry, "feature_versions", None)
            if (
                not isinstance(feature_set_name, str)
                or not feature_set_name
                or type(feature_set_version) is not int
                or not callable(feature_versions)
            ):
                raise LegacyRuntimeResolutionError("legacy feature registry metadata is invalid")
            versions = feature_versions()
            if not isinstance(versions, Mapping):
                raise LegacyRuntimeResolutionError("legacy feature versions are invalid")
            ordered_versions = tuple(
                (name, int(versions[name])) for name in feature_names if name in versions
            )
            if tuple(name for name, _ in ordered_versions) != feature_names:
                raise LegacyRuntimeResolutionError(
                    "legacy feature versions do not match ordered feature names"
                )
            recipe_path = recipe.path
            configured_recipe_path = features.__dict__.get("FEATURE_RECIPE_PATH")
            if (
                not isinstance(configured_recipe_path, (str, Path))
                or Path(configured_recipe_path).resolve() != recipe_path
            ):
                raise LegacyRuntimeResolutionError(
                    "legacy feature registry did not load the trusted recipe source"
                )
            fixed_params = repair_cases.__dict__.get("FIXED_TRAINING_PARAMS")
            if not isinstance(fixed_params, Mapping):
                raise LegacyRuntimeResolutionError(
                    "legacy runtime fixed training params are missing"
                )
            fixed_param_items = tuple(
                sorted((str(key), value) for key, value in fixed_params.items())
            )

            return LegacyRuntime(
                scripts_dir=trusted_dir,
                registry=LegacyRegistryMetadata(
                    feature_names=feature_names,
                    feature_set_name=feature_set_name,
                    feature_set_version=feature_set_version,
                    feature_versions=ordered_versions,
                    feature_recipe_path=recipe_path,
                ),
                compile_options_type=compile_options_type,
                compile_repair_inputs=_required_callable(
                    repair_cases, "compile_repair_inputs"
                ),
                load_dataset=_required_callable(common, "load_dataset"),
                load_compiled_production_cases=_required_callable(
                    repair_cases, "load_compiled_production_cases"
                ),
                merge_training_frames=_required_callable(
                    repair_cases, "merge_training_frames"
                ),
                train_model_from_frame=_required_callable(
                    repair_cases, "train_model_from_frame"
                ),
                load_model=_required_callable(repair_cases, "load_model"),
                evaluate_model_on_split=_required_callable(
                    repair_cases, "evaluate_model_on_split"
                ),
                evaluate_model_by_domain=_required_callable(
                    repair_cases, "evaluate_model_by_domain"
                ),
                evaluate_cases=_required_callable(repair_cases, "evaluate_cases"),
                load_gates=_required_callable(repair_cases, "load_gates"),
                code_manifest_files=tuple(source.path for source in sources),
                fixed_training_params=fixed_param_items,
                code_manifest_hash_at_load=manifest_hash,
                feature_recipe_hash_at_load=recipe.content_hash,
            )
        except (Exception, SystemExit):
            _evict_legacy_modules()
            raise
