"""Read-only import of pre-package Reckless state into an immutable release."""

from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Mapping

from .contracts import ArtifactRef, PromotionReceipt
from .errors import ArtifactIntegrityError
from .hashing import canonical_json_hash, sha256_file
from .release_store import BootstrapReleaseSnapshot, FileReleaseStore


_REQUIRED_LEGACY_FILES = {
    "legacy-ledger": "ledger.json",
    "legacy-gates": "gates.jsonl",
    "legacy-promoted-samples": "promoted_repair_samples.csv",
    "legacy-current-model": "current_model.json",
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"legacy {label} is unavailable",
            stage="MIGRATION",
            details={"path": str(path)},
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactIntegrityError(
            f"legacy {label} must be a regular file",
            stage="MIGRATION",
            details={"path": str(path)},
        )
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"legacy {label} cannot be resolved",
            stage="MIGRATION",
            details={"path": str(path)},
        ) from exc


def _artifact(artifact_type: str, path: Path) -> ArtifactRef:
    source = _regular_file(path, artifact_type)
    return ArtifactRef(
        artifact_type=artifact_type,
        path=source,
        content_hash=sha256_file(source),
        size_bytes=source.stat().st_size,
    )


def _read_current_model(path: Path) -> Mapping[str, object]:
    source = _regular_file(path, "current model pointer")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            "legacy current model pointer is invalid",
            stage="MIGRATION",
            details={"path": str(source)},
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(
            "legacy current model pointer must be a JSON object",
            stage="MIGRATION",
        )
    return value


def _resolve_referenced_file(
    raw_path: object,
    *,
    legacy_dir: Path,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ArtifactIntegrityError(
            f"legacy current model pointer has no {label} path",
            stage="MIGRATION",
        )
    path = Path(raw_path).expanduser()
    candidates = (
        (path,)
        if path.is_absolute()
        else (
            legacy_dir.parent / path,
            legacy_dir.parent.parent / path,
            legacy_dir / path,
        )
    )
    for candidate in candidates:
        try:
            return _regular_file(candidate, label)
        except ArtifactIntegrityError:
            continue
    raise ArtifactIntegrityError(
        f"legacy current model {label} file is unavailable",
        stage="MIGRATION",
        details={"path": raw_path},
    )


def build_legacy_snapshot(legacy_dir: Path) -> BootstrapReleaseSnapshot:
    """Hash legacy state without modifying a single legacy file."""

    root = Path(legacy_dir).expanduser()
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ArtifactIntegrityError(
            "legacy state directory is unavailable",
            stage="MIGRATION",
            details={"path": str(root)},
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ArtifactIntegrityError(
            "legacy state directory must be a non-symlink directory",
            stage="MIGRATION",
            details={"path": str(root)},
        )
    root = root.resolve()
    artifacts = [
        _artifact(artifact_type, root / filename)
        for artifact_type, filename in sorted(_REQUIRED_LEGACY_FILES.items())
    ]
    pointer = _read_current_model(root / "current_model.json")
    model = _resolve_referenced_file(
        pointer.get("model_path"),
        legacy_dir=root,
        label="model",
    )
    artifacts.append(_artifact("legacy-model", model))
    metadata_path = pointer.get("metadata_path")
    if metadata_path is not None:
        artifacts.append(
            _artifact(
                "legacy-model-metadata",
                _resolve_referenced_file(
                    metadata_path,
                    legacy_dir=root,
                    label="model metadata",
                ),
            )
        )
    ordered_artifacts = tuple(sorted(artifacts, key=lambda artifact: artifact.artifact_type))
    content_hash = canonical_json_hash(
        [
            {
                "artifact_type": artifact.artifact_type,
                "content_hash": artifact.content_hash,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in ordered_artifacts
        ]
    )
    return BootstrapReleaseSnapshot(
        run_id=f"legacy-{content_hash[:24]}",
        artifacts=ordered_artifacts,
        content_hash=content_hash,
    )


def migrate_legacy_state(
    legacy_dir: Path,
    release_store: FileReleaseStore,
) -> PromotionReceipt:
    """Import legacy state once, leaving its original files untouched."""

    if not isinstance(release_store, FileReleaseStore):
        raise TypeError("release_store must be a FileReleaseStore")
    snapshot = build_legacy_snapshot(legacy_dir)
    return release_store.import_bootstrap_release(
        snapshot,
        source="legacy-migration",
    )


__all__ = ["build_legacy_snapshot", "migrate_legacy_state"]
