"""Consistent local workspace backup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree_if_present(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    elif source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def create_backup(data_dir: Path, output_path: Path) -> Path:
    data_root = Path(data_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".heuriboost-backup-", dir=output.parent) as tmp:
        staging = Path(tmp) / "backup"
        staging.mkdir()
        db_path = data_root / "heuriboost.db"
        if db_path.exists():
            _sqlite_backup(db_path, staging / "heuriboost.db")
        _copy_tree_if_present(data_root / "datasets", staging / "datasets")
        _copy_tree_if_present(data_root / "releases", staging / "releases")
        manifest: dict[str, object] = {"schema_version": 1, "files": {}}
        files = manifest["files"]
        assert isinstance(files, dict)
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            if relative == "backup_manifest.json":
                continue
            files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        manifest_path = staging / "backup_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_archive = output.with_name(f".{output.name}.{next(tempfile._get_candidate_names())}.tmp")
        try:
            with tarfile.open(temporary_archive, "w:gz") as tar:
                for item in sorted(staging.rglob("*")):
                    tar.add(item, arcname=item.relative_to(staging).as_posix(), recursive=False)
            output.unlink(missing_ok=True)
            temporary_archive.replace(output)
        finally:
            if temporary_archive.exists():
                temporary_archive.unlink()
    return output


__all__ = ["create_backup"]
