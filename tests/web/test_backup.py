from __future__ import annotations

from pathlib import Path
import tarfile
import tempfile
import unittest

from heuriboost_rag.web.backup import create_backup
from heuriboost_rag.web.stores.sqlite import SQLiteStore


class BackupTests(unittest.TestCase):
    def test_backup_contains_sqlite_datasets_releases_and_manifest_without_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            store = SQLiteStore(data_dir / "heuriboost.db")
            store.migrate()
            store.append_audit_event("backup.fixture", {"ok": True})
            dataset = data_dir / "datasets" / "dataset-1"
            dataset.mkdir(parents=True)
            (dataset / "normalized.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
            release = data_dir / "releases" / "releases" / "run-1"
            release.mkdir(parents=True)
            (release / "artifact.json").write_text('{"model":true}\n', encoding="utf-8")
            (data_dir / "releases" / "current_model.json").write_text('{"run_id":"run-1"}\n', encoding="utf-8")
            upload = data_dir / "uploads" / "upload-1"
            upload.mkdir(parents=True)
            (upload / "source.csv").write_text("not backed up\n", encoding="utf-8")

            archive = create_backup(data_dir, Path(tmp) / "backup.tar.gz")

            self.assertTrue(archive.is_file())
            with tarfile.open(archive, "r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("heuriboost.db", names)
            self.assertIn("datasets/dataset-1/normalized.jsonl", names)
            self.assertIn("releases/current_model.json", names)
            self.assertIn("releases/releases/run-1/artifact.json", names)
            self.assertIn("backup_manifest.json", names)
            self.assertNotIn("uploads/upload-1/source.csv", names)


if __name__ == "__main__":
    unittest.main()
