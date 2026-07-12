from dataclasses import replace
from datetime import datetime
import errno
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
import unittest
from unittest import mock

from heuriboost_rag.reckless.contracts import (
    ArtifactRef,
    DatasetRef,
    RepairRequest,
    RunRecord,
    StageManifest,
    to_plain_data,
)
from heuriboost_rag.reckless.errors import (
    ArtifactIntegrityError,
    HeuriBoostError,
    InputBlockedError,
)
from heuriboost_rag.reckless.hashing import (
    ExecutionIdentity,
    ExecutionIdentityProvider,
    atomic_write_json,
    build_run_fingerprint,
    canonical_json_hash,
    sha256_file,
)
from heuriboost_rag.reckless.policy import RecklessPolicy
from heuriboost_rag.reckless.state import RunState
import heuriboost_rag.reckless.storage as storage_module
from heuriboost_rag.reckless.storage import (
    JsonDatasetRepository,
    JsonRunRepository,
    LocalArtifactStore,
    ResumeInspection,
)


class RecklessHashingTests(unittest.TestCase):
    def test_sha256_file_reads_files_larger_than_one_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.bin"
            content = (b"0123456789abcdef" * 70_000) + b"tail"
            path.write_bytes(content)

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(content).hexdigest(),
            )

    def test_canonical_hash_ignores_mapping_key_order(self):
        left = {
            "request": {"workspace": "alpha", "options": {"b": 2, "a": 1}},
            "features": ["title", "body"],
        }
        right = {
            "features": ["title", "body"],
            "request": {"options": {"a": 1, "b": 2}, "workspace": "alpha"},
        }

        self.assertEqual(canonical_json_hash(left), canonical_json_hash(right))

    def test_canonical_hash_changes_for_semantic_difference(self):
        self.assertNotEqual(
            canonical_json_hash({"features": ["title", "body"]}),
            canonical_json_hash({"features": ["body", "title"]}),
        )

    def test_canonical_hash_uses_contract_plain_data_and_rejects_nan(self):
        request = RepairRequest(
            workspace_id="workspace",
            base_dataset_id="base",
            production_cases_id="cases",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"nested": {"value": 1}},
        )
        self.assertEqual(
            canonical_json_hash(request),
            canonical_json_hash(
                {
                    "workspace_id": "workspace",
                    "base_dataset_id": "base",
                    "production_cases_id": "cases",
                    "policy_version": "1",
                    "backend_name": "fake",
                    "requested_by": "tester",
                    "run_options": {"nested": {"value": 1}},
                }
            ),
        )
        with self.assertRaises(ValueError):
            canonical_json_hash({"metric": float("nan")})

    def test_run_fingerprint_is_stable_for_equivalent_inputs_and_paths(self):
        request = self._request(run_options={"b": 2, "a": 1})
        equivalent_request = self._request(run_options={"a": 1, "b": 2})
        base = self._dataset("base", "base-content", "base-schema", "/a/base.csv")
        moved_base = replace(base, path=Path("/different/location/base.csv"))
        cases = self._dataset(
            "cases", "cases-content", "cases-schema", "/a/cases.csv"
        )
        moved_cases = replace(cases, path=Path("/different/location/cases.csv"))
        identity = self._identity(training_params={"eta": 0.1, "depth": 4})
        reordered_identity = self._identity(
            training_params={"depth": 4, "eta": 0.1}
        )

        first = build_run_fingerprint(
            request,
            RecklessPolicy.default(),
            base,
            cases,
            identity,
        )
        second = build_run_fingerprint(
            equivalent_request,
            RecklessPolicy.default(),
            moved_base,
            moved_cases,
            reordered_identity,
        )

        self.assertEqual(first, second)

    def test_run_fingerprint_changes_for_each_stable_execution_input(self):
        request = self._request(run_options={"rounds": 10})
        policy = RecklessPolicy.default()
        base = self._dataset("base", "base-content", "base-schema", "/base.csv")
        cases = self._dataset(
            "cases", "cases-content", "cases-schema", "/cases.csv"
        )
        identity = self._identity()
        baseline = build_run_fingerprint(
            request,
            policy,
            base,
            cases,
            identity,
        )
        variations = {
            "workspace_id": (
                replace(request, workspace_id="another-workspace"),
                policy,
                base,
                cases,
                identity,
            ),
            "base_dataset_id": (
                replace(request, base_dataset_id="another-base"),
                policy,
                base,
                cases,
                identity,
            ),
            "production_cases_id": (
                replace(request, production_cases_id="another-cases"),
                policy,
                base,
                cases,
                identity,
            ),
            "effective_policy_version": (
                replace(request, policy_version="2"),
                policy,
                base,
                cases,
                identity,
            ),
            "policy_content": (
                request,
                RecklessPolicy(acceptance_level="weak"),
                base,
                cases,
                identity,
            ),
            "base_content": (
                request,
                policy,
                replace(base, content_hash="changed"),
                cases,
                identity,
            ),
            "base_schema": (
                request,
                policy,
                replace(base, schema_hash="changed"),
                cases,
                identity,
            ),
            "production_content": (
                request,
                policy,
                base,
                replace(cases, content_hash="changed"),
                identity,
            ),
            "production_schema": (
                request,
                policy,
                base,
                replace(cases, schema_hash="changed"),
                identity,
            ),
            "backend_name": (
                replace(request, backend_name="other"),
                policy,
                base,
                cases,
                identity,
            ),
        }
        identity_variations = {
            "backend_version": replace(identity, backend_version="2.2.0"),
            "feature_names": replace(
                identity, feature_names=("title", "query")
            ),
            "feature_version": replace(identity, feature_version="features-v4"),
            "code_commit": replace(identity, code_commit="def456"),
            "training_params": replace(identity, training_params={"eta": 0.2}),
            "random_seed": replace(identity, random_seed=8),
        }
        for name, changed_identity in identity_variations.items():
            variations[name] = (request, policy, base, cases, changed_identity)

        for name, inputs in variations.items():
            with self.subTest(name=name):
                (
                    changed_request,
                    changed_policy,
                    changed_base,
                    changed_cases,
                    changed_identity,
                ) = inputs
                self.assertNotEqual(
                    baseline,
                    build_run_fingerprint(
                        changed_request,
                        changed_policy,
                        changed_base,
                        changed_cases,
                        changed_identity,
                    ),
                )

    def test_run_fingerprint_ignores_request_audit_and_arbitrary_options(self):
        base = self._dataset("base", "base-content", "base-schema", "/base.csv")
        cases = self._dataset(
            "cases", "cases-content", "cases-schema", "/cases.csv"
        )
        first_request = replace(
            self._request(run_options={}),
            requested_by="first-operator",
            run_options={
                "path": Path("/volatile/first.csv"),
                "timestamp": "2026-07-10T10:00:00+08:00",
                "nested": {"arbitrary": ["first", {"value": 1}]},
            },
        )
        second_request = replace(
            first_request,
            requested_by="second-operator",
            run_options={
                "path": Path("/volatile/second.csv"),
                "timestamp": "2030-01-01T00:00:00Z",
                "nested": {"arbitrary": ["second", {"value": 999}]},
            },
        )

        first = build_run_fingerprint(
            first_request,
            RecklessPolicy.default(),
            base,
            cases,
            self._identity(),
        )
        second = build_run_fingerprint(
            second_request,
            RecklessPolicy.default(),
            base,
            cases,
            self._identity(),
        )

        self.assertEqual(first, second)

    def test_run_fingerprint_rejects_missing_execution_identity(self):
        request = self._request(run_options={})
        base = self._dataset("base", "base-content", "base-schema", "/base.csv")
        cases = self._dataset(
            "cases", "cases-content", "cases-schema", "/cases.csv"
        )

        with self.assertRaisesRegex(ValueError, "execution_identity"):
            build_run_fingerprint(
                request,
                RecklessPolicy.default(),
                base,
                cases,
            )

    def test_execution_identity_requires_complete_stable_material(self):
        invalid_values = {
            "backend_version": {"backend_version": ""},
            "feature_names_empty": {"feature_names": ()},
            "feature_names_blank": {"feature_names": ("query", " ")},
            "feature_version": {"feature_version": " "},
            "code_commit": {"code_commit": ""},
            "training_params": {"training_params": None},
            "random_seed": {"random_seed": None},
        }
        for name, changes in invalid_values.items():
            with self.subTest(name=name):
                values = {
                    "backend_version": "2.1.0",
                    "feature_names": ("query", "title"),
                    "feature_version": "features-v3",
                    "code_commit": "abc123",
                    "training_params": {},
                    "random_seed": 7,
                    **changes,
                }
                with self.assertRaises((TypeError, ValueError)):
                    ExecutionIdentity(**values)

        identity = self._identity(training_params={})
        self.assertEqual(dict(identity.training_params), {})

    def test_execution_identity_provider_is_runtime_checkable(self):
        identity = self._identity()

        class MinimalProvider:
            def execution_identity(self):
                return identity

        provider = MinimalProvider()
        self.assertIsInstance(provider, ExecutionIdentityProvider)
        self.assertIs(provider.execution_identity(), identity)

    def test_atomic_write_json_is_strict_deterministic_and_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "record.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"old":true}\n', encoding="utf-8")

            atomic_write_json(path, {"z": "中文", "a": {"b": 2, "a": 1}})

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"a":{"a":1,"b":2},"z":"中文"}\n',
            )
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"metric": float("inf")})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"a": {"a": 1, "b": 2}, "z": "中文"},
            )

    def test_atomic_write_json_cleans_up_and_leaves_no_partial_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            for target_exists in (False, True):
                with self.subTest(target_exists=target_exists):
                    path = parent / f"record-{target_exists}.json"
                    if target_exists:
                        path.write_text('{"old":true}\n', encoding="utf-8")
                    with mock.patch(
                        "heuriboost_rag.reckless.hashing.os.replace",
                        side_effect=OSError("replace failed"),
                    ):
                        with self.assertRaises(OSError):
                            atomic_write_json(path, {"new": True})

                    if target_exists:
                        self.assertEqual(
                            path.read_text(encoding="utf-8"),
                            '{"old":true}\n',
                        )
                    else:
                        self.assertFalse(path.exists())
                    self.assertEqual(
                        list(parent.glob(f".{path.name}.*.tmp")),
                        [],
                    )

    def test_atomic_write_json_propagates_file_fsync_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            with mock.patch(
                "heuriboost_rag.reckless.hashing.os.fsync",
                side_effect=OSError(errno.EIO, "file fsync failed"),
            ):
                with self.assertRaisesRegex(OSError, "file fsync failed"):
                    atomic_write_json(path, {"value": 1})

            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).glob(".record.json.*.tmp")), [])

    def test_atomic_write_json_propagates_directory_fsync_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            real_fsync = os.fsync
            call_count = 0

            def fail_directory_fsync(descriptor):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError(errno.EIO, "directory fsync failed")
                return real_fsync(descriptor)

            with mock.patch(
                "heuriboost_rag.reckless.hashing.os.fsync",
                side_effect=fail_directory_fsync,
            ):
                with self.assertRaisesRegex(OSError, "directory fsync failed"):
                    atomic_write_json(path, {"value": 1})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(list(Path(tmp).glob(".record.json.*.tmp")), [])

    def test_atomic_write_json_fsyncs_new_directory_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one" / "two" / "record.json"
            real_fsync = os.fsync
            with mock.patch(
                "heuriboost_rag.reckless.hashing.os.fsync",
                wraps=real_fsync,
            ) as fsync:
                atomic_write_json(path, {"value": 1})

            self.assertGreaterEqual(fsync.call_count, 6)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})

    def test_atomic_write_json_handles_concurrent_missing_parent_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = (
                root / "shared" / "nested" / "first.json",
                root / "shared" / "nested" / "second.json",
            )
            create_barrier = threading.Barrier(2)
            writer_barrier = threading.Barrier(2)
            errors = []
            real_mkdir = os.mkdir

            def synchronized_mkdir(path, mode=0o777, *, dir_fd=None):
                if path == "shared":
                    create_barrier.wait(timeout=5)
                return real_mkdir(path, mode=mode, dir_fd=dir_fd)

            def write(path, value):
                try:
                    writer_barrier.wait(timeout=5)
                    atomic_write_json(path, {"value": value})
                except BaseException as exc:
                    errors.append(exc)

            supported_dir_fd = set(os.supports_dir_fd)
            supported_dir_fd.add(synchronized_mkdir)
            with mock.patch(
                "heuriboost_rag.reckless.hashing.os.mkdir",
                new=synchronized_mkdir,
            ), mock.patch(
                "heuriboost_rag.reckless.hashing.os.supports_dir_fd",
                new=frozenset(supported_dir_fd),
            ):
                threads = (
                    threading.Thread(target=write, args=(paths[0], "first")),
                    threading.Thread(target=write, args=(paths[1], "second")),
                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8")), {"value": "first"})
            self.assertEqual(json.loads(paths[1].read_text(encoding="utf-8")), {"value": "second"})

    @staticmethod
    def _request(*, run_options):
        return RepairRequest(
            workspace_id="workspace",
            base_dataset_id="base",
            production_cases_id="cases",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options=run_options,
        )

    @staticmethod
    def _dataset(dataset_id, content_hash, schema_hash, path):
        return DatasetRef(
            dataset_id=dataset_id,
            role=dataset_id,
            path=Path(path),
            content_hash=content_hash,
            schema_hash=schema_hash,
            metadata={"source": "test"},
        )

    @staticmethod
    def _identity(*, training_params=None):
        return ExecutionIdentity(
            backend_version="2.1.0",
            feature_names=("query", "title"),
            feature_version="features-v3",
            code_commit="abc123",
            training_params=(
                {"eta": 0.1} if training_params is None else training_params
            ),
            random_seed=7,
        )


class JsonDatasetRepositoryTests(unittest.TestCase):
    def test_dataset_repository_rejects_symlinked_root_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"
            outside.mkdir()
            root_link = base / "root-link"
            root_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                JsonDatasetRepository(root_link)

            root = base / "root"
            root.mkdir()
            (root / "datasets").symlink_to(outside, target_is_directory=True)
            repository = JsonDatasetRepository(root)
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                repository.save(self._dataset())
            self.assertEqual(list(outside.iterdir()), [])

    def test_dataset_repository_rejects_symlinked_record_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = JsonDatasetRepository(root)
            dataset = self._dataset()
            repository.save(dataset)
            record = root / "datasets" / dataset.dataset_id / "dataset.json"
            external = root / "external-record.json"
            record.replace(external)
            record.symlink_to(external)

            with self.assertRaises((OSError, RuntimeError, ValueError)):
                repository.get(dataset.dataset_id)

    def test_dataset_round_trip_preserves_contract_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonDatasetRepository(Path(tmp))
            dataset = self._dataset()

            saved = repository.save(dataset)
            loaded = repository.get(dataset.dataset_id)

            self.assertEqual(saved, dataset)
            self.assertEqual(loaded, dataset)
            self.assertIsInstance(loaded, DatasetRef)
            self.assertIsInstance(loaded.path, Path)
            self.assertEqual(dict(loaded.metadata), {"rows": 3, "source": "test"})
            record_path = Path(tmp) / "datasets" / "base-v1" / "dataset.json"
            self.assertEqual(
                json.loads(record_path.read_text(encoding="utf-8"))["schema_version"],
                1,
            )

    def test_dataset_records_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonDatasetRepository(Path(tmp))
            dataset = self._dataset()
            repository.save(dataset)

            with self.assertRaises(FileExistsError):
                repository.save(dataset)
            with self.assertRaises(FileExistsError):
                repository.save(replace(dataset, content_hash="changed"))

    def test_dataset_repository_rejects_invalid_contract_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonDatasetRepository(Path(tmp))
            invalid = replace(self._dataset(), role=42)

            with self.assertRaises(ValueError):
                repository.save(invalid)

            self.assertFalse((Path(tmp) / "datasets" / "base-v1").exists())

    def test_dataset_repository_rejects_unsafe_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonDatasetRepository(Path(tmp))
            for dataset_id in (
                "",
                ".",
                "..",
                "../escape",
                "nested/id",
                "nested\\id",
                "/absolute",
                "has space",
            ):
                with self.subTest(dataset_id=dataset_id):
                    with self.assertRaises(ValueError):
                        repository.get(dataset_id)
                    with self.assertRaises(ValueError):
                        repository.save(replace(self._dataset(), dataset_id=dataset_id))

    def test_dataset_repository_rejects_corrupt_records(self):
        corrupt_records = {
            "malformed": "{not-json",
            "non_finite": (
                '{"record":{"content_hash":"content","dataset_id":"base-v1",'
                '"metadata":{"rows":NaN},"path":"/data/base.csv",'
                '"role":"base","schema_hash":"schema"},"schema_version":1}'
            ),
            "overflow_float": (
                '{"record":{"content_hash":"content","dataset_id":"base-v1",'
                '"metadata":{"score":1e9999},"path":"/data/base.csv",'
                '"role":"base","schema_hash":"schema"},"schema_version":1}'
            ),
            "duplicate_key": (
                '{"record":{"content_hash":"content","dataset_id":"base-v1",'
                '"dataset_id":"other","metadata":{},"path":"/data/base.csv",'
                '"role":"base","schema_hash":"schema"},"schema_version":1}'
            ),
            "wrong_schema": json.dumps(
                {
                    "schema_version": 2,
                    "record": self._dataset_payload(),
                }
            ),
            "unknown_field": json.dumps(
                {
                    "schema_version": 1,
                    "record": {**self._dataset_payload(), "unexpected": True},
                }
            ),
            "wrong_type": json.dumps(
                {
                    "schema_version": 1,
                    "record": {**self._dataset_payload(), "path": 42},
                }
            ),
            "id_mismatch": json.dumps(
                {
                    "schema_version": 1,
                    "record": {
                        **self._dataset_payload(),
                        "dataset_id": "other-v1",
                    },
                }
            ),
        }
        for name, content in corrupt_records.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                repository = JsonDatasetRepository(Path(tmp))
                record_path = (
                    Path(tmp) / "datasets" / "base-v1" / "dataset.json"
                )
                record_path.parent.mkdir(parents=True)
                record_path.write_text(content, encoding="utf-8")

                with self.assertRaises(ValueError):
                    repository.get("base-v1")

    def test_dataset_repository_bounds_record_size_and_depth(self):
        records = {
            "oversized": (b" " * 1_100_000, "exceeds"),
            "too_deep": (
                ("[" * 70 + "0" + "]" * 70).encode("ascii"),
                "nesting depth",
            ),
        }
        for name, (content, message) in records.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repository = JsonDatasetRepository(root)
                record_path = root / "datasets" / "base-v1" / "dataset.json"
                record_path.parent.mkdir(parents=True)
                record_path.write_bytes(content)

                with self.assertRaisesRegex(ValueError, message):
                    repository.get("base-v1")

    @staticmethod
    def _dataset():
        return DatasetRef(
            dataset_id="base-v1",
            role="base",
            path=Path("/data/base.csv"),
            content_hash="content",
            schema_hash="schema",
            metadata={"source": "test", "rows": 3},
        )

    @staticmethod
    def _dataset_payload():
        return {
            "dataset_id": "base-v1",
            "role": "base",
            "path": "/data/base.csv",
            "content_hash": "content",
            "schema_hash": "schema",
            "metadata": {"source": "test", "rows": 3},
        }


class JsonRunRepositoryTests(unittest.TestCase):
    def test_run_repository_rejects_symlinked_root_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"
            outside.mkdir()
            root_link = base / "root-link"
            root_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                JsonRunRepository(root_link)

            root = base / "root"
            root.mkdir()
            (root / "runs").symlink_to(outside, target_is_directory=True)
            repository = JsonRunRepository(root)
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                repository.create(self._request(), "policy", "input")
            self.assertEqual(list(outside.iterdir()), [])

    def test_run_repository_rejects_symlinked_record_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = JsonRunRepository(root)
            record = repository.create(self._request(), "policy", "input")
            record_path = root / "runs" / record.run_id / "run.json"
            external = root / "external-run.json"
            record_path.replace(external)
            record_path.symlink_to(external)

            with self.assertRaises((OSError, RuntimeError, ValueError)):
                repository.get(record.run_id)

    def test_create_and_get_round_trip_preserves_contract_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            created = repository.create(
                self._request(),
                policy_hash="policy-hash",
                input_hash="input-hash",
            )
            loaded = repository.get(created.run_id)

            self.assertEqual(loaded, created)
            self.assertIsInstance(loaded, RunRecord)
            self.assertIsInstance(loaded.request, RepairRequest)
            self.assertEqual(created.state, RunState.RECEIVED.value)
            self.assertEqual(created.version, 1)
            self.assertTrue(created.run_id.startswith("run-"))
            self.assertTrue(
                (Path(tmp) / "runs" / created.run_id / "run.json").is_file()
            )

    def test_create_generates_collision_resistant_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            run_ids = {
                repository.create(self._request(), "policy", "input").run_id
                for _ in range(50)
            }
            self.assertEqual(len(run_ids), 50)

    def test_create_rejects_invalid_request_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            invalid_request = replace(
                self._request(),
                base_dataset_id="../escape",
            )

            with self.assertRaises(ValueError):
                repository.create(invalid_request, "policy", "input")

            runs_dir = Path(tmp) / "runs"
            self.assertEqual(
                [] if not runs_dir.exists() else list(runs_dir.glob("run-*")),
                [],
            )

    def test_save_increments_version_and_rejects_stale_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_repository = JsonRunRepository(Path(tmp))
            second_repository = JsonRunRepository(Path(tmp))
            created = first_repository.create(self._request(), "policy", "input")
            first_reader = first_repository.get(created.run_id)
            stale_reader = second_repository.get(created.run_id)

            saved = first_repository.save(
                replace(first_reader, metadata={"writer": "first"})
            )

            self.assertEqual(saved.version, first_reader.version + 1)
            self.assertEqual(dict(saved.metadata), {"writer": "first"})
            with self.assertRaisesRegex(ValueError, "stale"):
                second_repository.save(
                    replace(stale_reader, metadata={"writer": "second"})
                )
            self.assertEqual(
                dict(first_repository.get(created.run_id).metadata),
                {"writer": "first"},
            )

    def test_save_rejects_noop_and_immutable_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            record = repository.create(self._request(), "policy", "input")

            with self.assertRaisesRegex(ValueError, "no changes"):
                repository.save(record)
            immutable_changes = (
                replace(record, run_id="run-other"),
                replace(record, request=replace(record.request, requested_by="other")),
                replace(record, policy_hash="other"),
                replace(record, input_hash="other"),
            )
            for changed in immutable_changes:
                with self.subTest(field=changed):
                    with self.assertRaises(ValueError):
                        repository.save(changed)

    def test_save_rejects_invalid_record_before_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            record = repository.create(self._request(), "policy", "input")
            invalid = replace(
                record,
                metadata={"attempted": True},
                error={"not": "a structured error"},
            )

            with self.assertRaises(ValueError):
                repository.save(invalid)

            self.assertEqual(repository.get(record.run_id), record)

    def test_legal_transition_merges_metadata_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            created = repository.create(self._request(), "policy", "input")
            saved = repository.save(replace(created, metadata={"existing": 1}))

            transitioned = repository.transition(
                saved.run_id,
                RunState.VALIDATING,
                {"stage": "validation"},
            )

            self.assertEqual(transitioned.state, RunState.VALIDATING.value)
            self.assertEqual(
                dict(transitioned.metadata),
                {"existing": 1, "stage": "validation"},
            )
            self.assertEqual(transitioned.request, created.request)
            self.assertEqual(transitioned.policy_hash, created.policy_hash)
            self.assertEqual(transitioned.input_hash, created.input_hash)

    def test_illegal_transition_and_terminal_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            created = repository.create(self._request(), "policy", "input")
            with self.assertRaisesRegex(ValueError, "Invalid run-state transition"):
                repository.transition(created.run_id, RunState.TRAINING)

            cancelled = repository.transition(created.run_id, RunState.CANCELLED)
            with self.assertRaisesRegex(ValueError, "Invalid run-state transition"):
                repository.transition(cancelled.run_id, RunState.VALIDATING)
            with self.assertRaisesRegex(ValueError, "Invalid run-state transition"):
                repository.save(replace(cancelled, state=RunState.RECEIVED.value))

    def test_fail_serializes_structured_matching_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            created = repository.create(self._request(), "policy", "input")
            validating = repository.transition(created.run_id, RunState.VALIDATING)
            error = InputBlockedError(
                "labels are missing",
                stage="VALIDATING",
                run_id=created.run_id,
                details={"missing": ["label"]},
                operator_action="Add authoritative labels.",
            )

            failed = repository.fail(
                validating.run_id,
                RunState.BLOCKED_INPUT,
                error,
            )

            self.assertEqual(failed.state, RunState.BLOCKED_INPUT.value)
            self.assertEqual(to_plain_data(failed.error), error.to_dict())
            loaded = repository.get(failed.run_id)
            self.assertEqual(to_plain_data(loaded.error), error.to_dict())
            with self.assertRaisesRegex(ValueError, "Invalid run-state transition"):
                repository.transition(failed.run_id, RunState.VALIDATING)

    def test_fail_rejects_non_failure_and_mismatched_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            created = repository.create(self._request(), "policy", "input")
            validating = repository.transition(created.run_id, RunState.VALIDATING)
            error = InputBlockedError("blocked", stage="VALIDATING")

            with self.assertRaises(ValueError):
                repository.fail(validating.run_id, RunState.COMPILED, error)
            with self.assertRaises(ValueError):
                repository.fail(
                    validating.run_id,
                    RunState.FAILED_INTERNAL,
                    error,
                )

    def test_run_repository_rejects_unsafe_ids_and_corrupt_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            for run_id in ("", ".", "..", "../escape", "nested/id", "has space"):
                with self.subTest(run_id=run_id):
                    with self.assertRaises(ValueError):
                        repository.get(run_id)

            created = repository.create(self._request(), "policy", "input")
            record_path = Path(tmp) / "runs" / created.run_id / "run.json"
            record_path.write_text('{"schema_version":1,"record":NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                repository.get(created.run_id)

    def test_run_repository_bounds_record_size_and_depth(self):
        records = {
            "oversized": (b" " * 1_100_000, "exceeds"),
            "too_deep": (
                ("[" * 70 + "0" + "]" * 70).encode("ascii"),
                "nesting depth",
            ),
        }
        for name, (content, message) in records.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repository = JsonRunRepository(root)
                record_path = root / "runs" / "run-1" / "run.json"
                record_path.parent.mkdir(parents=True)
                record_path.write_bytes(content)

                with self.assertRaisesRegex(ValueError, message):
                    repository.get("run-1")

    def test_run_record_schema_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            created = repository.create(self._request(), "policy", "input")
            record_path = Path(tmp) / "runs" / created.run_id / "run.json"
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            payload["record"]["request"]["unexpected"] = True
            record_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                repository.get(created.run_id)

    @staticmethod
    def _request():
        return RepairRequest(
            workspace_id="workspace",
            base_dataset_id="base",
            production_cases_id="cases",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"rounds": 10},
        )


class LocalArtifactStoreTests(unittest.TestCase):
    def test_complete_stage_is_create_once_and_equivalent_retry_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "artifact.bin"
            artifact.write_bytes(b"content")

            first = store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"artifact": artifact},
            )
            manifest_path = (
                root
                / "runs"
                / "run-1"
                / "stages"
                / "COMPILED"
                / "stage_manifest.json"
            )
            first_bytes = manifest_path.read_bytes()
            first_artifact = first.artifacts[0]
            self.assertEqual(
                first_artifact.path.name,
                f"artifact-{first_artifact.content_hash}.snapshot",
            )
            second = store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"artifact": artifact},
            )

            self.assertEqual(second, first)
            self.assertEqual(manifest_path.read_bytes(), first_bytes)

    def test_complete_stage_rejects_conflicting_input_artifact_and_status(self):
        conflict_kinds = ("input", "artifact", "status")
        for conflict_kind in conflict_kinds:
            with self.subTest(conflict_kind=conflict_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = LocalArtifactStore(root)
                artifact = store.run_dir("run-1") / "artifact.bin"
                artifact.write_bytes(b"first")
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": artifact},
                )
                manifest_path = (
                    root
                    / "runs"
                    / "run-1"
                    / "stages"
                    / "COMPILED"
                    / "stage_manifest.json"
                )
                if conflict_kind == "artifact":
                    artifact.write_bytes(b"second")
                    changed_input = "input-hash"
                elif conflict_kind == "status":
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload["status"] = "FAILED"
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    changed_input = "input-hash"
                else:
                    changed_input = "changed-input"
                existing = manifest_path.read_bytes()
                snapshot_dir = manifest_path.parent / "artifacts"
                existing_snapshots = sorted(path.name for path in snapshot_dir.iterdir())

                with self.assertRaises(ArtifactIntegrityError):
                    store.complete_stage(
                        "run-1",
                        "COMPILED",
                        changed_input,
                        {"artifact": artifact},
                    )

                self.assertEqual(manifest_path.read_bytes(), existing)
                self.assertEqual(
                    sorted(path.name for path in snapshot_dir.iterdir()),
                    existing_snapshots,
                )

    def test_complete_stage_serializes_concurrent_conflicting_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_store = LocalArtifactStore(root)
            second_store = LocalArtifactStore(root)
            run_dir = first_store.run_dir("run-1")
            first_artifact = run_dir / "first.bin"
            second_artifact = run_dir / "second.bin"
            first_artifact.write_bytes(b"first")
            second_artifact.write_bytes(b"second")
            barrier = threading.Barrier(2)
            results = []

            def complete(store, input_hash, artifact):
                barrier.wait()
                try:
                    results.append(
                        store.complete_stage(
                            "run-1",
                            "COMPILED",
                            input_hash,
                            {"artifact": artifact},
                        )
                    )
                except BaseException as exc:
                    results.append(exc)

            threads = (
                threading.Thread(
                    target=complete,
                    args=(first_store, "first-input", first_artifact),
                ),
                threading.Thread(
                    target=complete,
                    args=(second_store, "second-input", second_artifact),
                ),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(sum(isinstance(item, StageManifest) for item in results), 1)
            self.assertEqual(
                sum(isinstance(item, ArtifactIntegrityError) for item in results),
                1,
            )

    def test_complete_stage_serializes_concurrent_equivalent_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_store = LocalArtifactStore(root)
            second_store = LocalArtifactStore(root)
            run_dir = first_store.run_dir("run-1")
            first_artifact = run_dir / "first.bin"
            second_artifact = run_dir / "second.bin"
            first_artifact.write_bytes(b"same-content")
            second_artifact.write_bytes(b"same-content")
            barrier = threading.Barrier(2)
            results = []

            def complete(store, artifact):
                barrier.wait()
                try:
                    results.append(
                        store.complete_stage(
                            "run-1",
                            "COMPILED",
                            "input-hash",
                            {"artifact": artifact},
                        )
                    )
                except BaseException as exc:
                    results.append(exc)

            threads = (
                threading.Thread(target=complete, args=(first_store, first_artifact)),
                threading.Thread(target=complete, args=(second_store, second_artifact)),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(results), 2)
            self.assertTrue(all(isinstance(item, StageManifest) for item in results))
            self.assertEqual(results[0], results[1])
            manifest_path = (
                root
                / "runs"
                / "run-1"
                / "stages"
                / "COMPILED"
                / "stage_manifest.json"
            )
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                to_plain_data(results[0]),
                {key: persisted[key] for key in to_plain_data(results[0])},
            )
            self.assertEqual(
                [path.name for path in (manifest_path.parent / "artifacts").iterdir()],
                [results[0].artifacts[0].path.name],
            )

    def test_stage_directory_lock_survives_lock_entry_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(Path(tmp))
            stage_descriptor = store._secure_root.open_dir(
                ("runs", "run-1", "stages", "COMPILED"),
                create=True,
            )
            first_descriptor = store._secure_root.open_dir(
                ("runs", "run-1", "stages", "COMPILED"),
                create=False,
            )
            second_descriptor = store._secure_root.open_dir(
                ("runs", "run-1", "stages", "COMPILED"),
                create=False,
            )
            first_entered = threading.Event()
            allow_first_exit = threading.Event()
            first_released = threading.Event()
            second_attempting = threading.Event()
            second_lock_acquired = threading.Event()
            second_entered = threading.Event()
            errors = []
            real_flock = storage_module.fcntl.flock

            def observed_flock(descriptor, operation):
                result = real_flock(descriptor, operation)
                if (
                    threading.current_thread().name == "replacement-lock-contender"
                    and operation == storage_module.fcntl.LOCK_EX
                ):
                    second_lock_acquired.set()
                return result

            def hold_first_lock():
                try:
                    with storage_module._locked_file(
                        first_descriptor,
                        ".stage.lock",
                    ):
                        first_entered.set()
                        if not allow_first_exit.wait(timeout=5):
                            errors.append(RuntimeError("first lock release timed out"))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    first_released.set()
                    os.close(first_descriptor)

            def contend_for_replaced_lock():
                second_attempting.set()
                try:
                    with storage_module._locked_file(
                        second_descriptor,
                        ".stage.lock",
                    ):
                        second_entered.set()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    os.close(second_descriptor)

            first_thread = threading.Thread(target=hold_first_lock)
            second_thread = threading.Thread(
                target=contend_for_replaced_lock,
                name="replacement-lock-contender",
            )
            try:
                with mock.patch.object(
                    storage_module.fcntl,
                    "flock",
                    side_effect=observed_flock,
                ):
                    first_thread.start()
                    self.assertTrue(first_entered.wait(timeout=5))
                    try:
                        os.unlink(".stage.lock", dir_fd=stage_descriptor)
                    except FileNotFoundError:
                        pass
                    replacement = os.open(
                        ".stage.lock",
                        os.O_RDWR
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=stage_descriptor,
                    )
                    os.close(replacement)
                    second_thread.start()
                    self.assertTrue(second_attempting.wait(timeout=5))
                    self.assertFalse(second_lock_acquired.wait(timeout=1))
                    self.assertFalse(second_entered.is_set())
            finally:
                allow_first_exit.set()
                first_thread.join(timeout=5)
                second_thread.join(timeout=5)
                os.close(stage_descriptor)

            self.assertTrue(first_released.is_set())
            self.assertTrue(second_entered.is_set())
            self.assertEqual(errors, [])

    def test_complete_stage_rejects_cross_run_artifact_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(Path(tmp))
            artifact = store.run_dir("run-2") / "artifact.bin"
            artifact.write_bytes(b"other-run")

            with self.assertRaises(ValueError):
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": artifact},
                )

    def test_artifact_store_rejects_symlinked_root_and_stage_parent_without_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"
            outside.mkdir()
            root_link = base / "root-link"
            root_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                LocalArtifactStore(root_link)

            root = base / "root"
            root.mkdir()
            store = LocalArtifactStore(root)
            run_dir = store.run_dir("run-1")
            artifact = run_dir / "artifact.bin"
            artifact.write_bytes(b"content")
            (run_dir / "stages").symlink_to(outside, target_is_directory=True)

            with self.assertRaises((OSError, RuntimeError, ValueError)):
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": artifact},
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_artifact_store_rejects_symlinked_artifact_leaf_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            run_dir = store.run_dir("run-1")
            outside = root / "outside"
            outside.mkdir()
            external = outside / "external.bin"
            external.write_bytes(b"external")

            leaf_link = run_dir / "leaf-link.bin"
            leaf_link.symlink_to(external)
            with self.assertRaises(ValueError):
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": leaf_link},
                )

            parent_link = run_dir / "parent-link"
            parent_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": parent_link / "external.bin"},
                )

    def test_artifact_store_rejects_external_symlinked_source_parent(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            store = LocalArtifactStore(Path(tmp))
            real_parent = Path(outside) / "real"
            real_parent.mkdir()
            source = real_parent / "artifact.bin"
            source.write_bytes(b"external")
            linked_parent = Path(outside) / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaises(ValueError):
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": linked_parent / "artifact.bin"},
                )

    def test_artifact_snapshot_rejects_same_size_concurrent_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "artifact.bin"
            artifact.write_bytes(b"original")
            real_read = os.read
            mutated = False

            def racing_read(descriptor, size):
                nonlocal mutated
                data = real_read(descriptor, size)
                if data and not mutated:
                    mutated = True
                    artifact.write_bytes(b"modified")
                return data

            with mock.patch(
                "heuriboost_rag.reckless.storage.os.read",
                side_effect=racing_read,
            ):
                with self.assertRaises(ArtifactIntegrityError):
                    store.complete_stage(
                        "run-1",
                        "COMPILED",
                        "input-hash",
                        {"artifact": artifact},
                    )

            self.assertFalse(
                (
                    root
                    / "runs"
                    / "run-1"
                    / "stages"
                    / "COMPILED"
                    / "stage_manifest.json"
                ).exists()
            )

    def test_manifest_binds_validated_bytes_across_source_symlink_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "artifact.bin"
            artifact.write_bytes(b"validated-bytes")
            replacement = store.run_dir("run-1") / "replacement.bin"
            replacement.write_bytes(b"different-bytes")
            original_hash_open_file = __import__(
                "heuriboost_rag.reckless.storage",
                fromlist=["_hash_open_file"],
            )._hash_open_file
            call_count = 0

            def swap_after_validation(descriptor):
                nonlocal call_count
                result = original_hash_open_file(descriptor)
                call_count += 1
                if call_count == 2:
                    artifact.unlink()
                    artifact.symlink_to(replacement)
                return result

            with mock.patch(
                "heuriboost_rag.reckless.storage._hash_open_file",
                side_effect=swap_after_validation,
            ):
                manifest = store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"artifact": artifact},
                )

            published = root / manifest.artifacts[0].path
            self.assertNotEqual(published, artifact)
            self.assertEqual(published.read_bytes(), b"validated-bytes")
            self.assertTrue(store.can_resume("run-1", "COMPILED", "input-hash"))

    def test_complete_stage_writes_sorted_root_relative_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            output_dir = store.run_dir("run-1") / "outputs"
            output_dir.mkdir(parents=True)
            zeta_path = output_dir / "zeta.bin"
            alpha_path = output_dir / "alpha.bin"
            zeta_path.write_bytes(b"zeta")
            alpha_path.write_bytes(b"alpha")

            manifest = store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"zeta": zeta_path, "alpha": alpha_path},
            )

            self.assertIsInstance(manifest, StageManifest)
            self.assertEqual(
                [artifact.artifact_type for artifact in manifest.artifacts],
                ["alpha", "zeta"],
            )
            self.assertTrue(
                all(isinstance(artifact, ArtifactRef) for artifact in manifest.artifacts)
            )
            self.assertEqual(
                [artifact.path.as_posix() for artifact in manifest.artifacts],
                [
                    "runs/run-1/stages/COMPILED/artifacts/"
                    f"alpha-{sha256_file(alpha_path)}.snapshot",
                    "runs/run-1/stages/COMPILED/artifacts/"
                    f"zeta-{sha256_file(zeta_path)}.snapshot",
                ],
            )
            self.assertEqual(
                [artifact.size_bytes for artifact in manifest.artifacts],
                [5, 4],
            )
            self.assertEqual(
                [artifact.content_hash for artifact in manifest.artifacts],
                [sha256_file(alpha_path), sha256_file(zeta_path)],
            )
            started = datetime.fromisoformat(manifest.started_at)
            completed = datetime.fromisoformat(manifest.completed_at)
            self.assertIsNotNone(started.tzinfo)
            self.assertIsNotNone(completed.tzinfo)
            self.assertLessEqual(started, completed)

            manifest_path = (
                root
                / "runs"
                / "run-1"
                / "stages"
                / "COMPILED"
                / "stage_manifest.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [artifact["artifact_type"] for artifact in payload["artifacts"]],
                ["alpha", "zeta"],
            )
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertIs(type(payload["duration_ms"]), int)
            self.assertGreaterEqual(payload["duration_ms"], 0)
            self.assertEqual(
                {key: payload[key] for key in to_plain_data(manifest)},
                to_plain_data(manifest),
            )

    def test_complete_stage_bounds_open_source_descriptors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(Path(tmp))
            run_dir = store.run_dir("run-1")
            artifacts = {}
            for index in range(8):
                path = run_dir / f"artifact-{index}.bin"
                path.write_bytes(f"content-{index}".encode("utf-8"))
                artifacts[f"artifact-{index}"] = path

            real_open_artifact_source = store._open_artifact_source
            real_close = os.close
            source_descriptors = set()
            peak_open_source_descriptors = 0

            def tracked_open_artifact_source(run_id, path):
                nonlocal peak_open_source_descriptors
                descriptor, source_path = real_open_artifact_source(run_id, path)
                source_descriptors.add(descriptor)
                peak_open_source_descriptors = max(
                    peak_open_source_descriptors,
                    len(source_descriptors),
                )
                return descriptor, source_path

            def tracked_close(descriptor):
                source_descriptors.discard(descriptor)
                return real_close(descriptor)

            with mock.patch.object(
                store,
                "_open_artifact_source",
                side_effect=tracked_open_artifact_source,
            ), mock.patch(
                "heuriboost_rag.reckless.storage.os.close",
                side_effect=tracked_close,
            ):
                manifest = store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    artifacts,
                )

            self.assertEqual(len(manifest.artifacts), len(artifacts))
            self.assertEqual(peak_open_source_descriptors, 1)
            self.assertEqual(source_descriptors, set())

    def test_can_resume_checks_input_stage_size_hash_and_existence(self):
        def overwrite(path, content):
            path.chmod(0o600)
            path.write_bytes(content)

        mutations = {
            "same_size_tamper": lambda path: overwrite(path, b"wxyz"),
            "size_tamper": lambda path: overwrite(path, b"longer"),
            "missing": lambda path: path.unlink(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = LocalArtifactStore(root)
                artifact_path = store.run_dir("run-1") / "artifact.bin"
                artifact_path.write_bytes(b"abcd")
                manifest = store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"compiled": artifact_path},
                )
                self.assertTrue(
                    store.can_resume("run-1", "COMPILED", "input-hash")
                )
                self.assertFalse(store.can_resume("run-1", "COMPILED", "changed"))
                self.assertFalse(store.can_resume("run-1", "TRAINED", "input-hash"))

                snapshot_path = root / manifest.artifacts[0].path
                mutate(snapshot_path)

                self.assertFalse(
                    store.can_resume("run-1", "COMPILED", "input-hash")
                )

    def test_load_completed_stage_returns_verified_root_relative_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "compiled.json"
            artifact.write_text('{"compiled": true}', encoding="utf-8")
            expected = store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"compiled-input": artifact},
            )

            loaded = store.load_completed_stage(
                "run-1",
                "COMPILED",
                "input-hash",
            )

            self.assertEqual(loaded, expected)
            self.assertFalse(loaded.artifacts[0].path.is_absolute())
            self.assertEqual(root / loaded.artifacts[0].path, root / expected.artifacts[0].path)

    def test_load_completed_stage_rejects_wrong_input_or_tampered_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "compiled.json"
            artifact.write_text('{"compiled": true}', encoding="utf-8")
            manifest = store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"compiled-input": artifact},
            )

            with self.assertRaises(ArtifactIntegrityError):
                store.load_completed_stage("run-1", "COMPILED", "other-input")

            snapshot = root / manifest.artifacts[0].path
            snapshot.chmod(0o600)
            snapshot.write_text('{"compiled": false}', encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                store.load_completed_stage("run-1", "COMPILED", "input-hash")

    def test_inspect_resume_reports_resumable_and_can_resume_delegates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "artifact.bin"
            artifact.write_bytes(b"content")
            store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"compiled": artifact},
            )

            inspection = store.inspect_resume(
                "run-1",
                "COMPILED",
                "input-hash",
            )

            self.assertEqual(
                inspection,
                ResumeInspection(
                    resumable=True,
                    outcome="resumable",
                    reason="stage manifest and artifacts are valid",
                ),
            )
            delegated = ResumeInspection(False, "io", "forced")
            with mock.patch.object(
                store,
                "inspect_resume",
                return_value=delegated,
            ) as inspect_resume:
                self.assertFalse(
                    store.can_resume("run-1", "COMPILED", "input-hash")
                )
            inspect_resume.assert_called_once_with(
                "run-1",
                "COMPILED",
                "input-hash",
            )

    def test_inspect_resume_missing_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            before = tuple(root.iterdir())

            inspection = store.inspect_resume(
                "missing-run",
                "COMPILED",
                "input-hash",
            )
            resumable = store.can_resume(
                "missing-run",
                "COMPILED",
                "input-hash",
            )

            self.assertFalse(inspection.resumable)
            self.assertEqual(inspection.outcome, "missing")
            self.assertIn("missing", inspection.reason)
            self.assertFalse(resumable)
            self.assertEqual(tuple(root.iterdir()), before)

    def test_inspect_resume_reports_corrupt_manifest_variants(self):
        variants = {
            "invalid_json": "{not-json",
            "oversized": " " * 1_100_000,
            "too_deep": "[" * 70 + "0" + "]" * 70,
            "bad_schema": '{"unexpected":true}',
        }
        for name, content in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = LocalArtifactStore(root)
                artifact = store.run_dir("run-1") / "artifact.bin"
                artifact.write_bytes(b"content")
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"compiled": artifact},
                )
                manifest_path = (
                    root
                    / "runs"
                    / "run-1"
                    / "stages"
                    / "COMPILED"
                    / "stage_manifest.json"
                )
                manifest_path.write_text(content, encoding="utf-8")

                inspection = store.inspect_resume(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                )

                self.assertFalse(inspection.resumable)
                self.assertEqual(inspection.outcome, "corrupt")
                self.assertTrue(inspection.reason)
                self.assertFalse(
                    store.can_resume("run-1", "COMPILED", "input-hash")
                )

    def test_inspect_resume_reports_integrity_mismatches(self):
        variants = ("input", "status", "artifact_hash", "artifact_missing")
        for name in variants:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = LocalArtifactStore(root)
                artifact = store.run_dir("run-1") / "artifact.bin"
                artifact.write_bytes(b"content")
                manifest = store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"compiled": artifact},
                )
                requested_input = "input-hash"
                if name == "input":
                    requested_input = "different-input"
                elif name == "status":
                    manifest_path = (
                        root
                        / "runs"
                        / "run-1"
                        / "stages"
                        / "COMPILED"
                        / "stage_manifest.json"
                    )
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload["status"] = "FAILED"
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                else:
                    snapshot = root / manifest.artifacts[0].path
                    if name == "artifact_hash":
                        snapshot.chmod(0o600)
                        snapshot.write_bytes(b"changed")
                    else:
                        snapshot.unlink()

                inspection = store.inspect_resume(
                    "run-1",
                    "COMPILED",
                    requested_input,
                )

                self.assertFalse(inspection.resumable)
                self.assertEqual(inspection.outcome, "integrity")
                self.assertTrue(inspection.reason)

    def test_inspect_resume_rejects_artifact_from_another_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact = store.run_dir("run-1") / "artifact.bin"
            artifact.write_bytes(b"content")
            compiled = store.complete_stage(
                "run-1",
                "COMPILED",
                "compiled-input",
                {"compiled": artifact},
            )
            store.complete_stage(
                "run-1",
                "TRAINED",
                "trained-input",
                {"trained": artifact},
            )
            trained_manifest_path = (
                root
                / "runs"
                / "run-1"
                / "stages"
                / "TRAINED"
                / "stage_manifest.json"
            )
            payload = json.loads(trained_manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"][0] = to_plain_data(compiled.artifacts[0])
            trained_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            inspection = store.inspect_resume(
                "run-1",
                "TRAINED",
                "trained-input",
            )

            self.assertFalse(inspection.resumable)
            self.assertEqual(inspection.outcome, "integrity")

    def test_inspect_resume_reports_permission_and_io_errors(self):
        errors = (
            PermissionError(errno.EACCES, "denied"),
            OSError(errno.EIO, "read failed"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = LocalArtifactStore(root)
                artifact = store.run_dir("run-1") / "artifact.bin"
                artifact.write_bytes(b"content")
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"compiled": artifact},
                )

                with mock.patch(
                    "heuriboost_rag.reckless.storage._strict_json_at",
                    side_effect=error,
                ):
                    inspection = store.inspect_resume(
                        "run-1",
                        "COMPILED",
                        "input-hash",
                    )

                self.assertFalse(inspection.resumable)
                self.assertEqual(inspection.outcome, "io")
                self.assertIn(str(error), inspection.reason)

    def test_can_resume_returns_false_for_missing_or_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            self.assertFalse(store.can_resume("run-1", "COMPILED", "input-hash"))

            artifact_path = store.run_dir("run-1") / "artifact.bin"
            artifact_path.write_bytes(b"content")
            store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"compiled": artifact_path},
            )
            manifest_path = (
                root
                / "runs"
                / "run-1"
                / "stages"
                / "COMPILED"
                / "stage_manifest.json"
            )
            valid = json.loads(manifest_path.read_text(encoding="utf-8"))
            malformed_payloads = {
                "invalid_json": "{not-json",
                "non_finite": '{"stage":NaN}',
                "deep_nesting": "[" * 2_000 + "0" + "]" * 2_000,
                "wrong_stage": json.dumps({**valid, "stage": "TRAINED"}),
                "unknown_field": json.dumps({**valid, "unexpected": True}),
                "bad_artifacts": json.dumps({**valid, "artifacts": {}}),
                "bad_hash": json.dumps(
                    {
                        **valid,
                        "artifacts": [
                            {**valid["artifacts"][0], "content_hash": "bad"}
                        ],
                    }
                ),
                "unsafe_artifact_type": json.dumps(
                    {
                        **valid,
                        "artifacts": [
                            {
                                **valid["artifacts"][0],
                                "artifact_type": "../compiled",
                            }
                        ],
                    }
                ),
                "unsafe_stage": json.dumps({**valid, "stage": "../COMPILED"}),
                "missing_status": json.dumps(
                    {key: value for key, value in valid.items() if key != "status"}
                ),
                "missing_duration": json.dumps(
                    {
                        key: value
                        for key, value in valid.items()
                        if key != "duration_ms"
                    }
                ),
                "bad_status_type": json.dumps({**valid, "status": 1}),
                "blank_status": json.dumps({**valid, "status": ""}),
                "negative_duration": json.dumps({**valid, "duration_ms": -1}),
                "float_duration": json.dumps({**valid, "duration_ms": 1.5}),
                "traversal": json.dumps(
                    {
                        **valid,
                        "artifacts": [
                            {**valid["artifacts"][0], "path": "../../escape"}
                        ],
                    }
                ),
            }
            for name, content in malformed_payloads.items():
                with self.subTest(name=name):
                    manifest_path.write_text(content, encoding="utf-8")
                    self.assertFalse(
                        store.can_resume("run-1", "COMPILED", "input-hash")
                    )

    def test_can_resume_requires_completed_manifest_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            artifact_path = store.run_dir("run-1") / "artifact.bin"
            artifact_path.write_bytes(b"content")
            store.complete_stage(
                "run-1",
                "COMPILED",
                "input-hash",
                {"compiled": artifact_path},
            )
            manifest_path = (
                root
                / "runs"
                / "run-1"
                / "stages"
                / "COMPILED"
                / "stage_manifest.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["status"] = "FAILED"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertFalse(
                store.can_resume("run-1", "COMPILED", "input-hash")
            )

    def test_store_rejects_path_traversal_and_symlinked_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            store = LocalArtifactStore(root)
            external = Path(outside) / "external.bin"
            external.write_bytes(b"external")

            for run_id in ("../escape", "nested/run", "has space"):
                with self.subTest(run_id=run_id):
                    with self.assertRaises(ValueError):
                        store.run_dir(run_id)
            for stage in ("../COMPILED", "nested/stage", "has space"):
                with self.subTest(stage=stage):
                    with self.assertRaises(ValueError):
                        store.complete_stage(
                            "run-1",
                            stage,
                            "input-hash",
                            {"external": external},
                        )
            manifest = store.complete_stage(
                "run-1",
                "EXTERNAL",
                "input-hash",
                {"external": external},
            )
            self.assertEqual(
                manifest.artifacts[0].path.parent,
                Path("runs", "run-1", "stages", "EXTERNAL", "artifacts"),
            )
            self.assertFalse(manifest.artifacts[0].path.is_absolute())
            self.assertNotEqual(manifest.artifacts[0].path, external)

            run_dir = store.run_dir("run-1")
            symlink = run_dir / "external-link.bin"
            try:
                symlink.symlink_to(external)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            with self.assertRaises(ValueError):
                store.complete_stage(
                    "run-1",
                    "COMPILED",
                    "input-hash",
                    {"external": symlink},
                )

    def test_inspect_resume_rejects_unsafe_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(Path(tmp))
            with self.assertRaises(ValueError):
                store.inspect_resume("../escape", "COMPILED", "hash")
            with self.assertRaises(ValueError):
                store.can_resume("run-1", "../escape", "hash")


class RecklessStorageTests(unittest.TestCase):
    def test_stage_manifest_round_trip_and_resume_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(Path(tmp))
            compiled_path = Path(tmp) / "compiled.csv"
            compiled_path.write_text("query_id,label\nq1,3\n", encoding="utf-8")
            manifest = store.complete_stage(
                run_id="run-1",
                stage="COMPILED",
                input_hash="input-hash",
                artifacts={"compiled": compiled_path},
            )
            self.assertTrue(store.can_resume("run-1", "COMPILED", "input-hash"))
            self.assertFalse(store.can_resume("run-1", "COMPILED", "changed"))
            self.assertEqual(manifest.stage, "COMPILED")
            self.assertEqual(
                manifest.artifacts[0].path.parent,
                Path("runs", "run-1", "stages", "COMPILED", "artifacts"),
            )
            self.assertFalse(manifest.artifacts[0].path.is_absolute())
            self.assertNotEqual(manifest.artifacts[0].path, compiled_path)

    def test_run_repository_rejects_non_monotonic_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = JsonRunRepository(Path(tmp))
            record = repository.create(
                RepairRequest(
                    workspace_id="workspace",
                    base_dataset_id="base",
                    production_cases_id="cases",
                    policy_version="1",
                    backend_name="fake",
                    requested_by="tester",
                ),
                policy_hash="policy-hash",
                input_hash="input-hash",
            )
            with self.assertRaises(ValueError):
                repository.save(record)
