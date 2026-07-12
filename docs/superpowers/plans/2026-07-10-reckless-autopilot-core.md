# HeuriBoost Reckless Autopilot Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable HeuriBoost Python package that runs Reckless repair from frozen inputs through validation, training, evaluation, immutable Pre Promote reporting, explicit approval, and atomic promotion.

**Architecture:** Introduce `src/heuriboost_rag` as the stable library boundary while retaining the current scripts as compatibility wrappers. The orchestrator depends on protocols for datasets, backends, run storage, and promotion targets; the existing RAG repair functions and the current untracked `ranking_api.py` become default implementations behind those protocols.

**Tech Stack:** Python 3.10, dataclasses, pathlib, PyYAML, pandas, XGBoost, unittest, self-contained HTML/JavaScript.

**Git policy:** Commit commands below are logical checkpoints. Do not run `git add` or `git commit` until the user explicitly authorizes commits.

---

### Task 1: Establish the installable package without losing the existing ranking API

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/pyproject.toml`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/__init__.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/backends/__init__.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/__init__.py`
- Modify: `plugins/heuriboost/skills/heuriboost-rag/SKILL.md`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing package import test**

```python
import unittest


class PackageImportTests(unittest.TestCase):
    def test_public_package_imports(self):
        import heuriboost_rag

        self.assertEqual(heuriboost_rag.__version__, "0.2.0")
```

- [ ] **Step 2: Run the test and confirm the package does not exist yet**

Run:

```bash
rtk conda run -n py310 python -m unittest tests.test_package_imports -v
```

Expected: `ModuleNotFoundError: No module named 'heuriboost_rag'`.

- [ ] **Step 3: Add package metadata and package roots**

Create `pyproject.toml` with this initial content:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "heuriboost-rag"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pandas",
  "pyyaml",
  "scikit-learn",
  "xgboost",
]

[project.optional-dependencies]
test = []

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
heuriboost_rag = [
  "reckless/templates/*.html",
  "reckless/i18n/*.json",
]
```

Create `src/heuriboost_rag/__init__.py`:

```python
__version__ = "0.2.0"
```

Create empty package `__init__.py` files only. Later tasks add and test public modules when their real implementation is introduced; do not create temporary API stubs.

- [ ] **Step 4: Install the package in editable mode and rerun the import test**

```bash
rtk conda run -n py310 python -m pip install -e 'plugins/heuriboost/skills/heuriboost-rag[test]'
rtk conda run -n py310 python -m unittest tests.test_package_imports -v
```

Expected: one passing test.

- [ ] **Step 5: Update the skill contract for the post-V0 package boundary**

Replace the V0-only guardrail `Do not add a formal Python package scaffold in V0` with:

```markdown
- Use `heuriboost_rag` for reusable training, Reckless orchestration, reporting, and promotion APIs.
- Keep scripts as thin CLI adapters; do not put new reusable business logic only in `scripts/*.py`.
```

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/pyproject.toml \
  plugins/heuriboost/skills/heuriboost-rag/src \
  plugins/heuriboost/skills/heuriboost-rag/SKILL.md \
  tests/test_package_imports.py
rtk git commit -m "feat: scaffold heuriboost rag package"
```

### Task 2: Define contracts, errors, and legal run-state transitions

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/contracts.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/errors.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/state.py`
- Test: `tests/test_reckless_contracts.py`
- Test: `tests/test_reckless_state.py`

- [ ] **Step 1: Write tests for immutable request and approval contracts**

```python
from dataclasses import FrozenInstanceError
import unittest

from heuriboost_rag.reckless.contracts import PromotionApproval, RepairRequest


class RecklessContractTests(unittest.TestCase):
    def test_repair_request_is_frozen(self):
        request = RepairRequest(
            workspace_id="prod-recog",
            base_dataset_id="base-v1",
            production_cases_id="cases-v2",
            policy_version="1",
            backend_name="fake",
            requested_by="tester",
            run_options={"rounds": 2},
        )
        with self.assertRaises(FrozenInstanceError):
            request.workspace_id = "changed"

    def test_approval_carries_stale-write_guards(self):
        approval = PromotionApproval(
            run_id="run-1",
            approved_by="tester",
            approved_at="2026-07-10T00:00:00Z",
            report_hash="report-hash",
            decision_hash="decision-hash",
            expected_current_model="run-0",
            idempotency_key="approval-1",
        )
        self.assertEqual(approval.expected_current_model, "run-0")
```

- [ ] **Step 2: Write tests for allowed and forbidden state transitions**

```python
import unittest

from heuriboost_rag.reckless.state import RunState, assert_transition


class RecklessStateTests(unittest.TestCase):
    def test_normal_transition_is_allowed(self):
        assert_transition(RunState.REPORTING, RunState.READY_FOR_PROMOTION)

    def test_blocked_run_cannot_be_reopened_in_place(self):
        with self.assertRaises(ValueError):
            assert_transition(RunState.BLOCKED_INPUT, RunState.VALIDATING)

    def test_failed_promotion_can_retry(self):
        assert_transition(RunState.PROMOTION_FAILED, RunState.PROMOTING)
```

- [ ] **Step 3: Run both tests and confirm missing symbols**

```bash
rtk conda run -n py310 python -m unittest \
  tests.test_reckless_contracts \
  tests.test_reckless_state -v
```

Expected: import failures for the new contracts and state symbols.

- [ ] **Step 4: Implement the stable dataclasses**

Start `contracts.py` with `from __future__ import annotations`, then implement these public types with `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class RepairRequest:
    workspace_id: str
    base_dataset_id: str
    production_cases_id: str
    policy_version: str
    backend_name: str
    requested_by: str
    run_options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionApproval:
    run_id: str
    approved_by: str
    approved_at: str
    report_hash: str
    decision_hash: str
    expected_current_model: str | None
    idempotency_key: str


@dataclass(frozen=True)
class GateCheck:
    check_id: str
    label: str
    passed: bool
    observed: object
    required: object
    reason: str


@dataclass(frozen=True)
class Decision:
    promotion_eligible: bool
    acceptance_level: str
    checks: tuple[GateCheck, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
```

Add the remaining public contracts with these exact fields:

```python
@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    role: str
    path: Path
    content_hash: str
    schema_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_type: str
    path: Path
    content_hash: str
    size_bytes: int


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    metadata: Mapping[str, object]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledInputs:
    artifacts: tuple[ArtifactRef, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class CandidateModel:
    model_path: Path
    artifacts: tuple[ArtifactRef, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactVerification:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TargetValidation:
    valid: bool
    current_model: str | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class PreparedActivation:
    run_id: str
    pointer_payload: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ActivationResult:
    current_model: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EvaluationResult:
    acceptance_level: str
    current_cases_passed: bool
    historical_gates_passed: bool
    global_metrics: Mapping[str, float]
    anchor_metrics: Mapping[str, float]
    touched_domains: Mapping[str, Mapping[str, float]]
    artifacts_valid: bool
    details: Mapping[str, object]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    datasets: Mapping[str, DatasetRef]
    options: Mapping[str, object]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    state: str
    version: int
    request: RepairRequest
    policy_hash: str
    input_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    error: Mapping[str, object] | None = None


@dataclass(frozen=True)
class StageManifest:
    stage: str
    input_hash: str
    artifacts: tuple[ArtifactRef, ...]
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class ReportArtifact:
    html_path: Path
    data_path: Path
    manifest_path: Path
    data_hash: str
    html_hash: str
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class ReleaseSnapshot:
    run_id: str
    artifacts: tuple[ArtifactRef, ...]
    manifest_hash: str
    previous_model: str | None


@dataclass(frozen=True)
class PromotionReceipt:
    run_id: str
    release_path: Path
    promoted_at: str
    approved_by: str
    previous_model: str | None
    current_model: str
    release_manifest_hash: str
    receipt_json_path: Path
    receipt_html_path: Path


@dataclass(frozen=True)
class RollbackReceipt:
    source_run_id: str
    rolled_back_at: str
    approved_by: str
    previous_model: str
    restored_model: str
    receipt_json_path: Path
    receipt_html_path: Path
```

Use tuples for immutable collections and `Mapping[str, object]` for versioned extension metadata.

- [ ] **Step 5: Implement structured errors**

```python
class HeuriBoostError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        run_id: str | None = None,
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
        operator_action: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.run_id = run_id
        self.retryable = retryable
        self.details = dict(details or {})
        self.operator_action = operator_action
```

Add concrete subclasses `InputBlockedError`, `NotEligibleError`, `EvaluationBlockedError`, `PromotionConflictError`, and `ArtifactIntegrityError` with stable error codes.

- [ ] **Step 6: Implement the transition table**

Use a closed `Enum` and an explicit mapping:

```python
ALLOWED_TRANSITIONS = {
    RunState.RECEIVED: {RunState.VALIDATING, RunState.CANCELLED},
    RunState.VALIDATING: {RunState.COMPILED, RunState.BLOCKED_INPUT, RunState.FAILED_INTERNAL},
    RunState.COMPILED: {RunState.TRAINING, RunState.CANCELLED},
    RunState.TRAINING: {RunState.TRAINED, RunState.INTERRUPTED, RunState.CANCELLED, RunState.FAILED_INTERNAL},
    RunState.TRAINED: {RunState.EVALUATING},
    RunState.EVALUATING: {RunState.REPORTING, RunState.BLOCKED_EVALUATION, RunState.BLOCKED_NOT_ELIGIBLE, RunState.FAILED_INTERNAL},
    RunState.REPORTING: {RunState.READY_FOR_PROMOTION, RunState.BLOCKED_NOT_ELIGIBLE, RunState.FAILED_INTERNAL},
    RunState.READY_FOR_PROMOTION: {RunState.PROMOTING},
    RunState.PROMOTING: {RunState.PROMOTED, RunState.PROMOTION_FAILED},
    RunState.PROMOTION_FAILED: {RunState.PROMOTING},
}
```

States absent from the mapping are terminal.

- [ ] **Step 7: Rerun tests**

```bash
rtk conda run -n py310 python -m unittest \
  tests.test_reckless_contracts \
  tests.test_reckless_state -v
```

Expected: all tests pass.

- [ ] **Step 8: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless \
  tests/test_reckless_contracts.py tests/test_reckless_state.py
rtk git commit -m "feat: define reckless run contracts"
```

### Task 3: Add versioned Reckless policy parsing and decision rules

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/policy.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/templates/reckless_policy.yml`
- Test: `tests/test_reckless_policy.py`

- [ ] **Step 1: Write policy tests**

```python
import tempfile
from pathlib import Path
import unittest

from heuriboost_rag.reckless.policy import RecklessPolicy, load_policy


class RecklessPolicyTests(unittest.TestCase):
    def test_default_policy_forbids_weak_promotion(self):
        policy = RecklessPolicy.default()
        self.assertEqual(policy.acceptance_level, "full")
        self.assertFalse(policy.promotion.allow_weak)
        self.assertTrue(policy.promotion.require_explicit_human_approval)

    def test_policy_hash_is_stable_for_equivalent_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.yml"
            second = Path(tmp) / "second.yml"
            first.write_text("version: 1\nacceptance_level: full\n", encoding="utf-8")
            second.write_text("acceptance_level: full\nversion: 1\n", encoding="utf-8")
            self.assertEqual(load_policy(first).content_hash, load_policy(second).content_hash)
```

- [ ] **Step 2: Run the test and confirm failure**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_policy -v
```

Expected: import failure for `RecklessPolicy`.

- [ ] **Step 3: Implement policy dataclasses and canonical hashing**

Define `InputPolicy`, `EvaluationPolicy`, `PromotionPolicy`, and `RecklessPolicy`. Parse YAML with `yaml.safe_load`, merge only documented defaults, reject unknown top-level keys, reject `acceptance_level` outside `full|weak`, and hash canonical JSON:

```python
canonical = json.dumps(policy_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Provide `evaluate_promotion_eligibility(policy, evaluation) -> Decision`; it must emit one `GateCheck` for current cases, historical gates, global nDCG, global MRR, touched domains, artifact integrity, and acceptance level.

- [ ] **Step 4: Add the committed default template**

Use the exact policy from the approved core design, including `allow_weak: false`, `allow_anchor_reset: false`, and `allow_gate_retirement: false`.

- [ ] **Step 5: Rerun policy tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_policy -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/policy.py \
  plugins/heuriboost/skills/heuriboost-rag/templates/reckless_policy.yml \
  tests/test_reckless_policy.py
rtk git commit -m "feat: add reckless policy engine"
```

### Task 4: Implement hashing, atomic JSON writes, run storage, and stage manifests

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/hashing.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/storage.py`
- Test: `tests/test_reckless_storage.py`

- [ ] **Step 1: Write storage tests**

```python
import json
import tempfile
from pathlib import Path
import unittest

from heuriboost_rag.reckless.contracts import RepairRequest
from heuriboost_rag.reckless.storage import JsonRunRepository, LocalArtifactStore


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
```

- [ ] **Step 2: Run the tests and confirm missing modules**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_storage -v
```

- [ ] **Step 3: Implement deterministic hashing and atomic writes**

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
```

Also implement canonical JSON hashing and combined run fingerprints.

- [ ] **Step 4: Implement repository and artifact protocols**

Define these exact boundaries:

```python
class DatasetRepository(Protocol):
    def get(self, dataset_id: str) -> DatasetRef: ...


class RunRepository(Protocol):
    def create(self, request: RepairRequest, policy_hash: str, input_hash: str) -> RunRecord: ...
    def get(self, run_id: str) -> RunRecord: ...
    def save(self, record: RunRecord) -> RunRecord: ...
    def transition(self, run_id: str, state: RunState, metadata: Mapping[str, object] | None = None) -> RunRecord: ...
    def fail(self, run_id: str, state: RunState, error: HeuriBoostError) -> RunRecord: ...


class ArtifactStore(Protocol):
    def run_dir(self, run_id: str) -> Path: ...
    def complete_stage(self, run_id: str, stage: str, input_hash: str, artifacts: Mapping[str, Path]) -> StageManifest: ...
    def can_resume(self, run_id: str, stage: str, input_hash: str) -> bool: ...


@dataclass(frozen=True)
class OrchestratorStores:
    datasets: DatasetRepository
    runs: RunRepository
    artifacts: ArtifactStore
```

`JsonDatasetRepository` stores immutable `DatasetRef` records. `JsonRunRepository` stores one versioned `run.json` per run and rejects stale writes. `LocalArtifactStore` writes stage manifests under `runs/<run_id>/stages/<stage>/stage_manifest.json`, hashes every declared artifact, and implements `can_resume` by checking the stored input hash plus current artifact hashes.

- [ ] **Step 5: Rerun storage tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_storage -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/hashing.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/storage.py \
  tests/test_reckless_storage.py
rtk git commit -m "feat: persist reckless runs and manifests"
```

### Task 5: Move the existing generic ranking API behind the package backend boundary

**Files:**
- Modify: `plugins/heuriboost/skills/heuriboost-rag/scripts/ranking_api.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/backends/base.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/backends/ranking.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/backends/legacy_runtime.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/backends/xgboost_rag.py`
- Modify: `tests/test_ranking_api.py`
- Test: `tests/test_backend_protocol.py`

- [ ] **Step 1: Preserve the current ranking API behavior in package-level tests**

Change `tests/test_ranking_api.py` to import:

```python
from heuriboost_rag.backends.ranking import (
    evaluate_xgboost_ranker,
    predict_xgboost_ranker,
    train_xgboost_ranker,
)
```

Keep the current assertions for `reranker.json`, metadata, feature names, grouped ordering, and `ndcg@10`. Add assertions that evaluation returns `mrr@10`, and prediction returns one finite score per row in original order.

- [ ] **Step 2: Write a protocol conformance test**

```python
import unittest

from heuriboost_rag.backends.base import RepairBackend
from heuriboost_rag.backends.xgboost_rag import XGBoostRagBackend
from heuriboost_rag.reckless.hashing import (
    ExecutionIdentity,
    ExecutionIdentityProvider,
)


class BackendProtocolTests(unittest.TestCase):
    def test_default_backend_exposes_required_operations(self):
        backend = XGBoostRagBackend()
        self.assertEqual(backend.name, "xgboost-rag")
        for name in (
            "execution_identity",
            "validate",
            "compile",
            "train",
            "evaluate",
            "verify_artifacts",
        ):
            self.assertTrue(callable(getattr(backend, name)))
        identity = backend.execution_identity()
        self.assertIsInstance(identity, ExecutionIdentity)
        self.assertIsInstance(backend, ExecutionIdentityProvider)
        self.assertIsInstance(backend, RepairBackend)
```

Mark `RepairBackend` with `@runtime_checkable`.

- [ ] **Step 3: Run tests before migration**

```bash
rtk conda run -n py310 python -m unittest tests.test_ranking_api tests.test_backend_protocol -v
```

Expected: package import failures.

- [ ] **Step 4: Move implementation, then leave a compatibility re-export**

Move the existing untracked `scripts/ranking_api.py` implementation into `heuriboost_rag/backends/ranking.py` without changing behavior. Replace the script body with:

```python
from heuriboost_rag.backends.ranking import (
    evaluate_xgboost_ranker,
    predict_xgboost_ranker,
    train_xgboost_ranker,
)

__all__ = [
    "train_xgboost_ranker",
    "evaluate_xgboost_ranker",
    "predict_xgboost_ranker",
]
```

This step must preserve any current user changes in `ranking_api.py`; compare the source before replacing it. Move `_dmatrix`, label mapping, nDCG, and metadata behavior unchanged. Add `predict_xgboost_ranker` as the shared load/predict function, and calculate `mrr@10` from the same grouped predictions used by `evaluate_xgboost_ranker`.

- [ ] **Step 5: Define backend protocols and the default adapter**

Use these exact runtime-checkable protocols:

```python
from heuriboost_rag.reckless.hashing import ExecutionIdentity


@runtime_checkable
class RepairBackend(Protocol):
    name: str

    def execution_identity(self) -> ExecutionIdentity: ...
    def validate(self, request: RepairRequest, context: RunContext) -> ValidationResult: ...
    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs: ...
    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel: ...
    def evaluate(self, candidate: CandidateModel, context: RunContext) -> EvaluationResult: ...
    def verify_artifacts(self, candidate: CandidateModel, context: RunContext) -> ArtifactVerification: ...


@runtime_checkable
class PromotionTarget(Protocol):
    name: str

    def validate_target(self, expected_current: str | None) -> TargetValidation: ...
    def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation: ...
    def activate(self, prepared: PreparedActivation) -> ActivationResult: ...
    def rollback(self, receipt: PromotionReceipt) -> ActivationResult: ...
```

`XGBoostRagBackend.execution_identity()` must return the concrete, validated identity used by fingerprints: backend version, ordered feature names and feature version, HeuriBoost code commit, effective training parameters, and random seed. It must not source any of these fields implicitly from `RepairRequest.run_options`.

`XGBoostRagBackend` calls the current `repair_cases` functions through one internal `legacy_runtime.py` resolver. The resolver may add the skill's `scripts/` directory to `sys.path`, but no consumer project may do so after this task.

The default backend must map existing outputs into `CompiledInputs`, `CandidateModel`, and `EvaluationResult` dataclasses without changing current full/weak semantics.

`PromotionTarget.prepare_release` and `activate` may validate or materialize immutable target-specific artifacts inside the staged release, but they must never write a mutable current-model pointer. `FileReleaseStore` remains the sole writer of `current_model.json`.

- [ ] **Step 6: Run ranking and backend tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_ranking_api tests.test_backend_protocol -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/scripts/ranking_api.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/backends \
  tests/test_ranking_api.py tests/test_backend_protocol.py
rtk git commit -m "refactor: expose ranking backend package api"
```

### Task 6: Build the resumable Reckless orchestrator

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/orchestrator.py`
- Test: `tests/test_reckless_orchestrator.py`

- [ ] **Step 1: Write a fake backend and normal-flow test**

```python
from heuriboost_rag.reckless.hashing import ExecutionIdentity


class FakeBackend:
    name = "fake"

    def execution_identity(self):
        return ExecutionIdentity(
            backend_version="test-v1",
            feature_names=("feature_a", "feature_b"),
            feature_version="test-features-v1",
            code_commit="test-commit",
            training_params={},
            random_seed=7,
        )

    def validate(self, request, context):
        return ValidationResult(valid=True, metadata={"validated": True})

    def compile(self, request, context):
        return CompiledInputs(artifacts=(), metadata={"compiled": True})

    def train(self, inputs, context):
        model_path = context.run_dir / "models/reranker.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        return CandidateModel(model_path=model_path, artifacts=(), metadata={})

    def evaluate(self, candidate, context):
        return EvaluationResult(
            acceptance_level="full",
            current_cases_passed=True,
            historical_gates_passed=True,
            global_metrics={"ndcg@10": 0.8, "mrr@10": 0.7},
            anchor_metrics={"ndcg@10": 0.7, "mrr@10": 0.6},
            touched_domains={"medical": {"ndcg@10": 0.8, "anchor_ndcg@10": 0.8}},
            artifacts_valid=True,
            details={},
        )

    def verify_artifacts(self, candidate, context):
        return ArtifactVerification(valid=True, errors=())
```

Test that this concrete call reaches `READY_FOR_PROMOTION`, writes all stage manifests, and returns a decision with no blockers:

```python
run = run_reckless_repair(request, FakeBackend(), stores, RecklessPolicy.default())
self.assertEqual(run.state, RunState.READY_FOR_PROMOTION.value)
```

- [ ] **Step 2: Write blocked and resume tests**

Add tests proving:

- invalid input becomes `BLOCKED_INPUT`;
- weak acceptance becomes `BLOCKED_NOT_ELIGIBLE`;
- failed metrics become `BLOCKED_EVALUATION`;
- an `INTERRUPTED` run reuses a completed compile stage only when its input hash still matches.

- [ ] **Step 3: Run tests and confirm missing orchestrator**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_orchestrator -v
```

- [ ] **Step 4: Implement stage-by-stage orchestration**

`run_reckless_repair` must:

```python
def run_reckless_repair(request, backend, stores, policy):
    base_dataset = stores.datasets.get(request.base_dataset_id)
    production_cases = stores.datasets.get(request.production_cases_id)
    execution_identity = backend.execution_identity()
    input_hash = build_run_fingerprint(
        request,
        policy,
        base_dataset,
        production_cases,
        execution_identity,
    )
    run = stores.runs.create(request, policy.content_hash, input_hash)
    context = RunContext(
        run_id=run.run_id,
        run_dir=stores.artifacts.run_dir(run.run_id),
        datasets={"base": base_dataset, "production_cases": production_cases},
        options=request.run_options,
    )
    validate_stage(run, context, backend, stores)
    compiled = compile_stage(run, context, backend, stores)
    candidate = train_stage(run, context, backend, stores, compiled)
    evaluation = evaluate_stage(run, context, backend, stores, candidate)
    decision = evaluate_promotion_eligibility(policy, evaluation)
    report_stage(run, context, stores, decision, evaluation)
    return stores.runs.get(run.run_id)
```

Each helper performs exactly one legal state transition, writes a stage manifest, and converts known domain exceptions into the matching `BLOCKED_*` state. Unexpected exceptions become `FAILED_INTERNAL` with a structured error record.

- [ ] **Step 5: Add resume entry point**

```python
def resume_reckless_repair(run_id, backend, stores, policy):
    run = stores.runs.get(run_id)
    if run.state != RunState.INTERRUPTED.value:
        raise ValueError("only INTERRUPTED runs can resume")
    return _continue_from_first_incomplete_stage(run, backend, stores, policy)
```

Resume must never reopen `BLOCKED_*` runs.

- [ ] **Step 6: Run orchestrator tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_orchestrator -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/orchestrator.py \
  tests/test_reckless_orchestrator.py
rtk git commit -m "feat: orchestrate reckless repair runs"
```

### Task 7: Generate immutable bilingual Pre Promote reports

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/report.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/templates/pre_promote.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/i18n/zh-CN.json`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/i18n/en.json`
- Test: `tests/test_pre_promote_report.py`

- [ ] **Step 1: Write report completeness and language-parity tests**

```python
import json
import tempfile
from html.parser import HTMLParser
from pathlib import Path
import unittest

from heuriboost_rag.reckless.report import build_localized_view, render_pre_promote_report


def sample_report_data():
    return {
        "schema_version": 1,
        "run": {"run_id": "run-1", "state": "READY_FOR_PROMOTION"},
        "decision": {"status": "READY_FOR_PROMOTION", "promotion_eligible": True},
        "data_lineage": {"base_dataset_id": "base-v1", "production_cases_id": "cases-v2"},
        "validation": {"passed": True, "warnings": []},
        "compilation": {"train_rows": 4, "validation_rows": 2, "test_rows": 2},
        "training": {"objective": "rank:ndcg", "rounds": 2},
        "evaluation": {"ndcg@10": 1.0, "mrr@10": 1.0},
        "gate_checks": [{"check_id": "historical_gates", "passed": True}],
        "warnings": [],
        "artifacts": [{"artifact_type": "model", "content_hash": "model-hash"}],
        "reproducibility": {"policy_hash": "policy-hash", "code_revision": "revision"},
    }


class JsonScriptParser(HTMLParser):
    def __init__(self, script_id):
        super().__init__()
        self.script_id = script_id
        self.capture = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.capture = tag == "script" and attributes.get("id") == self.script_id

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.capture = False


def extract_json_script(html, script_id):
    parser = JsonScriptParser(script_id)
    parser.feed(html)
    return json.loads("".join(parser.parts))


class PrePromoteReportTests(unittest.TestCase):
    def test_report_contains_decision_training_lineage_and_machine_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_pre_promote_report(sample_report_data(), Path(tmp), locale="zh-CN")
            html = result.html_path.read_text(encoding="utf-8")
            embedded = extract_json_script(html, "heuriboost-pre-promote-data")
            self.assertEqual(embedded["decision"]["status"], "READY_FOR_PROMOTION")
            self.assertIn("training", embedded)
            self.assertIn("data_lineage", embedded)
            self.assertIn("gate_checks", embedded)
            self.assertEqual(result.data_hash, result.manifest["data_hash"])

    def test_locale_switch_does_not_change_numeric_data(self):
        zh = build_localized_view(sample_report_data(), "zh-CN")
        en = build_localized_view(sample_report_data(), "en")
        self.assertEqual(zh["raw_data"], en["raw_data"])
```

- [ ] **Step 2: Run the tests and confirm missing renderer**

```bash
rtk conda run -n py310 python -m unittest tests.test_pre_promote_report -v
```

- [ ] **Step 3: Implement the report data builder**

`build_report_data(run, evaluation, decision, artifacts)` must produce these top-level keys with schema version `1`:

```python
{
    "schema_version": 1,
    "run": {"run_id": "run-1", "state": "READY_FOR_PROMOTION"},
    "decision": {"status": "READY_FOR_PROMOTION", "promotion_eligible": True},
    "data_lineage": {"base_dataset_id": "base-v1", "production_cases_id": "cases-v2"},
    "validation": {"passed": True, "warnings": []},
    "compilation": {"train_rows": 100, "validation_rows": 20, "test_rows": 20},
    "training": {"objective": "rank:ndcg", "rounds": 40, "duration_seconds": 12.5},
    "evaluation": {"ndcg@10": 0.8, "mrr@10": 0.7},
    "gate_checks": [{"check_id": "historical_gates", "passed": True}],
    "warnings": [],
    "artifacts": [{"artifact_type": "model", "content_hash": "model-hash"}],
    "reproducibility": {"policy_hash": "policy-hash", "code_revision": "revision"},
}
```

Reject report generation if any mandatory key is absent.

- [ ] **Step 4: Implement the self-contained HTML renderer**

Load the packaged template with `importlib.resources`, embed escaped JSON using:

```python
json_text = json.dumps(report_data, ensure_ascii=False).replace("<", "\\u003c")
html = template.replace("__REPORT_JSON__", json_text)
```

The template must implement the approved process-first layout with the key decision summary at the top, Chinese default, an English toggle, collapsible stage details, print styles, and no network assets.

- [ ] **Step 5: Write immutable report artifacts**

Write `pre_promote_report_data.json`, `pre_promote_report.html`, and `pre_promote_report_manifest.json` with atomic writes. The manifest contains data, HTML, decision, model, and policy hashes. If any target already exists with different content, raise `ArtifactIntegrityError` instead of overwriting it.

- [ ] **Step 6: Rerun report tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_pre_promote_report -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/report.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/templates \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/i18n \
  tests/test_pre_promote_report.py
rtk git commit -m "feat: render pre promote reports"
```

### Task 8: Implement immutable releases and idempotent promotion

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/release_store.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/promotion.py`
- Test: `tests/test_reckless_promotion.py`

- [ ] **Step 1: Write promotion safety tests**

Add these concrete tests. The shared `make_ready_fixture()` helper must create a `READY_FOR_PROMOTION` run, matching report/decision/model hashes, a fake current pointer, and a recording PromotionTarget:

```python
def test_promotion_rejects_changed_report_hash(self):
    fixture = self.make_ready_fixture()
    approval = replace(fixture.approval, report_hash="changed")
    with self.assertRaises(ArtifactIntegrityError):
        promote_repair(fixture.run_id, approval, fixture.target, fixture.stores)


def test_promotion_rejects_changed_current_model(self):
    fixture = self.make_ready_fixture(current_model="run-0")
    approval = replace(fixture.approval, expected_current_model="older-run")
    with self.assertRaises(PromotionConflictError):
        promote_repair(fixture.run_id, approval, fixture.target, fixture.stores)


def test_duplicate_idempotency_key_returns_same_receipt(self):
    fixture = self.make_ready_fixture()
    first = promote_repair(fixture.run_id, fixture.approval, fixture.target, fixture.stores)
    second = promote_repair(fixture.run_id, fixture.approval, fixture.target, fixture.stores)
    self.assertEqual(first, second)
    self.assertEqual(fixture.target.activations, 1)


def test_failure_before_pointer_swap_keeps_current_model(self):
    fixture = self.make_ready_fixture(fail_before_pointer_swap=True)
    with self.assertRaises(RuntimeError):
        promote_repair(fixture.run_id, fixture.approval, fixture.target, fixture.stores)
    self.assertEqual(fixture.stores.releases.read_current_model(), "run-0")


def test_two_concurrent_promotions_create_one_release(self):
    fixture = self.make_ready_fixture()
    receipts = fixture.promote_concurrently(count=2)
    self.assertEqual(receipts[0], receipts[1])
    self.assertEqual(fixture.target.activations, 1)


def test_rollback_restores_previous_model_and_writes_receipt(self):
    fixture = self.make_ready_fixture(current_model="run-0")
    receipt = promote_repair(fixture.run_id, fixture.approval, fixture.target, fixture.stores)
    rollback = rollback_release(receipt, fixture.target, fixture.stores, approved_by="tester")
    self.assertEqual(rollback.restored_model, "run-0")
    self.assertTrue(rollback.receipt_json_path.exists())
```

Use temporary directories and a fake `PromotionTarget`; do not require XGBoost.

- [ ] **Step 2: Run tests and confirm missing promotion modules**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_promotion -v
```

- [ ] **Step 3: Implement the file release store**

`FileReleaseStore` must:

- acquire a workspace lock using `fcntl.flock`;
- build `releases/.staging-<run_id>-<uuid>`;
- copy declared artifacts and write a manifest;
- verify every hash;
- atomically rename the staging directory to `releases/<run_id>`;
- atomically replace `current_model.json` last;
- leave an existing release immutable.

Before pointer replacement, call `target.validate_target`, `target.prepare_release`, and `target.activate`; use the returned activation metadata to build the final pointer payload. Target implementations must obey the no-mutable-pointer rule defined in Task 5. If any target call fails, remove only the staging directory and leave the current pointer unchanged.

Use an injected failure hook in tests immediately before pointer replacement.

- [ ] **Step 4: Implement server-side revalidation and idempotency**

Define the promotion storage boundary in `promotion.py`:

```python
class PromotionRepository(Protocol):
    def find_by_idempotency_key(self, key: str) -> PromotionReceipt | None: ...
    def save(self, receipt: PromotionReceipt, idempotency_key: str) -> None: ...


@dataclass(frozen=True)
class PromotionStores:
    runs: RunRepository
    artifacts: ArtifactStore
    promotions: PromotionRepository
    releases: FileReleaseStore
```

```python
def promote_repair(run_id, approval, target, stores):
    existing = stores.promotions.find_by_idempotency_key(approval.idempotency_key)
    if existing:
        return existing
    run = stores.runs.get(run_id)
    assert_ready_and_unchanged(run, approval, stores.artifacts)
    stores.runs.transition(run_id, RunState.PROMOTING)
    try:
        receipt = stores.releases.promote(run, approval, target)
    except Exception as exc:
        stores.runs.fail_promotion(run_id, exc)
        raise
    stores.promotions.save(receipt, approval.idempotency_key)
    stores.runs.transition(run_id, RunState.PROMOTED)
    return receipt
```

`assert_ready_and_unchanged` must check state, full acceptance, all gate checks, report hash, decision hash, model/schema hashes, and expected current model.

Implement `rollback_release(receipt, target, stores, approved_by)` under the same workspace lock. It may only point to `receipt.previous_model`, must return `RollbackReceipt`, create a new append-only rollback receipt/audit record, and never mutate or delete either release directory.

- [ ] **Step 5: Generate Promotion Receipt artifacts**

Reuse the report renderer's static layout utilities to write `promotion_receipt.json` and `promotion_receipt.html`. Include approver, timestamps, old/new model refs, release manifest hash, gate/ledger snapshot hashes, and rollback target.

- [ ] **Step 6: Rerun promotion tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_promotion -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/release_store.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/promotion.py \
  tests/test_reckless_promotion.py
rtk git commit -m "feat: promote immutable reckless releases"
```

### Task 9: Convert existing production-repair scripts into compatibility wrappers

**Files:**
- Modify: `plugins/heuriboost/skills/heuriboost-rag/scripts/repair_reranker.py`
- Modify: `plugins/heuriboost/skills/heuriboost-rag/scripts/promote_repair.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/scripts/reckless_autopilot.py`
- Test: `tests/test_reckless_cli.py`

- [ ] **Step 1: Write CLI compatibility tests**

Use `subprocess.run` against the current example repair files and assert:

- `repair_reranker.py --reckless` still accepts the existing flags;
- `reckless_autopilot.py run` produces `READY_FOR_PROMOTION` for a passing fixture;
- `promote_repair.py --output-dir examples/fiqa/output` delegates to the package API;
- `--acceptance-level weak` cannot Promote.

- [ ] **Step 2: Run the CLI tests before refactoring**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_cli -v
```

Expected: failures for the new autopilot command and package delegation assertions.

- [ ] **Step 3: Add the new CLI entry point**

Support:

```text
reckless_autopilot.py run --base-dataset examples/fiqa/repair/base_dataset_minimal.csv --production-cases examples/fiqa/repair/production_cases_full.csv --output-dir examples/fiqa/output --policy plugins/heuriboost/skills/heuriboost-rag/templates/reckless_policy.yml
reckless_autopilot.py resume --run-id RUN-20260710-01 --output-dir examples/fiqa/output
reckless_autopilot.py report --run-id RUN-20260710-01 --output-dir examples/fiqa/output --locale zh-CN
reckless_autopilot.py promote --run-id RUN-20260710-01 --output-dir examples/fiqa/output --approved-by local-maintainer
```

The CLI creates `DatasetRef` records from validated local paths, then calls only package APIs.

- [ ] **Step 4: Refactor legacy CLIs**

Keep current argument names and defaults. Replace the bodies of `repair_reranker.main` and `promote_repair.main` with adapters that build requests and call `run_reckless_repair` / `promote_repair`. Preserve current console summaries and exit non-zero on blocked runs.

- [ ] **Step 5: Run CLI compatibility tests**

```bash
rtk conda run -n py310 python -m unittest tests.test_reckless_cli -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/scripts/repair_reranker.py \
  plugins/heuriboost/skills/heuriboost-rag/scripts/promote_repair.py \
  plugins/heuriboost/skills/heuriboost-rag/scripts/reckless_autopilot.py \
  tests/test_reckless_cli.py
rtk git commit -m "refactor: delegate reckless cli to core api"
```

### Task 10: Add legacy-state migration, documentation, and full verification

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/migration.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/scripts/migrate_reckless_state.py`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/REFERENCE.md`
- Modify: `docs/REFERENCE.zh-CN.md`
- Test: `tests/test_reckless_migration.py`

- [ ] **Step 1: Write migration tests**

Create a temporary legacy `.heuriboost` directory containing `ledger.json`, `gates.jsonl`, `promoted_repair_samples.csv`, and `current_model.json`. Assert migration creates exactly one immutable bootstrap release, preserves hashes, and refuses a second migration with different content.

- [ ] **Step 2: Implement read-only legacy import**

```python
def migrate_legacy_state(legacy_dir: Path, release_store: FileReleaseStore) -> PromotionReceipt:
    snapshot = build_legacy_snapshot(legacy_dir)
    return release_store.import_bootstrap_release(snapshot, source="legacy-migration")
```

The command must not delete or rewrite legacy files. Until migration succeeds, package APIs may read legacy state but must reject new Promote operations.

- [ ] **Step 3: Update English and Chinese docs**

Document package installation, the stable Python API, CLI compatibility, policy file, report paths, explicit approval, migration, recovery, and the rule that Anchor reset / gate retirement remain separate administrative operations.

- [ ] **Step 4: Run the complete upstream test suite**

```bash
rtk conda run -n py310 python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Compile all changed Python modules**

```bash
rtk conda run -n py310 python -m compileall \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag \
  plugins/heuriboost/skills/heuriboost-rag/scripts
```

Expected: no syntax errors.

- [ ] **Step 6: Run new-file and repository whitespace checks**

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; status lists only intended HeuriBoost changes plus pre-existing user work.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/reckless/migration.py \
  plugins/heuriboost/skills/heuriboost-rag/scripts/migrate_reckless_state.py \
  README.md README.zh-CN.md docs/REFERENCE.md docs/REFERENCE.zh-CN.md \
  tests/test_reckless_migration.py
rtk git commit -m "docs: document reckless autopilot workflow"
```
