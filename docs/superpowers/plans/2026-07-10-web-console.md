# HeuriBoost Web Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first browser console for importing labeled CSV/JSONL/XLSX data, running the Reckless pipeline, inspecting progress and reports, and performing one-click human-approved promotion.

**Architecture:** Add an optional FastAPI/Jinja2 web package on top of the Reckless Core plan. Persist operational metadata and the job queue in SQLite, keep artifacts in the Core `LocalArtifactStore`, execute one training run at a time in a supervised child process, and drive the UI from durable API/SSE state rather than in-memory browser state.

**Tech Stack:** Python 3.10, FastAPI, Uvicorn, Jinja2, sqlite3, python-multipart, pandas, pyarrow, openpyxl, native JavaScript, SSE, unittest, FastAPI TestClient, Playwright.

**Prerequisite:** Complete `docs/superpowers/plans/2026-07-10-reckless-autopilot-core.md` first.

**Git policy:** Commit commands below are logical checkpoints. Do not run `git add` or `git commit` until the user explicitly authorizes commits.

---

### Task 1: Add optional Web dependencies, configuration, and a local-only application shell

**Files:**
- Modify: `plugins/heuriboost/skills/heuriboost-rag/pyproject.toml`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/__init__.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/config.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/app.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/__main__.py`
- Test: `tests/web/test_app_bootstrap.py`

- [ ] **Step 1: Write failing app bootstrap tests**

```python
import tempfile
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from heuriboost_rag.web.app import create_app
from heuriboost_rag.web.config import WebConfig


class AppBootstrapTests(unittest.TestCase):
    def test_health_endpoint_and_local_session_cookie(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = WebConfig.for_test(Path(tmp))
            client = TestClient(create_app(config))
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    def test_non_loopback_bind_requires_explicit_shared_mode(self):
        with self.assertRaises(ValueError):
            WebConfig(data_dir=Path("/tmp/hb"), host="0.0.0.0", shared_mode=False)
```

- [ ] **Step 2: Run the test and confirm missing web extra/package**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_app_bootstrap -v
```

Expected: import failures for FastAPI or `heuriboost_rag.web`.

- [ ] **Step 3: Add optional dependencies and console entry point**

Extend `pyproject.toml`:

```toml
[project.optional-dependencies]
test = []
web = [
  "fastapi>=0.115,<1.0",
  "uvicorn>=0.30,<1.0",
  "jinja2>=3.1,<4.0",
  "python-multipart>=0.0.9,<1.0",
  "openpyxl>=3.1,<4.0",
  "pyarrow>=16,<20",
]
web-test = [
  "httpx>=0.27,<1.0",
  "playwright>=1.50,<2.0",
]

[project.scripts]
heuriboost-web = "heuriboost_rag.web.__main__:main"
```

- [ ] **Step 4: Implement immutable Web configuration**

```python
@dataclass(frozen=True)
class WebConfig:
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8787
    shared_mode: bool = False
    max_upload_bytes: int = 100 * 1024 * 1024
    max_xlsx_uncompressed_bytes: int = 512 * 1024 * 1024
    max_xlsx_sheets: int = 32
    max_xlsx_rows: int = 250_000
    max_xlsx_columns: int = 256
    session_token_ttl_seconds: int = 3600

    def __post_init__(self):
        if self.host not in {"127.0.0.1", "localhost"} and not self.shared_mode:
            raise ValueError("non-loopback host requires shared_mode")
```

`for_test` must create a deterministic temporary data directory and test token.

- [ ] **Step 5: Implement the application factory and CLI**

`create_app(config)` stores configuration in `app.state`, adds `/health`, and prepares dependency hooks for stores and services. `__main__.py` parses `--config`, `--data-dir`, `--host`, and `--port`, prints a tokenized local URL, then starts Uvicorn.

- [ ] **Step 6: Install Web extras and rerun tests**

```bash
rtk conda run -n py310 python -m pip install -e 'plugins/heuriboost/skills/heuriboost-rag[web,web-test]'
rtk conda run -n py310 python -m unittest tests.web.test_app_bootstrap -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/pyproject.toml \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web \
  tests/web/test_app_bootstrap.py
rtk git commit -m "feat: bootstrap heuriboost web console"
```

### Task 2: Add versioned SQLite migrations and repositories

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/migrations/0001_initial.sql`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/stores/sqlite.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/stores/__init__.py`
- Test: `tests/web/test_sqlite_store.py`

- [ ] **Step 1: Write migration and append-only audit tests**

```python
import sqlite3
import tempfile
from pathlib import Path
import unittest

from heuriboost_rag.web.stores.sqlite import SQLiteStore


class SQLiteStoreTests(unittest.TestCase):
    def test_migration_creates_all_required_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            names = store.table_names()
            self.assertTrue({
                "workspaces", "uploads", "datasets", "import_profiles", "runs",
                "run_stages", "jobs", "artifacts", "approvals", "promotions",
                "audit_events", "schema_migrations",
            }.issubset(names))

    def test_audit_events_are_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "heuriboost.db")
            store.migrate()
            event_id = store.append_audit_event("run.created", {"run_id": "run-1"})
            with self.assertRaises(sqlite3.IntegrityError):
                store.update_audit_event(event_id, {"changed": True})
```

- [ ] **Step 2: Run tests and confirm missing store**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_sqlite_store -v
```

- [ ] **Step 3: Create the initial schema migration**

The SQL file must create the twelve tables listed above, foreign keys, unique idempotency keys, run/job status indexes, and triggers that abort `UPDATE` or `DELETE` on `audit_events`:

```sql
CREATE TRIGGER audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
  SELECT RAISE(ABORT, 'audit_events are append-only');
END;
```

Use UTC ISO timestamps and integer schema versions. JSON columns are stored as validated text with an adjacent `schema_version` where extensible.

- [ ] **Step 4: Implement `SQLiteStore` and repository adapters**

Use `sqlite3.connect(self.db_path, isolation_level=None)` plus explicit transactions. Implement Core `RunRepository` and promotion lookup/save interfaces, and Web repositories for uploads, datasets, jobs, artifacts, approvals, and audit events. Every stale run update must use `WHERE id = ? AND version = ?` and fail when `rowcount != 1`.

- [ ] **Step 5: Rerun store tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_sqlite_store -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/migrations \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/stores \
  tests/web/test_sqlite_store.py
rtk git commit -m "feat: persist web console state in sqlite"
```

### Task 3: Implement canonical CSV and JSONL importers

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers/__init__.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers/base.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers/csv_importer.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers/jsonl_importer.py`
- Test: `tests/web/test_tabular_importers.py`

- [ ] **Step 1: Write CSV/JSONL inspection and normalization tests**

Test that both formats normalize equivalent records to the same Parquet rows and semantic hash. Add explicit failures for invalid UTF-8 CSV, unsupported delimiter without an option, JSONL top-level arrays, and malformed JSON with the exact line number.

```python
self.assertEqual(csv_result.semantic_hash, jsonl_result.semantic_hash)
self.assertEqual(csv_result.columns, REQUIRED_COLUMNS)
```

- [ ] **Step 2: Run tests and confirm missing importers**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_tabular_importers -v
```

- [ ] **Step 3: Define import contracts**

```python
@dataclass(frozen=True)
class ImportOptions:
    delimiter: str | None = None
    sheet_name: str | None = None
    header_row: int = 1


@dataclass(frozen=True)
class FieldMapping:
    source_to_target: Mapping[str, str]


@dataclass(frozen=True)
class NormalizedDataset:
    parquet_path: Path
    source_hash: str
    semantic_hash: str
    schema_hash: str
    columns: tuple[str, ...]
    rows: int
    warnings: tuple[str, ...]
```

`DatasetImporter` exposes `inspect`, `preview`, and `normalize`.

- [ ] **Step 4: Implement strict CSV and JSONL readers**

CSV reads UTF-8 with no replacement characters and only auto-accepts comma. JSONL reads one object per non-empty line and reports exact line numbers. Both apply the supplied `FieldMapping`, preserve the original source file, sort columns by the canonical contract, write Parquet, and compute semantic hashes from canonical row JSON.

- [ ] **Step 5: Rerun importer tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_tabular_importers -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers \
  tests/web/test_tabular_importers.py
rtk git commit -m "feat: import csv and jsonl datasets"
```

### Task 4: Add secure XLSX inspection, preview, mapping, and normalization

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers/xlsx_importer.py`
- Test: `tests/web/test_xlsx_importer.py`

- [ ] **Step 1: Write XLSX behavior and security tests**

Use `openpyxl.Workbook` fixtures for:

- multiple visible/hidden worksheets;
- a selectable header row;
- automatic mapping suggestions;
- formula cells with and without cached values;
- duplicate headers;
- row/column limit breaches;
- a fake `.xlsm` name;
- a ZIP entry set whose declared uncompressed size exceeds the configured limit.

Assert hidden sheets are reported but not selected by default, missing cached values in required columns block normalization, and macro extensions are rejected before opening the workbook.

- [ ] **Step 2: Run tests and confirm missing XLSX importer**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_xlsx_importer -v
```

- [ ] **Step 3: Implement ZIP preflight and workbook inspection**

Before `openpyxl.load_workbook`, inspect the ZIP central directory and enforce compressed bytes, summed uncompressed bytes, member count, and unsafe member names. Load with:

```python
load_workbook(path, read_only=True, data_only=True, keep_links=False)
```

Return worksheet name, visibility, dimensions, and a bounded preview.

- [ ] **Step 4: Implement mapping and normalization**

Normalize header cells to trimmed strings, reject duplicates after normalization, apply saved or user-provided mappings, stream rows into a DataFrame in bounded chunks, and write the same `NormalizedDataset` contract as CSV/JSONL. Never evaluate formulas.

- [ ] **Step 5: Rerun XLSX tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_xlsx_importer -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/importers/xlsx_importer.py \
  tests/web/test_xlsx_importer.py
rtk git commit -m "feat: import labeled xlsx datasets"
```

### Task 5: Build upload, preview, mapping, and dataset APIs plus the workflow-first workbench

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/import_service.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/imports.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/datasets.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/pages.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/base.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/workbench.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/static/console.css`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/static/console.js`
- Test: `tests/web/test_import_api.py`

- [ ] **Step 1: Write API tests for all three formats**

Use `TestClient` multipart uploads and assert:

- `POST /api/imports` creates an upload without trusting the supplied filename as a path;
- XLSX exposes `/sheets` and `/preview`;
- normalization saves mapping and creates a versioned dataset;
- validation errors return structured `{code, message, details, operator_action}`;
- the workbench lists the active base dataset and recent runs.

- [ ] **Step 2: Run tests and confirm missing routes**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_import_api -v
```

- [ ] **Step 3: Implement `ImportService`**

Store uploads as `uploads/<generated_id>/source.<validated_extension>`, stream bytes while enforcing `max_upload_bytes`, calculate SHA-256, dispatch by content/extension, and persist inspection results. Normalization writes a new immutable dataset directory and database row; an existing semantic hash may be reused but never overwritten.

- [ ] **Step 4: Implement routes and structured errors**

Add the exact endpoints from the spec. Convert `HeuriBoostError` to JSON with stable HTTP mappings: input errors `422`, conflicts `409`, missing IDs `404`, unauthorized/CSRF `403`, internal failures `500` with safe detail.

- [ ] **Step 5: Build the workbench layout**

The first screen must be the actual workflow: left navigation, active base dataset selector, CSV/JSONL/XLSX upload, XLSX worksheet/header/mapping controls, preflight results, policy selector, “校验并启动完整流程”, and a recent-run table. Keep cards at `8px` radius or less, use a quiet neutral palette with green only for success/action, and ensure all labels wrap at mobile widths.

- [ ] **Step 6: Rerun API tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_import_api -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/import_service.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/static \
  tests/web/test_import_api.py
rtk git commit -m "feat: add web dataset workbench"
```

### Task 6: Implement the durable single-worker job executor and crash recovery

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/jobs/executor.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/jobs/worker.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/jobs/supervisor.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/jobs/__init__.py`
- Test: `tests/web/test_job_executor.py`

- [ ] **Step 1: Write queue, cancellation, and restart tests**

Use a fake run function that records stages. Assert only one job is `RUNNING`, queued jobs remain after a store reopen, stale heartbeats become `INTERRUPTED`, cancellation is observed at stage boundaries, and retry creates a new attempt without changing the original job history.

- [ ] **Step 2: Run tests and confirm missing executor**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_job_executor -v
```

- [ ] **Step 3: Define the executor protocol and job statuses**

```python
class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class JobExecutor(Protocol):
    def enqueue(self, run_id: str) -> str: ...
    def request_cancel(self, job_id: str) -> None: ...
    def retry(self, job_id: str) -> str: ...
```

- [ ] **Step 4: Implement claim and heartbeat transactions**

Use one SQLite transaction to select the oldest queued job and update it to `CLAIMED`. The worker spawns one child process, records PID, marks `RUNNING`, updates heartbeat every five seconds, and records exit/result status. The child calls `run_reckless_repair` using startup-configured backend and repositories.

- [ ] **Step 5: Implement supervisor recovery**

At startup, mark jobs with stale heartbeat and no live PID as `INTERRUPTED`. Do not auto-resume them; expose an explicit retry action that delegates to the Core resume logic after fingerprint checks.

- [ ] **Step 6: Rerun executor tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_job_executor -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/jobs \
  tests/web/test_job_executor.py
rtk git commit -m "feat: execute web runs in durable worker"
```

### Task 7: Add run application services, APIs, and resumable SSE events

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/run_service.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/runs.py`
- Test: `tests/web/test_run_api.py`
- Test: `tests/web/test_sse.py`

- [ ] **Step 1: Write run lifecycle API tests**

Assert `POST /api/runs` freezes dataset/policy/backend refs and enqueues a job; `GET` returns durable state; cancel/retry/clone use `Idempotency-Key`; blocked runs cannot retry in place; interrupted runs can resume when fingerprints match.

- [ ] **Step 2: Write SSE replay tests**

Create audit/run events with IDs `1..3`, connect with `Last-Event-ID: 1`, and assert events `2` and `3` are streamed in order. Disconnect and reconnect to prove no in-memory queue is required.

- [ ] **Step 3: Run tests and confirm missing services/routes**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_run_api tests.web.test_sse -v
```

- [ ] **Step 4: Implement `RunService`**

`create_run` validates dataset roles and statuses, resolves only configured backend/policy IDs, creates the Core `RepairRequest`, appends `run.created`, and enqueues a job. `cancel`, `retry`, and `clone` append immutable audit events and return the persisted result of the idempotent operation.

- [ ] **Step 5: Implement APIs and SSE**

SSE polls durable events after the provided event ID, emits heartbeat comments every fifteen seconds, and sends JSON event data with run ID, type, stage, status, progress, message, and UTC timestamp. Apply per-response pagination to avoid unbounded event reads.

- [ ] **Step 6: Rerun lifecycle tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_run_api tests.web.test_sse -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/run_service.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/runs.py \
  tests/web/test_run_api.py tests/web/test_sse.py
rtk git commit -m "feat: expose durable reckless run api"
```

### Task 8: Build the run-detail browser workflow

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/run_detail.html`
- Modify: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/static/console.js`
- Modify: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/static/console.css`
- Modify: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/pages.py`
- Test: `tests/web/test_run_pages.py`

- [ ] **Step 1: Write server-rendered page tests**

Assert the run page contains all seven stages, current status text, artifact links, log region, cancel/retry controls appropriate to state, and the Pre Promote entry only after report generation. Verify status text exists independently of CSS color.

- [ ] **Step 2: Run tests and confirm missing page**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_run_pages -v
```

- [ ] **Step 3: Implement the dense operational layout**

Render a stable two-column work surface: stage navigation on the left; current stage summary, progress, structured events, logs, inputs, outputs, duration, and fingerprints on the right. Use fixed track sizes and responsive stacking below `800px` so status updates do not shift controls.

- [ ] **Step 4: Add resilient SSE client behavior**

The JavaScript uses `EventSource`, updates only known DOM targets, stores the last event ID, reconnects with backoff, and falls back to periodic `GET /api/runs/{id}` polling after repeated SSE failures. It must never infer a run state locally.

- [ ] **Step 5: Rerun page tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_run_pages -v
```

Expected: all tests pass.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/run_detail.html \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/static \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/pages.py \
  tests/web/test_run_pages.py
rtk git commit -m "feat: add reckless run detail page"
```

### Task 9: Serve the bilingual Pre Promote report and one-click promotion

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/report_service.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/promotion_service.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/reports.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/promotions.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/pre_promote_live.html`
- Test: `tests/web/test_promotion_api.py`

- [ ] **Step 1: Write report and promotion API tests**

Assert:

- the offline report route returns the immutable Core HTML without a usable Promote form;
- the live report page defaults to Chinese and references the same report hash;
- only `READY_FOR_PROMOTION` renders “批准并 Promote”;
- POST requires CSRF, one-time approval token, hashes, expected current model, approver, and `Idempotency-Key`;
- stale or duplicate requests produce the Core-defined behavior;
- success returns the receipt and new current model.

- [ ] **Step 2: Run tests and confirm missing routes/services**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_promotion_api -v
```

- [ ] **Step 3: Implement report service boundaries**

`ReportService` reads and verifies the immutable report manifest, returns the static file for download, and provides report data plus a short-lived single-use approval token to the live template. It does not regenerate or mutate the report.

- [ ] **Step 4: Implement promotion service and route**

Resolve approver identity from the local session, verify the token exactly once, construct Core `PromotionApproval`, and call `promote_repair`. Append approval-requested, promotion-succeeded, or promotion-failed audit events. Never call `shutil.copy2` directly in the route.

- [ ] **Step 5: Implement the approved report presentation**

Top section: decision, blockers/warnings, global deltas, production cases, historical gates, domain and artifact checks. Main section: chronological data intake, validation, compilation, training, evaluation, decision, artifacts. The Promote action stays in the decision section with exact run and target model visible. On success, replace the action region with receipt and rollback target without changing the underlying report file.

- [ ] **Step 6: Rerun promotion tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_promotion_api -v
```

Expected: all tests pass.

- [ ] **Step 7: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/report_service.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/promotion_service.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/reports.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/promotions.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/pre_promote_live.html \
  tests/web/test_promotion_api.py
rtk git commit -m "feat: approve and promote from web report"
```

### Task 10: Add datasets, models, gates, audit, and settings views

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/datasets.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/models.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/gates.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/audit.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates/settings.html`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/assets.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/audit.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/releases.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/release_service.py`
- Test: `tests/web/test_asset_pages.py`
- Test: `tests/web/test_rollback_api.py`

- [ ] **Step 1: Write page and authorization tests**

Assert datasets and model releases are versioned, gate pages are read-only, audit events paginate in stable order, and settings expose configured values without allowing adapter code edits. In `test_rollback_api.py`, assert `POST /api/releases/{run_id}/rollback` requires session, CSRF, explicit approver and `Idempotency-Key`, only restores the previous immutable release, and returns a Core `RollbackReceipt`.

- [ ] **Step 2: Implement read models and pages**

Keep these pages table-oriented and scan-friendly. Provide filters and direct links to immutable manifests/reports. Do not put page sections in floating cards or nest cards. Gate retirement and Anchor reset are displayed as unavailable in the normal UI, with links to documented administrative commands.

Implement `ReleaseService.rollback` by resolving the existing Promotion Receipt and calling Core `rollback_release`; append requested/succeeded/failed audit events. The HTTP route must not edit `current_model.json` directly.

- [ ] **Step 3: Rerun asset-page tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_asset_pages tests.web.test_rollback_api -v
```

Expected: all tests pass.

- [ ] **Step 4: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/templates \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/assets.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/audit.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/routes/releases.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/services/release_service.py \
  tests/web/test_asset_pages.py tests/web/test_rollback_api.py
rtk git commit -m "feat: add heuriboost asset and audit views"
```

### Task 11: Harden local security and verify responsive UI with Playwright

**Files:**
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/security.py`
- Modify: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/app.py`
- Test: `tests/web/test_security.py`
- Test: `tests/web/test_console_e2e.py`

- [ ] **Step 1: Write security regression tests**

Cover missing/expired session token, CSRF mismatch, cross-origin writes, path traversal filenames, symlink escape, oversized uploads, ZIP bomb preflight, arbitrary adapter names, unsafe error details, and replayed approval tokens.

- [ ] **Step 2: Implement local security middleware and helpers**

Define the team-upgrade identity boundary:

```python
class IdentityProvider(Protocol):
    def current_identity(self, request: Request) -> str: ...


class LocalIdentityProvider:
    def __init__(self, username: str):
        self.username = username

    def current_identity(self, request: Request) -> str:
        return self.username
```

Generate a cryptographically random session token at startup, accept it only once from the launch URL, then store an HttpOnly/SameSite=Strict cookie. Generate per-session CSRF tokens. Set `Content-Security-Policy`, `X-Content-Type-Options`, `Referrer-Policy`, and frame denial headers. Do not install permissive CORS middleware. Promotion and rollback services obtain the approver only through `IdentityProvider`.

- [ ] **Step 3: Write the browser end-to-end test**

With a fake backend and temporary data directory, Playwright must:

1. open the workbench;
2. upload an XLSX fixture;
3. select a worksheet and mapping;
4. start a run;
5. observe stage updates;
6. open the Chinese report;
7. click “批准并 Promote” once;
8. verify the Receipt and current model.

Capture screenshots at `1440x900`, `1024x768`, and `390x844`. Assert no horizontal body overflow and no intersecting bounding boxes for navigation, action controls, status banner, and report sections.

- [ ] **Step 4: Run security tests**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_security -v
```

- [ ] **Step 5: Install Chromium and run Playwright**

```bash
rtk conda run -n py310 python -m playwright install chromium
rtk conda run -n py310 python -m unittest tests.web.test_console_e2e -v
```

Expected: all tests pass and screenshots show no overlap or blank content.

- [ ] **Step 6: Record the logical checkpoint**

```bash
rtk git add plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/security.py \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/app.py \
  tests/web/test_security.py tests/web/test_console_e2e.py
rtk git commit -m "test: harden heuriboost web workflow"
```

### Task 12: Document startup, backup, recovery, and run the complete Web verification

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/REFERENCE.md`
- Modify: `docs/REFERENCE.zh-CN.md`
- Create: `plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/backup.py`
- Create: `plugins/heuriboost/skills/heuriboost-rag/templates/heuriboost-workspace.yml`
- Test: `tests/web/test_backup.py`

- [ ] **Step 1: Add the workspace template**

```yaml
workspace_id: default-rag
backend: xgboost-rag
promotion_target: file-pointer
policy: reckless_policy.yml
data_dir: ~/.heuriboost
web:
  host: 127.0.0.1
  port: 8787
  default_locale: zh-CN
  max_concurrent_jobs: 1
```

- [ ] **Step 2: Write and run a consistent-backup test**

The test creates a live SQLite database, one dataset, one release, and `current_model.json`, then calls `create_backup(data_dir, output_path)`. Assert the archive contains a SQLite backup made through `Connection.backup`, datasets, releases, current pointer, a SHA-256 manifest, and no uploads or run staging directories.

```bash
rtk conda run -n py310 python -m unittest tests.web.test_backup -v
```

Expected: failure because `create_backup` does not exist.

- [ ] **Step 3: Implement the backup command**

`backup.py` must acquire the workspace read lock, copy SQLite with the SQLite backup API, copy immutable datasets/releases/current pointer into a temporary directory, write and verify `backup_manifest.json`, then atomically rename the final `.tar.gz`. Add a `backup` subcommand to `heuriboost-web`:

```bash
rtk conda run -n py310 heuriboost-web backup \
  --data-dir ~/.heuriboost \
  --output /tmp/heuriboost-backup.tar.gz
```

- [ ] **Step 4: Rerun the backup test**

```bash
rtk conda run -n py310 python -m unittest tests.web.test_backup -v
```

Expected: all tests pass.

- [ ] **Step 5: Document operation and recovery**

Document installation with `[web]`, launch URL/token behavior, CSV/JSONL/XLSX limits, data-directory layout, backup of SQLite/datasets/releases/current pointer, interrupted-run recovery, Receipt and rollback, and team-upgrade interfaces.

- [ ] **Step 6: Run all HeuriBoost tests**

```bash
rtk conda run -n py310 python -m unittest discover -s tests -v
```

Expected: all Core and Web tests pass.

- [ ] **Step 7: Compile the package and scripts**

```bash
rtk conda run -n py310 python -m compileall \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag \
  plugins/heuriboost/skills/heuriboost-rag/scripts
```

Expected: no syntax errors.

- [ ] **Step 8: Start the real local server and smoke-test it**

```bash
rtk conda run -n py310 heuriboost-web \
  --config plugins/heuriboost/skills/heuriboost-rag/templates/heuriboost-workspace.yml \
  --data-dir /tmp/heuriboost-web-smoke \
  --host 127.0.0.1 \
  --port 8787
```

Expected: the command prints a tokenized local URL; `/health` returns `{"status":"ok"}` and the first screen is the run workbench.

- [ ] **Step 9: Check repository formatting and scope**

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; no generated SQLite, uploads, models, reports, screenshots, or Playwright browser files are staged.

- [ ] **Step 10: Record the logical checkpoint**

```bash
rtk git add README.md README.zh-CN.md docs/REFERENCE.md docs/REFERENCE.zh-CN.md \
  plugins/heuriboost/skills/heuriboost-rag/src/heuriboost_rag/web/backup.py \
  tests/web/test_backup.py \
  plugins/heuriboost/skills/heuriboost-rag/templates/heuriboost-workspace.yml
rtk git commit -m "docs: add heuriboost web console operations"
```
