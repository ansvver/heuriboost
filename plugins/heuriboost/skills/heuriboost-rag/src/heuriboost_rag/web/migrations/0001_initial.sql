CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE workspaces (
    id TEXT PRIMARY KEY,
    backend_name TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE uploads (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL UNIQUE,
    format_name TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    content_hash TEXT NOT NULL,
    inspection_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    workspace_id TEXT,
    upload_id TEXT,
    role TEXT NOT NULL,
    semantic_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    normalized_path TEXT NOT NULL UNIQUE,
    metadata_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
    FOREIGN KEY (upload_id) REFERENCES uploads(id),
    UNIQUE (workspace_id, role, semantic_hash)
);

CREATE TABLE import_profiles (
    id TEXT PRIMARY KEY,
    workspace_id TEXT,
    format_name TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    metadata_json TEXT NOT NULL,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE run_stages (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    state TEXT NOT NULL,
    input_hash TEXT,
    output_hash TEXT,
    details_json TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    UNIQUE (run_id, stage)
);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_job_id TEXT,
    status TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    pid INTEGER,
    heartbeat_at TEXT,
    cancel_requested_at TEXT,
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (parent_job_id) REFERENCES jobs(id)
);

CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    UNIQUE (run_id, stage, artifact_type, content_hash)
);

CREATE TABLE approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    report_hash TEXT NOT NULL,
    decision_hash TEXT NOT NULL,
    expected_current_model TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    approval_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE promotions (
    idempotency_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    current_model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE INDEX runs_state_updated_at_idx ON runs (state, updated_at DESC);
CREATE INDEX jobs_status_created_at_idx ON jobs (status, created_at ASC);
CREATE INDEX run_stages_run_id_idx ON run_stages (run_id, id);
CREATE INDEX artifacts_run_id_idx ON artifacts (run_id, id);
CREATE INDEX audit_events_occurred_at_idx ON audit_events (occurred_at ASC, id ASC);

CREATE TRIGGER audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events are append-only');
END;

CREATE TRIGGER audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events are append-only');
END;
