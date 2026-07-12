# Core 6 Artifact Boundary

`XGBoostRagBackend` emits deterministic run-local `compiled-input-binding` and
`candidate-binding` artifacts. The compile binding covers the input
`DatasetRef` semantics, active execution identity, compiled metadata (including
`touched_domains`), and compiled content artifacts. The candidate binding
covers the inherited compile binding, candidate metadata, model bytes, model
metadata, and every non-self candidate artifact; the outer
`candidate_artifact_set_hash` also covers the binding artifact itself.

These bindings detect ordinary modification of metadata, model references, or
artifact refs. They do not authenticate a caller that can fabricate every
artifact, ref, metadata object, and binding consistently. That trust boundary
belongs to Core 6, not to an adapter-local signing scheme.

Core 6 must seal both stage results with `LocalArtifactStore`:

- After compile, materialize canonical compiled stage-result metadata and the
  compile binding as stage artifacts, then call `complete_stage()` with the
  complete set.
- After train, materialize canonical candidate result metadata and the
  candidate binding alongside the model and model metadata, then call
  `complete_stage()` with the complete set.
- On resume, accept `CompiledInputs` or `CandidateModel` only after rebuilding
  them from a verified immutable stage manifest and its artifact snapshots.
  Do not resume from caller-supplied in-memory metadata or path refs.
  Manifest artifact paths are relative to `LocalArtifactStore.root`; rebase
  them under that root before rebuilding the adapter objects. Verify the full
  manifest, read Core 6 stage-result metadata from its own artifact, and pass
  only adapter-declared artifact refs into `CompiledInputs` or
  `CandidateModel`. Do not pass orchestration-only result-metadata artifacts
  to adapter artifact verification. When restoring a `CandidateModel`, also
  set `model_path` to the rebased `xgboost-model` artifact path; rebasing only
  `candidate.artifacts` leaves the model reference inconsistent.

Core 6 must continue to call `verify_artifacts(candidate, context)` with the
run's resolved context after manifest verification. Do not add an adapter-local
signature as a substitute for immutable stage manifests.
