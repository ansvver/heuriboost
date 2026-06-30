# Implement: Production-case repair entrypoint

## Phase 0: Planning Review

- [x] Review `prd.md` with the user.
- [x] Review `design.md` with the user.
- [x] Confirm implementation scope for the first slice.
- [x] Run `task.py start` only after review approval.

## Phase 1: Compiler Foundation

1. [x] Add `repair_cases.py`
   - input loading for CSV and JSONL
   - column alias normalization
   - relevance/verdict normalization
   - stable synthetic id helpers
   - domain-scoped id helpers
   - deterministic query-group split helper
   - compile report data structures

2. [x] Add `compile_cases.py`
   - CLI arguments:
     - `--base-dataset`
     - `--production-cases`
     - `--output-dir`
     - `--resplit`
     - split ratio / seed options
     - strict sufficiency options
   - write compiled canonical artifacts under `output/.heuriboost/compiled/`
   - print concise summary and warnings

3. [x] Validate compiler behavior
   - minimal base dataset compiles
   - recommended base dataset compiles
   - minimal production cases compile
   - default domain is applied
   - synthetic ids are stable
   - split is respected when present
   - auto-split is deterministic when missing
   - same doc good+bad conflict hard-fails

## Phase 2: Case and Gate Evaluation

1. [x] Add case/gate candidate ranking helpers
   - rank self-contained candidate pools with existing `extract_features` and
     `rank_by_model`
   - evaluate full and weak semantics
   - support multiple-good any-hit and multiple-bad all-suppressed checks

2. [x] Add gate snapshot IO
   - read/write `output/.heuriboost/gates.jsonl`
   - validate gate snapshot schema
   - evaluate all gates every repair run

3. [x] Validate case/gate behavior
   - full pass
   - full fail missing good
   - full fail bad in top_k
   - weak pass bad suppressed
   - weak cannot be promotion eligible
   - all historical gates block promotion when any fail

## Phase 3: Ledger and Anchors

1. [x] Extend ledger helpers additively
   - global anchor remains supported
   - per-domain anchors added under `anchor.domains`
   - repair round metadata records touched domains, acceptance level, gate
     results, and promotion eligibility

2. [x] Add auto-anchor initialization
   - train/eval baseline from base dataset only when no anchor exists
   - write `anchor_baseline.json`
   - refuse to overwrite existing anchor unless reset flag is explicit

3. [x] Validate anchor behavior
   - missing anchor initializes automatically
   - existing anchor is reused
   - reset flag overwrites only when explicitly set
   - global improvement required
   - touched domain non-regression required

## Phase 4: Strict Repair Orchestrator

1. [x] Add `repair_reranker.py`
   - CLI arguments:
     - `--base-dataset`
     - `--production-cases`
     - `--output-dir`
     - `--reckless` required for strict repair
     - `--acceptance-level full|weak` default full
     - `--case-top-k`
     - sufficiency threshold options
     - anchor reset / debug artifact options
   - call compiler
   - initialize/reuse anchor
   - build candidate training CSV from base train + promoted memory + current
     repair samples
   - train one user-visible candidate model into `output/models/`
   - evaluate current cases, gates, global test, and touched domains
   - write `reports/repair_report.md`
   - hard-fail on strict acceptance failure

2. [x] Validate orchestrator behavior
   - one output model is produced
   - baseline model is not exposed as a second user-visible model
   - full acceptance succeeds only with good target + bad suppression
   - weak acceptance can pass but marks promotion ineligible
   - insufficient test hard-fails
   - touched domain regression hard-fails

## Phase 5: Promotion

1. [x] Add `promote_repair.py`
   - read latest repair report / machine-readable repair metadata
   - refuse failed or weak runs
   - update current model state/pointer
   - refresh anchors
   - append full case gate snapshots
   - append promoted repair samples
   - leave user inputs unchanged

2. [x] Validate promotion behavior
   - eligible full run promotes
   - weak run refuses promotion
   - failed run refuses promotion
   - gates are self-contained snapshots
   - promoted repair memory is domain-scoped

## Phase 6: Documentation and Specs

1. [x] Update docs
   - `README.md`
   - `README.zh-CN.md`
   - `docs/REFERENCE.md`
   - `docs/REFERENCE.zh-CN.md`
   - `CODEBUDDY.md`

2. [x] Update Trellis backend spec
   - document production-case repair entrypoint contracts
   - document two-table user contract
   - document gate/test/anchor separation
   - document weak/full acceptance and promotion eligibility

## Validation Commands

```bash
python3 -m py_compile skills/heuriboost-rag/scripts/*.py

python3 skills/heuriboost-rag/scripts/compile_cases.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_minimal.csv \
  --output-dir examples/fiqa/output

python3 skills/heuriboost-rag/scripts/repair_reranker.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_full.csv \
  --output-dir examples/fiqa/output \
  --reckless

python3 skills/heuriboost-rag/scripts/repair_reranker.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_bad_only.csv \
  --output-dir examples/fiqa/output \
  --reckless \
  --acceptance-level weak

python3 skills/heuriboost-rag/scripts/promote_repair.py \
  --output-dir examples/fiqa/output
```

Use purpose-built tiny fixtures for unit/smoke validation rather than relying
only on the committed FiQA demo.

## Risks / Rollback Points

- Ledger shape changes can affect existing reckless eval. Keep additions
  backward-compatible and test old ledger files.
- Auto-anchor initialization trains a baseline model internally. Keep temporary
  artifacts hidden and make reports explicit.
- Production case rows entering training is intentional only for the new repair
  entrypoint. Do not weaken legacy `--case-sets` isolation.
- Domain-scoped ids can break joins if source/internal ids are mixed. Preserve
  both fields and use internal ids consistently after compilation.
- Weak acceptance must never become promotion-eligible.
