# Backend Development Guidelines

> Best practices for backend development in this project.

---

## Overview

This directory contains guidelines for backend development. Fill in each file with your project's specific conventions.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Module organization and file layout | V0 runtime documented |
| [Database Guidelines](./database-guidelines.md) | ORM patterns, queries, migrations | To fill |
| [Error Handling](./error-handling.md) | Error types, handling strategies | To fill |
| [Quality Guidelines](./quality-guidelines.md) | Code standards, forbidden patterns | V0 runtime documented |
| [Logging Guidelines](./logging-guidelines.md) | Structured logging, log levels | To fill |
| [FiQA Demo & Feature-Set Contracts](./fiqa-demo-contracts.md) | FiQA build obligations, label/feature leakage rules, FEATURE_NAMES consistency, reckless production-case repair entrypoint | Documented 2026-06-30 |
| [Feature Recipe Registry Contracts](./feature-registry-contracts.md) | FeatureRecipe registry load/validate contracts, ALLOWED_INPUTS, Option C shared impl, eager load | Documented 2026-06-29 |
| [HPO Engine Contracts](./hpo-contracts.md) | HPOEngine adapter signatures, raw-label nDCG consistency, nthread=1 determinism, test-blind search + post-hoc eval, overfit finding | Documented 2026-06-29 |
| [Ablation Framework Contracts](./ablation-contracts.md) | A/B/C/D 4-cell runner, candidate probe contract, dual val+test+gate promotion rule, unified training procedure | Documented 2026-06-29 |
| [LLM Candidate Discovery Contracts](./discovery-contracts.md) | One-shot JSON-mode LLM call, full context, static ast validation (no importlib), drop+warn, safety stance | Documented 2026-06-29 |

---

## How to Fill These Guidelines

For each guideline file:

1. Document your project's **actual conventions** (not ideals)
2. Include **code examples** from your codebase
3. List **forbidden patterns** and why
4. Add **common mistakes** your team has made

The goal is to help AI assistants and new team members understand how YOUR project works.

---

**Language**: All documentation should be written in **English**.
