#!/usr/bin/env python3
"""Production-case repair helpers for the HeuriBoost reranker.

This module owns the user-facing two-table compiler and the production-case
repair acceptance helpers. The existing train/eval scripts stay as low-level
canonical CSV entrypoints; this layer compiles friendly inputs into that
canonical shape.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import (
    FEATURE_NAMES,
    ensure_output_dirs,
    evaluate_ranked_frame,
    extract_features,
    group_sizes,
    rank_by_model,
    relevance_labels,
    require_dependencies,
    sort_for_ranking,
    split_frame,
    validate_dataset_frame,
    write_json,
)
from features import REGISTRY


DEFAULT_DOMAIN = "default"
DEFAULT_CASE_TOP_K = 3
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_SPLIT_SEED = 42
DEFAULT_MIN_GLOBAL_TEST_QUERIES = 10
DEFAULT_MIN_DOMAIN_TEST_QUERIES = 3
DEFAULT_MIN_DOCS_PER_QUERY = 2

LABEL_ALIASES = {
    "3": 3,
    "good": 3,
    "positive": 3,
    "2": 2,
    "partial": 2,
    "1": 1,
    "weak": 1,
    "0": 0,
    "irrelevant": 0,
    "negative": 0,
    "-1": -1,
    "bad": -1,
    "hard_negative": -1,
    "hard-negative": -1,
}

VERDICT_ALIASES = {
    "good": "good",
    "positive": "good",
    "accepted": "good",
    "bad": "bad",
    "negative": "bad",
    "wrong": "bad",
    "unknown": "unknown",
    "": "unknown",
}

BASE_ALIASES = {
    "domain": ("domain",),
    "query_id": ("query_id", "qid"),
    "query_text": ("query", "query_text"),
    "doc_id": ("doc_id", "id"),
    "doc_text": ("text", "doc_text", "document", "document_text"),
    "label": ("relevance", "label", "verdict"),
    "split": ("split",),
    "rank": ("rank",),
    "score": ("score",),
    "dense_rank": ("dense_rank",),
    "dense_score": ("dense_score",),
    "sparse_rank": ("sparse_rank",),
    "sparse_score": ("sparse_score",),
}

CASE_ALIASES = {
    "domain": ("domain",),
    "case_id": ("case_id",),
    "query_text": ("query", "query_text"),
    "doc_id": ("shown_doc_id", "doc_id", "id"),
    "doc_text": ("shown_doc_text", "doc_text", "text", "document_text"),
    "verdict": ("user_verdict", "verdict"),
    "rank": ("rank",),
    "score": ("score",),
}


@dataclass(frozen=True)
class CompileOptions:
    output_dir: Path
    resplit: bool = False
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS
    split_seed: int = DEFAULT_SPLIT_SEED
    case_top_k: int = DEFAULT_CASE_TOP_K
    strict: bool = False
    acceptance_level: str = "full"
    min_global_test_queries: int = DEFAULT_MIN_GLOBAL_TEST_QUERIES
    min_domain_test_queries: int = DEFAULT_MIN_DOMAIN_TEST_QUERIES
    min_docs_per_query: int = DEFAULT_MIN_DOCS_PER_QUERY


@dataclass(frozen=True)
class CompileResult:
    output_dir: Path
    heuriboost_dir: Path
    compiled_dir: Path
    base_dataset_path: Path
    regression_cases_path: Path
    case_sets_dir: Path
    production_cases_json_path: Path
    compile_report_path: Path
    base_df: Any
    repair_samples_df: Any
    production_cases: list[dict]
    touched_domains: list[str]
    warnings: list[str]


def utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return " ".join(str(value).strip().split())


def normalize_key(value: Any) -> str:
    return normalize_text(value).lower()


def stable_hash(*parts: Any, length: int = 12) -> str:
    joined = "\x1f".join(normalize_key(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def synthetic_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{stable_hash(*parts)}"


def scoped_id(domain: str, value: Any) -> str:
    raw = normalize_text(value)
    if not raw:
        raw = synthetic_id("id", domain)
    return f"{domain}::{raw}"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return normalize_text(value) == ""


def _row_value(row: dict, aliases: Iterable[str]) -> Any:
    for name in aliases:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return None


def _load_rows(path: str | Path) -> list[dict]:
    pd = _load_pandas()
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict] = []
        for line_no, line in enumerate(input_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
            rows.extend(_flatten_json_object(obj))
        return rows

    if suffix == ".json":
        try:
            data = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON file {input_path}: {exc}") from exc
        if isinstance(data, list):
            rows = []
            for obj in data:
                rows.extend(_flatten_json_object(obj))
            return rows
        return _flatten_json_object(data)

    try:
        frame = pd.read_csv(input_path)
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV {input_path}: {exc}") from exc
    return frame.to_dict(orient="records")


def _flatten_json_object(obj: Any) -> list[dict]:
    if not isinstance(obj, dict):
        raise SystemExit("JSON/JSONL rows must be objects.")

    if isinstance(obj.get("shown"), list):
        base = {key: value for key, value in obj.items() if key != "shown"}
        rows = []
        for shown in obj["shown"]:
            if not isinstance(shown, dict):
                raise SystemExit("JSON field 'shown' must contain objects.")
            row = dict(base)
            row.update(
                {
                    "shown_doc_id": shown.get("id", shown.get("doc_id")),
                    "shown_doc_text": shown.get("text", shown.get("doc_text")),
                    "user_verdict": shown.get("verdict", shown.get("user_verdict")),
                    "rank": shown.get("rank", row.get("rank")),
                    "score": shown.get("score", row.get("score")),
                }
            )
            rows.append(row)
        return rows

    if isinstance(obj.get("documents"), list):
        base = {key: value for key, value in obj.items() if key != "documents"}
        rows = []
        for doc in obj["documents"]:
            if not isinstance(doc, dict):
                raise SystemExit("JSON field 'documents' must contain objects.")
            row = dict(base)
            row.update(
                {
                    "doc_id": doc.get("id", doc.get("doc_id")),
                    "text": doc.get("text", doc.get("doc_text")),
                    "relevance": doc.get("relevance", doc.get("label")),
                    "rank": doc.get("rank", row.get("rank")),
                    "score": doc.get("score", row.get("score")),
                }
            )
            rows.append(row)
        return rows

    return [obj]


def _load_pandas():
    require_dependencies("pandas")
    import pandas as pd

    return pd


def _normalize_label(value: Any) -> int:
    token = normalize_key(value)
    if token in LABEL_ALIASES:
        return LABEL_ALIASES[token]
    try:
        parsed = int(float(token))
    except (TypeError, ValueError):
        raise SystemExit(
            "Unsupported relevance/label value: "
            f"{value!r}. Expected one of {sorted(LABEL_ALIASES)}."
        )
    if parsed not in {-1, 0, 1, 2, 3}:
        raise SystemExit(
            f"Unsupported numeric label: {parsed}. Expected -1, 0, 1, 2, or 3."
        )
    return parsed


def _normalize_verdict(value: Any) -> str:
    token = normalize_key(value)
    if token in VERDICT_ALIASES:
        return VERDICT_ALIASES[token]
    raise SystemExit(
        "Unsupported user_verdict value: "
        f"{value!r}. Expected good, bad, or unknown."
    )


def _normalize_split(value: Any) -> str:
    token = normalize_key(value)
    if token in {"train", "validation", "test"}:
        return token
    if token in {"valid", "val"}:
        return "validation"
    raise SystemExit(
        f"Unsupported split value: {value!r}. Expected train, validation, or test."
    )


def compile_repair_inputs(
    base_dataset: str | Path,
    production_cases: str | Path,
    options: CompileOptions,
) -> CompileResult:
    pd = _load_pandas()
    output_dir = Path(options.output_dir)
    heuriboost_dir = output_dir / ".heuriboost"
    compiled_dir = heuriboost_dir / "compiled"
    case_sets_dir = compiled_dir / "case_sets"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    case_sets_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    base_df = _compile_base_dataset(base_dataset, options, warnings)
    case_rows = _compile_case_rows(production_cases, warnings)

    base_domains = set(base_df["domain"].astype(str))
    case_domains = {row["domain"] for row in case_rows}
    missing_domains = sorted(case_domains - base_domains)
    if missing_domains:
        raise SystemExit(
            "production_cases contains domain(s) absent from base_dataset: "
            + ", ".join(missing_domains)
        )

    production_case_list = _build_production_cases(
        case_rows, base_df, options, warnings
    )
    touched_domains = sorted({case["domain"] for case in production_case_list})
    repair_samples_df = _repair_samples_frame(production_case_list)

    if options.strict:
        _validate_base_sufficiency(base_df, touched_domains, options)
        _validate_production_cases(production_case_list, options)

    base_dataset_path = compiled_dir / "query_doc_examples.csv"
    regression_cases_path = compiled_dir / "regression_cases.yaml"
    production_cases_json_path = compiled_dir / "production_cases.json"
    compile_report_path = compiled_dir / "compile_report.md"
    case_set_path = case_sets_dir / "current_production_cases.csv"

    base_df.to_csv(base_dataset_path, index=False)
    repair_samples_df.to_csv(case_set_path, index=False)
    _write_regression_cases_yaml(regression_cases_path, production_case_list)
    write_json(production_cases_json_path, production_case_list)
    _write_compile_report(
        compile_report_path,
        base_df,
        production_case_list,
        repair_samples_df,
        warnings,
        options,
    )

    return CompileResult(
        output_dir=output_dir,
        heuriboost_dir=heuriboost_dir,
        compiled_dir=compiled_dir,
        base_dataset_path=base_dataset_path,
        regression_cases_path=regression_cases_path,
        case_sets_dir=case_sets_dir,
        production_cases_json_path=production_cases_json_path,
        compile_report_path=compile_report_path,
        base_df=base_df,
        repair_samples_df=repair_samples_df,
        production_cases=production_case_list,
        touched_domains=touched_domains,
        warnings=warnings,
    )


def _compile_base_dataset(
    base_dataset: str | Path,
    options: CompileOptions,
    warnings: list[str],
):
    pd = _load_pandas()
    raw_rows = _load_rows(base_dataset)
    if not raw_rows:
        raise SystemExit("base_dataset is empty.")

    compiled_rows = []
    split_present = any(not _is_missing(_row_value(row, BASE_ALIASES["split"])) for row in raw_rows)
    for idx, row in enumerate(raw_rows, start=1):
        domain = normalize_text(_row_value(row, BASE_ALIASES["domain"])) or DEFAULT_DOMAIN
        query_text = normalize_text(_row_value(row, BASE_ALIASES["query_text"]))
        doc_text = normalize_text(_row_value(row, BASE_ALIASES["doc_text"]))
        label_value = _row_value(row, BASE_ALIASES["label"])
        if not query_text:
            raise SystemExit(f"base_dataset row {idx} is missing query text.")
        if not doc_text:
            raise SystemExit(f"base_dataset row {idx} is missing document text.")
        if _is_missing(label_value):
            raise SystemExit(f"base_dataset row {idx} is missing relevance/label.")
        label = _normalize_label(label_value)

        source_query_id = normalize_text(_row_value(row, BASE_ALIASES["query_id"]))
        if not source_query_id:
            source_query_id = synthetic_id("q", domain, query_text)
            warnings.append(f"Generated synthetic query_id for base row {idx}.")
        source_doc_id = normalize_text(_row_value(row, BASE_ALIASES["doc_id"]))
        if not source_doc_id:
            source_doc_id = synthetic_id("d", domain, doc_text)
            warnings.append(f"Generated synthetic doc_id for base row {idx}.")

        split = None
        if split_present and not options.resplit:
            raw_split = _row_value(row, BASE_ALIASES["split"])
            if _is_missing(raw_split):
                raise SystemExit(
                    "base_dataset has a split column but row "
                    f"{idx} is missing split. Use --resplit to replace splits."
                )
            split = _normalize_split(raw_split)

        compiled = {
            "domain": domain,
            "source_query_id": source_query_id,
            "query_id": scoped_id(domain, source_query_id),
            "query_text": query_text,
            "source_doc_id": source_doc_id,
            "doc_id": scoped_id(domain, source_doc_id),
            "doc_text": doc_text,
            "label": label,
            "split": split or "",
        }
        _copy_optional_numeric(row, compiled)
        compiled_rows.append(compiled)

    frame = pd.DataFrame(compiled_rows)
    if options.resplit or not split_present:
        if split_present and options.resplit:
            warnings.append("Existing split column was replaced because --resplit was used.")
        if not split_present:
            warnings.append("Split auto-generated by query_id with a fixed seed.")
        frame = _assign_splits(frame, options)

    validate_dataset_frame(frame)
    _warn_single_doc_queries(frame, warnings)
    return frame


def _copy_optional_numeric(source: dict, target: dict) -> None:
    for canonical in (
        "rank",
        "score",
        "dense_rank",
        "dense_score",
        "sparse_rank",
        "sparse_score",
    ):
        value = _row_value(source, BASE_ALIASES.get(canonical, (canonical,)))
        if not _is_missing(value):
            target[canonical] = value
    if "rank" in target and "dense_rank" not in target:
        target["dense_rank"] = target["rank"]
    if "score" in target and "dense_score" not in target:
        target["dense_score"] = target["score"]


def _assign_splits(frame, options: CompileOptions):
    split_by_query: dict[str, str] = {}
    for domain, domain_frame in frame.groupby("domain", sort=True):
        query_ids = sorted(domain_frame["query_id"].astype(str).unique().tolist())
        seed = options.split_seed + int(stable_hash(domain), 16)
        rng = random.Random(seed)
        rng.shuffle(query_ids)
        assignments = _split_query_ids(query_ids, options.split_ratios)
        split_by_query.update(assignments)
    assigned = frame.copy()
    assigned["split"] = assigned["query_id"].astype(str).map(split_by_query)
    return assigned


def _split_query_ids(
    query_ids: list[str],
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    if not query_ids:
        return {}
    if len(query_ids) == 1:
        return {query_ids[0]: "train"}
    if len(query_ids) == 2:
        return {query_ids[0]: "train", query_ids[1]: "validation"}

    train_ratio, valid_ratio, test_ratio = ratios
    total_ratio = train_ratio + valid_ratio + test_ratio
    if total_ratio <= 0:
        raise SystemExit("Split ratios must sum to a positive value.")

    n = len(query_ids)
    test_count = max(1, int(round(n * test_ratio / total_ratio)))
    valid_count = max(1, int(round(n * valid_ratio / total_ratio)))
    if test_count + valid_count >= n:
        test_count = 1
        valid_count = 1
    train_count = n - valid_count - test_count

    assignments = {}
    for query_id in query_ids[:train_count]:
        assignments[query_id] = "train"
    for query_id in query_ids[train_count : train_count + valid_count]:
        assignments[query_id] = "validation"
    for query_id in query_ids[train_count + valid_count :]:
        assignments[query_id] = "test"
    return assignments


def _warn_single_doc_queries(frame, warnings: list[str]) -> None:
    singletons = frame.groupby("query_id").size()
    count = int((singletons < 2).sum())
    if count:
        warnings.append(f"{count} query group(s) have only one candidate document.")


def _compile_case_rows(production_cases: str | Path, warnings: list[str]) -> list[dict]:
    raw_rows = _load_rows(production_cases)
    if not raw_rows:
        raise SystemExit("production_cases is empty.")

    rows = []
    for idx, row in enumerate(raw_rows, start=1):
        domain = normalize_text(_row_value(row, CASE_ALIASES["domain"])) or DEFAULT_DOMAIN
        query_text = normalize_text(_row_value(row, CASE_ALIASES["query_text"]))
        doc_text = normalize_text(_row_value(row, CASE_ALIASES["doc_text"]))
        verdict_value = _row_value(row, CASE_ALIASES["verdict"])
        if not query_text:
            raise SystemExit(f"production_cases row {idx} is missing query text.")
        if not doc_text:
            raise SystemExit(f"production_cases row {idx} is missing shown document text.")
        if _is_missing(verdict_value):
            raise SystemExit(f"production_cases row {idx} is missing user_verdict.")
        verdict = _normalize_verdict(verdict_value)

        case_id = normalize_text(_row_value(row, CASE_ALIASES["case_id"]))
        if not case_id:
            case_id = synthetic_id("case", domain, query_text, doc_text, verdict)
            warnings.append(f"Generated synthetic case_id for production row {idx}.")

        source_doc_id = normalize_text(_row_value(row, CASE_ALIASES["doc_id"]))
        if not source_doc_id:
            source_doc_id = synthetic_id("d", domain, doc_text)
            warnings.append(f"Generated synthetic shown_doc_id for production row {idx}.")

        compiled = {
            "domain": domain,
            "case_id": scoped_id(domain, case_id),
            "source_case_id": case_id,
            "query_text": query_text,
            "source_doc_id": source_doc_id,
            "doc_id": scoped_id(domain, source_doc_id),
            "doc_text": doc_text,
            "verdict": verdict,
        }
        for field in ("rank", "score"):
            value = _row_value(row, CASE_ALIASES[field])
            if not _is_missing(value):
                compiled[field] = value
        rows.append(compiled)

    if all(row["verdict"] == "unknown" for row in rows):
        raise SystemExit("production_cases contains only unknown verdicts.")
    return rows


def _build_production_cases(
    rows: list[dict],
    base_df,
    options: CompileOptions,
    warnings: list[str],
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)

    production_cases = []
    for case_id, case_rows in grouped.items():
        domains = sorted({row["domain"] for row in case_rows})
        if len(domains) != 1:
            raise SystemExit(f"Case {case_id} spans multiple domains: {domains}")
        queries = sorted({row["query_text"] for row in case_rows})
        if len(queries) != 1:
            raise SystemExit(f"Case {case_id} has multiple query texts.")
        domain = domains[0]
        query_text = queries[0]

        role_by_doc: dict[str, str] = {}
        candidates = []
        for row in case_rows:
            role = {
                "good": "good",
                "bad": "bad",
                "unknown": "context",
            }[row["verdict"]]
            existing = role_by_doc.get(row["doc_id"])
            if existing and existing != role and "context" not in {existing, role}:
                raise SystemExit(
                    f"Case {case_id} marks doc {row['doc_id']} as both "
                    f"{existing} and {role}."
                )
            if existing == "context" and role in {"good", "bad"}:
                role_by_doc[row["doc_id"]] = role
            else:
                role_by_doc.setdefault(row["doc_id"], role)
            candidates.append(_candidate_from_case_row(row, role))

        candidates = _dedupe_candidates(candidates)
        candidates = _complete_candidate_pool(candidates, base_df, domain, query_text)

        good_ids = [
            candidate["doc_id"] for candidate in candidates if candidate["role"] == "good"
        ]
        bad_ids = [
            candidate["doc_id"] for candidate in candidates if candidate["role"] == "bad"
        ]
        context_ids = [
            candidate["doc_id"] for candidate in candidates if candidate["role"] == "context"
        ]

        if not good_ids and bad_ids:
            warnings.append(
                f"Case {case_id} has only bad evidence; it requires weak acceptance."
            )

        production_cases.append(
            {
                "case_id": case_id,
                "source_case_id": case_rows[0]["source_case_id"],
                "domain": domain,
                "query": query_text,
                "top_k": int(options.case_top_k),
                "good_doc_ids": good_ids,
                "bad_doc_ids": bad_ids,
                "context_doc_ids": context_ids,
                "candidates": candidates,
            }
        )

    return production_cases


def _candidate_from_case_row(row: dict, role: str) -> dict:
    candidate = {
        "doc_id": row["doc_id"],
        "source_doc_id": row["source_doc_id"],
        "text": row["doc_text"],
        "role": role,
    }
    if "rank" in row:
        candidate["rank"] = row["rank"]
    if "score" in row:
        candidate["score"] = row["score"]
    return candidate


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    order = {"good": 3, "bad": 2, "context": 1}
    by_doc: dict[str, dict] = {}
    for candidate in candidates:
        existing = by_doc.get(candidate["doc_id"])
        if existing is None or order[candidate["role"]] > order[existing["role"]]:
            by_doc[candidate["doc_id"]] = candidate
    return list(by_doc.values())


def _complete_candidate_pool(
    candidates: list[dict],
    base_df,
    domain: str,
    query_text: str,
    limit: int = 20,
) -> list[dict]:
    existing_ids = {candidate["doc_id"] for candidate in candidates}
    completed = list(candidates)

    domain_frame = base_df[base_df["domain"].astype(str) == domain]
    if domain_frame.empty:
        return completed

    # Prefer exact same-query rows, then same-domain rows. Auto-completion only
    # adds base rows labeled as non-positive evidence, and added rows are
    # context only; they never synthesize a good target or upgrade weak cases to
    # full acceptance.
    same_query = domain_frame[
        domain_frame["query_text"].astype(str).map(normalize_key) == normalize_key(query_text)
    ]
    pools = [same_query, domain_frame]
    for pool in pools:
        for _, row in pool.iterrows():
            if int(row.get("label", 0)) > 0:
                continue
            doc_id = str(row["doc_id"])
            if doc_id in existing_ids:
                continue
            completed.append(
                {
                    "doc_id": doc_id,
                    "source_doc_id": str(row.get("source_doc_id", doc_id)),
                    "text": str(row["doc_text"]),
                    "role": "context",
                }
            )
            existing_ids.add(doc_id)
            if len(completed) >= limit:
                return completed
    return completed


def _repair_samples_frame(production_cases: list[dict]):
    pd = _load_pandas()
    rows = []
    for case in production_cases:
        query_id = f"{case['case_id']}::repair"
        for candidate in case["candidates"]:
            role = candidate["role"]
            if role not in {"good", "bad"}:
                continue
            rows.append(
                {
                    "domain": case["domain"],
                    "source_query_id": case["source_case_id"],
                    "query_id": query_id,
                    "query_text": case["query"],
                    "source_doc_id": candidate.get("source_doc_id", candidate["doc_id"]),
                    "doc_id": candidate["doc_id"],
                    "doc_text": candidate["text"],
                    "label": 3 if role == "good" else -1,
                    "split": "train",
                    "source_case_id": case["case_id"],
                }
            )
    return pd.DataFrame(rows)


def _write_regression_cases_yaml(path: Path, production_cases: list[dict]) -> None:
    require_dependencies("yaml")
    import yaml

    cases = []
    for case in production_cases:
        cases.append(
            {
                "case_id": case["case_id"],
                "query_id": f"{case['case_id']}::repair",
                "status": "gate",
                "query": case["query"],
                "must_include_doc_ids": case["good_doc_ids"],
                "must_not_include_doc_ids": case["bad_doc_ids"],
                "top_k": case["top_k"],
                "failure_type": "production_case",
                "domain": case["domain"],
            }
        )
    path.write_text(
        yaml.dump({"cases": cases}, sort_keys=False, allow_unicode=True)
    )


def _write_compile_report(
    path: Path,
    base_df,
    production_cases: list[dict],
    repair_samples_df,
    warnings: list[str],
    options: CompileOptions,
) -> None:
    lines = [
        "# Production Case Compile Report",
        "",
        "## Summary",
        "",
        f"- Base rows: {len(base_df)}",
        f"- Base query groups: {base_df['query_id'].nunique()}",
        f"- Domains: {', '.join(sorted(base_df['domain'].astype(str).unique()))}",
        f"- Production cases: {len(production_cases)}",
        f"- Repair sample rows: {len(repair_samples_df)}",
        f"- Acceptance level: {options.acceptance_level}",
        "",
        "## Splits",
        "",
    ]
    for split, count in sorted(base_df["split"].value_counts().items()):
        lines.append(f"- {split}: {int(count)} rows")

    lines.extend(["", "## Cases", ""])
    for case in production_cases:
        lines.append(
            "- `{case_id}` domain={domain} good={good} bad={bad} context={context}".format(
                case_id=case["case_id"],
                domain=case["domain"],
                good=len(case["good_doc_ids"]),
                bad=len(case["bad_doc_ids"]),
                context=len(case["context_doc_ids"]),
            )
        )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("(none)")
    path.write_text("\n".join(lines) + "\n")


def _validate_base_sufficiency(
    base_df,
    touched_domains: list[str],
    options: CompileOptions,
) -> None:
    _require_split_labels(base_df, "train", require_doc_groups=False)
    _require_split_labels(base_df, "validation", require_doc_groups=True, options=options)
    _require_split_labels(base_df, "test", require_doc_groups=True, options=options)

    test_df = split_frame(base_df, "test")
    global_queries = int(test_df["query_id"].nunique())
    if global_queries < options.min_global_test_queries:
        raise SystemExit(
            "Strict repair requires at least "
            f"{options.min_global_test_queries} test query groups; found {global_queries}."
        )

    for domain in touched_domains:
        domain_test = test_df[test_df["domain"].astype(str) == str(domain)]
        query_count = int(domain_test["query_id"].nunique())
        if query_count < options.min_domain_test_queries:
            raise SystemExit(
                f"Touched domain {domain!r} requires at least "
                f"{options.min_domain_test_queries} test query groups; found {query_count}."
            )
        _require_positive_and_negative(domain_test, f"test split for domain {domain!r}")
        _require_min_docs_per_query(
            domain_test, f"test split for domain {domain!r}", options.min_docs_per_query
        )


def _require_split_labels(
    base_df,
    split: str,
    require_doc_groups: bool,
    options: CompileOptions | None = None,
) -> None:
    frame = split_frame(base_df, split)
    if frame.empty:
        raise SystemExit(f"Strict repair requires a non-empty {split} split.")
    _require_positive_and_negative(frame, f"{split} split")
    if require_doc_groups:
        min_docs = options.min_docs_per_query if options else DEFAULT_MIN_DOCS_PER_QUERY
        _require_min_docs_per_query(frame, f"{split} split", min_docs)


def _require_positive_and_negative(frame, label: str) -> None:
    labels = frame["label"].astype(int)
    if not (labels > 0).any():
        raise SystemExit(f"Strict repair requires at least one positive label in {label}.")
    if not (labels <= 0).any():
        raise SystemExit(f"Strict repair requires at least one negative label in {label}.")


def _require_min_docs_per_query(frame, label: str, min_docs: int) -> None:
    sizes = frame.groupby("query_id").size()
    too_small = sizes[sizes < min_docs]
    if not too_small.empty:
        sample = ", ".join(str(idx) for idx in too_small.index[:5])
        raise SystemExit(
            f"Strict repair requires every {label} query to have at least "
            f"{min_docs} candidate docs. Offenders: {sample}"
        )


def _validate_production_cases(
    production_cases: list[dict],
    options: CompileOptions,
) -> None:
    actionable = [
        case
        for case in production_cases
        if case["good_doc_ids"] or case["bad_doc_ids"]
    ]
    if not actionable:
        raise SystemExit("Strict repair requires at least one actionable production case.")

    for case in actionable:
        if options.acceptance_level == "full" and not case["good_doc_ids"]:
            raise SystemExit(
                "Full acceptance requires every actionable production case to "
                f"include at least one good target. Case {case['case_id']} has none."
            )
        if not case["candidates"]:
            raise SystemExit(f"Case {case['case_id']} has no candidate pool.")


def load_compiled_production_cases(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"Compiled production cases not found: {source}")
    try:
        data = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Compiled production cases JSON is corrupt: {source}") from exc
    if not isinstance(data, list):
        raise SystemExit("Compiled production cases JSON must be a list.")
    return data


def case_candidates_frame(case: dict):
    pd = _load_pandas()
    rows = []
    for candidate in case.get("candidates", []):
        rows.append(
            {
                "query_id": case["case_id"],
                "query_text": case["query"],
                "doc_id": candidate["doc_id"],
                "doc_text": candidate["text"],
                "label": _role_label(candidate.get("role", "context")),
                "split": "test",
                "role": candidate.get("role", "context"),
                "domain": case.get("domain", DEFAULT_DOMAIN),
            }
        )
    return pd.DataFrame(rows)


def _role_label(role: str) -> int:
    if role == "good":
        return 3
    if role == "bad":
        return -1
    return 0


def rank_case_with_model(case: dict, model):
    import xgboost as xgb

    frame = case_candidates_frame(case)
    if frame.empty:
        return frame
    features = extract_features(frame)
    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
    scores = model.predict(dmatrix)
    return rank_by_model(frame, scores)


def evaluate_repair_case(case: dict, model, acceptance_level: str = "full") -> dict:
    ranked = rank_case_with_model(case, model)
    top_k = int(case.get("top_k", DEFAULT_CASE_TOP_K))
    ranked_doc_ids = ranked["doc_id"].astype(str).tolist() if not ranked.empty else []
    top_k_doc_ids = ranked_doc_ids[:top_k]

    good_ids = [str(doc_id) for doc_id in case.get("good_doc_ids", [])]
    bad_ids = [str(doc_id) for doc_id in case.get("bad_doc_ids", [])]

    good_hit = any(doc_id in top_k_doc_ids for doc_id in good_ids)
    bad_present = [doc_id for doc_id in bad_ids if doc_id in top_k_doc_ids]
    ranks = {doc_id: rank for rank, doc_id in enumerate(ranked_doc_ids, start=1)}

    if acceptance_level == "weak" and not good_ids:
        passed = not bad_present
        level = "weak"
    else:
        passed = bool(good_ids) and good_hit and not bad_present
        level = "full"

    return {
        "case_id": case["case_id"],
        "domain": case.get("domain", DEFAULT_DOMAIN),
        "acceptance_level": level,
        "passed": passed,
        "good_hit": good_hit,
        "bad_present": bad_present,
        "good_ranks": {doc_id: ranks.get(doc_id) for doc_id in good_ids},
        "bad_ranks": {doc_id: ranks.get(doc_id) for doc_id in bad_ids},
        "top_k": top_k,
    }


def evaluate_cases(cases: list[dict], model, acceptance_level: str = "full") -> list[dict]:
    return [
        evaluate_repair_case(case, model, acceptance_level=acceptance_level)
        for case in cases
        if case.get("good_doc_ids") or case.get("bad_doc_ids")
    ]


def gate_snapshot_from_case(case: dict, source_run_id: str) -> dict:
    return {
        "gate_id": case["case_id"],
        "source_case_id": case.get("source_case_id", case["case_id"]),
        "domain": case.get("domain", DEFAULT_DOMAIN),
        "query": case["query"],
        "top_k": int(case.get("top_k", DEFAULT_CASE_TOP_K)),
        "acceptance_level": "full",
        "source_run_id": source_run_id,
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidates": case.get("candidates", []),
        "good_doc_ids": case.get("good_doc_ids", []),
        "bad_doc_ids": case.get("bad_doc_ids", []),
        "context_doc_ids": case.get("context_doc_ids", []),
    }


def load_gates(path: str | Path) -> list[dict]:
    gate_path = Path(path)
    if not gate_path.exists():
        return []
    gates = []
    for line_no, line in enumerate(gate_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            gates.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid gates JSONL at {gate_path}:{line_no}: {exc}") from exc
    return gates


def append_gates(path: str | Path, gates: list[dict]) -> None:
    gate_path = Path(path)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {gate.get("gate_id") for gate in load_gates(gate_path)}
    with gate_path.open("a") as handle:
        for gate in gates:
            if gate.get("gate_id") in existing:
                continue
            handle.write(json.dumps(gate, ensure_ascii=False) + "\n")


def read_ledger(path: str | Path) -> dict:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"anchor": None, "rounds": []}
    try:
        data = json.loads(ledger_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Ledger JSON is corrupt: {ledger_path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Ledger must be a JSON object: {ledger_path}")
    data.setdefault("anchor", None)
    data.setdefault("rounds", [])
    return data


def write_ledger(path: str | Path, ledger: dict) -> None:
    write_json(path, ledger)


def metric_deltas(current: dict, anchor: dict) -> dict:
    deltas = {}
    for metric in ("ndcg@10", "mrr@10"):
        current_value = float(current.get(metric, 0.0))
        anchor_value = float(anchor.get(metric, 0.0))
        deltas[metric] = {
            "current": current_value,
            "anchor": anchor_value,
            "delta": current_value - anchor_value,
        }
    return deltas


def metrics_improved(current: dict, anchor: dict) -> bool:
    return all(
        float(current.get(metric, 0.0)) > float(anchor.get(metric, 0.0))
        for metric in ("ndcg@10", "mrr@10")
    )


def metrics_not_regressed(current: dict, anchor: dict) -> bool:
    return all(
        float(current.get(metric, 0.0)) >= float(anchor.get(metric, 0.0))
        for metric in ("ndcg@10", "mrr@10")
    )


def build_anchor(round_id: str, global_metrics: dict, domain_metrics: dict, set_by: str) -> dict:
    return {
        "round_id": round_id,
        "global": {
            "ndcg@10": float(global_metrics.get("ndcg@10", 0.0)),
            "mrr@10": float(global_metrics.get("mrr@10", 0.0)),
        },
        "domains": {
            domain: {
                "ndcg@10": float(metrics.get("ndcg@10", 0.0)),
                "mrr@10": float(metrics.get("mrr@10", 0.0)),
            }
            for domain, metrics in sorted(domain_metrics.items())
        },
        "set_by": set_by,
    }


def train_model_from_frame(frame, output_dir: str | Path, rounds: int = 40):
    require_dependencies("numpy", "xgboost")
    import xgboost as xgb

    validate_dataset_frame(frame)
    train_df = split_frame(frame, "train")
    valid_df = split_frame(frame, "validation")
    if train_df.empty:
        raise SystemExit("Training split is empty.")
    if valid_df.empty:
        raise SystemExit("Validation split is empty.")

    x_train = extract_features(train_df)
    y_train = relevance_labels(train_df)
    x_valid = extract_features(valid_df)
    y_valid = relevance_labels(valid_df)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_NAMES)
    dtrain.set_group(group_sizes(train_df))
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=FEATURE_NAMES)
    dvalid.set_group(group_sizes(valid_df))

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.08,
        "max_depth": 3,
        "min_child_weight": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        verbose_eval=False,
    )

    models_dir, _ = ensure_output_dirs(output_dir)
    model_path = models_dir / "reranker.json"
    model.save_model(model_path)
    write_json(
        models_dir / "reranker_metadata.json",
        {
            "feature_names": FEATURE_NAMES,
            "feature_set_name": REGISTRY.feature_set_name,
            "feature_set_version": REGISTRY.feature_set_version,
            "feature_versions": REGISTRY.feature_versions(),
            "params": params,
            "rounds": rounds,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "train_groups": int(train_df["query_id"].nunique()),
            "validation_groups": int(valid_df["query_id"].nunique()),
        },
    )
    return model


def load_model(model_path: str | Path):
    require_dependencies("xgboost")
    import xgboost as xgb

    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    model = xgb.Booster()
    model.load_model(path)
    return model


def evaluate_model_on_split(model, frame, split: str) -> tuple[dict, Any]:
    require_dependencies("xgboost")
    import xgboost as xgb

    eval_df = split_frame(frame, split)
    if eval_df.empty:
        raise SystemExit(f"Split is empty: {split}")
    features = extract_features(eval_df)
    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
    scores = model.predict(dmatrix)
    ranked = rank_by_model(eval_df, scores)
    return evaluate_ranked_frame(ranked), ranked


def evaluate_model_by_domain(model, frame, split: str) -> dict[str, dict]:
    metrics = {}
    for domain, domain_frame in frame[frame["split"].astype(str) == split].groupby(
        "domain", sort=True
    ):
        if domain_frame.empty:
            continue
        domain_metrics, _ = evaluate_model_on_frame(model, domain_frame)
        metrics[str(domain)] = domain_metrics
    return metrics


def evaluate_model_on_frame(model, frame) -> tuple[dict, Any]:
    require_dependencies("xgboost")
    import xgboost as xgb

    sorted_frame = sort_for_ranking(frame.copy())
    features = extract_features(sorted_frame)
    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
    scores = model.predict(dmatrix)
    ranked = rank_by_model(sorted_frame, scores)
    return evaluate_ranked_frame(ranked), ranked


def merge_training_frames(base_df, repair_samples_df, promoted_samples_path: str | Path | None = None):
    pd = _load_pandas()
    frames = [base_df]
    if promoted_samples_path is not None and Path(promoted_samples_path).exists():
        promoted = pd.read_csv(promoted_samples_path)
        if not promoted.empty:
            frames.append(promoted.reindex(columns=base_df.columns))
    if repair_samples_df is not None and not repair_samples_df.empty:
        frames.append(repair_samples_df.reindex(columns=base_df.columns))
    return pd.concat(frames, ignore_index=True)


def append_promoted_samples(path: str | Path, repair_samples_df) -> int:
    pd = _load_pandas()
    sample_path = Path(path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if repair_samples_df is None or repair_samples_df.empty:
        return 0
    added = len(repair_samples_df)
    if sample_path.exists():
        existing = pd.read_csv(sample_path)
        combined = pd.concat([existing, repair_samples_df], ignore_index=True)
    else:
        combined = repair_samples_df.copy()
    dedupe_columns = [
        column
        for column in ("source_case_id", "query_id", "doc_id")
        if column in combined.columns
    ]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="last")
    combined.to_csv(sample_path, index=False)
    return int(added)


def copy_model_artifacts(source_dir: str | Path, dest_dir: str | Path) -> None:
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("reranker.json", "reranker_metadata.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)
