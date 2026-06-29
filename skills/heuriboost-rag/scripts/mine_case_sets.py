#!/usr/bin/env python3
"""Mine case_sets: same-pattern training samples for pending regression cases.

For each ``pending`` case in ``regression_cases.yaml``, this script finds
non-case FiQA queries whose docs exhibit the same failure shape (a hard
negative near the top + a positive buried below), filters them by semantic
similarity to the case's query, enforces B+C isolation (no case query_id, no
case doc_id), and writes the surviving rows to
``examples/fiqa/case_sets/<case_id>.csv``.

Mining rule = a+b+c intersection:

  (c) failure_type match — a candidate query must share the case's
      ``failure_type``. Today all pending cases are ``semantic_hard_negative``
      and the CSV has no per-query failure_type, so this is approximated as
      "has both a -1 row and a 3 row" (exhibits the hard-negative/positive
      structure at all). The explicit failure_type check is kept for future
      multi-type extension.
  (b) failure shape — a candidate query has a hard-negative row (label == -1)
      with ``dense_rank <= SHAPE_RANK`` AND a positive row (label == 3) with
      ``dense_rank >= SHAPE_POS_GAP``.
  (a) semantic similarity — among queries passing (c)+(b), keep the K most
      similar to the case's ``query_text`` by cosine of all-MiniLM-L6-v2
      embeddings.

B+C isolation:
  B — drop any candidate whose ``query_id`` is in the set of ALL case
      ``query_id`` values (any status).
  C — drop specific ROWS whose ``doc_id`` is in the set of ALL case doc_ids
      (must_include + must_not_include across all cases). A query can still
      contribute its other docs.

The output CSV has the same schema as the main dataset plus a
``source_case_id`` column (train ignores it; it preserves traceability).

sentence-transformers is a BUILD dependency (requirements-build.txt), not a
runtime dependency. Mining reuses ``examples/fiqa/.cache/query_embeddings.npz``
when present; otherwise it encodes on the fly and saves the cache for reuse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import require_dependencies


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_cases(path: str | Path) -> list[dict]:
    require_dependencies("yaml")
    import yaml

    case_path = Path(path)
    if not case_path.exists():
        raise SystemExit(f"Regression cases file not found: {case_path}")
    data = yaml.safe_load(case_path.read_text()) or {}
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("regression_cases.yaml must contain a top-level cases list.")
    return cases


def build_case_denylist(cases: list[dict]) -> tuple[set[str], set[str]]:
    """Return (case_query_ids, case_doc_ids) across ALL cases (any status).

    B isolation uses case_query_ids; C isolation uses case_doc_ids.
    """
    case_query_ids: set[str] = set()
    case_doc_ids: set[str] = set()
    for case in cases:
        qid = case.get("query_id")
        if qid is not None:
            case_query_ids.add(str(qid))
        for key in ("must_include_doc_ids", "must_not_include_doc_ids"):
            for did in case.get(key, []) or []:
                case_doc_ids.add(str(did))
    return case_query_ids, case_doc_ids


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_cached_embeddings(cache_path: Path) -> dict[str, object]:
    """Load query_embeddings.npz if it exists.

    Returns a dict ``{query_id: embedding_vector}`` (empty if absent/corrupt).
    """
    import numpy as np

    if not cache_path.exists():
        return {}
    try:
        data = np.load(cache_path, allow_pickle=False)
    except Exception as exc:
        print(f"mine_case_sets: failed to load cache {cache_path}: {exc}", file=sys.stderr)
        return {}
    if "query_ids" not in data or "embeddings" not in data:
        return {}
    query_ids = [str(q) for q in data["query_ids"]]
    embeddings = data["embeddings"]
    if embeddings.shape[0] != len(query_ids):
        return {}
    return {qid: embeddings[i] for i, qid in enumerate(query_ids)}


def save_cached_embeddings(cache_path: Path, embedding_map: dict[str, object]) -> None:
    """Save query embeddings to query_embeddings.npz keyed by query_id."""
    import numpy as np

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    query_ids = list(embedding_map.keys())
    matrix = np.array([embedding_map[qid] for qid in query_ids])
    np.savez(cache_path, query_ids=np.array(query_ids), embeddings=matrix)


def encode_texts(texts: list[str]) -> object:
    """Encode a list of texts with all-MiniLM-L6-v2, returning a normalized
    numpy array (one row per text)."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(embeddings)


def ensure_query_embeddings(
    query_text_by_id: dict[str, str],
    cache_path: Path,
) -> dict[str, object]:
    """Build a complete ``{query_id: embedding}`` map for all CSV queries.

    Reuses cached embeddings when present; encodes the rest in one batch and
    saves the updated cache for future reuse.
    """
    cached = load_cached_embeddings(cache_path)
    if cached:
        print(f"Loaded cached query embeddings from {cache_path} ({len(cached)} queries)")

    missing_qids = [qid for qid in query_text_by_id if qid not in cached]
    if missing_qids:
        print(f"Encoding {len(missing_qids)} new query text(s) with all-MiniLM-L6-v2...")
        texts = [query_text_by_id[qid] for qid in missing_qids]
        embs = encode_texts(texts)
        for i, qid in enumerate(missing_qids):
            cached[qid] = embs[i]
        save_cached_embeddings(cache_path, cached)
        print(f"Saved {len(cached)} query embeddings to {cache_path}")

    return cached


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def candidate_passes_shape(group, shape_rank: int, shape_pos_gap: int) -> bool:
    """(b) shape filter: has a -1 row with dense_rank <= shape_rank AND a 3
    row with dense_rank >= shape_pos_gap."""
    has_hard_neg_top = False
    has_pos_buried = False
    for _, row in group.iterrows():
        label = int(row["label"])
        try:
            dense_rank = float(row["dense_rank"])
        except (TypeError, ValueError):
            continue
        if label == -1 and dense_rank <= shape_rank:
            has_hard_neg_top = True
        if label == 3 and dense_rank >= shape_pos_gap:
            has_pos_buried = True
    return has_hard_neg_top and has_pos_buried


def failure_type_match(_case: dict, _candidate_group) -> bool:
    """(c) failure_type match.

    Today all pending cases are ``semantic_hard_negative`` and the CSV has no
    per-query failure_type field, so this is a no-op pass-through. The check
    exists for future multi-type extension (e.g. when cases carry distinct
    failure_types and the CSV is extended with a matching field).
    """
    return True


def find_shape_candidates(
    df,
    shape_rank: int,
    shape_pos_gap: int,
) -> list[str]:
    """Return query_ids that pass (c) + (b): have both -1 and 3 labels AND
    the failure shape (hard negative near top, positive buried)."""
    candidates: list[str] = []
    for qid, group in df.groupby("query_id", sort=False):
        labels = set(group["label"].astype(int))
        if -1 not in labels or 3 not in labels:
            continue  # (c): must exhibit hard-negative/positive structure
        if not candidate_passes_shape(group, shape_rank, shape_pos_gap):
            continue  # (b): must have the failure shape
        candidates.append(str(qid))
    return candidates


def mine_for_case(
    case: dict,
    df,
    case_query_ids: set[str],
    case_doc_ids: set[str],
    shape_candidates: list[str],
    query_embeddings: dict[str, object],
    shape_rank: int,
    shape_pos_gap: int,
    top_k_similar: int,
) -> list[dict]:
    """Mine candidate rows for a single pending case.

    Returns a list of row dicts (same schema as the CSV plus source_case_id).
    """
    import numpy as np

    case_id = case.get("case_id", "<missing>")
    case_query_text = str(case.get("query", ""))

    # Step 1: (c) + (b) candidates are pre-computed. Apply failure_type_match
    # (no-op today) for future extension.
    candidate_qids = [
        qid for qid in shape_candidates
        if failure_type_match(case, None)
    ]

    # Step 2: B isolation — drop candidate queries whose query_id is any
    # case's query_id.
    candidate_qids = [qid for qid in candidate_qids if qid not in case_query_ids]
    if not candidate_qids:
        return []

    # Step 3: (a) semantic similarity — encode the case query, compute cosine
    # against candidate embeddings, keep top-k.
    case_vec = encode_texts([case_query_text])[0]
    emb_matrix = np.array([query_embeddings[qid] for qid in candidate_qids])
    sims = emb_matrix @ np.asarray(case_vec)

    k = min(top_k_similar, len(candidate_qids))
    # argsort is ascending; reverse for descending similarity. Use a stable
    # sort and preserve the resulting order as a LIST (not a set) so the mined
    # CSV rows are deterministic across runs (most-similar query first).
    top_indices = np.argsort(sims)[::-1][:k]
    selected_qids = [candidate_qids[i] for i in top_indices]

    # Step 4: collect rows from selected queries, applying C isolation
    # (row-level drop of doc_ids in the case doc_id denylist).
    mined_rows: list[dict] = []
    for qid in selected_qids:
        group = df[df["query_id"].astype(str) == qid]
        for _, row in group.iterrows():
            doc_id = str(row["doc_id"])
            if doc_id in case_doc_ids:
                continue  # C isolation
            row_dict = {col: row[col] for col in df.columns}
            row_dict["source_case_id"] = case_id
            mined_rows.append(row_dict)

    return mined_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CASE_SET_COLUMNS = [
    "query_id",
    "query_text",
    "doc_id",
    "chunk_id",
    "doc_text",
    "dense_rank",
    "dense_score",
    "sparse_rank",
    "sparse_score",
    "label",
    "split",
    "source_case_id",
]


def write_case_set_csv(path: Path, rows: list[dict]) -> None:
    """Write mined rows to a case_set CSV. If rows is empty, write header only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd = __import__("pandas")
    frame = pd.DataFrame(rows, columns=CASE_SET_COLUMNS)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="examples/fiqa/query_doc_examples.csv",
        help="Path to the main query_doc_examples.csv",
    )
    parser.add_argument(
        "--cases",
        default="examples/fiqa/regression_cases.yaml",
        help="Path to regression_cases.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="examples/fiqa/case_sets",
        help="Directory for mined case_set CSVs + manifest.json",
    )
    parser.add_argument("--shape-rank", type=int, default=3, help="(b) max dense_rank for a hard negative")
    parser.add_argument("--shape-pos-gap", type=int, default=5, help="(b) min dense_rank for a buried positive")
    parser.add_argument("--top-k-similar", type=int, default=10, help="(a) max candidates per case")
    parser.add_argument(
        "--cache-dir",
        default="examples/fiqa/.cache",
        help="Directory for query_embeddings.npz reuse",
    )
    args = parser.parse_args()

    require_dependencies("pandas", "yaml", "numpy")
    import pandas as pd

    # Load dataset + cases.
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)

    cases = load_cases(args.cases)
    case_query_ids, case_doc_ids = build_case_denylist(cases)

    # Pending cases are the ones to mine for.
    pending_cases = [c for c in cases if c.get("status", "gate") == "pending"]
    if not pending_cases:
        print("No pending cases to mine for. Nothing to do.")
        return

    # Build query_id -> query_text map for candidate encoding.
    query_text_by_id: dict[str, str] = {}
    for qid, group in df.groupby("query_id", sort=False):
        query_text_by_id[str(qid)] = str(group.iloc[0]["query_text"])

    # Pre-encode all CSV query texts (reusing cache) so similarity lookups
    # are fast and the cache is complete for future runs.
    cache_path = Path(args.cache_dir) / "query_embeddings.npz"
    query_embeddings = ensure_query_embeddings(query_text_by_id, cache_path)

    # Pre-compute (c)+(b) shape candidates once (same for all cases today,
    # since failure_type_match is a no-op pass-through).
    shape_candidates = find_shape_candidates(df, args.shape_rank, args.shape_pos_gap)
    print(f"Shape candidates ((c)+(b)): {len(shape_candidates)} queries")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mining_params": {
            "shape_rank": args.shape_rank,
            "shape_pos_gap": args.shape_pos_gap,
            "top_k_similar": args.top_k_similar,
            "dataset": str(dataset_path),
            "cases": str(Path(args.cases)),
        },
        "cases": [],
    }

    total_rows = 0
    for case in pending_cases:
        case_id = case.get("case_id", "<missing>")
        mined_rows = mine_for_case(
            case,
            df,
            case_query_ids,
            case_doc_ids,
            shape_candidates,
            query_embeddings,
            args.shape_rank,
            args.shape_pos_gap,
            args.top_k_similar,
        )

        case_set_path = out_dir / f"{case_id}.csv"
        write_case_set_csv(case_set_path, mined_rows)
        total_rows += len(mined_rows)

        mined_query_ids = sorted({str(r["query_id"]) for r in mined_rows})
        manifest["cases"].append({
            "case_id": case_id,
            "query_id": str(case.get("query_id", "")),
            "failure_type": case.get("failure_type", ""),
            "mined_rows": len(mined_rows),
            "mined_query_ids": mined_query_ids,
        })

        if mined_rows:
            print(
                f"  {case_id}: mined {len(mined_rows)} rows from "
                f"{len(mined_query_ids)} query/queries -> {case_set_path}"
            )
        else:
            print(
                f"  {case_id}: WARNING mined 0 rows (empty file written) -> {case_set_path}",
                file=sys.stderr,
            )

    # Write manifest.
    import json

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote manifest: {manifest_path}")
    print(
        f"Done. Mined for {len(pending_cases)} pending case(s): "
        f"{total_rows} total rows."
    )


if __name__ == "__main__":
    main()
