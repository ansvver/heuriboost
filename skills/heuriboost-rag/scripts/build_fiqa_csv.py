#!/usr/bin/env python3
"""Offline generator for the HeuriBoost FiQA demo CSV.

This script regenerates ``examples/fiqa/query_doc_examples.csv`` from a slice of
the BEIR/FiQA-2018 financial question-answering dataset. It is an AUTHOR-ONLY
tool: it is run locally by a maintainer and is NOT part of the demo runtime path
(validate/train/eval). It is never run in CI or by the agent.

Requirements (heavy; kept out of the lightweight runtime requirements.txt):

    python -m pip install -r skills/heuriboost-rag/requirements-build.txt

It also requires:

  * Network access to download FiQA (corpus, queries, qrels) via the Hugging
    Face ``datasets`` library. Downloaded data goes into ``--cache-dir`` (which
    is gitignored) and the standard Hugging Face cache.
  * An LLM API key for build-time label judging, ONLY when --label-mode llm
    (the default). With --label-mode heuristic, no key or network LLM is needed.
    For llm mode this script uses an OpenAI-compatible client; by default it
    targets DeepSeek
    (``--base-url https://api.deepseek.com``, model ``deepseek-chat``) and reads
    ``DEEPSEEK_API_KEY`` (falling back to ``OPENAI_API_KEY``). To use OpenAI,
    pass ``--base-url ""`` (or the OpenAI base URL) and ``--judge-model``
    accordingly. If the client or key is missing, the script exits with a clear,
    actionable error instead of a stack trace.

What it does:

  1. Loads FiQA-2018 corpus, queries, and qrels honoring the native
     train/validation/test split.
  2. Slices to small per-split query caps for a committable demo.
  3. Retrieves candidates per query with BM25 (rank_bm25, "sparse") and a dense
     encoder (sentence-transformers all-MiniLM-L6-v2, "dense"), computes an RRF
     ordering to pick the candidate union, and records dense/sparse ranks/scores.
     NOTE: FiQA does not ship a candidate set, so this script runs retrieval
     itself to produce dense/sparse scores. That is expected.
  4. Assigns labels {3,2,1,0,-1}. qrel positives are always seeded as 3. For
     the remaining candidates, two modes are available via --label-mode:
       * llm (default): an LLM judge grades each candidate on the full 5-level
         scale. Needs an API key; ~thousands of calls for the default caps.
       * heuristic: zero-cost, deterministic, no LLM. A non-positive ranked
         highly by the dense retriever (dense_rank <= --hard-negative-rank) is
         labeled -1 (semantic hard negative); everything else is 0. FiQA qrels
         are sparse, so some -1s may be unlabeled-relevant; this is a demo
         approximation, documented in the data card, not a benchmark.
  5. Writes a CSV with the fixed HeuriBoost column schema; doc_text is truncated
     to ``--max-doc-chars``.

Nothing is written outside ``--output`` and ``--cache-dir``.

Verify syntax only with ``python3 -m py_compile``; do NOT execute in CI.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency / client helpers (imports are deferred so --help works anywhere)
# ---------------------------------------------------------------------------

def _fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"build_fiqa_csv: {message}", file=sys.stderr)
    raise SystemExit(2)


def _require_build_deps() -> None:
    missing = []
    for module in ("rank_bm25", "sentence_transformers", "datasets"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        _fail(
            "missing build dependencies: "
            + ", ".join(missing)
            + ". Install them with: python -m pip install -r "
            "skills/heuriboost-rag/requirements-build.txt"
        )


class LLMJudge:
    """Thin wrapper around an LLM client used only at dataset build time.

    The client is imported and instantiated lazily so the rest of the build can
    run (e.g. ``--dry-run`` retrieval inspection) without an API key, and so a
    missing client/key produces a clear error rather than a stack trace.
    """

    GRADES = {3, 2, 1, 0, -1}

    PROMPT_TEMPLATE = (
        "You are grading whether a retrieved passage supports the answer to a "
        "financial question. Return ONLY a single integer grade from this scale:\n"
        "  3  = directly supports the answer\n"
        "  2  = partially supports the answer\n"
        "  1  = related but weak evidence\n"
        "  0  = irrelevant\n"
        " -1  = misleading hard negative (same financial topic but answers a "
        "different entity/situation, so it cannot support the answer)\n\n"
        "Question:\n{query}\n\nPassage:\n{doc}\n\n"
        "Grade (one of 3, 2, 1, 0, -1):"
    )

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            _fail(
                "the openai package is required for label judging. Install it "
                "with: python -m pip install -r "
                "skills/heuriboost-rag/requirements-build.txt"
            )
        # DeepSeek and other providers expose an OpenAI-compatible API. Read the
        # key from DEEPSEEK_API_KEY first, then fall back to OPENAI_API_KEY.
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            _fail(
                "No API key found. Export DEEPSEEK_API_KEY (or OPENAI_API_KEY) "
                "before generating labels, e.g. export DEEPSEEK_API_KEY=sk-..."
            )
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def grade(self, query_text: str, doc_text: str) -> int:
        client = self._ensure_client()
        prompt = self.PROMPT_TEMPLATE.format(query=query_text, doc=doc_text)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # network/auth/etc.
            _fail(f"LLM judge call failed: {exc}")
        return self._parse_grade(raw)

    @classmethod
    def _parse_grade(cls, raw: str) -> int:
        token = raw.replace("grade", "").strip().split()[0] if raw.strip() else ""
        try:
            value = int(token)
        except ValueError:
            # Unparseable judge output is treated as irrelevant rather than crashing.
            return 0
        return value if value in cls.GRADES else 0


# ---------------------------------------------------------------------------
# Heuristic labeling (no LLM)
# ---------------------------------------------------------------------------


def heuristic_label(dense_rank, hard_negative_rank: int) -> int:
    """Assign a label without an LLM, for a candidate that is NOT a qrel positive.

    Rule (deterministic, zero-cost):
      * a candidate the dense retriever ranked highly (dense_rank <=
        hard_negative_rank) but that is not a qrel positive is treated as a
        semantic hard negative -> -1 (the retriever was fooled by topical
        similarity).
      * everything else -> 0 (irrelevant).

    Caveat: FiQA qrels are sparsely annotated, so some -1s may actually be
    unlabeled-relevant. This is an approximation suitable for a demo, not a
    benchmark; it is documented in the data card.
    """
    try:
        rank = int(dense_rank)
    except (TypeError, ValueError):
        # No dense rank (e.g. a seeded positive that fell outside top-k); not a
        # retriever-fooled hard negative.
        return 0
    return -1 if 0 < rank <= hard_negative_rank else 0


# ---------------------------------------------------------------------------
# FiQA loading
# ---------------------------------------------------------------------------

# We load FiQA-2018 through the Hugging Face ``datasets`` library using the BeIR
# mirror, which exposes three configurations:
#   * BeIR/fiqa            "corpus"  -> {_id, title, text}
#   * BeIR/fiqa            "queries" -> {_id, text}
#   * BeIR/fiqa-qrels                -> {query-id, corpus-id, score} with native
#                                       train/validation/test splits.
# BeIR/fiqa-qrels exposes the dev split under the name "validation" on the HF
# Hub, so we load "validation" directly (no "dev" alias exists there).

QREL_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}


def load_fiqa(cache_dir: str):
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/fiqa", "corpus", cache_dir=cache_dir)["corpus"]
    queries_ds = load_dataset("BeIR/fiqa", "queries", cache_dir=cache_dir)["queries"]

    corpus = {}
    for row in corpus_ds:
        text = (row.get("title") or "").strip()
        body = (row.get("text") or "").strip()
        full = (text + " " + body).strip() if text else body
        corpus[str(row["_id"])] = full

    queries = {str(row["_id"]): str(row["text"]) for row in queries_ds}

    # qrels: positive (query_id -> set(doc_id)) keyed by mapped split name.
    qrels: dict[str, dict[str, set[str]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for native_split, mapped in QREL_SPLIT_MAP.items():
        try:
            qrel_ds = load_dataset(
                "BeIR/fiqa-qrels", split=native_split, cache_dir=cache_dir
            )
        except Exception as exc:
            _fail(f"failed to load FiQA qrels split '{native_split}': {exc}")
        for row in qrel_ds:
            if int(row["score"]) <= 0:
                continue
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            qrels[mapped].setdefault(qid, set()).add(did)

    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def build_retrievers(corpus_ids: list[str], corpus_texts: list[str]):
    """Build BM25 and dense encoders over the (sliced) corpus."""
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    tokenized = [simple_tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_embeddings = encoder.encode(
        corpus_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return bm25, encoder, doc_embeddings


def retrieve_candidates(
    query_text: str,
    corpus_ids: list[str],
    bm25,
    encoder,
    doc_embeddings,
    top_k: int,
):
    """Return ordered candidate doc_ids plus dense/sparse rank/score maps.

    Candidate union is the RRF-fused top-k of dense and sparse rankings.
    """
    import numpy as np

    # Sparse (BM25)
    sparse_scores = bm25.get_scores(simple_tokenize(query_text))
    sparse_order = list(np.argsort(sparse_scores)[::-1])
    sparse_rank = {idx: rank for rank, idx in enumerate(sparse_order, start=1)}

    # Dense (cosine on normalized embeddings == dot product)
    query_emb = encoder.encode(
        [query_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]
    dense_scores = doc_embeddings @ query_emb
    dense_order = list(np.argsort(dense_scores)[::-1])
    dense_rank = {idx: rank for rank, idx in enumerate(dense_order, start=1)}

    # RRF over the two rankings to choose the candidate union.
    rrf = {}
    for idx in range(len(corpus_ids)):
        score = 0.0
        if idx in dense_rank:
            score += 1.0 / (60.0 + dense_rank[idx])
        if idx in sparse_rank:
            score += 1.0 / (60.0 + sparse_rank[idx])
        rrf[idx] = score
    union = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]

    candidates = []
    for idx in union:
        candidates.append(
            {
                "doc_id": corpus_ids[idx],
                "dense_rank": dense_rank[idx],
                "dense_score": float(dense_scores[idx]),
                "sparse_rank": sparse_rank[idx],
                "sparse_score": float(sparse_scores[idx]),
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
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
]


def select_query_ids(qrels_for_split: dict[str, set[str]], cap: int) -> list[str]:
    # Deterministic: sort query ids, take the first `cap` that have a positive.
    return sorted(qrels_for_split.keys())[:cap]


def dump_query_embeddings(
    cache_dir: str,
    selected: dict[str, list[str]],
    queries: dict[str, str],
    encoder,
) -> None:
    """Dump per-query MiniLM embeddings to ``<cache_dir>/query_embeddings.npz``.

    This is an optional convenience: ``mine_case_sets.py`` reuses this cache to
    avoid a second encoding pass. If it fails, mining falls back to encoding on
    the fly, so a failure here is non-fatal (warned on stderr).
    """
    import numpy as np

    query_ids: list[str] = []
    query_texts: list[str] = []
    for query_ids_for_split in selected.values():
        for qid in query_ids_for_split:
            text = queries.get(qid)
            if text:
                query_ids.append(qid)
                query_texts.append(text)

    if not query_ids:
        return

    try:
        embeddings = encoder.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cache_path = Path(cache_dir) / "query_embeddings.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            query_ids=np.array(query_ids),
            embeddings=np.asarray(embeddings),
        )
        print(
            f"Dumped {len(query_ids)} query embeddings to {cache_path} "
            "(reused by mine_case_sets.py)",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"build_fiqa_csv: failed to dump query embeddings: {exc}", file=sys.stderr)


def build(args) -> None:
    _require_build_deps()

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print("Loading FiQA-2018 (corpus, queries, qrels)...", file=sys.stderr)
    corpus, queries, qrels = load_fiqa(cache_dir)

    split_caps = {
        "train": args.max_train,
        "validation": args.max_val,
        "test": args.max_test,
    }

    # Collect the queries we will keep, and the corpus subset they reference plus
    # retrieved candidates. To keep retrieval cheap and the corpus committable in
    # spirit, we build retrievers over the full corpus but only emit sliced rows.
    selected = {}  # split -> [query_id, ...]
    for split, cap in split_caps.items():
        selected[split] = select_query_ids(qrels.get(split, {}), cap)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    print(
        f"Building retrievers over {len(corpus_ids)} passages "
        "(BM25 + all-MiniLM-L6-v2)...",
        file=sys.stderr,
    )
    bm25, encoder, doc_embeddings = build_retrievers(corpus_ids, corpus_texts)

    judge = (
        LLMJudge(args.judge_model, base_url=args.base_url)
        if args.label_mode == "llm"
        else None
    )

    rows = []
    for split, query_ids in selected.items():
        for query_id in query_ids:
            query_text = queries.get(query_id)
            if not query_text:
                continue
            positives = qrels[split].get(query_id, set())
            candidates = retrieve_candidates(
                query_text, corpus_ids, bm25, encoder, doc_embeddings, args.top_k
            )

            # Ensure qrel positives are in the candidate set so the demo can show
            # the supporting passage rising.
            present = {c["doc_id"] for c in candidates}
            for did in positives:
                if did not in present and did in corpus:
                    candidates.append(
                        {
                            "doc_id": did,
                            "dense_rank": "",
                            "dense_score": "",
                            "sparse_rank": "",
                            "sparse_score": "",
                        }
                    )

            for cand in candidates:
                did = cand["doc_id"]
                doc_text = corpus.get(did, "")[: args.max_doc_chars]
                if did in positives:
                    # Seed qrel positives as directly-supporting in both modes.
                    label = 3
                elif args.label_mode == "heuristic":
                    label = heuristic_label(cand["dense_rank"], args.hard_negative_rank)
                else:
                    label = judge.grade(query_text, doc_text)
                rows.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "doc_id": did,
                        "chunk_id": did,  # FiQA passages: chunk_id == doc_id
                        "doc_text": doc_text,
                        "dense_rank": cand["dense_rank"],
                        "dense_score": cand["dense_score"],
                        "sparse_rank": cand["sparse_rank"],
                        "sparse_score": cand["sparse_score"],
                        "label": label,
                        "split": split,
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(
        f"Wrote {len(rows)} rows across "
        f"{sum(len(v) for v in selected.values())} queries to {output_path}",
        file=sys.stderr,
    )

    # Dump query embeddings so mine_case_sets.py can reuse them without a
    # second encoding pass. Optional: mining works without it.
    dump_query_embeddings(cache_dir, selected, queries, encoder)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=150, help="Max train queries")
    parser.add_argument("--max-val", type=int, default=40, help="Max validation queries")
    parser.add_argument("--max-test", type=int, default=40, help="Max test queries")
    parser.add_argument(
        "--top-k", type=int, default=20, help="Candidate documents per query"
    )
    parser.add_argument(
        "--max-doc-chars", type=int, default=400, help="Truncate doc_text to this length"
    )
    parser.add_argument(
        "--cache-dir",
        default="examples/fiqa/.cache",
        help="Gitignored directory for downloaded corpus/intermediates",
    )
    parser.add_argument(
        "--output",
        default="examples/fiqa/query_doc_examples.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--label-mode",
        choices=["llm", "heuristic"],
        default="llm",
        help=(
            "How to label non-positive candidates. 'llm' grades each with an "
            "LLM (5 levels, needs an API key). 'heuristic' is zero-cost and "
            "deterministic: high dense-rank non-positives -> -1, else 0."
        ),
    )
    parser.add_argument(
        "--hard-negative-rank",
        type=int,
        default=5,
        help=(
            "Heuristic mode only: a non-positive with dense_rank <= this value "
            "is labeled a hard negative (-1)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-chat",
        help="LLM model name used for build-time label judging (llm mode)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help=(
            "OpenAI-compatible API base URL for the judge. Defaults to DeepSeek; "
            "pass an empty string '' or https://api.openai.com/v1 for OpenAI."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
