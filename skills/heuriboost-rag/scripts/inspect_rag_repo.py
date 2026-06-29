#!/usr/bin/env python3
"""Read-only scan for likely RAG retrieval and evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path


KEYWORDS = {
    "retrieval": ("retriever", "retrieve", "vector", "bm25", "embedding", "rerank"),
    "evaluation": ("eval", "metric", "ndcg", "mrr", "recall"),
    "labels": ("label", "golden", "judgment", "annotation"),
    "logs": ("log", "trace", "feedback", "citation"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Repository root to inspect")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    matches = {key: [] for key in KEYWORDS}
    for path in root.rglob("*"):
        if any(part in {".git", "node_modules", ".venv", "__pycache__"} for part in path.parts):
            continue
        if not path.is_file():
            continue
        lowered = str(path.relative_to(root)).lower()
        for category, keywords in KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                matches[category].append(path.relative_to(root))

    print("HeuriBoost RAG audit")
    for category, paths in matches.items():
        print(f"\n{category}:")
        if not paths:
            print("  none found")
            continue
        for path in paths[:20]:
            print(f"  {path}")
        if len(paths) > 20:
            print(f"  ... {len(paths) - 20} more")

    print("\nNext step:")
    print("  Export a CSV with query_id, query_text, doc_id, doc_text, label, split.")


if __name__ == "__main__":
    main()
