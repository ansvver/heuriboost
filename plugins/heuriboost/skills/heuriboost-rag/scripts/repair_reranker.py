#!/usr/bin/env python3
"""Run strict production-case repair for a HeuriBoost reranker."""

from __future__ import annotations

from heuriboost_rag.adapters.cli import (
    build_legacy_repair_parser as build_parser,
    legacy_repair_main,
)


def main() -> None:
    raise SystemExit(legacy_repair_main())


if __name__ == "__main__":
    main()
