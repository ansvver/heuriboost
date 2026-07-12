#!/usr/bin/env python3
"""Promote an eligible production-case repair run."""

from __future__ import annotations

from heuriboost_rag.adapters.cli import (
    build_legacy_promote_parser as build_parser,
    legacy_promote_main,
)


def main() -> None:
    raise SystemExit(legacy_promote_main())


if __name__ == "__main__":
    main()
