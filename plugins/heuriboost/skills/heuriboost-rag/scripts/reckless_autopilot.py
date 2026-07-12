#!/usr/bin/env python3
"""Run, report, resume, and promote immutable Reckless repairs."""

from __future__ import annotations

from heuriboost_rag.adapters.cli import autopilot_main, build_autopilot_parser as build_parser


def main() -> None:
    raise SystemExit(autopilot_main())


if __name__ == "__main__":
    main()
