"""Command-line entry point for the local HeuriBoost Web Console."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import uvicorn

from .app import create_app
from .backup import create_backup
from .config import WebConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the HeuriBoost Web Console.")
    subparsers = parser.add_subparsers(dest="command")
    backup = subparsers.add_parser("backup", help="Create a consistent Web Console backup.")
    backup.add_argument("--data-dir", type=Path, required=True, help="Console data directory")
    backup.add_argument("--output", type=Path, required=True, help="Backup .tar.gz path")
    parser.add_argument("--config", type=Path, help="Workspace configuration YAML")
    parser.add_argument("--data-dir", type=Path, help="Console data directory")
    parser.add_argument("--host", help="Bind host")
    parser.add_argument("--port", type=int, help="Bind port")
    return parser


def _config_from_args(args: argparse.Namespace) -> WebConfig:
    if args.config:
        return WebConfig.from_file(
            args.config,
            data_dir=args.data_dir,
            host=args.host,
            port=args.port,
        )
    return WebConfig(
        data_dir=args.data_dir or Path("~/.heuriboost"),
        host=args.host or "127.0.0.1",
        port=args.port or 8787,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "backup":
        archive = create_backup(args.data_dir, args.output)
        print(str(archive), flush=True)
        return 0
    config = _config_from_args(args)
    print(f"HeuriBoost Web Console: {config.launch_url}", flush=True)
    uvicorn.run(create_app(config), host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
