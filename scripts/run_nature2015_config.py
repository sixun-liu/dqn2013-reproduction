#!/usr/bin/env python3
"""Run the Nature 2015 executor from a complete JSON configuration."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.dqn2015_nature_breakout import Args, run


def load_config(path: Path) -> Args:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a JSON object")
    fields = {field.name for field in dataclasses.fields(Args)}
    missing = sorted(fields - values.keys())
    unknown = sorted(values.keys() - fields)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError("; ".join(parts))
    return Args(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("config", "cpu", "cuda"), default="config")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = load_config(cli.config.resolve())
    if cli.output_dir is not None:
        args = dataclasses.replace(args, output_dir=str(cli.output_dir.resolve()))
    if cli.device == "cpu":
        args = dataclasses.replace(args, cuda=False)
    elif cli.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        args = dataclasses.replace(args, cuda=True)
    run(args)


if __name__ == "__main__":
    main()

