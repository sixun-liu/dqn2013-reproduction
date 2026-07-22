#!/usr/bin/env python3
"""Evaluate a Nature 2015 checkpoint with the frozen full-game protocol."""

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

from src.dqn2015_nature_breakout import Args, QNetwork, StopController, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-agent-decisions", type=int, default=135_000)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-epsilon", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
    device = torch.device(cli.device)
    payload = torch.load(cli.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    checkpoint_args = payload.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("checkpoint does not contain expanded args")
    args = Args(**checkpoint_args)
    args = dataclasses.replace(
        args,
        cuda=cli.device == "cuda",
        eval_agent_decisions=cli.eval_agent_decisions,
        eval_seed=cli.eval_seed,
        eval_epsilon=cli.eval_epsilon,
    )
    network = QNetwork(action_count=4).to(device)
    network.load_state_dict(payload["online_network"])
    network.eval()
    result = evaluate(
        args,
        network,
        device,
        int(payload["completed_agent_decisions"]),
        StopController(),
    )
    result["type"] = "independent_checkpoint_evaluation"
    result["checkpoint"] = str(cli.checkpoint.resolve())
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

