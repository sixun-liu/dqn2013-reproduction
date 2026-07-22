#!/usr/bin/env python3
"""Verify structural completion signals for a DQN run directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {label}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{label}[{index}]")


def verify(run_dir: Path, mode: str) -> dict[str, Any]:
    required = [".started", ".completed", "resolved_config.json", "runtime.json", "metrics.jsonl"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing run files: {', '.join(missing)}")
    if (run_dir / ".failed").exists():
        raise RuntimeError(f"run contains failure marker: {run_dir / '.failed'}")

    config = read_json(run_dir / "resolved_config.json")
    runtime = read_json(run_dir / "runtime.json")
    completed = read_json(run_dir / ".completed")
    records = read_jsonl(run_dir / "metrics.jsonl")
    assert_finite(records, "metrics")

    expected_decisions = int(config["total_agent_decisions"])
    if int(completed["completed_agent_decisions"]) != expected_decisions:
        raise ValueError("completion decision count does not match resolved config")
    if runtime.get("network_parameter_count") != 1_686_180:
        raise ValueError("unexpected network parameter count")
    if runtime.get("action_count") != 4 or runtime.get("observation_shape") != [4, 84, 84]:
        raise ValueError("unexpected Breakout action or observation shape")
    wrapper_types = set(runtime.get("wrapper_types", []))
    required_wrappers = {"NoopResetEnv", "MaxAndSkipEnv", "EpisodicLifeEnv", "ClipRewardEnv", "FrameStack"}
    if not required_wrappers.issubset(wrapper_types):
        raise ValueError(f"wrapper stack is incomplete: {sorted(wrapper_types)}")

    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(str(record.get("type")), []).append(record)
    for record_type in ("start", "progress", "heldout_capture", "heldout_q", "completed"):
        if not by_type.get(record_type):
            raise ValueError(f"missing metrics record type: {record_type}")
    last_progress = by_type["progress"][-1]
    if int(last_progress.get("optimizer_updates", 0)) <= 0:
        raise ValueError("optimizer never updated")
    if int(last_progress.get("target_syncs", 0)) <= 0:
        raise ValueError("target network never synchronized")
    if mode == "full":
        expected_evaluations = expected_decisions // int(config["eval_every_agent_decisions"])
        if len(by_type.get("evaluation", [])) != expected_evaluations:
            raise ValueError("formal run does not contain the expected evaluation count")

    return {
        "status": "ok",
        "mode": mode,
        "run_dir": str(run_dir),
        "completed_agent_decisions": expected_decisions,
        "optimizer_updates": int(last_progress["optimizer_updates"]),
        "target_syncs": int(last_progress["target_syncs"]),
        "evaluation_count": len(by_type.get("evaluation", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve(), args.mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

