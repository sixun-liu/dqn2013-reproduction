#!/usr/bin/env python3
"""Extract the frozen 40-checkpoint DQN value and representation panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if int(os.environ.get("MKL_NUM_THREADS", "0") or 0) <= 0:
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dqn2015_nature_breakout import QNetwork
from src.dqn2015_offline_analysis import (
    behavior_records,
    centered_linear_cka,
    discover_evaluation_checkpoints,
    extract_q_and_features,
    read_jsonl,
    representation_spectrum,
    sha256_file,
    summarize_q_values,
)


DEFAULT_RUN = Path(
    "/root/autodl-tmp/runs/EXP-0004__breakout__s000__10m-dec__20260720T174600Z"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-states", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def heldout_metric_map(metrics_path: Path) -> dict[int, float]:
    return {
        int(record["completed_agent_decisions"]): float(record["mean_max_q"])
        for record in read_jsonl(metrics_path)
        if record.get("type") == "heldout_q"
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (output_dir / ".completed").exists():
        raise SystemExit(f"Refusing to overwrite completed output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_states <= 0:
        raise ValueError("batch-states must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    states_path = run_dir / "heldout_states.npy"
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "resolved_config.json"
    runtime_path = run_dir / "runtime.json"
    checkpoint_index_path = run_dir / "checkpoints.jsonl"
    for path in (states_path, metrics_path, config_path, runtime_path, checkpoint_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    states = np.load(states_path, allow_pickle=False)
    if states.shape != (500, 4, 84, 84) or states.dtype != np.uint8:
        raise ValueError(f"Unexpected heldout states: {states.shape}, {states.dtype}")
    checkpoints = discover_evaluation_checkpoints(checkpoint_index_path)
    heldout_metrics = heldout_metric_map(metrics_path)
    evaluations, episodes = behavior_records(metrics_path)

    state_manifest = [
        {
            "state_id": state_id,
            "observation_sha256": __import__("hashlib").sha256(state.tobytes()).hexdigest(),
        }
        for state_id, state in enumerate(states)
    ]
    checkpoint_rows = []
    stage_rows = []
    q_panel = []
    feature_panel = []
    maximum_forward_error = 0.0
    maximum_heldout_parity_error = 0.0

    for index, checkpoint in enumerate(checkpoints):
        print(
            json.dumps(
                {
                    "event": "checkpoint",
                    "index": index + 1,
                    "count": len(checkpoints),
                    "completed_agent_decisions": checkpoint.completed_agent_decisions,
                }
            ),
            flush=True,
        )
        payload = torch.load(checkpoint.path, map_location=device, weights_only=False)
        if int(payload["completed_agent_decisions"]) != checkpoint.completed_agent_decisions:
            raise ValueError(f"Checkpoint payload step mismatch: {checkpoint.path}")
        network = QNetwork(action_count=4).to(device)
        network.load_state_dict(payload["online_network"], strict=True)
        q_values, features, forward_error = extract_q_and_features(
            network, states, device, args.batch_states
        )
        if q_values.shape != (500, 4) or features.shape != (500, 512):
            raise ValueError(f"Unexpected output shapes: {q_values.shape}, {features.shape}")
        if not np.isfinite(q_values).all() or not np.isfinite(features).all():
            raise ValueError(f"Non-finite outputs at {checkpoint.completed_agent_decisions}")
        maximum_forward_error = max(maximum_forward_error, forward_error)

        q_summary = summarize_q_values(q_values)
        original_mean = heldout_metrics.get(checkpoint.completed_agent_decisions)
        if original_mean is None:
            raise ValueError(f"Missing original heldout metric at {checkpoint.completed_agent_decisions}")
        parity_error = abs(q_summary["max_q_mean"] - original_mean)
        maximum_heldout_parity_error = max(maximum_heldout_parity_error, parity_error)
        stage_rows.append(
            {
                "stage_index": index,
                "completed_agent_decisions": checkpoint.completed_agent_decisions,
                **q_summary,
                "original_heldout_mean_max_q": original_mean,
                "heldout_mean_max_q_abs_error": parity_error,
                "feature_l2_mean": float(np.linalg.norm(features, axis=1).mean()),
                "feature_zero_fraction": float((features == 0).mean()),
                **representation_spectrum(features),
            }
        )
        checkpoint_rows.append(
            {
                "stage_index": index,
                "completed_agent_decisions": checkpoint.completed_agent_decisions,
                "optimizer_updates": checkpoint.optimizer_updates,
                "reason": checkpoint.reason,
                "path": str(checkpoint.path),
                "bytes": checkpoint.path.stat().st_size,
                "sha256": sha256_file(checkpoint.path),
                "forward_parity_max_abs_error": forward_error,
            }
        )
        q_panel.append(q_values)
        feature_panel.append(features)
        del network, payload

    q_array = np.stack(q_panel).astype(np.float32, copy=False)
    feature_array = np.stack(feature_panel).astype(np.float32, copy=False)
    final_features = feature_array[-1]
    for index, row in enumerate(stage_rows):
        row["linear_cka_to_final"] = centered_linear_cka(feature_array[index], final_features)
        evaluation = evaluations[index]
        if evaluation["completed_agent_decisions"] != row["completed_agent_decisions"]:
            raise ValueError("Behavior stage alignment failed")
        row["behavior_mean_episode_return"] = evaluation["mean_episode_return"]
        row["behavior_median_episode_return"] = evaluation["median_episode_return"]
        row["behavior_completed_games"] = evaluation["completed_games"]

    if maximum_forward_error > 1e-6:
        raise ValueError(f"Forward parity failed: {maximum_forward_error}")
    if maximum_heldout_parity_error > 1e-5:
        raise ValueError(f"Heldout parity failed: {maximum_heldout_parity_error}")

    np.save(output_dir / "q_values.npy", q_array, allow_pickle=False)
    np.save(output_dir / "features.npy", feature_array, allow_pickle=False)
    write_csv(output_dir / "states.csv", state_manifest)
    write_csv(output_dir / "checkpoints.csv", checkpoint_rows)
    write_csv(output_dir / "stages.csv", stage_rows)
    write_csv(output_dir / "behavior_evaluations.csv", evaluations)
    write_csv(output_dir / "behavior_episodes.csv", episodes)

    source_manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0005",
        "analysis_commit": git_head(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_module_sha256": sha256_file(
            PROJECT_ROOT / "src/dqn2015_offline_analysis.py"
        ),
        "device": str(device),
        "run_dir": str(run_dir),
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in {
                "heldout_states": states_path,
                "metrics": metrics_path,
                "resolved_config": config_path,
                "runtime": runtime_path,
                "checkpoint_index": checkpoint_index_path,
            }.items()
        },
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_manifest": checkpoint_rows,
    }
    atomic_json(output_dir / "source_manifest.json", source_manifest)
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0005",
        "checkpoint_count": len(checkpoints),
        "state_count": len(states),
        "q_panel_shape": list(q_array.shape),
        "feature_panel_shape": list(feature_array.shape),
        "behavior_evaluation_count": len(evaluations),
        "behavior_episode_count": len(episodes),
        "maximum_forward_parity_abs_error": maximum_forward_error,
        "maximum_heldout_mean_max_q_abs_error": maximum_heldout_parity_error,
        "all_finite": bool(np.isfinite(q_array).all() and np.isfinite(feature_array).all()),
        "known_answer_gate_passed": True,
    }
    atomic_json(output_dir / "summary.json", summary)
    completed = {
        "experiment_id": "EXP-0005",
        "completed": True,
        "analysis_commit": git_head(),
        "known_answer_gate_passed": True,
    }
    atomic_json(output_dir / ".completed", completed)
    print(json.dumps({"event": "completed", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
