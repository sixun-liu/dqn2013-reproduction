#!/usr/bin/env python3
"""Collect greedy DQN trajectories under the frozen training reward semantics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if int(os.environ.get("MKL_NUM_THREADS", "0") or 0) <= 0:
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dqn2015_calibration import (
    bootstrap_interval,
    discounted_returns,
    paired_bootstrap_difference,
    update_reservoir,
)
from src.dqn2015_nature_breakout import (
    Args,
    QNetwork,
    StopController,
    make_atari_env,
    scalar,
    wrapper_type_names,
)
from src.dqn2015_offline_analysis import discover_evaluation_checkpoints, sha256_file


DEFAULT_CONFIG = PROJECT_ROOT / "configs/exp0006_dqn_calibration.json"
TRACE_FIELDS = (
    "checkpoint_decisions",
    "stage_step_index",
    "game_id",
    "life_id",
    "life_step_index",
    "q_noop",
    "q_fire",
    "q_right",
    "q_left",
    "max_q",
    "action_margin",
    "action",
    "clipped_reward",
    "discounted_return",
    "q_minus_return",
    "abs_q_minus_return",
    "terminated",
    "truncated",
    "life_loss",
    "real_game_end",
    "lives_before",
    "lives_after",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, action="append")
    parser.add_argument("--target-games", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_executor_args(source_run: Path) -> Args:
    values = json.loads((source_run / "resolved_config.json").read_text(encoding="utf-8"))
    return Args(**values)


def checkpoint_map(source_run: Path) -> dict[int, Path]:
    return {
        record.completed_agent_decisions: record.path
        for record in discover_evaluation_checkpoints(source_run / "checkpoints.jsonl")
    }


def complete_life(
    records: list[dict[str, Any]],
    *,
    gamma: float,
    writer: csv.DictWriter,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot complete an empty life trajectory")
    returns = discounted_returns(
        np.asarray([record["clipped_reward"] for record in records]), gamma
    )
    gaps = []
    for record, discounted in zip(records, returns, strict=True):
        gap = float(record["max_q"] - discounted)
        record["discounted_return"] = float(discounted)
        record["q_minus_return"] = gap
        record["abs_q_minus_return"] = abs(gap)
        writer.writerow({field: record[field] for field in TRACE_FIELDS})
        gaps.append(gap)
    margins = np.asarray([record["action_margin"] for record in records])
    return {
        "checkpoint_decisions": records[0]["checkpoint_decisions"],
        "game_id": records[0]["game_id"],
        "life_id": records[0]["life_id"],
        "agent_decisions": len(records),
        "clipped_reward_sum": float(sum(record["clipped_reward"] for record in records)),
        "start_discounted_return": float(returns[0]),
        "start_max_q": float(records[0]["max_q"]),
        "start_q_minus_return": float(gaps[0]),
        "median_q_minus_return": float(np.median(gaps)),
        "mean_q_minus_return": float(np.mean(gaps)),
        "rmse_q_minus_return": float(np.sqrt(np.mean(np.square(gaps)))),
        "overestimate_fraction": float(np.mean(np.asarray(gaps) > 0)),
        "mean_action_margin": float(margins.mean()),
        "p05_action_margin": float(np.quantile(margins, 0.05)),
        "real_game_end": bool(records[-1]["real_game_end"]),
    }


def summarize_game(
    checkpoint_decisions: int,
    game_id: int,
    life_rows: list[dict[str, Any]],
    step_records: list[dict[str, Any]],
    info: dict[str, Any],
) -> dict[str, Any]:
    gaps = np.asarray([record["q_minus_return"] for record in step_records])
    margins = np.asarray([record["action_margin"] for record in step_records])
    max_q = np.asarray([record["max_q"] for record in step_records])
    returns = np.asarray([record["discounted_return"] for record in step_records])
    return {
        "checkpoint_decisions": checkpoint_decisions,
        "game_id": game_id,
        "life_count": len(life_rows),
        "agent_decisions": len(step_records),
        "raw_game_return": scalar(info["episode"]["r"]),
        "raw_game_length": int(scalar(info["episode"]["l"])),
        "median_q_minus_return": float(np.median(gaps)),
        "mean_q_minus_return": float(gaps.mean()),
        "rmse_q_minus_return": float(np.sqrt(np.mean(np.square(gaps)))),
        "p95_abs_q_minus_return": float(np.quantile(np.abs(gaps), 0.95)),
        "overestimate_fraction": float(np.mean(gaps > 0)),
        "mean_action_margin": float(margins.mean()),
        "p05_action_margin": float(np.quantile(margins, 0.05)),
        "mean_max_q": float(max_q.mean()),
        "mean_discounted_return": float(returns.mean()),
    }


@torch.inference_mode()
def collect_stage(
    *,
    executor_args: Args,
    checkpoint_decisions: int,
    checkpoint_path: Path,
    output_dir: Path,
    target_games: int,
    max_decisions: int,
    gamma: float,
    rollout_seed: int,
    reservoir_size: int,
    device: torch.device,
    stop: StopController,
    expected_wrapper_prefix: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, list[dict[str, int]], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(payload["completed_agent_decisions"]) != checkpoint_decisions:
        raise ValueError(f"Checkpoint payload mismatch: {checkpoint_path}")
    if payload["args"] != asdict(executor_args):
        raise ValueError(f"Checkpoint executor args drifted: {checkpoint_path}")
    network = QNetwork(action_count=4).to(device)
    network.load_state_dict(payload["online_network"], strict=True)
    network.eval()

    env = make_atari_env(executor_args, evaluation=False)
    wrappers = wrapper_type_names(env)
    if wrappers[: len(expected_wrapper_prefix)] != expected_wrapper_prefix:
        raise ValueError(
            f"Training wrapper order drifted: {wrappers[:len(expected_wrapper_prefix)]}"
        )
    observation, _ = env.reset(seed=rollout_seed)
    reservoir = np.empty((reservoir_size, 4, 84, 84), dtype=np.uint8)
    reservoir_metadata: list[dict[str, int]] = []
    reservoir_rng = np.random.default_rng(rollout_seed + checkpoint_decisions)
    trace_path = output_dir / f"trace_{checkpoint_decisions:08d}.csv.gz"
    trace_temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
    life_rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    current_life: list[dict[str, Any]] = []
    current_game_steps: list[dict[str, Any]] = []
    current_game_lives: list[dict[str, Any]] = []
    game_id = 0
    life_id = 0
    stage_step = 0
    started = time.monotonic()

    try:
        with gzip.open(trace_temporary, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            while game_id < target_games and stage_step < max_decisions:
                if stop.requested:
                    raise InterruptedError(f"Signal {stop.signal_number} requested")
                state = np.asarray(observation)
                if state.shape != (4, 84, 84) or state.dtype != np.uint8:
                    raise ValueError(f"Observation semantics drifted: {state.shape}, {state.dtype}")
                update_reservoir(
                    reservoir,
                    reservoir_metadata,
                    state,
                    {
                        "checkpoint_decisions": checkpoint_decisions,
                        "stage_step_index": stage_step,
                        "game_id": game_id,
                        "life_id": life_id,
                        "life_step_index": len(current_life),
                    },
                    seen=stage_step,
                    rng=reservoir_rng,
                )
                tensor = torch.as_tensor(state[None], device=device)
                q_values = network(tensor)[0].detach().cpu().numpy()
                sorted_q = np.sort(q_values)
                action = int(np.argmax(q_values))
                lives_before = int(env.unwrapped.ale.lives())
                next_observation, reward, terminated, truncated, info = env.step(action)
                lives_after = int(env.unwrapped.ale.lives())
                done = bool(terminated or truncated)
                real_game_end = bool("episode" in info)
                life_loss = lives_after < lives_before
                if float(reward) not in (-1.0, 0.0, 1.0):
                    raise ValueError(f"Reward is not clipped: {reward}")
                if life_loss != done and not bool(truncated):
                    raise ValueError(
                        f"Life-loss terminal mismatch: loss={life_loss}, done={done}"
                    )
                if real_game_end and not done:
                    raise ValueError("Raw game ended without an outer terminal")
                record = {
                    "checkpoint_decisions": checkpoint_decisions,
                    "stage_step_index": stage_step,
                    "game_id": game_id,
                    "life_id": life_id,
                    "life_step_index": len(current_life),
                    "q_noop": float(q_values[0]),
                    "q_fire": float(q_values[1]),
                    "q_right": float(q_values[2]),
                    "q_left": float(q_values[3]),
                    "max_q": float(sorted_q[-1]),
                    "action_margin": float(sorted_q[-1] - sorted_q[-2]),
                    "action": action,
                    "clipped_reward": float(reward),
                    "discounted_return": None,
                    "q_minus_return": None,
                    "abs_q_minus_return": None,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "life_loss": life_loss,
                    "real_game_end": real_game_end,
                    "lives_before": lives_before,
                    "lives_after": lives_after,
                }
                current_life.append(record)
                stage_step += 1
                observation = next_observation
                if done:
                    life_summary = complete_life(current_life, gamma=gamma, writer=writer)
                    life_rows.append(life_summary)
                    current_game_lives.append(life_summary)
                    current_game_steps.extend(current_life)
                    current_life = []
                    if real_game_end:
                        game_rows.append(
                            summarize_game(
                                checkpoint_decisions,
                                game_id,
                                current_game_lives,
                                current_game_steps,
                                info,
                            )
                        )
                        game_id += 1
                        life_id = 0
                        current_game_steps = []
                        current_game_lives = []
                    else:
                        life_id += 1
                    observation, _ = env.reset()
        if game_id != target_games:
            raise RuntimeError(
                f"Stage {checkpoint_decisions} completed {game_id}/{target_games} games "
                f"within {stage_step}/{max_decisions} decisions"
            )
        if len(reservoir_metadata) != reservoir_size:
            raise RuntimeError(f"Reservoir incomplete: {len(reservoir_metadata)}/{reservoir_size}")
        os.replace(trace_temporary, trace_path)
    finally:
        env.close()

    stage_runtime = {
        "checkpoint_decisions": checkpoint_decisions,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "agent_decisions": stage_step,
        "completed_games": len(game_rows),
        "completed_lives": len(life_rows),
        "wall_seconds": time.monotonic() - started,
        "wrapper_types": wrappers,
    }
    return life_rows, game_rows, reservoir, reservoir_metadata, stage_runtime


def stage_summaries(
    games: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        ("mean_q_minus_return", np.mean),
        ("median_q_minus_return", np.median),
        ("rmse_q_minus_return", np.median),
        ("p95_abs_q_minus_return", np.median),
        ("overestimate_fraction", np.mean),
        ("mean_action_margin", np.mean),
        ("p05_action_margin", np.median),
        ("raw_game_return", np.mean),
    )
    rows = []
    for stage_index, (checkpoint, frame) in enumerate(games.groupby("checkpoint_decisions")):
        row: dict[str, Any] = {
            "checkpoint_decisions": int(checkpoint),
            "game_count": len(frame),
            "life_count": int(frame["life_count"].sum()),
            "agent_decisions": int(frame["agent_decisions"].sum()),
        }
        for metric_index, (name, estimator) in enumerate(metrics):
            estimate, low, high = bootstrap_interval(
                frame[name].to_numpy(),
                estimator=estimator,
                resamples=resamples,
                seed=seed + stage_index * 100 + metric_index,
            )
            row[name] = estimate
            row[f"{name}_ci95_low"] = low
            row[f"{name}_ci95_high"] = high
        rows.append(row)
    return rows


def stage_comparisons(
    games: pd.DataFrame,
    *,
    focal_stage: int,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    focal = games.loc[games["checkpoint_decisions"] == focal_stage].sort_values("game_id")
    if focal.empty:
        raise ValueError(f"Focal stage is absent: {focal_stage}")
    metrics = (
        ("mean_q_minus_return", np.mean),
        ("median_q_minus_return", np.median),
        ("mean_action_margin", np.mean),
        ("overestimate_fraction", np.mean),
        ("raw_game_return", np.mean),
    )
    rows = []
    comparison_stages = sorted(set(games["checkpoint_decisions"]) - {focal_stage})
    for stage_index, checkpoint in enumerate(comparison_stages):
        comparison = games.loc[
            games["checkpoint_decisions"] == checkpoint
        ].sort_values("game_id")
        if not np.array_equal(focal["game_id"].to_numpy(), comparison["game_id"].to_numpy()):
            raise ValueError(f"Game-ID pairing failed for {focal_stage} vs {checkpoint}")
        row: dict[str, Any] = {
            "focal_checkpoint_decisions": focal_stage,
            "comparison_checkpoint_decisions": int(checkpoint),
            "paired_game_count": len(focal),
        }
        for metric_index, (name, estimator) in enumerate(metrics):
            estimate, low, high = paired_bootstrap_difference(
                focal[name].to_numpy(),
                comparison[name].to_numpy(),
                estimator=estimator,
                resamples=resamples,
                seed=seed + stage_index * 100 + metric_index,
            )
            row[f"{name}_focal_minus_comparison"] = estimate
            row[f"{name}_difference_ci95_low"] = low
            row[f"{name}_difference_ci95_high"] = high
        rows.append(row)
    return rows


def main() -> int:
    cli = parse_args()
    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["schema_version"]) != 1:
        raise ValueError("Unsupported calibration config schema")
    source_run = Path(config["source_run"]).resolve()
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_record = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
        "argv": sys.argv,
    }
    atomic_json(output_dir / ".started", started_record)

    stop = StopController()
    signal.signal(signal.SIGTERM, stop.request)
    signal.signal(signal.SIGINT, stop.request)
    try:
        stages = [int(value) for value in (cli.stage or config["checkpoint_decisions"])]
        if len(stages) != len(set(stages)):
            raise ValueError("Checkpoint stages must be unique")
        target_games = int(cli.target_games or config["target_complete_games_per_stage"])
        max_decisions = int(cli.max_decisions or config["max_agent_decisions_per_stage"])
        if target_games <= 1 or max_decisions <= 0:
            raise ValueError("Calibration requires at least two games and a positive decision cap")
        if cli.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cli.device)
        random.seed(int(config["rollout_seed"]))
        np.random.seed(int(config["rollout_seed"]))
        torch.manual_seed(int(config["rollout_seed"]))
        torch.backends.cudnn.deterministic = True

        executor_args = load_executor_args(source_run)
        if float(config["discount"]) != executor_args.discount:
            raise ValueError("Calibration discount does not match the frozen training config")
        checkpoints = checkpoint_map(source_run)
        unknown = [stage for stage in stages if stage not in checkpoints]
        if unknown:
            raise ValueError(f"Missing checkpoint stages: {unknown}")

        all_lives: list[dict[str, Any]] = []
        all_games: list[dict[str, Any]] = []
        all_reservoirs = []
        all_reservoir_metadata: list[dict[str, int]] = []
        runtime_rows = []
        for stage in stages:
            lives, games, reservoir, reservoir_metadata, runtime = collect_stage(
                executor_args=executor_args,
                checkpoint_decisions=stage,
                checkpoint_path=checkpoints[stage],
                output_dir=output_dir,
                target_games=target_games,
                max_decisions=max_decisions,
                gamma=float(config["discount"]),
                rollout_seed=int(config["rollout_seed"]),
                reservoir_size=int(config["reservoir_states_per_stage"]),
                device=device,
                stop=stop,
                expected_wrapper_prefix=list(config["expected_wrapper_prefix"]),
            )
            all_lives.extend(lives)
            all_games.extend(games)
            all_reservoirs.append(reservoir)
            all_reservoir_metadata.extend(reservoir_metadata)
            runtime_rows.append(runtime)
            print(json.dumps({"event": "stage_completed", **runtime}), flush=True)

        games_frame = pd.DataFrame(all_games)
        summaries = stage_summaries(
            games_frame,
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]),
        )
        comparisons = (
            stage_comparisons(
                games_frame,
                focal_stage=9_250_000,
                resamples=int(config["bootstrap_resamples"]),
                seed=int(config["bootstrap_seed"]) + 10_000,
            )
            if 9_250_000 in stages and len(stages) > 1
            else []
        )
        write_csv(output_dir / "lives.csv", all_lives)
        write_csv(output_dir / "games.csv", all_games)
        write_csv(output_dir / "stage_summary.csv", summaries)
        if comparisons:
            write_csv(output_dir / "stage_comparisons.csv", comparisons)
        write_csv(output_dir / "reservoir_metadata.csv", all_reservoir_metadata)
        reservoir_array = np.stack(all_reservoirs)
        np.save(output_dir / "reservoir_states.npy", reservoir_array, allow_pickle=False)

        script_path = Path(__file__).resolve()
        inputs = {
            "calibration_config": config_path,
            "training_config": source_run / "resolved_config.json",
            "checkpoint_index": source_run / "checkpoints.jsonl",
        }
        manifest = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "analysis_commit": subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "analysis_script": str(script_path),
            "analysis_script_sha256": sha256_file(script_path),
            "calibration_module_sha256": sha256_file(PROJECT_ROOT / "src/dqn2015_calibration.py"),
            "device": str(device),
            "stages": stages,
            "target_complete_games_per_stage": target_games,
            "max_agent_decisions_per_stage": max_decisions,
            "rollout_seed": int(config["rollout_seed"]),
            "discount": float(config["discount"]),
            "executor_args": asdict(executor_args),
            "inputs": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for name, path in inputs.items()
            },
            "stage_runtime": runtime_rows,
            "outputs": {
                path.name: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (
                    output_dir / "lives.csv",
                    output_dir / "games.csv",
                    output_dir / "stage_summary.csv",
                    output_dir / "reservoir_metadata.csv",
                    output_dir / "reservoir_states.npy",
                    *(output_dir / f"trace_{stage:08d}.csv.gz" for stage in stages),
                    *((output_dir / "stage_comparisons.csv",) if comparisons else ()),
                )
            },
        }
        atomic_json(output_dir / "source_manifest.json", manifest)
        completed = {
            "experiment_id": config["experiment_id"],
            "completed": True,
            "checkpoint_stages": stages,
            "game_count": len(all_games),
            "life_count": len(all_lives),
            "reservoir_shape": list(reservoir_array.shape),
            "all_finite": bool(
                np.isfinite(games_frame.select_dtypes(include=[np.number]).to_numpy()).all()
                and np.isfinite(reservoir_array).all()
            ),
        }
        atomic_json(output_dir / ".completed", completed)
        print(json.dumps({"event": "completed", **completed}), flush=True)
        return 0
    except InterruptedError as error:
        atomic_json(output_dir / ".stopped", {"error": str(error), "stopped_at": time.time()})
        raise
    except BaseException as error:
        atomic_json(
            output_dir / ".failed",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
