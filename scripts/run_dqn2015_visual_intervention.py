#!/usr/bin/env python3
"""Run frozen-state frame ablation and Greydanus blur perturbations for DQN."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if int(os.environ.get("MKL_NUM_THREADS", "0") or 0) <= 0:
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dqn2015_calibration import bootstrap_interval, paired_bootstrap_difference
from src.dqn2015_nature_breakout import QNetwork
from src.dqn2015_offline_analysis import discover_evaluation_checkpoints, sha256_file
from src.dqn2015_visual_intervention import (
    adjacent_frame_replacements,
    author_q_score,
    blur_state,
    map_statistics,
    occlude_with_blur,
    saliency_grid,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/exp0008_dqn_visual_intervention.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, action="append")
    parser.add_argument("--state-count", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


@torch.inference_mode()
def infer_q(
    network: QNetwork,
    states: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    batches = []
    for offset in range(0, len(states), batch_size):
        tensor = torch.as_tensor(states[offset : offset + batch_size], device=device)
        batches.append(network(tensor).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches)


def checkpoint_map(source_run: Path) -> dict[int, Path]:
    return {
        record.completed_agent_decisions: record.path
        for record in discover_evaluation_checkpoints(source_run / "checkpoints.jsonl")
    }


def summarize_stages(
    state_rows: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        ("saliency_mean", np.mean),
        ("saliency_normalized_mean", np.mean),
        ("saliency_p95", np.mean),
        ("saliency_entropy_normalized", np.mean),
        ("top_decile_concentration", np.mean),
        ("top_to_random_mean_ratio", np.median),
        ("spatial_action_switch_fraction", np.mean),
        ("spatial_original_action_delta_mean", np.mean),
        ("global_blur_q_score", np.mean),
        ("global_blur_q_score_normalized", np.mean),
        ("global_blur_action_switch", np.mean),
        ("frame_ablation_q_score_mean", np.mean),
        ("frame_ablation_q_score_normalized_mean", np.mean),
        ("frame_ablation_action_switch_fraction", np.mean),
    )
    rows = []
    for stage_index, (checkpoint, frame) in enumerate(
        state_rows.groupby("checkpoint_decisions", sort=True)
    ):
        row: dict[str, Any] = {
            "checkpoint_decisions": int(checkpoint),
            "state_count": len(frame),
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


def compare_focal_stage(
    state_rows: pd.DataFrame,
    *,
    focal_stage: int,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    focal = state_rows.loc[state_rows["checkpoint_decisions"] == focal_stage].sort_values(
        "state_id"
    )
    if focal.empty:
        raise ValueError(f"Focal stage missing: {focal_stage}")
    metrics = (
        ("saliency_mean", np.mean),
        ("saliency_normalized_mean", np.mean),
        ("saliency_entropy_normalized", np.mean),
        ("top_decile_concentration", np.mean),
        ("spatial_action_switch_fraction", np.mean),
        ("global_blur_q_score", np.mean),
        ("global_blur_q_score_normalized", np.mean),
        ("frame_ablation_q_score_mean", np.mean),
        ("frame_ablation_q_score_normalized_mean", np.mean),
    )
    rows = []
    stages = sorted(set(state_rows["checkpoint_decisions"]) - {focal_stage})
    for stage_index, checkpoint in enumerate(stages):
        comparison = state_rows.loc[
            state_rows["checkpoint_decisions"] == checkpoint
        ].sort_values("state_id")
        if not np.array_equal(focal["state_id"].to_numpy(), comparison["state_id"].to_numpy()):
            raise ValueError(f"State pairing failed for {focal_stage} vs {checkpoint}")
        row: dict[str, Any] = {
            "focal_checkpoint_decisions": focal_stage,
            "comparison_checkpoint_decisions": int(checkpoint),
            "paired_state_count": len(focal),
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
        raise ValueError("Unsupported visual intervention config schema")
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_dir / ".started",
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.time(),
            "argv": sys.argv,
        },
    )
    try:
        source_run = Path(config["source_run"]).resolve()
        panel_dir = Path(config["offline_panel"]).resolve()
        if not (panel_dir / ".completed").is_file():
            raise ValueError("EXP-0005 offline panel is incomplete")
        stages = [int(value) for value in (cli.stage or config["checkpoint_decisions"])]
        if len(stages) != len(set(stages)):
            raise ValueError("Checkpoint stages must be unique")
        formal_count = int(config["state_count"])
        state_count = int(cli.state_count or formal_count)
        if state_count <= 1 or state_count > formal_count:
            raise ValueError("state-count must be between 2 and the frozen formal count")
        if cli.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cli.device)

        heldout_path = source_run / "heldout_states.npy"
        all_states = np.load(heldout_path, allow_pickle=False)
        if all_states.shape != (500, 4, 84, 84) or all_states.dtype != np.uint8:
            raise ValueError(f"Heldout state drift: {all_states.shape}, {all_states.dtype}")
        state_rng = np.random.default_rng(int(config["state_selection_seed"]))
        formal_ids = np.sort(state_rng.choice(len(all_states), size=formal_count, replace=False))
        state_ids = formal_ids[:state_count]
        states = all_states[state_ids]

        centers, masks = saliency_grid(
            (84, 84), int(config["grid_stride"]), float(config["mask_radius"])
        )
        grid_count = len(centers)
        random_count = max(int(np.ceil(float(config["random_cell_fraction"]) * grid_count)), 1)
        random_rng = np.random.default_rng(int(config["random_cell_seed"]))
        random_indices = np.stack(
            [random_rng.choice(grid_count, size=random_count, replace=False) for _ in state_ids]
        )

        checkpoints = checkpoint_map(source_run)
        missing = [stage for stage in stages if stage not in checkpoints]
        if missing:
            raise ValueError(f"Missing checkpoint stages: {missing}")
        panel_stages = pd.read_csv(panel_dir / "stages.csv")
        panel_q = np.load(panel_dir / "q_values.npy", mmap_mode="r", allow_pickle=False)
        panel_index = {
            int(step): index for index, step in enumerate(panel_stages["completed_agent_decisions"])
        }

        shape = (len(stages), state_count, grid_count)
        saliency_scores = np.empty(shape, dtype=np.float32)
        saliency_action_delta = np.empty(shape, dtype=np.float32)
        saliency_switch = np.empty(shape, dtype=np.bool_)
        frame_q_score = np.empty((len(stages), state_count, 4), dtype=np.float32)
        frame_action_delta = np.empty((len(stages), state_count, 4), dtype=np.float32)
        frame_switch = np.empty((len(stages), state_count, 4), dtype=np.bool_)
        global_q_score = np.empty((len(stages), state_count), dtype=np.float32)
        global_action_delta = np.empty((len(stages), state_count), dtype=np.float32)
        global_switch = np.empty((len(stages), state_count), dtype=np.bool_)
        state_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        checkpoint_rows = []
        max_panel_parity_error = 0.0
        batch_size = int(config["perturbation_batch_size"])

        for stage_index, stage in enumerate(stages):
            payload = torch.load(checkpoints[stage], map_location=device, weights_only=False)
            if int(payload["completed_agent_decisions"]) != stage:
                raise ValueError(f"Checkpoint payload drift: {checkpoints[stage]}")
            network = QNetwork(action_count=4).to(device)
            network.load_state_dict(payload["online_network"], strict=True)
            network.eval()
            baseline_q = infer_q(network, states, device, batch_size)
            parity_error = float(np.max(np.abs(baseline_q - panel_q[panel_index[stage], state_ids])))
            max_panel_parity_error = max(max_panel_parity_error, parity_error)
            if parity_error > 1e-6:
                raise ValueError(f"EXP-0005 baseline Q parity failed at {stage}: {parity_error}")

            for local_index, (state_id, state, base_q) in enumerate(
                zip(state_ids, states, baseline_q, strict=True)
            ):
                original_action = int(np.argmax(base_q))
                sorted_base_q = np.sort(base_q)
                baseline_q_energy = float(0.5 * np.square(base_q).sum())
                blurred = blur_state(state, float(config["gaussian_blur_sigma"]))
                perturbed_q_batches = []
                for offset in range(0, grid_count, batch_size):
                    perturbed_states = occlude_with_blur(
                        state, blurred, masks[offset : offset + batch_size]
                    )
                    perturbed_q_batches.append(
                        infer_q(network, perturbed_states, device, batch_size)
                    )
                perturbed_q = np.concatenate(perturbed_q_batches)
                scores = author_q_score(base_q, perturbed_q)
                normalized_scores = scores / (baseline_q_energy + 1e-12)
                action_delta = base_q[original_action] - perturbed_q[:, original_action]
                switches = perturbed_q.argmax(axis=1) != original_action
                saliency_scores[stage_index, local_index] = scores
                saliency_action_delta[stage_index, local_index] = action_delta
                saliency_switch[stage_index, local_index] = switches

                global_q = infer_q(network, blurred[None], device, batch_size)[0]
                global_q_score[stage_index, local_index] = author_q_score(
                    base_q, global_q[None]
                )[0]
                global_action_delta[stage_index, local_index] = (
                    base_q[original_action] - global_q[original_action]
                )
                global_switch[stage_index, local_index] = (
                    int(np.argmax(global_q)) != original_action
                )

                replacements = adjacent_frame_replacements(state)
                replacement_q = infer_q(network, replacements, device, batch_size)
                frame_q_score[stage_index, local_index] = author_q_score(base_q, replacement_q)
                frame_action_delta[stage_index, local_index] = (
                    base_q[original_action] - replacement_q[:, original_action]
                )
                frame_switch[stage_index, local_index] = (
                    replacement_q.argmax(axis=1) != original_action
                )

                statistics = map_statistics(scores, random_indices[local_index])
                state_rows.append(
                    {
                        "checkpoint_decisions": stage,
                        "state_id": int(state_id),
                        "original_action": original_action,
                        "baseline_max_q": float(base_q.max()),
                        "baseline_action_margin": float(sorted_base_q[-1] - sorted_base_q[-2]),
                        "baseline_q_energy": baseline_q_energy,
                        **statistics,
                        "saliency_normalized_mean": float(normalized_scores.mean()),
                        "saliency_normalized_p95": float(np.quantile(normalized_scores, 0.95)),
                        "spatial_action_switch_fraction": float(switches.mean()),
                        "spatial_original_action_delta_mean": float(action_delta.mean()),
                        "spatial_original_action_delta_p95": float(
                            np.quantile(action_delta, 0.95)
                        ),
                        "global_blur_q_score": float(global_q_score[stage_index, local_index]),
                        "global_blur_q_score_normalized": float(
                            global_q_score[stage_index, local_index] / (baseline_q_energy + 1e-12)
                        ),
                        "global_blur_original_action_delta": float(
                            global_action_delta[stage_index, local_index]
                        ),
                        "global_blur_action_switch": bool(global_switch[stage_index, local_index]),
                        "frame_ablation_q_score_mean": float(
                            frame_q_score[stage_index, local_index].mean()
                        ),
                        "frame_ablation_q_score_normalized_mean": float(
                            frame_q_score[stage_index, local_index].mean()
                            / (baseline_q_energy + 1e-12)
                        ),
                        "frame_ablation_action_switch_fraction": float(
                            frame_switch[stage_index, local_index].mean()
                        ),
                    }
                )
                for channel in range(4):
                    frame_rows.append(
                        {
                            "checkpoint_decisions": stage,
                            "state_id": int(state_id),
                            "frame_channel": channel,
                            "replacement_channel": channel + 1 if channel < 3 else channel - 1,
                            "q_score": float(frame_q_score[stage_index, local_index, channel]),
                            "q_score_normalized": float(
                                frame_q_score[stage_index, local_index, channel]
                                / (baseline_q_energy + 1e-12)
                            ),
                            "original_action_delta": float(
                                frame_action_delta[stage_index, local_index, channel]
                            ),
                            "action_switch": bool(
                                frame_switch[stage_index, local_index, channel]
                            ),
                        }
                    )
            checkpoint_rows.append(
                {
                    "checkpoint_decisions": stage,
                    "path": str(checkpoints[stage]),
                    "bytes": checkpoints[stage].stat().st_size,
                    "sha256": sha256_file(checkpoints[stage]),
                    "baseline_panel_q_max_abs_error": parity_error,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "stage_completed",
                        "checkpoint_decisions": stage,
                        "state_count": state_count,
                        "grid_count": grid_count,
                    }
                ),
                flush=True,
            )

        for array in (
            saliency_scores,
            saliency_action_delta,
            frame_q_score,
            frame_action_delta,
            global_q_score,
            global_action_delta,
        ):
            if not np.isfinite(array).all():
                raise ValueError("Visual intervention output contains non-finite values")

        state_frame = pd.DataFrame(state_rows)
        stage_rows = summarize_stages(
            state_frame,
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]),
        )
        comparison_rows = compare_focal_stage(
            state_frame,
            focal_stage=9_250_000,
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]) + 10_000,
        ) if 9_250_000 in stages and len(stages) > 1 else []

        arrays = {
            "selected_state_ids.npy": state_ids,
            "saliency_scores.npy": saliency_scores,
            "saliency_action_delta.npy": saliency_action_delta,
            "saliency_switch.npy": saliency_switch,
            "frame_ablation_q_score.npy": frame_q_score,
            "frame_ablation_action_delta.npy": frame_action_delta,
            "frame_ablation_switch.npy": frame_switch,
            "global_blur_q_score.npy": global_q_score,
            "global_blur_action_delta.npy": global_action_delta,
            "global_blur_switch.npy": global_switch,
            "grid_centers.npy": centers,
            "random_cell_indices.npy": random_indices,
        }
        for name, array in arrays.items():
            np.save(output_dir / name, array, allow_pickle=False)
        write_csv(output_dir / "state_summary.csv", state_rows)
        write_csv(output_dir / "frame_interventions.csv", frame_rows)
        write_csv(output_dir / "stage_summary.csv", stage_rows)
        if comparison_rows:
            write_csv(output_dir / "stage_comparisons.csv", comparison_rows)
        write_csv(output_dir / "checkpoints.csv", checkpoint_rows)

        script_path = Path(__file__).resolve()
        author_repo = Path("/root/autodl-tmp/third_party/visualize_atari")
        author_commit = subprocess.check_output(
            ["git", "-C", str(author_repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if author_commit != config["visualize_atari_commit"]:
            raise ValueError(
                f"Greydanus author repository drift: {author_commit}"
            )
        input_paths = {
            "config": config_path,
            "heldout_states": heldout_path,
            "checkpoint_index": source_run / "checkpoints.jsonl",
            "offline_q_panel": panel_dir / "q_values.npy",
            "offline_stages": panel_dir / "stages.csv",
            "author_saliency_source": author_repo / "saliency.py",
        }
        output_paths = [
            *(output_dir / name for name in arrays),
            output_dir / "state_summary.csv",
            output_dir / "frame_interventions.csv",
            output_dir / "stage_summary.csv",
            output_dir / "checkpoints.csv",
            *((output_dir / "stage_comparisons.csv",) if comparison_rows else ()),
        ]
        manifest = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "analysis_commit": subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "analysis_script": str(script_path),
            "analysis_script_sha256": sha256_file(script_path),
            "analysis_module_sha256": sha256_file(
                PROJECT_ROOT / "src/dqn2015_visual_intervention.py"
            ),
            "author_repository": str(author_repo),
            "author_commit": author_commit,
            "author_license": "MIT declared in README and source headers",
            "device": str(device),
            "runtime_versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "checkpoint_stages": stages,
            "state_count": state_count,
            "state_selection_seed": int(config["state_selection_seed"]),
            "grid_shape": [len(range(0, 84, int(config["grid_stride"]))) ] * 2,
            "grid_count": grid_count,
            "gaussian_blur_sigma": float(config["gaussian_blur_sigma"]),
            "mask_radius": float(config["mask_radius"]),
            "grid_stride": int(config["grid_stride"]),
            "maximum_exp0005_q_parity_error": max_panel_parity_error,
            "inputs": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for name, path in input_paths.items()
            },
            "outputs": {
                path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in output_paths
            },
        }
        atomic_json(output_dir / "source_manifest.json", manifest)
        completed = {
            "experiment_id": config["experiment_id"],
            "completed": True,
            "checkpoint_stages": stages,
            "state_count": state_count,
            "grid_count": grid_count,
            "all_finite": True,
            "maximum_exp0005_q_parity_error": max_panel_parity_error,
        }
        atomic_json(output_dir / ".completed", completed)
        print(json.dumps({"event": "completed", **completed}), flush=True)
        return 0
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
