#!/usr/bin/env python3
"""Recompute EXP-0004 metrics, optionally reevaluate, and build review artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dqn2015_nature_breakout import Args, QNetwork, StopController, evaluate


REFERENCE_SCORE = 316.8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def validate_metrics(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluations = [row for row in rows if row.get("type") == "evaluation"]
    heldout_q = [row for row in rows if row.get("type") == "heldout_q"]
    progress = [row for row in rows if row.get("type") == "progress"]
    completed = [row for row in rows if row.get("type") == "completed"]

    if len(evaluations) != 40 or len(progress) != 10_000 or len(completed) != 1:
        raise RuntimeError(
            f"unexpected completion counts: evaluations={len(evaluations)}, "
            f"progress={len(progress)}, completed={len(completed)}"
        )
    if completed[0]["completed_agent_decisions"] != 10_000_000:
        raise RuntimeError("formal run did not complete 10M agent decisions")

    for evaluation in evaluations:
        if evaluation["interrupted"] or evaluation["eval_agent_decisions_executed"] != 135_000:
            raise RuntimeError(f"incomplete evaluator at {evaluation['completed_agent_decisions']}")
        returns = evaluation["episode_returns"]
        if evaluation["completed_games"] != len(returns) or not returns:
            raise RuntimeError(f"invalid completed-game set at {evaluation['completed_agent_decisions']}")
        recomputed = float(np.mean(returns))
        if not math.isclose(recomputed, evaluation["mean_episode_return"], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"mean mismatch at {evaluation['completed_agent_decisions']}")

    finite_progress_fields = (
        "epsilon",
        "loss_mean",
        "mean_abs_td_error",
        "predicted_q_mean",
        "active_train_decisions_per_second",
        "wall_decisions_per_second",
    )
    for row in progress:
        for field in finite_progress_fields:
            value = row.get(field)
            if value is not None and not math.isfinite(value):
                raise RuntimeError(f"non-finite {field} at {row['completed_agent_decisions']}")
    if not heldout_q or not all(math.isfinite(row["mean_max_q"]) for row in heldout_q):
        raise RuntimeError("held-out Q trajectory is missing or non-finite")
    return evaluations, heldout_q, progress


def reevaluate_checkpoint(
    run_dir: Path,
    output_dir: Path,
    evaluations: list[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["completed_agent_decisions"] != 10_000_000 or checkpoint["reason"] != "complete":
        raise RuntimeError("final checkpoint has unexpected decision or reason")
    for name in ("online_network", "target_network"):
        if not all(torch.isfinite(tensor).all().item() for tensor in checkpoint[name].values()):
            raise RuntimeError(f"non-finite tensor in {name}")

    executor_args = Args(**checkpoint["args"])
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if executor_args.cuda and torch.cuda.is_available() else "cpu")
    network = QNetwork(int(runtime["action_count"])).to(device)
    network.load_state_dict(checkpoint["online_network"])
    controller = StopController()
    result = evaluate(executor_args, network, device, 10_000_000, controller)
    original = evaluations[-1]
    result["type"] = "independent_final_checkpoint_evaluation"
    result["checkpoint"] = str(checkpoint_path)
    result["checkpoint_sha256"] = sha256_file(checkpoint_path)
    result["original_mean_episode_return"] = original["mean_episode_return"]
    result["mean_difference"] = result["mean_episode_return"] - original["mean_episode_return"]
    result["episode_returns_exact_match"] = result["episode_returns"] == original["episode_returns"]
    result["checkpoint_finite"] = True
    (output_dir / "independent_final_eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_figure(
    output_path: Path,
    evaluations: list[dict[str, Any]],
    heldout_q: list[dict[str, Any]],
    historical_evaluations: list[dict[str, Any]],
) -> None:
    decisions_m = np.asarray([row["completed_agent_decisions"] for row in evaluations]) / 1e6
    scores = np.asarray([row["mean_episode_return"] for row in evaluations])
    peak_index = int(np.argmax(scores))
    q_decisions_m = np.asarray([row["completed_agent_decisions"] for row in heldout_q]) / 1e6
    q_values = np.asarray([row["mean_max_q"] for row in heldout_q])

    figure, axes = plt.subplots(3, 1, figsize=(11.2, 11.0), gridspec_kw={"height_ratios": [2.1, 1.2, 1.2]})
    figure.patch.set_facecolor("#f7f7f5")
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.grid(color="#d7d9d7", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(decisions_m, scores, color="#165c4c", marker="o", markersize=4.5, linewidth=2.0)
    axes[0].axhline(REFERENCE_SCORE, color="#b43a32", linestyle="--", linewidth=1.8, label="Nature Table 3: 316.8")
    axes[0].scatter(
        decisions_m[peak_index], scores[peak_index], marker="*", s=190, color="#d69b21", edgecolor="#202421", zorder=5
    )
    axes[0].annotate(
        f"peak {scores[peak_index]:.2f} @ {decisions_m[peak_index]:.2f}M",
        (decisions_m[peak_index], scores[peak_index]),
        xytext=(-205, -58),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#202421"},
        fontsize=10,
    )
    axes[0].set_title("Nature 2015 DQN Breakout partial reproduction (single seed, modern ALE)", fontsize=14)
    axes[0].set_ylabel("Mean episode return")
    axes[0].set_xlim(0, 10.2)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_xlabel(
        "Training agent decisions (millions)\n"
        "Each point: epsilon=0.05, 135K decisions, complete games only; "
        "paper and local values are trajectory peaks.",
        fontsize=9,
    )

    axes[1].plot(q_decisions_m, q_values, color="#d17a22", linewidth=1.8)
    axes[1].set_title("Fixed held-out replay states", fontsize=12)
    axes[1].set_ylabel("Mean max Q")
    axes[1].set_xlim(0, 10.2)
    axes[1].set_ylim(bottom=0)

    old_decisions_m = np.asarray([row["step"] for row in historical_evaluations]) / 1e6
    old_scores = np.asarray([row["mean_return"] for row in historical_evaluations])
    axes[2].plot(
        old_decisions_m, old_scores, color="#58606b", marker="s", markersize=4.5, linewidth=1.7
    )
    axes[2].set_title(
        "2013 independent reimplementation: historical context only (not a causal no-target control)",
        fontsize=12,
    )
    axes[2].set_xlabel("Training agent decisions (millions)")
    axes[2].set_ylabel("Mean episode return")
    axes[2].set_xlim(0, 2.6)
    axes[2].set_ylim(bottom=0)
    axes[2].text(
        0.01,
        0.88,
        "Different architecture, preprocessing, evaluation length, and budget semantics.",
        transform=axes[2].transAxes,
        fontsize=9,
        color="#454b48",
    )

    figure.tight_layout(pad=2.0)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--historical-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reevaluate-checkpoint", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.run_dir / "metrics.jsonl")
    evaluations, heldout_q, progress = validate_metrics(rows)
    historical_rows = load_jsonl(args.historical_run_dir / "metrics.jsonl")
    historical_evaluations = [row for row in historical_rows if row.get("type") == "evaluation"]

    evaluation_rows = []
    for row in evaluations:
        evaluation_rows.append(
            {
                "completed_agent_decisions": row["completed_agent_decisions"],
                "nominal_training_emulator_frames": row["nominal_training_emulator_frames"],
                "completed_games": row["completed_games"],
                "mean_episode_return": row["mean_episode_return"],
                "median_episode_return": row["median_episode_return"],
                "min_episode_return": row["min_episode_return"],
                "max_episode_return": row["max_episode_return"],
                "reference_score": REFERENCE_SCORE,
                "difference_from_reference": row["mean_episode_return"] - REFERENCE_SCORE,
                "eval_wall_seconds": row["eval_wall_seconds"],
            }
        )
    write_csv(args.output_dir / "evaluations.csv", evaluation_rows, list(evaluation_rows[0]))
    write_csv(
        args.output_dir / "heldout_q.csv",
        heldout_q,
        ["completed_agent_decisions", "nominal_training_emulator_frames", "mean_max_q"],
    )
    build_figure(
        args.output_dir / "nature2015_breakout_replication.png",
        evaluations,
        heldout_q,
        historical_evaluations,
    )

    independent_result = None
    if args.reevaluate_checkpoint:
        independent_result = reevaluate_checkpoint(
            args.run_dir, args.output_dir, evaluations, args.reevaluate_checkpoint.resolve()
        )
    elif (args.output_dir / "independent_final_eval.json").exists():
        independent_result = json.loads(
            (args.output_dir / "independent_final_eval.json").read_text(encoding="utf-8")
        )

    peak = max(evaluations, key=lambda row: row["mean_episode_return"])
    completed = json.loads((args.run_dir / ".completed").read_text(encoding="utf-8"))
    started = json.loads((args.run_dir / ".started").read_text(encoding="utf-8"))
    finite_progress = [row for row in progress if row["loss_mean"] is not None]
    summary = {
        "experiment_id": "EXP-0004",
        "reproduction_kind": "independent_reimplementation",
        "reference_score": REFERENCE_SCORE,
        "evaluation_count": len(evaluations),
        "evaluation_games_total": sum(row["completed_games"] for row in evaluations),
        "peak_mean_episode_return": peak["mean_episode_return"],
        "peak_completed_agent_decisions": peak["completed_agent_decisions"],
        "peak_relative_difference": peak["mean_episode_return"] / REFERENCE_SCORE - 1.0,
        "final_mean_episode_return": evaluations[-1]["mean_episode_return"],
        "final_median_episode_return": evaluations[-1]["median_episode_return"],
        "evaluations_ge_100": sum(row["mean_episode_return"] >= 100 for row in evaluations),
        "evaluations_ge_200": sum(row["mean_episode_return"] >= 200 for row in evaluations),
        "evaluations_ge_300": sum(row["mean_episode_return"] >= 300 for row in evaluations),
        "last_five_means": [row["mean_episode_return"] for row in evaluations[-5:]],
        "heldout_q_count": len(heldout_q),
        "heldout_q_final": heldout_q[-1]["mean_max_q"],
        "heldout_q_max": max(row["mean_max_q"] for row in heldout_q),
        "loss_mean_max": max(row["loss_mean"] for row in finite_progress),
        "mean_abs_td_error_max": max(row["mean_abs_td_error"] for row in finite_progress),
        "predicted_q_mean_max": max(row["predicted_q_mean"] for row in finite_progress),
        "optimizer_updates": completed["optimizer_updates"],
        "wall_hours": (completed["completed_at"] - started["started_at"]) / 3600,
        "metrics_sha256": sha256_file(args.run_dir / "metrics.jsonl"),
        "final_checkpoint": completed["checkpoint"],
        "final_checkpoint_sha256": sha256_file(Path(completed["checkpoint"])),
        "independent_final_eval": None,
        "limitations": [
            "single fixed training seed",
            "modern Gymnasium/ALE-Py rather than the original ALE stack",
            "fixed learning rate rather than selecting the best of three rates",
            "paper raw periodic curve and seed-selection details are unavailable",
        ],
    }
    if independent_result:
        summary["independent_final_eval"] = {
            "mean_episode_return": independent_result["mean_episode_return"],
            "completed_games": independent_result["completed_games"],
            "episode_returns_exact_match": independent_result["episode_returns_exact_match"],
            "mean_difference": independent_result["mean_difference"],
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
