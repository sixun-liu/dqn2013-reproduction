#!/usr/bin/env python3
"""Plot DQN training and epsilon-greedy evaluation returns."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values, window):
    if len(values) < window:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-episodes", type=int, default=20)
    args = parser.parse_args()

    train = []
    evaluations = []
    progress = []
    with args.metrics.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "train_episode":
                train.append(row)
            elif row.get("type") == "evaluation":
                evaluations.append(row)
            elif row.get("type") == "progress":
                progress.append(row)
    if not train:
        raise ValueError("No training episodes in metrics")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_x = np.asarray([row["emulator_frames"] for row in train], dtype=float)
    train_y = np.asarray([row["return"] for row in train], dtype=float)
    rolling = rolling_mean(train_y, args.rolling_episodes)

    with (args.output_dir / "train_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emulator_frames", "return", f"rolling_{args.rolling_episodes}"])
        writer.writerows(zip(train_x, train_y, rolling))
    with (args.output_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["emulator_frames", "episodes", "mean_return", "median_return", "min_return", "max_return"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluations)

    finite_rolling = rolling[np.isfinite(rolling)]
    summary = {
        "train_episode_count": len(train),
        "last_emulator_frame": float(train_x[-1]),
        "train_return_max": float(train_y.max()),
        "train_return_p99_5": float(np.percentile(train_y, 99.5)),
        "latest_rolling_return": float(finite_rolling[-1]) if len(finite_rolling) else None,
        "evaluation_count": len(evaluations),
        "latest_evaluation": evaluations[-1] if evaluations else None,
        "latest_sps": progress[-1].get("sps") if progress else None,
        "paper_references": {"random": 1.2, "average_dqn": 168, "best_dqn": 225},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    valid_eval = [row for row in evaluations if row.get("mean_return") is not None]
    eval_x = [row["emulator_frames"] for row in valid_eval]
    eval_y = [row["mean_return"] for row in valid_eval]

    fig, (ax_local, ax_paper) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax in (ax_local, ax_paper):
        ax.scatter(train_x, train_y, color="#9aa5b1", alpha=0.18, s=7,
                   label="Training episodes")
        if len(finite_rolling):
            ax.plot(train_x, rolling, color="#b33a3a", linewidth=2.0,
                    label=f"Training rolling mean ({args.rolling_episodes} episodes)")
        if valid_eval:
            ax.plot(eval_x, eval_y, color="#315b8a", marker="o", linewidth=2.0,
                    label="Evaluation epsilon=0.05 mean")
        ax.axhline(1.2, color="#4f7f52", linestyle="--", linewidth=1.2,
                   label="Paper random: 1.2")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.22)

    local_max = max(
        [10.0, float(np.percentile(train_y, 99.5))]
        + ([float(finite_rolling.max())] if len(finite_rolling) else [])
        + eval_y
    )
    full_max = max([10.0, float(train_y.max())] + eval_y)
    ax_local.set_ylim(0, local_max * 1.15)
    ax_local.set(xlabel="Emulator frames", ylabel="Raw episode return",
                 title="Local learning scale (train p99.5)")
    ax_local.legend(loc="upper left", frameon=True, fontsize=7)

    ax_paper.axhline(168, color="#d58b2a", linestyle="--", linewidth=1.2,
                     label="Paper DQN average: 168")
    ax_paper.axhline(225, color="#6f4a8e", linestyle=":", linewidth=1.2,
                     label="Paper DQN best: 225")
    ax_paper.set_ylim(0, max(240, full_max * 1.15))
    ax_paper.set(xlabel="Emulator frames", title="Paper reference scale")
    ax_paper.legend(loc="upper left", frameon=True, fontsize=7)
    fig.suptitle("2013-style DQN Breakout reproduction")
    fig.tight_layout()
    fig.savefig(args.output_dir / "breakout_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
