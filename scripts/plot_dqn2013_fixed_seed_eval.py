#!/usr/bin/env python3
"""Plot fixed-seed DQN checkpoint evaluation results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["per_seed"]
    seeds = [row["seed"] for row in rows]
    means = np.asarray([row["mean_return"] for row in rows], dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "seed", "eval_steps", "episodes", "mean_return", "median_return",
            "min_return", "max_return", "wall_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(seeds))
    ax.bar(positions, means, width=0.7, color="#315b8a", label="Final checkpoint seed mean")
    ax.axhline(1.2, color="#4f7f52", linestyle="--", linewidth=1.4,
               label="Paper random reference: 1.2")
    ax.axhline(7.0, color="#d58b2a", linestyle="--", linewidth=1.4,
               label="Preregistered low-policy boundary: 7")
    ax.axhline(10.9048, color="#6f4a8e", linestyle=":", linewidth=1.4,
               label="EXP-0001 9M-frame evaluation mean: 10.90")
    ax.set_xticks(positions, [str(seed) for seed in seeds], rotation=35, ha="right")
    ax.set_ylim(0, 12)
    ax.set_xlabel("Fixed evaluation seed")
    ax.set_ylabel("Mean raw episode return")
    ax.set_title("DQN final checkpoint: fixed-seed reevaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(
        0.5,
        0.01,
        f"median seed mean={np.median(means):.2f}; range={means.min():.2f}-{means.max():.2f}; "
        f"pooled episodes={summary['completed_episode_count']}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output_dir / "fixed_seed_evaluation.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
