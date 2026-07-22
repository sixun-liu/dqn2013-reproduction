#!/usr/bin/env python3
"""Build the preregistered review sheet for DQN calibration trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dqn2015_offline_analysis import sha256_file


DEFAULT_INPUT = Path("/root/autodl-tmp/artifacts/dqn2013/EXP-0006")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/artifacts/dqn2013/review/EXP-0006-dqn-925m-trajectory-calibration"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def errorbar_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    x = summary["checkpoint_decisions"].to_numpy() / 1_000_000
    center = summary[metric].to_numpy()
    low = summary[f"{metric}_ci95_low"].to_numpy()
    high = summary[f"{metric}_ci95_high"].to_numpy()
    ax.errorbar(
        x,
        center,
        yerr=np.vstack([center - low, high - center]),
        color=color,
        marker="o",
        capsize=4,
        linewidth=1.8,
    )
    ax.axvline(9.25, color="#dd8452", linestyle="--", alpha=0.6)
    ax.set(title=title, xlabel="checkpoint decisions (M)", ylabel=ylabel)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (input_dir / ".completed").is_file():
        raise ValueError("Calibration collection is incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(input_dir / "games.csv")
    lives = pd.read_csv(input_dir / "lives.csv")
    summary = pd.read_csv(input_dir / "stage_summary.csv").sort_values(
        "checkpoint_decisions"
    )
    comparisons = pd.read_csv(input_dir / "stage_comparisons.csv")
    trace_paths = sorted(input_dir.glob("trace_*.csv.gz"))
    if len(trace_paths) != 4 or len(summary) != 4:
        raise ValueError("Formal calibration review requires exactly four stages")
    traces = pd.concat(
        [pd.read_csv(path) for path in trace_paths], ignore_index=True
    )
    if not np.isfinite(
        traces[["max_q", "discounted_return", "q_minus_return", "action_margin"]].to_numpy()
    ).all():
        raise ValueError("Calibration trace contains non-finite values")

    stages = summary["checkpoint_decisions"].astype(int).tolist()
    labels = [f"{stage / 1_000_000:g}M" for stage in stages]
    colors = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    ax = axes[0, 0]
    distributions = [
        games.loc[games["checkpoint_decisions"] == stage, "raw_game_return"].to_numpy()
        for stage in stages
    ]
    boxes = ax.boxplot(distributions, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set(title="A. Greedy raw full-game behavior", ylabel="raw game return")

    errorbar_panel(
        axes[0, 1],
        summary,
        "mean_q_minus_return",
        title="B. Game-cluster mean calibration gap",
        ylabel="Q(s,a*) - sampled G",
        color="#4c72b0",
    )
    axes[0, 1].axhline(0, color="#333333", linewidth=0.8)

    errorbar_panel(
        axes[1, 0],
        summary,
        "overestimate_fraction",
        title="C. Fraction of steps with Q above sampled G",
        ylabel="game-cluster mean fraction",
        color="#c44e52",
    )
    axes[1, 0].set_ylim(0, 1)

    errorbar_panel(
        axes[1, 1],
        summary,
        "mean_action_margin",
        title="D. Greedy action confidence",
        ylabel="top-2 Q margin",
        color="#55a868",
    )

    ax = axes[2, 0]
    for stage, color, label in zip(stages, colors, labels, strict=True):
        frame = traces.loc[traces["checkpoint_decisions"] == stage]
        stride = max(len(frame) // 2_000, 1)
        sample = frame.iloc[::stride].head(2_000)
        ax.scatter(
            sample["discounted_return"],
            sample["max_q"],
            s=7,
            alpha=0.22,
            color=color,
            label=label,
        )
    bounds = np.quantile(
        np.concatenate([traces["discounted_return"], traces["max_q"]]), [0.01, 0.99]
    )
    ax.plot(bounds, bounds, color="#222222", linestyle="--", linewidth=1, label="Q = G")
    ax.set(
        title="E. Per-step Q versus sampled discounted return",
        xlabel="sampled clipped return G",
        ylabel="max-Q",
        xlim=tuple(bounds),
        ylim=tuple(bounds),
    )
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax = axes[2, 1]
    for stage, color, label in zip(stages, colors, labels, strict=True):
        frame = games.loc[games["checkpoint_decisions"] == stage]
        ax.scatter(
            frame["mean_q_minus_return"],
            frame["raw_game_return"],
            color=color,
            s=26,
            alpha=0.72,
            label=label,
        )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set(
        title="F. Calibration gap and behavior by complete game",
        xlabel="within-game mean Q - G",
        ylabel="raw game return",
    )
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.2, linewidth=0.6)
    fig.suptitle(
        "EXP-0006 | DQN life-terminal calibration | fixed rollout seed | diagnostic",
        fontsize=15,
    )
    figure_path = output_dir / "EXP-0006_calibration_review.png"
    temporary = figure_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180, facecolor="white")
    plt.close(fig)
    os.replace(temporary, figure_path)

    input_paths = [
        input_dir / ".completed",
        input_dir / "source_manifest.json",
        input_dir / "games.csv",
        input_dir / "lives.csv",
        input_dir / "stage_summary.csv",
        input_dir / "stage_comparisons.csv",
        *trace_paths,
    ]
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0006",
        "analysis_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_script": str(script_path),
        "analysis_script_sha256": sha256_file(script_path),
        "inputs": {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "outputs": {
            figure_path.name: {
                "path": str(figure_path),
                "bytes": figure_path.stat().st_size,
                "sha256": sha256_file(figure_path),
            }
        },
    }
    atomic_json(output_dir / "review_manifest.json", manifest)
    review_summary = {
        "experiment_id": "EXP-0006",
        "checkpoint_stages": stages,
        "game_count_per_stage": {
            str(int(key)): int(value)
            for key, value in games.groupby("checkpoint_decisions").size().items()
        },
        "life_count_per_stage": {
            str(int(key)): int(value)
            for key, value in lives.groupby("checkpoint_decisions").size().items()
        },
        "stage_summary": summary.to_dict(orient="records"),
        "focal_comparisons": comparisons.to_dict(orient="records"),
        "figure": str(figure_path),
        "figure_sha256": sha256_file(figure_path),
    }
    atomic_json(output_dir / "review_summary.json", review_summary)
    atomic_json(
        output_dir / ".completed",
        {
            "experiment_id": "EXP-0006",
            "completed": True,
            "figure": str(figure_path),
            "figure_sha256": sha256_file(figure_path),
        },
    )
    print(json.dumps(review_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
