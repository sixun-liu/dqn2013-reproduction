#!/usr/bin/env python3
"""Build deterministic statistics and a review figure from the frozen DQN panel."""

from __future__ import annotations

import argparse
import csv
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
from src.dqn2015_offline_statistics import (
    bootstrap_binary_mean,
    circular_block_bootstrap_spearman,
    paired_bootstrap_summary,
    spearman_correlation,
)


DEFAULT_INPUT = Path("/root/autodl-tmp/artifacts/dqn2013/EXP-0005")
DEFAULT_REVIEW = Path(
    "/root/autodl-tmp/artifacts/dqn2013/review/"
    "EXP-0005-fixed-state-value-representation-atlas"
)
SELECTED_DECISIONS = (250_000, 1_000_000, 2_500_000, 5_000_000, 7_500_000, 9_250_000)
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20_260_722
BLOCK_LENGTH = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalized_action_entropy(actions: np.ndarray, action_count: int) -> float:
    probabilities = np.bincount(actions, minlength=action_count) / len(actions)
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(action_count))


def input_manifest(input_dir: Path) -> dict[str, dict[str, object]]:
    names = (
        ".completed",
        "summary.json",
        "source_manifest.json",
        "q_values.npy",
        "features.npy",
        "stages.csv",
        "behavior_evaluations.csv",
        "behavior_episodes.csv",
    )
    result = {}
    for name in names:
        path = input_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def build_stage_rows(stages: pd.DataFrame, q_values: np.ndarray) -> list[dict[str, object]]:
    final_actions = q_values[-1].argmax(axis=1)
    rows = []
    for index, stage in stages.iterrows():
        actions = q_values[index].argmax(axis=1)
        rows.append(
            {
                **stage.to_dict(),
                "action_agreement_to_final": float(np.mean(actions == final_actions)),
                "greedy_action_entropy_normalized": normalized_action_entropy(
                    actions, q_values.shape[2]
                ),
            }
        )
    return rows


def paired_rows(stages: pd.DataFrame, q_values: np.ndarray) -> list[dict[str, object]]:
    step_to_index = {
        int(value): index for index, value in enumerate(stages["completed_agent_decisions"])
    }
    final_q = q_values[-1]
    final_max = final_q.max(axis=1)
    final_sorted = np.sort(final_q, axis=1)
    final_margin = final_sorted[:, -1] - final_sorted[:, -2]
    final_actions = final_q.argmax(axis=1)
    rows = []
    for comparison_index, step in enumerate(SELECTED_DECISIONS):
        index = step_to_index[step]
        q_stage = q_values[index]
        sorted_stage = np.sort(q_stage, axis=1)
        stage_metrics = {
            "max_q": q_stage.max(axis=1),
            "action_margin": sorted_stage[:, -1] - sorted_stage[:, -2],
        }
        final_metrics = {"max_q": final_max, "action_margin": final_margin}
        row: dict[str, object] = {
            "completed_agent_decisions": step,
            "reference_decisions": 10_000_000,
        }
        for metric_index, name in enumerate(("max_q", "action_margin")):
            summary = paired_bootstrap_summary(
                stage_metrics[name] - final_metrics[name],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED + comparison_index * 10 + metric_index,
            )
            for field, value in summary.items():
                row[f"{name}_{field}"] = value
        switch, low, high = bootstrap_binary_mean(
            (q_stage.argmax(axis=1) != final_actions).astype(np.float64),
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED + comparison_index * 10 + 2,
        )
        row.update(
            {
                "greedy_action_switch_fraction": switch,
                "greedy_action_switch_ci95_low": low,
                "greedy_action_switch_ci95_high": high,
            }
        )
        rows.append(row)
    return rows


def correlation_rows(stages: pd.DataFrame) -> list[dict[str, object]]:
    behavior = stages["behavior_mean_episode_return"].to_numpy()
    metrics = (
        "max_q_mean",
        "action_margin_mean",
        "effective_rank",
        "participation_ratio",
        "pca_top10_fraction",
        "feature_zero_fraction",
        "linear_cka_to_final",
    )
    rows = []
    for index, name in enumerate(metrics):
        values = stages[name].to_numpy()
        rho, low, high = circular_block_bootstrap_spearman(
            behavior,
            values,
            block_length=BLOCK_LENGTH,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED + 100 + index,
        )
        rows.append(
            {
                "metric": name,
                "spearman_rho_to_behavior_mean": rho,
                "block_bootstrap_ci95_low": low,
                "block_bootstrap_ci95_high": high,
                "block_length_checkpoints": BLOCK_LENGTH,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "first_difference_spearman_exploratory": spearman_correlation(
                    np.diff(behavior), np.diff(values)
                ),
            }
        )
    return rows


def render_figure(stages: pd.DataFrame, q_values: np.ndarray, output: Path) -> None:
    x = stages["completed_agent_decisions"].to_numpy() / 1_000_000
    colors = ("#666666", "#c44e52", "#4c72b0", "#55a868")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, stages["behavior_mean_episode_return"], color="#111111", label="mean")
    ax.plot(x, stages["behavior_median_episode_return"], color="#4c72b0", label="median")
    ax.set(title="A. Frozen behavior evaluations", ylabel="raw full-game return")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(x, stages["max_q_mean"], color="#4c72b0", label="mean max-Q")
    ax.fill_between(x, stages["max_q_p05"], stages["max_q_p95"], color="#4c72b0", alpha=0.15)
    margin_axis = ax.twinx()
    margin_axis.plot(x, stages["action_margin_mean"], color="#c44e52", label="mean margin")
    ax.scatter([9.25], [stages.loc[x == 9.25, "max_q_mean"].iloc[0]], color="#dd8452", zorder=4)
    ax.set(title="B. Fixed-state value scale and confidence", ylabel="max-Q")
    margin_axis.set_ylabel("top-2 action margin")
    lines = ax.lines + margin_axis.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")

    ax = axes[1, 0]
    ax.plot(x, stages["effective_rank"], color="#4c72b0", label="effective rank")
    ax.plot(x, stages["participation_ratio"], color="#55a868", label="participation ratio")
    geometry_axis = ax.twinx()
    geometry_axis.plot(x, stages["linear_cka_to_final"], color="#c44e52", label="CKA to final")
    ax.set(title="C. FC512 representation geometry", ylabel="spectrum dimension")
    geometry_axis.set_ylabel("linear CKA")
    lines = ax.lines + geometry_axis.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")

    ax = axes[1, 1]
    fractions = np.vstack(
        [stages[f"greedy_action_{action}_fraction"].to_numpy() for action in range(4)]
    )
    ax.stackplot(x, fractions, labels=("NOOP", "FIRE", "RIGHT", "LEFT"), colors=colors, alpha=0.9)
    ax.set(title="D. Greedy actions on the same 500 states", ylabel="state fraction", ylim=(0, 1))
    ax.legend(frameon=False, ncol=4, loc="lower center")

    ax = axes[2, 0]
    scatter = ax.scatter(
        stages["max_q_mean"],
        stages["behavior_mean_episode_return"],
        c=x,
        cmap="viridis",
        s=42,
    )
    for step in (9.25, 10.0):
        row = stages.loc[x == step].iloc[0]
        ax.annotate(
            f"{step:.2g}M",
            (row["max_q_mean"], row["behavior_mean_episode_return"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    fig.colorbar(scatter, ax=ax, label="training decisions (M)")
    ax.set(title="E. Q scale does not determine behavior", xlabel="mean max-Q", ylabel="mean return")

    ax = axes[2, 1]
    final_actions = q_values[-1].argmax(axis=1)
    agreement = [np.mean(q.argmax(axis=1) == final_actions) for q in q_values]
    entropy = [normalized_action_entropy(q.argmax(axis=1), q_values.shape[2]) for q in q_values]
    ax.plot(x, agreement, color="#4c72b0", label="action agreement to final")
    ax.plot(x, entropy, color="#55a868", label="normalized action entropy")
    ax.plot(x, stages["feature_zero_fraction"], color="#c44e52", label="FC512 zero fraction")
    ax.set(title="F. Policy agreement and activation sparsity", ylabel="fraction / normalized value", ylim=(0, 1))
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.set_xlabel("training decisions (M)")
        ax.grid(alpha=0.2, linewidth=0.6)
    fig.suptitle("EXP-0005 | Nature DQN fixed-state offline atlas | seed 0 diagnostic", fontsize=15)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = input_manifest(input_dir)

    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("known_answer_gate_passed"):
        raise ValueError("Fixed-state known-answer gate did not pass")
    q_values = np.load(input_dir / "q_values.npy", allow_pickle=False)
    features = np.load(input_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    stages = pd.read_csv(input_dir / "stages.csv")
    if q_values.shape != (40, 500, 4) or features.shape != (40, 500, 512):
        raise ValueError(f"Unexpected panel shapes: {q_values.shape}, {features.shape}")
    if len(stages) != 40:
        raise ValueError(f"Expected 40 stage rows, got {len(stages)}")

    stage_rows = build_stage_rows(stages, q_values)
    paired = paired_rows(stages, q_values)
    correlations = correlation_rows(stages)
    write_csv(output_dir / "stage_derived.csv", stage_rows)
    write_csv(output_dir / "paired_to_final.csv", paired)
    write_csv(output_dir / "stage_correlations.csv", correlations)
    figure_path = output_dir / "EXP-0005_offline_atlas.png"
    render_figure(pd.DataFrame(stage_rows), q_values, figure_path)

    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0005",
        "analysis_phase": "post-run deterministic review",
        "analysis_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_script": str(source_path),
        "analysis_script_sha256": sha256_file(source_path),
        "statistics_module_sha256": sha256_file(
            PROJECT_ROOT / "src/dqn2015_offline_statistics.py"
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "checkpoint_block_length": BLOCK_LENGTH,
        "inputs": inputs,
        "outputs": {},
    }
    for path in (
        output_dir / "stage_derived.csv",
        output_dir / "paired_to_final.csv",
        output_dir / "stage_correlations.csv",
        figure_path,
    ):
        manifest["outputs"][path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_json(output_dir / "analysis_manifest.json", manifest)
    completed = {
        "experiment_id": "EXP-0005",
        "completed": True,
        "figure": str(figure_path),
        "figure_sha256": sha256_file(figure_path),
    }
    atomic_json(output_dir / ".completed", completed)
    print(json.dumps(completed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
