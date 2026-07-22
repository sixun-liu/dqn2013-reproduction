#!/usr/bin/env python3
"""Build preregistered summary and result-blind contact sheets for EXP-0008."""

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


DEFAULT_INPUT = Path("/root/autodl-tmp/artifacts/dqn2013/EXP-0008")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/artifacts/dqn2013/review/EXP-0008-dqn-visual-intervention"
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
        marker="o",
        capsize=4,
        color=color,
        linewidth=1.8,
    )
    ax.axvline(9.25, color="#dd8452", linestyle="--", alpha=0.6)
    ax.set(title=title, xlabel="checkpoint decisions (M)", ylabel=ylabel)


def render_summary(
    summary: pd.DataFrame,
    frame_rows: pd.DataFrame,
    output: Path,
) -> None:
    summary = summary.sort_values("checkpoint_decisions")
    stages = summary["checkpoint_decisions"].astype(int).tolist()
    labels = [f"{stage / 1_000_000:g}M" for stage in stages]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    errorbar_panel(
        axes[0, 0], summary, "saliency_mean",
        title="A. Mean local perturbation Q score", ylabel="0.5 ||Q - Q'||^2", color="#4c72b0"
    )
    errorbar_panel(
        axes[0, 1], summary, "spatial_action_switch_fraction",
        title="B. Local perturbation action switches", ylabel="grid-cell switch fraction", color="#c44e52"
    )
    errorbar_panel(
        axes[1, 0], summary, "top_decile_concentration",
        title="C. Spatial concentration", ylabel="top-decile score mass", color="#55a868"
    )
    errorbar_panel(
        axes[1, 1], summary, "global_blur_q_score",
        title="D. Global-blur positive control", ylabel="0.5 ||Q - Q'||^2", color="#8172b3"
    )

    q_heatmap = frame_rows.pivot_table(
        index="checkpoint_decisions", columns="frame_channel", values="q_score", aggfunc="mean"
    ).reindex(stages)
    switch_heatmap = frame_rows.pivot_table(
        index="checkpoint_decisions", columns="frame_channel", values="action_switch", aggfunc="mean"
    ).reindex(stages)
    image = axes[2, 0].imshow(q_heatmap.to_numpy(), aspect="auto", cmap="viridis")
    axes[2, 0].set(
        title="E. Adjacent-frame replacement Q score",
        xlabel="replaced stack channel (oldest -> newest)",
        ylabel="checkpoint",
        xticks=np.arange(4),
        yticks=np.arange(len(labels)),
        yticklabels=labels,
    )
    fig.colorbar(image, ax=axes[2, 0], shrink=0.8)

    image = axes[2, 1].imshow(
        switch_heatmap.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1
    )
    axes[2, 1].set(
        title="F. Adjacent-frame replacement switches",
        xlabel="replaced stack channel (oldest -> newest)",
        ylabel="checkpoint",
        xticks=np.arange(4),
        yticks=np.arange(len(labels)),
        yticklabels=labels,
    )
    fig.colorbar(image, ax=axes[2, 1], shrink=0.8)
    for ax in axes[:2].flat:
        ax.grid(alpha=0.2, linewidth=0.6)
    fig.suptitle(
        "EXP-0008 | Fixed-state DQN visual interventions | diagnostic only", fontsize=15
    )
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def render_contact_sheet(
    states: np.ndarray,
    state_ids: np.ndarray,
    saliency_scores: np.ndarray,
    stages: list[int],
    output: Path,
) -> None:
    review_count = min(6, len(state_ids))
    fig, axes = plt.subplots(
        review_count,
        len(stages) + 1,
        figsize=(14, 2.45 * review_count),
        constrained_layout=True,
    )
    if review_count == 1:
        axes = axes[None]
    for row in range(review_count):
        latest = states[row, -1]
        row_max = float(saliency_scores[:, row].max())
        axes[row, 0].imshow(latest, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_ylabel(f"state {int(state_ids[row])}")
        for column, stage in enumerate(stages, 1):
            axes[row, column].imshow(latest, cmap="gray", vmin=0, vmax=255)
            axes[row, column].imshow(
                saliency_scores[column - 1, row],
                cmap="inferno",
                alpha=0.62,
                vmin=0,
                vmax=row_max if row_max > 0 else 1.0,
                extent=(0, 84, 84, 0),
                interpolation="bilinear",
            )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("latest input frame")
    for column, stage in enumerate(stages, 1):
        axes[0, column].set_title(f"{stage / 1_000_000:g}M saliency")
    fig.suptitle(
        "Result-blind fixed state IDs | Greydanus blur perturbation overlays", fontsize=14
    )
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    completed_path = input_dir / ".completed"
    if not completed_path.is_file():
        raise ValueError("Visual intervention collection is incomplete")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    experiment_id = str(completed["experiment_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_dir / "stage_summary.csv")
    comparisons = pd.read_csv(input_dir / "stage_comparisons.csv")
    frame_rows = pd.read_csv(input_dir / "frame_interventions.csv")
    state_ids = np.load(input_dir / "selected_state_ids.npy", allow_pickle=False)
    saliency_scores = np.load(input_dir / "saliency_scores.npy", allow_pickle=False)
    manifest = json.loads((input_dir / "source_manifest.json").read_text(encoding="utf-8"))
    all_states = np.load(
        manifest["inputs"]["heldout_states"]["path"], allow_pickle=False
    )
    states = all_states[state_ids]
    stages = [int(value) for value in completed["checkpoint_stages"]]
    if saliency_scores.shape[:2] != (len(stages), len(state_ids)):
        raise ValueError(f"Saliency shape drift: {saliency_scores.shape}")

    summary_figure = output_dir / f"{experiment_id}_visual_summary.png"
    contact_figure = output_dir / f"{experiment_id}_saliency_contact_sheet.png"
    render_summary(summary, frame_rows, summary_figure)
    render_contact_sheet(states, state_ids, saliency_scores, stages, contact_figure)

    source_path = Path(__file__).resolve()
    input_paths = [
        completed_path,
        input_dir / "source_manifest.json",
        input_dir / "stage_summary.csv",
        input_dir / "stage_comparisons.csv",
        input_dir / "frame_interventions.csv",
        input_dir / "selected_state_ids.npy",
        input_dir / "saliency_scores.npy",
    ]
    review_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "analysis_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_script": str(source_path),
        "analysis_script_sha256": sha256_file(source_path),
        "inputs": {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "outputs": {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (summary_figure, contact_figure)
        },
    }
    atomic_json(output_dir / "review_manifest.json", review_manifest)
    review_summary = {
        "experiment_id": experiment_id,
        "checkpoint_stages": stages,
        "state_count": len(state_ids),
        "stage_summary": summary.to_dict(orient="records"),
        "focal_comparisons": comparisons.to_dict(orient="records"),
        "summary_figure": str(summary_figure),
        "contact_sheet": str(contact_figure),
    }
    atomic_json(output_dir / "review_summary.json", review_summary)
    atomic_json(
        output_dir / ".completed",
        {
            "experiment_id": experiment_id,
            "completed": True,
            "summary_figure": str(summary_figure),
            "summary_figure_sha256": sha256_file(summary_figure),
            "contact_sheet": str(contact_figure),
            "contact_sheet_sha256": sha256_file(contact_figure),
        },
    )
    print(json.dumps(review_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
