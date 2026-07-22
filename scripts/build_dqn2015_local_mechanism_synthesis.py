#!/usr/bin/env python3
"""Build a cross-cycle local mechanism synthesis for the frozen Nature DQN baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP0005 = Path("/root/autodl-tmp/artifacts/dqn2013/EXP-0005")
DEFAULT_EXP0008 = Path("/root/autodl-tmp/artifacts/dqn2013/EXP-0008")
DEFAULT_EXP0008_REVIEW = Path(
    "/root/autodl-tmp/artifacts/dqn2013/review/EXP-0008-dqn-visual-intervention"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/artifacts/dqn2013/review/DQN2015-local-mechanism-synthesis"
)
STAGES = [9_000_000, 9_250_000, 9_500_000, 10_000_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp0005", type=Path, default=DEFAULT_EXP0005)
    parser.add_argument("--exp0008", type=Path, default=DEFAULT_EXP0008)
    parser.add_argument("--exp0008-review", type=Path, default=DEFAULT_EXP0008_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require_completed(path: Path) -> None:
    if not (path / ".completed").is_file():
        raise ValueError(f"Incomplete source artifact: {path}")


def load_stage_table(
    exp0005: Path,
    exp0008: Path,
    exp0008_review: Path,
) -> pd.DataFrame:
    for source in (exp0005, exp0008, exp0008_review):
        require_completed(source)

    atlas = pd.read_csv(exp0005 / "stages.csv")
    atlas = atlas.loc[atlas["completed_agent_decisions"].isin(STAGES)].copy()
    atlas = atlas[
        [
            "completed_agent_decisions",
            "behavior_mean_episode_return",
            "behavior_median_episode_return",
            "behavior_completed_games",
            "max_q_mean",
            "action_margin_mean",
            "effective_rank",
            "feature_zero_fraction",
            "linear_cka_to_final",
        ]
    ].rename(
        columns={
            "completed_agent_decisions": "checkpoint_decisions",
            "max_q_mean": "fixed500_mean_max_q",
            "action_margin_mean": "fixed500_mean_action_margin",
            "effective_rank": "fc512_effective_rank",
            "feature_zero_fraction": "fc512_zero_fraction",
            "linear_cka_to_final": "fc512_cka_to_10m",
        }
    )

    visual = pd.read_csv(exp0008 / "stage_summary.csv")[
        [
            "checkpoint_decisions",
            "saliency_mean",
            "saliency_mean_ci95_low",
            "saliency_mean_ci95_high",
            "top_decile_concentration",
            "top_decile_concentration_ci95_low",
            "top_decile_concentration_ci95_high",
            "spatial_action_switch_fraction",
            "spatial_action_switch_fraction_ci95_low",
            "spatial_action_switch_fraction_ci95_high",
            "frame_ablation_action_switch_fraction",
        ]
    ].rename(columns={"saliency_mean": "author_local_q_sensitivity"})
    controls = pd.read_csv(exp0008_review / "EXP-0008_stage_controls.csv")[
        [
            "checkpoint_decisions",
            "baseline_action_margin",
            "margin_switch_spearman",
            "saliency_normalized_median",
            "saliency_to_q_energy_ratio_of_means",
        ]
    ].rename(columns={"baseline_action_margin": "fixed128_mean_action_margin"})

    merged = atlas.merge(visual, on="checkpoint_decisions", validate="one_to_one")
    merged = merged.merge(controls, on="checkpoint_decisions", validate="one_to_one")
    merged = merged.sort_values("checkpoint_decisions").reset_index(drop=True)
    if merged["checkpoint_decisions"].astype(int).tolist() != STAGES:
        raise ValueError(f"Stage alignment failed: {merged['checkpoint_decisions'].tolist()}")
    if not np.isfinite(merged.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Synthesis table contains non-finite values")
    return merged


def column_zscores(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float64)
    scales = values.std(axis=0)
    if (scales == 0).any():
        raise ValueError("Cannot standardize a constant synthesis metric")
    return (values - values.mean(axis=0)) / scales


def render_figure(frame: pd.DataFrame, output: Path) -> None:
    x = frame["checkpoint_decisions"].to_numpy() / 1_000_000
    labels = [f"{value:g}M" for value in x]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        x,
        frame["behavior_mean_episode_return"],
        marker="o",
        linewidth=2,
        color="#c44e52",
        label="mean game return",
    )
    ax.set(xlabel="checkpoint decisions (M)", ylabel="mean game return")
    twin = ax.twinx()
    twin.plot(
        x,
        frame["fixed500_mean_max_q"],
        marker="s",
        linewidth=2,
        color="#4c72b0",
        label="fixed-state mean max-Q",
    )
    twin.set_ylabel("fixed500 mean max-Q")
    ax.set_title("A. Behavior and value scale diverge at 9.25M")
    ax.legend(loc="lower right")
    twin.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(
        x,
        frame["fixed500_mean_action_margin"],
        marker="o",
        linewidth=2,
        color="#55a868",
        label="mean action margin",
    )
    ax.set(xlabel="checkpoint decisions (M)", ylabel="fixed500 action margin")
    twin = ax.twinx()
    twin.plot(
        x,
        frame["fc512_cka_to_10m"],
        marker="s",
        linewidth=2,
        color="#8172b3",
        label="FC512 CKA to 10M",
    )
    twin.set_ylabel("linear CKA to 10M")
    ax.set_title("B. Decision margin and representation alignment")
    ax.legend(loc="upper left")
    twin.legend(loc="lower right")

    ax = axes[1, 0]
    center = frame["author_local_q_sensitivity"].to_numpy()
    low = frame["saliency_mean_ci95_low"].to_numpy()
    high = frame["saliency_mean_ci95_high"].to_numpy()
    ax.errorbar(
        x,
        center,
        yerr=np.vstack([center - low, high - center]),
        marker="o",
        capsize=4,
        linewidth=2,
        color="#dd8452",
        label="author local Q sensitivity",
    )
    ax.set(xlabel="checkpoint decisions (M)", ylabel="0.5 ||Q - Q'||^2")
    twin = ax.twinx()
    twin.plot(
        x,
        frame["top_decile_concentration"],
        marker="s",
        linewidth=2,
        color="#937860",
        label="top-decile concentration",
    )
    twin.plot(
        x,
        frame["spatial_action_switch_fraction"],
        marker="^",
        linewidth=2,
        color="#64b5cd",
        label="local action switch",
    )
    twin.set_ylabel("fraction")
    twin.set_ylim(0, 1)
    ax.set_title("C. Lower sensitivity amplitude, higher concentration")
    ax.legend(loc="upper left")
    twin.legend(loc="upper right")

    heatmap_columns = [
        "behavior_mean_episode_return",
        "fixed500_mean_max_q",
        "fixed500_mean_action_margin",
        "fc512_effective_rank",
        "fc512_cka_to_10m",
        "author_local_q_sensitivity",
        "top_decile_concentration",
        "spatial_action_switch_fraction",
    ]
    heatmap_labels = [
        "return",
        "max-Q",
        "margin",
        "FC rank",
        "FC CKA",
        "local sens.",
        "concentr.",
        "switch",
    ]
    zscores = column_zscores(frame, heatmap_columns)
    image = axes[1, 1].imshow(zscores, cmap="coolwarm", vmin=-1.7, vmax=1.7, aspect="auto")
    axes[1, 1].set(
        title="D. Column-wise z-scores (descriptive, four stages only)",
        xticks=np.arange(len(heatmap_labels)),
        xticklabels=heatmap_labels,
        yticks=np.arange(len(labels)),
        yticklabels=labels,
    )
    axes[1, 1].tick_params(axis="x", rotation=35)
    for row in range(zscores.shape[0]):
        for column in range(zscores.shape[1]):
            axes[1, 1].text(
                column,
                row,
                f"{zscores[row, column]:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(zscores[row, column]) > 0.9 else "black",
            )
    fig.colorbar(image, ax=axes[1, 1], shrink=0.8, label="within-column z-score")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axvline(9.25, color="#888888", linestyle="--", alpha=0.35)
        ax.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle(
        "Frozen Nature DQN baseline | Local mechanism synthesis | single seed",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.002,
        "Fixed early-replay states and OOD blur interventions; stage co-variation is not causality.",
        ha="center",
        fontsize=10,
    )
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def build_readme(frame: pd.DataFrame, figure: Path, table: Path) -> str:
    focal = frame.loc[frame["checkpoint_decisions"] == 9_250_000].iloc[0]
    neighbors = frame.loc[frame["checkpoint_decisions"] != 9_250_000]
    return f"""# Nature 2015 DQN 局部机制综合

本页把同一个冻结单 seed 基线的三个已关闭周期合并到四个共同 checkpoint：EXP-0004 的行为、
EXP-0005 的固定状态 Q/FC512、EXP-0008 的视觉扰动。原始单位表见 `{table.name}`，图见
`{figure.name}`。

## 直接观察

- 9.25M 行为均分为 {focal['behavior_mean_episode_return']:.2f}，三个邻近阶段范围为
  {neighbors['behavior_mean_episode_return'].min():.2f}--{neighbors['behavior_mean_episode_return'].max():.2f}。
- 同时 fixed500 mean max-Q 升到 {focal['fixed500_mean_max_q']:.3f}，mean action margin 降到
  {focal['fixed500_mean_action_margin']:.4f}，FC512 到 10M 的 CKA 降到
  {focal['fc512_cka_to_10m']:.3f}。
- 作者原始 local Q sensitivity 为 {focal['author_local_q_sensitivity']:.3f}，低于三个邻近阶段；
  top-decile concentration 为 {focal['top_decile_concentration']:.3f}，高于三个邻近阶段；
  action-switch fraction 为 {focal['spatial_action_switch_fraction']:.3f}，没有配对显著升高。

## 受限解释

这组局部证据支持“9.25M 是高 Q 尺度、低动作间隔、表征对齐回撤和更集中视觉敏感性的瞬时共现”，
不支持“9.25M 因为更广泛依赖视觉而退化”。这些量共享训练阶段，不能由共变升级为相互因果。

## 边界

- 单训练 seed，行为 episode 不是独立训练重复。
- 固定状态来自训练早期 replay，不代表四个策略各自的 on-policy 分布。
- blur 与相邻帧替换是 OOD 诊断；显著图不证明对象语义注意。
- Q-return calibration 未完成；训练 Q 与完整游戏 raw return 不做直接误差解释。
- 图 D 的 z-score 仅帮助读取四阶段共现，不是显著性检验。

人工确认：pending。
"""


def main() -> int:
    args = parse_args()
    exp0005 = args.exp0005.resolve()
    exp0008 = args.exp0008.resolve()
    exp0008_review = args.exp0008_review.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_table = load_stage_table(exp0005, exp0008, exp0008_review)
    table_path = output_dir / "mechanism_stage_table.csv"
    figure_path = output_dir / "DQN2015_local_mechanism_synthesis.png"
    readme_path = output_dir / "README.md"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "source_manifest.json"
    completed_path = output_dir / ".completed"
    if completed_path.exists():
        raise SystemExit(f"Refusing to overwrite completed synthesis: {output_dir}")

    atomic_csv(table_path, stage_table)
    render_figure(stage_table, figure_path)
    atomic_text(readme_path, build_readme(stage_table, figure_path, table_path))
    summary = {
        "schema_version": 1,
        "title": "Nature 2015 DQN fixed-baseline local mechanism synthesis",
        "stages": STAGES,
        "source_experiments": ["EXP-0004", "EXP-0005", "EXP-0008"],
        "observation": (
            "9.25M combines lower behavior, higher Q scale, lower action margin, lower "
            "FC512 alignment, lower perturbation amplitude, and higher spatial concentration."
        ),
        "interpretation": (
            "The local evidence rejects broad visual fragility and retains low-margin, "
            "concentrated sensitivity as a non-causal candidate pattern."
        ),
        "human_visual_confirmation": "pending",
    }
    atomic_json(summary_path, summary)

    input_paths = [
        exp0005 / ".completed",
        exp0005 / "stages.csv",
        exp0005 / "source_manifest.json",
        exp0008 / ".completed",
        exp0008 / "stage_summary.csv",
        exp0008 / "source_manifest.json",
        exp0008_review / ".completed",
        exp0008_review / "EXP-0008_stage_controls.csv",
        exp0008_review / "review_manifest.json",
    ]
    output_paths = [table_path, figure_path, readme_path, summary_path]
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "analysis_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_script": str(script_path),
        "analysis_script_sha256": sha256_file(script_path),
        "inputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        completed_path,
        {
            "completed": True,
            "stages": STAGES,
            "table_sha256": sha256_file(table_path),
            "figure_sha256": sha256_file(figure_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    latest_path = output_dir.parent / "LATEST.md"
    atomic_text(
        latest_path,
        "# DQN 最新人工复核入口\n\n"
        f"- 当前综合：`{output_dir.name}/README.md`\n"
        "- 来源实验：`EXP-0004`、`EXP-0005`、`EXP-0008`\n"
        "- 人工确认：`pending`\n",
    )
    print(json.dumps({"output": str(output_dir), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
