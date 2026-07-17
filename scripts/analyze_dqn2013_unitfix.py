#!/usr/bin/env python3
"""Compare EXP-0003's unit correction with the historical EXP-0001 run."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def finite_max(rows, key):
    values = [row.get(key) for row in rows]
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return max(values) if values else None


def finite_min(rows, key):
    values = [row.get(key) for row in rows]
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return min(values) if values else None


def select_progress(rows, start, end):
    return [
        row for row in rows
        if row.get("type") == "progress" and start <= int(row["step"]) <= end
    ]


def evaluation_map(rows, kind=None):
    selected = {}
    for row in rows:
        if row.get("type") != "evaluation":
            continue
        if kind is not None and row.get("eval_kind") != kind:
            continue
        selected[int(row["step"])] = row
    return selected


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-metrics", type=Path, required=True)
    parser.add_argument("--new-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=1_450_000)
    parser.add_argument("--window-end", type=int, default=1_500_000)
    args = parser.parse_args()

    old_rows = read_metrics(args.old_metrics)
    new_rows = read_metrics(args.new_metrics)
    if not any(row.get("type") == "completed" for row in new_rows):
        raise ValueError("New run has no completed record; refusing partial analysis")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_eval = evaluation_map(old_rows)
    new_legacy = evaluation_map(new_rows, "legacy_seed")
    new_fixed = evaluation_map(new_rows, "fixed_seed")
    steps = sorted(set(old_eval) & set(new_legacy))
    paired_rows = []
    for step in steps:
        paired_rows.append({
            "decision_step": step,
            "emulator_frames": step * 4,
            "old_legacy_mean": old_eval[step].get("mean_return"),
            "old_legacy_episodes": old_eval[step].get("episodes"),
            "new_legacy_mean": new_legacy[step].get("mean_return"),
            "new_legacy_episodes": new_legacy[step].get("episodes"),
            "new_fixed_mean": new_fixed.get(step, {}).get("mean_return"),
            "new_fixed_episodes": new_fixed.get(step, {}).get("episodes"),
        })
    write_csv(
        args.output_dir / "paired_evaluations.csv",
        [
            "decision_step", "emulator_frames", "old_legacy_mean", "old_legacy_episodes",
            "new_legacy_mean", "new_legacy_episodes", "new_fixed_mean", "new_fixed_episodes",
        ],
        paired_rows,
    )

    old_window = select_progress(old_rows, args.window_start, args.window_end)
    new_window = select_progress(new_rows, args.window_start, args.window_end)
    if not old_window or not new_window:
        raise ValueError("Comparison window is missing progress samples")

    diagnostic_fields = [
        "step", "emulator_frames", "epsilon", "q_mean", "q_min", "q_max", "target_mean",
        "target_min", "target_max", "loss", "td_abs_mean", "td_abs_max", "grad_norm",
        "parameter_norm", "replay_size", "sps", "wall_seconds",
    ]
    write_csv(args.output_dir / "new_final_window_diagnostics.csv", diagnostic_fields, new_window)

    final_step = args.window_end
    old_final = old_eval.get(final_step)
    new_final = new_legacy.get(final_step)
    fixed_final = new_fixed.get(final_step)
    if old_final is None or new_final is None or fixed_final is None:
        raise ValueError("Paired final evaluations are incomplete")

    summary = {
        "comparison": {
            "changed_variables": {
                "replay_size_transitions": {"old": 250_000, "new": 1_000_000},
                "epsilon_decay_decisions": {"old": 250_000, "new": 1_000_000},
            },
            "window_decisions": [args.window_start, args.window_end],
            "old_window_logged_samples": len(old_window),
            "new_window_logged_samples": len(new_window),
        },
        "old_window": {
            "q_mean_max": finite_max(old_window, "q_mean"),
            "q_mean_min": finite_min(old_window, "q_mean"),
            "loss_max": finite_max(old_window, "loss"),
            "final_legacy_mean_return": old_final.get("mean_return"),
            "final_legacy_median_return": old_final.get("median_return"),
            "final_legacy_episodes": old_final.get("episodes"),
        },
        "new_window": {
            "q_mean_max": finite_max(new_window, "q_mean"),
            "q_mean_min": finite_min(new_window, "q_mean"),
            "q_max_max": finite_max(new_window, "q_max"),
            "loss_max": finite_max(new_window, "loss"),
            "td_abs_max": finite_max(new_window, "td_abs_max"),
            "grad_norm_max": finite_max(new_window, "grad_norm"),
            "parameter_norm_max": finite_max(new_window, "parameter_norm"),
            "final_legacy_mean_return": new_final.get("mean_return"),
            "final_legacy_median_return": new_final.get("median_return"),
            "final_legacy_episodes": new_final.get("episodes"),
            "final_fixed_mean_return": fixed_final.get("mean_return"),
            "final_fixed_median_return": fixed_final.get("median_return"),
            "final_fixed_episodes": fixed_final.get("episodes"),
        },
        "preregistered_gates": {
            "support": "new q_mean max < 20 and new final legacy mean >= 5",
            "insufficient": "new q_mean max >= 20 or new final legacy mean <= 3",
            "support_passed": (
                finite_max(new_window, "q_mean") < 20
                and float(new_final["mean_return"]) >= 5
            ),
            "insufficient_triggered": (
                finite_max(new_window, "q_mean") >= 20
                or float(new_final["mean_return"]) <= 3
            ),
        },
        "completion": {
            "completed_step": final_step,
            "new_evaluation_steps": sorted(new_legacy),
            "paired_evaluation_steps": steps,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    old_progress = [row for row in old_rows if row.get("type") == "progress"]
    new_progress = [row for row in new_rows if row.get("type") == "progress"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.5))

    ax = axes[0]
    paired_steps = [row["decision_step"] for row in paired_rows]
    ax.plot(paired_steps, [row["old_legacy_mean"] for row in paired_rows], marker="o",
            color="#7a7f87", label="EXP-0001 legacy seed")
    ax.plot(paired_steps, [row["new_legacy_mean"] for row in paired_rows], marker="o",
            color="#236e9b", label="EXP-0003 paired legacy seed")
    ax.plot(paired_steps, [row["new_fixed_mean"] for row in paired_rows], marker="s",
            color="#b15c2e", label="EXP-0003 fixed seed 20000")
    ax.axhline(5, color="#38814a", linestyle="--", linewidth=1.2, label="Final support gate: 5")
    ax.set(title="Paired Breakout evaluations", ylabel="Mean episode return")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot([row["step"] for row in old_progress], [row["q_mean"] for row in old_progress],
            color="#7a7f87", alpha=0.75, label="EXP-0001 q mean")
    ax.plot([row["step"] for row in new_progress], [row["q_mean"] for row in new_progress],
            color="#236e9b", alpha=0.9, label="EXP-0003 q mean")
    ax.axvspan(args.window_start, args.window_end, color="#d9e6d4", alpha=0.55,
               label="Preregistered final window")
    ax.axhline(20, color="#9d2f2f", linestyle="--", linewidth=1.2, label="Q support gate: 20")
    ax.set(title="Logged Q mean stability", ylabel="Q mean")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([row["step"] for row in new_progress], [row.get("loss", math.nan) for row in new_progress],
            color="#b15c2e", alpha=0.75, label="MSE loss")
    ax.plot([row["step"] for row in new_progress],
            [row.get("td_abs_max", math.nan) for row in new_progress],
            color="#6f4c91", alpha=0.75, label="Max absolute TD error")
    ax.plot([row["step"] for row in new_progress],
            [row.get("grad_norm", math.nan) for row in new_progress],
            color="#38814a", alpha=0.75, label="Gradient norm")
    ax.axvspan(args.window_start, args.window_end, color="#d9e6d4", alpha=0.55)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set(title="EXP-0003 tail diagnostics", xlabel="Agent decisions", ylabel="Symlog value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("DQN replay/epsilon agent-step unit correction")
    fig.tight_layout()
    fig.savefig(args.output_dir / "paired_stability.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
