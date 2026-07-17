#!/usr/bin/env python3
"""Evaluate a frozen DQN checkpoint on a fixed list of ALE seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dqn2013_breakout import Args, QNetwork, make_env  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_seed(args: Args, model: QNetwork, device: torch.device, seed: int, steps: int):
    env = make_env(args, seed)
    obs, _ = env.reset(seed=seed)
    rng = random.Random(seed)
    returns = []
    started = time.time()

    for _ in range(steps):
        if rng.random() < args.eval_epsilon:
            action = env.action_space.sample()
        else:
            tensor = torch.as_tensor(np.asarray(obs)[None], device=device)
            action = int(model(tensor).argmax(dim=1).item())
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if "episode" in info:
                returns.append(float(np.asarray(info["episode"]["r"]).item()))
            obs, _ = env.reset()

    env.close()
    values = np.asarray(returns, dtype=np.float64)
    return {
        "seed": seed,
        "eval_steps": steps,
        "episodes": int(values.size),
        "mean_return": float(values.mean()) if values.size else None,
        "median_return": float(np.median(values)) if values.size else None,
        "min_return": float(values.min()) if values.size else None,
        "max_return": float(values.max()) if values.size else None,
        "returns": returns,
        "wall_seconds": time.time() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args_cli = parser.parse_args()

    checkpoint_path = args_cli.checkpoint.resolve()
    output_dir = args_cli.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.eval_steps <= 0:
        raise SystemExit("--eval-steps must be positive")
    if len(set(args_cli.seeds)) != len(args_cli.seeds):
        raise SystemExit("--seeds must not contain duplicates")
    if args_cli.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = Args(**checkpoint["args"])
    device = torch.device(args_cli.device)
    torch.backends.cudnn.deterministic = train_args.torch_deterministic
    model = QNetwork(4).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    records = []
    metrics_path = output_dir / "evaluations.jsonl"
    for seed in args_cli.seeds:
        record = evaluate_seed(train_args, model, device, seed, args_cli.eval_steps)
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    all_returns = np.asarray(
        [value for record in records for value in record["returns"]], dtype=np.float64
    )
    if not all_returns.size:
        raise RuntimeError("No complete episodes were observed on any evaluation seed")
    seed_means = np.asarray([record["mean_return"] for record in records], dtype=np.float64)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_reason": checkpoint["reason"],
        "device": str(device),
        "torch": torch.__version__,
        "eval_epsilon": train_args.eval_epsilon,
        "eval_steps_per_seed": args_cli.eval_steps,
        "seeds": args_cli.seeds,
        "seed_count": len(records),
        "completed_episode_count": int(all_returns.size),
        "pooled_mean_return": float(all_returns.mean()),
        "pooled_median_return": float(np.median(all_returns)),
        "pooled_min_return": float(all_returns.min()),
        "pooled_max_return": float(all_returns.max()),
        "mean_of_seed_means": float(seed_means.mean()),
        "median_of_seed_means": float(np.median(seed_means)),
        "min_seed_mean": float(seed_means.min()),
        "max_seed_mean": float(seed_means.max()),
        "per_seed": [{key: value for key, value in record.items() if key != "returns"}
                     for record in records],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"type": "completed", **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
