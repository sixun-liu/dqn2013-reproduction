#!/usr/bin/env python3
"""Instrumented executor for controlled follow-ups to the frozen DQN run."""

import json
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dqn2013_breakout import (  # noqa: E402
    Args as BaseArgs,
    QNetwork,
    append_jsonl,
    epsilon_at,
    evaluate,
    make_env,
    save_checkpoint,
)


@dataclass
class Args(BaseArgs):
    fixed_eval_seed: int = -1
    preserve_checkpoints: bool = True


def tensor_l2_norm(tensors):
    values = [tensor.detach().float().pow(2).sum() for tensor in tensors if tensor is not None]
    if not values:
        return None
    return float(torch.stack(values).sum().sqrt().cpu())


def main():
    args = tyro.cli(Args)
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.jsonl"
    (outdir / "config.json").write_text(
        json.dumps(asdict(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([lambda: make_env(args, args.seed)])
    model = QNetwork(envs.single_action_space.n).to(device)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        alpha=args.rmsprop_alpha,
        eps=args.rmsprop_eps,
        momentum=0.0,
        centered=False,
    )
    start_step = 0
    if args.resume_model:
        checkpoint = torch.load(args.resume_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        append_jsonl(metrics_path, {
            "type": "resume",
            "step": start_step,
            "limitation": "Replay is not restored; resumed evidence is not protocol-equivalent.",
        })

    replay = ReplayBuffer(
        args.replay_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    stop_requested = {"value": False, "signal": None}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        stop_requested["signal"] = signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    obs, _ = envs.reset(seed=args.seed)
    started = time.time()
    last_loss = None
    last_q = None
    diagnostics = {}
    print(json.dumps({
        "event": "start",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "step": start_step,
        "executor": "instrumented",
    }), flush=True)

    for step in range(start_step, args.total_steps):
        epsilon = epsilon_at(args, step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                actions = model(torch.as_tensor(obs, device=device)).argmax(dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    append_jsonl(metrics_path, {
                        "type": "train_episode",
                        "step": step + 1,
                        "emulator_frames": (step + 1) * args.action_repeat,
                        "return": float(np.asarray(info["episode"]["r"]).item()),
                        "length": int(np.asarray(info["episode"]["l"]).item()),
                    })

        real_next_obs = next_obs.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[index] = infos["final_observation"][index]
        replay.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        current = step + 1
        if step >= args.learning_starts and step % args.train_frequency == 0:
            data = replay.sample(args.batch_size)
            with torch.no_grad():
                next_q = model(data.next_observations).max(dim=1).values
                target = data.rewards.flatten() + args.gamma * next_q * (1 - data.dones.flatten())
            predicted = model(data.observations).gather(1, data.actions.long()).squeeze(1)
            td_error = target - predicted
            loss = F.mse_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if current % args.log_every == 0:
                diagnostics = {
                    "q_min": float(predicted.detach().min().cpu()),
                    "q_max": float(predicted.detach().max().cpu()),
                    "target_mean": float(target.detach().mean().cpu()),
                    "target_min": float(target.detach().min().cpu()),
                    "target_max": float(target.detach().max().cpu()),
                    "td_abs_mean": float(td_error.detach().abs().mean().cpu()),
                    "td_abs_max": float(td_error.detach().abs().max().cpu()),
                    "grad_norm": tensor_l2_norm([param.grad for param in model.parameters()]),
                }
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            last_q = float(predicted.detach().mean().cpu())

        if current % args.log_every == 0:
            elapsed = time.time() - started
            record = {
                "type": "progress",
                "step": current,
                "emulator_frames": current * args.action_repeat,
                "epsilon": epsilon,
                "sps": (current - start_step) / max(elapsed, 1e-6),
                "loss": last_loss,
                "q_mean": last_q,
                "replay_size": int(replay.size()),
                "wall_seconds": elapsed,
                "parameter_norm": tensor_l2_norm(list(model.parameters())),
                **diagnostics,
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)

        if args.eval_every and current % args.eval_every == 0:
            legacy_seed = args.seed + 10_000 + current
            result = evaluate(args, model, device, legacy_seed)
            append_jsonl(metrics_path, {
                "type": "evaluation",
                "eval_kind": "legacy_seed",
                "seed": legacy_seed,
                "step": current,
                "emulator_frames": current * args.action_repeat,
                **result,
            })
            if args.fixed_eval_seed >= 0:
                fixed_result = evaluate(args, model, device, args.fixed_eval_seed)
                append_jsonl(metrics_path, {
                    "type": "evaluation",
                    "eval_kind": "fixed_seed",
                    "seed": args.fixed_eval_seed,
                    "step": current,
                    "emulator_frames": current * args.action_repeat,
                    **fixed_result,
                })

        if args.checkpoint_every and current % args.checkpoint_every == 0:
            latest_path = outdir / "checkpoint_latest.pt"
            save_checkpoint(latest_path, model, optimizer, current, args, "periodic")
            if args.preserve_checkpoints:
                checkpoint_path = outdir / "checkpoints" / f"step_{current:09d}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(checkpoint_path, model, optimizer, current, args, "periodic")

        if stop_requested["value"]:
            save_checkpoint(
                outdir / "checkpoint_latest.pt", model, optimizer, current, args, "signal"
            )
            append_jsonl(metrics_path, {
                "type": "stopped",
                "step": current,
                "signal": stop_requested["signal"],
            })
            break
    else:
        save_checkpoint(
            outdir / "checkpoint_latest.pt", model, optimizer, args.total_steps, args, "complete"
        )
        if args.preserve_checkpoints:
            checkpoint_path = outdir / "checkpoints" / f"step_{args.total_steps:09d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(checkpoint_path, model, optimizer, args.total_steps, args, "complete")
        append_jsonl(metrics_path, {"type": "completed", "step": args.total_steps})

    envs.close()


if __name__ == "__main__":
    main()
