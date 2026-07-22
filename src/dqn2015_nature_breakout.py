#!/usr/bin/env python3
"""Auditable Nature 2015 DQN executor for Breakout.

The execution structure and Atari wrapper order are adapted from CleanRL's
``dqn_atari.py`` at commit fe8d8a03c41a7ef5b523e2e354bd01c363e786bb.
CleanRL is Copyright (c) 2019--2023 CleanRL contributors and licensed under
the MIT License. This independent implementation does not contain DeepMind's
limited-license Lua source.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The host image exports these as 0, which libgomp treats as invalid.
if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if int(os.environ.get("MKL_NUM_THREADS", "0") or 0) <= 0:
    os.environ["MKL_NUM_THREADS"] = "1"

import ale_py
import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer


CONFIG_SCHEMA_VERSION = 1
PAPER_SHA256 = "cc811007a48aea14fcc135158ed96d01982930f415045a19f89474bfa1a74eb5"
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
DEEPMIND_DQN_COMMIT = "9d9b1d13a2b491d6ebd4d046740c511c662bbe0f"


@dataclass(frozen=True)
class Args:
    config_schema_version: int = CONFIG_SCHEMA_VERSION
    output_dir: str = "runs/nature2015-breakout"
    env_id: str = "BreakoutNoFrameskip-v4"
    train_seed: int = 0
    eval_seed: int = 10_000
    total_agent_decisions: int = 10_000_000
    action_repeat: int = 4
    replay_capacity_transitions: int = 1_000_000
    learning_starts_agent_decisions: int = 50_000
    batch_transitions: int = 32
    train_every_agent_decisions: int = 4
    target_sync_agent_decisions: int = 10_000
    discount: float = 0.99
    learning_rate: float = 2.5e-4
    rmsprop_gradient_momentum: float = 0.95
    rmsprop_squared_gradient_momentum: float = 0.95
    rmsprop_denominator_epsilon: float = 0.01
    start_epsilon: float = 1.0
    end_epsilon: float = 0.1
    epsilon_decay_agent_decisions: int = 1_000_000
    noop_max: int = 30
    eval_every_agent_decisions: int = 250_000
    eval_agent_decisions: int = 135_000
    eval_epsilon: float = 0.05
    heldout_state_count: int = 500
    q_diagnostic_every_agent_decisions: int = 50_000
    q_diagnostic_batch_states: int = 128
    checkpoint_every_agent_decisions: int = 250_000
    log_every_agent_decisions: int = 1_000
    cuda: bool = True
    torch_deterministic: bool = True


class QNetwork(nn.Module):
    def __init__(self, action_count: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float() / 255.0)


class DeepMindCenteredRMSprop(torch.optim.Optimizer):
    """Centered RMSProp with epsilon inside sqrt, matching DQN 3.0."""

    def __init__(
        self,
        params,
        lr: float = 2.5e-4,
        gradient_momentum: float = 0.95,
        squared_gradient_momentum: float = 0.95,
        denominator_epsilon: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            gradient_momentum=gradient_momentum,
            squared_gradient_momentum=squared_gradient_momentum,
            denominator_epsilon=denominator_epsilon,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            grad_alpha = group["gradient_momentum"]
            squared_alpha = group["squared_gradient_momentum"]
            epsilon = group["denominator_epsilon"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("DeepMindCenteredRMSprop does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["gradient_average"] = torch.zeros_like(parameter)
                    state["squared_gradient_average"] = torch.zeros_like(parameter)
                gradient_average = state["gradient_average"]
                squared_gradient_average = state["squared_gradient_average"]
                gradient_average.mul_(grad_alpha).add_(gradient, alpha=1.0 - grad_alpha)
                squared_gradient_average.mul_(squared_alpha).addcmul_(
                    gradient, gradient, value=1.0 - squared_alpha
                )
                variance = (squared_gradient_average - gradient_average.square()).clamp_min_(0.0)
                denominator = variance.add(epsilon).sqrt_()
                parameter.addcdiv_(gradient, denominator, value=-group["lr"])
        return loss


class StopController:
    def __init__(self):
        self.requested = False
        self.signal_number: int | None = None

    def request(self, signal_number: int, _frame=None) -> None:
        self.requested = True
        self.signal_number = signal_number


def epsilon_at(args: Args, completed_agent_decisions: int) -> float:
    anneal_progress = max(completed_agent_decisions - args.learning_starts_agent_decisions, 0)
    fraction = min(anneal_progress / args.epsilon_decay_agent_decisions, 1.0)
    return args.start_epsilon + fraction * (args.end_epsilon - args.start_epsilon)


def should_train(args: Args, completed_agent_decisions: int) -> bool:
    return (
        completed_agent_decisions > args.learning_starts_agent_decisions
        and completed_agent_decisions % args.train_every_agent_decisions == 0
    )


def should_sync_target(args: Args, completed_agent_decisions: int) -> bool:
    return (
        completed_agent_decisions > 0
        and completed_agent_decisions % args.target_sync_agent_decisions == 0
    )


def clipped_td_loss(predicted_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    # Sum reduction matches the author implementation's direct minibatch backward pass.
    return F.smooth_l1_loss(predicted_q, target_q, reduction="sum", beta=1.0)


def wrapper_type_names(env: gym.Env) -> list[str]:
    names = []
    current: Any = env
    while True:
        names.append(type(current).__name__)
        if not hasattr(current, "env"):
            return names
        current = current.env


def make_atari_env(args: Args, *, evaluation: bool) -> gym.Env:
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=args.noop_max)
    env = MaxAndSkipEnv(env, skip=args.action_repeat)
    if not evaluation:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if not evaluation:
        env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.action_space.seed(args.eval_seed if evaluation else args.train_seed)
    return env


def validate_args(args: Args) -> None:
    if args.config_schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema: {args.config_schema_version}")
    positive = {
        "total_agent_decisions": args.total_agent_decisions,
        "action_repeat": args.action_repeat,
        "replay_capacity_transitions": args.replay_capacity_transitions,
        "batch_transitions": args.batch_transitions,
        "train_every_agent_decisions": args.train_every_agent_decisions,
        "target_sync_agent_decisions": args.target_sync_agent_decisions,
        "epsilon_decay_agent_decisions": args.epsilon_decay_agent_decisions,
        "heldout_state_count": args.heldout_state_count,
        "q_diagnostic_batch_states": args.q_diagnostic_batch_states,
        "log_every_agent_decisions": args.log_every_agent_decisions,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"values must be positive: {', '.join(invalid)}")
    if args.replay_capacity_transitions <= args.learning_starts_agent_decisions:
        raise ValueError("replay capacity must exceed the learning-start boundary")
    if args.heldout_state_count > args.learning_starts_agent_decisions:
        raise ValueError("held-out set cannot exceed transitions available at capture")
    if not (0.0 <= args.end_epsilon <= args.start_epsilon <= 1.0):
        raise ValueError("epsilon bounds must satisfy 0 <= end <= start <= 1")


def atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any) -> float:
    return float(np.asarray(value).item())


def checkpoint_payload(
    args: Args,
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: DeepMindCenteredRMSprop,
    completed_agent_decisions: int,
    optimizer_updates: int,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "checkpoint_schema_version": 1,
        "completed_agent_decisions": completed_agent_decisions,
        "nominal_training_emulator_frames": completed_agent_decisions * args.action_repeat,
        "optimizer_updates": optimizer_updates,
        "online_network": online_network.state_dict(),
        "target_network": target_network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": asdict(args),
        "reason": reason,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "resume_limitation": (
            "ALE state and replay are not serialized; this checkpoint is for evaluation, "
            "not protocol-equivalent resume."
        ),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(
    checkpoint_dir: Path,
    index_path: Path,
    args: Args,
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: DeepMindCenteredRMSprop,
    completed_agent_decisions: int,
    optimizer_updates: int,
    reason: str,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"decision_{completed_agent_decisions:010d}__{reason}.pt"
    temp_path = path.with_suffix(".pt.tmp")
    torch.save(
        checkpoint_payload(
            args,
            online_network,
            target_network,
            optimizer,
            completed_agent_decisions,
            optimizer_updates,
            reason,
        ),
        temp_path,
    )
    os.replace(temp_path, path)
    record = {
        "path": str(path),
        "completed_agent_decisions": completed_agent_decisions,
        "optimizer_updates": optimizer_updates,
        "reason": reason,
        "bytes": path.stat().st_size,
        "created_at": time.time(),
    }
    append_jsonl(index_path, record)
    atomic_write_json(checkpoint_dir / "LATEST.json", record)
    return path


def capture_heldout_states(
    replay: ReplayBuffer,
    count: int,
    seed: int,
    output_path: Path,
) -> np.ndarray:
    available = replay.size()
    if available < count:
        raise RuntimeError(f"held-out capture needs {count} states, replay has {available}")
    generator = np.random.default_rng(seed)
    indices = generator.choice(available, size=count, replace=False)
    states = replay.observations[indices, 0].copy()
    np.save(output_path, states, allow_pickle=False)
    return states


@torch.no_grad()
def heldout_mean_max_q(
    network: QNetwork,
    states: np.ndarray,
    device: torch.device,
    batch_states: int,
) -> float:
    was_training = network.training
    network.eval()
    maxima = []
    for offset in range(0, len(states), batch_states):
        batch = torch.as_tensor(states[offset : offset + batch_states], device=device)
        maxima.append(network(batch).max(dim=1).values.cpu().numpy())
    network.train(was_training)
    return float(np.concatenate(maxima).mean())


@torch.no_grad()
def evaluate(
    args: Args,
    network: QNetwork,
    device: torch.device,
    completed_agent_decisions: int,
    stop_controller: StopController,
) -> dict[str, Any]:
    env = make_atari_env(args, evaluation=True)
    observation, _ = env.reset(seed=args.eval_seed)
    action_rng = random.Random(args.eval_seed)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    terminated_games = 0
    truncated_games = 0
    evaluation_started = time.monotonic()
    was_training = network.training
    network.eval()

    executed = 0
    for executed in range(1, args.eval_agent_decisions + 1):
        if action_rng.random() < args.eval_epsilon:
            action = env.action_space.sample()
        else:
            tensor = torch.as_tensor(np.asarray(observation)[None], device=device)
            action = int(network(tensor).argmax(dim=1).item())
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if "episode" in info:
                episode_returns.append(scalar(info["episode"]["r"]))
                episode_lengths.append(int(scalar(info["episode"]["l"])))
            terminated_games += int(terminated)
            truncated_games += int(truncated)
            observation, _ = env.reset()
        if stop_controller.requested:
            break

    env.close()
    network.train(was_training)
    return {
        "type": "evaluation",
        "completed_agent_decisions": completed_agent_decisions,
        "nominal_training_emulator_frames": completed_agent_decisions * args.action_repeat,
        "eval_agent_decisions_requested": args.eval_agent_decisions,
        "eval_agent_decisions_executed": executed,
        "eval_epsilon": args.eval_epsilon,
        "eval_seed": args.eval_seed,
        "completed_games": len(episode_returns),
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else None,
        "median_episode_return": float(np.median(episode_returns)) if episode_returns else None,
        "min_episode_return": float(np.min(episode_returns)) if episode_returns else None,
        "max_episode_return": float(np.max(episode_returns)) if episode_returns else None,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "terminated_games": terminated_games,
        "truncated_games": truncated_games,
        "interrupted": stop_controller.requested,
        "eval_wall_seconds": time.monotonic() - evaluation_started,
    }


def runtime_record(args: Args, env: gym.Env, device: torch.device, parameter_count: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gymnasium": gym.__version__,
        "ale_py": ale_py.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "action_meanings": [str(name) for name in env.unwrapped.get_action_meanings()],
        "action_count": int(env.action_space.n),
        "observation_shape": list(env.observation_space.shape),
        "wrapper_types": wrapper_type_names(env),
        "network_parameter_count": parameter_count,
        "paper_sha256": PAPER_SHA256,
        "cleanrl_reference_commit": CLEANRL_COMMIT,
        "deepmind_dqn_oracle_commit": DEEPMIND_DQN_COMMIT,
        "executor_class": "independent_reimplementation",
        "total_nominal_training_emulator_frames": args.total_agent_decisions * args.action_repeat,
    }


def run(args: Args) -> None:
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_path = output_dir / ".started"
    started_record = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
        "argv": sys.argv,
    }
    with started_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(started_record, indent=2, sort_keys=True) + "\n")

    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_index_path = output_dir / "checkpoints.jsonl"
    atomic_write_json(output_dir / "resolved_config.json", asdict(args))

    stop_controller = StopController()
    signal.signal(signal.SIGTERM, stop_controller.request)
    signal.signal(signal.SIGINT, stop_controller.request)

    try:
        random.seed(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        env = make_atari_env(args, evaluation=False)
        observation, _ = env.reset(seed=args.train_seed)
        online_network = QNetwork(env.action_space.n).to(device)
        target_network = QNetwork(env.action_space.n).to(device)
        target_network.load_state_dict(online_network.state_dict())
        target_network.eval()
        optimizer = DeepMindCenteredRMSprop(
            online_network.parameters(),
            lr=args.learning_rate,
            gradient_momentum=args.rmsprop_gradient_momentum,
            squared_gradient_momentum=args.rmsprop_squared_gradient_momentum,
            denominator_epsilon=args.rmsprop_denominator_epsilon,
        )
        replay = ReplayBuffer(
            args.replay_capacity_transitions,
            env.observation_space,
            env.action_space,
            device,
            n_envs=1,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        parameter_count = sum(parameter.numel() for parameter in online_network.parameters())
        runtime = runtime_record(args, env, device, parameter_count)
        atomic_write_json(output_dir / "runtime.json", runtime)
        append_jsonl(metrics_path, {"type": "start", **runtime})
        print(json.dumps({"event": "start", **runtime}), flush=True)

        started_monotonic = time.monotonic()
        evaluation_wall_seconds = 0.0
        optimizer_updates = 0
        target_syncs = 0
        last_loss_mean = None
        last_mean_abs_td_error = None
        last_predicted_q_mean = None
        heldout_states = None
        heldout_path = output_dir / "heldout_states.npy"

        for completed_agent_decisions in range(1, args.total_agent_decisions + 1):
            epsilon = epsilon_at(args, completed_agent_decisions - 1)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    tensor = torch.as_tensor(np.asarray(observation)[None], device=device)
                    action = int(online_network(tensor).argmax(dim=1).item())

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(
                np.asarray(observation)[None],
                np.asarray(next_observation)[None],
                np.asarray([action]),
                np.asarray([reward], dtype=np.float32),
                np.asarray([done]),
                [info],
            )
            observation = next_observation

            if done:
                if "episode" in info:
                    append_jsonl(
                        metrics_path,
                        {
                            "type": "train_game",
                            "completed_agent_decisions": completed_agent_decisions,
                            "nominal_training_emulator_frames": completed_agent_decisions * args.action_repeat,
                            "raw_episode_return": scalar(info["episode"]["r"]),
                            "raw_episode_length": int(scalar(info["episode"]["l"])),
                        },
                    )
                observation, _ = env.reset()

            if should_train(args, completed_agent_decisions):
                batch = replay.sample(args.batch_transitions)
                with torch.no_grad():
                    next_q = target_network(batch.next_observations).max(dim=1).values
                    target_q = batch.rewards.flatten() + args.discount * next_q * (1.0 - batch.dones.flatten())
                predicted_q = online_network(batch.observations).gather(1, batch.actions.long()).squeeze(1)
                td_error = target_q - predicted_q
                loss = clipped_td_loss(predicted_q, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_updates += 1
                last_loss_mean = float(loss.detach().cpu()) / args.batch_transitions
                last_mean_abs_td_error = float(td_error.detach().abs().mean().cpu())
                last_predicted_q_mean = float(predicted_q.detach().mean().cpu())

            if should_sync_target(args, completed_agent_decisions):
                target_network.load_state_dict(online_network.state_dict())
                target_syncs += 1

            if (
                heldout_states is None
                and completed_agent_decisions > args.learning_starts_agent_decisions
            ):
                heldout_states = capture_heldout_states(
                    replay,
                    args.heldout_state_count,
                    args.train_seed + 20_000,
                    heldout_path,
                )
                append_jsonl(
                    metrics_path,
                    {
                        "type": "heldout_capture",
                        "completed_agent_decisions": completed_agent_decisions,
                        "state_count": len(heldout_states),
                        "path": str(heldout_path),
                        "sha256": sha256_file(heldout_path),
                    },
                )

            if (
                heldout_states is not None
                and completed_agent_decisions % args.q_diagnostic_every_agent_decisions == 0
            ):
                append_jsonl(
                    metrics_path,
                    {
                        "type": "heldout_q",
                        "completed_agent_decisions": completed_agent_decisions,
                        "nominal_training_emulator_frames": completed_agent_decisions * args.action_repeat,
                        "mean_max_q": heldout_mean_max_q(
                            online_network,
                            heldout_states,
                            device,
                            args.q_diagnostic_batch_states,
                        ),
                    },
                )

            evaluation_due = (
                args.eval_every_agent_decisions > 0
                and completed_agent_decisions % args.eval_every_agent_decisions == 0
            )
            checkpoint_due = (
                args.checkpoint_every_agent_decisions > 0
                and completed_agent_decisions % args.checkpoint_every_agent_decisions == 0
            )
            if evaluation_due or checkpoint_due:
                save_checkpoint(
                    checkpoint_dir,
                    checkpoint_index_path,
                    args,
                    online_network,
                    target_network,
                    optimizer,
                    completed_agent_decisions,
                    optimizer_updates,
                    "evaluation" if evaluation_due else "periodic",
                )

            if evaluation_due:
                result = evaluate(
                    args,
                    online_network,
                    device,
                    completed_agent_decisions,
                    stop_controller,
                )
                evaluation_wall_seconds += result["eval_wall_seconds"]
                append_jsonl(metrics_path, result)
                print(json.dumps(result), flush=True)

            if completed_agent_decisions % args.log_every_agent_decisions == 0:
                wall_seconds = time.monotonic() - started_monotonic
                active_train_seconds = max(wall_seconds - evaluation_wall_seconds, 1e-9)
                progress = {
                    "type": "progress",
                    "completed_agent_decisions": completed_agent_decisions,
                    "nominal_training_emulator_frames": completed_agent_decisions * args.action_repeat,
                    "optimizer_updates": optimizer_updates,
                    "target_syncs": target_syncs,
                    "epsilon": epsilon,
                    "loss_mean": last_loss_mean,
                    "mean_abs_td_error": last_mean_abs_td_error,
                    "predicted_q_mean": last_predicted_q_mean,
                    "replay_transitions": replay.size(),
                    "wall_seconds": wall_seconds,
                    "evaluation_wall_seconds": evaluation_wall_seconds,
                    "active_train_decisions_per_second": completed_agent_decisions / active_train_seconds,
                    "wall_decisions_per_second": completed_agent_decisions / max(wall_seconds, 1e-9),
                }
                append_jsonl(metrics_path, progress)
                print(json.dumps(progress), flush=True)

            if stop_controller.requested:
                path = save_checkpoint(
                    checkpoint_dir,
                    checkpoint_index_path,
                    args,
                    online_network,
                    target_network,
                    optimizer,
                    completed_agent_decisions,
                    optimizer_updates,
                    "signal",
                )
                stopped = {
                    "type": "stopped",
                    "completed_agent_decisions": completed_agent_decisions,
                    "signal_number": stop_controller.signal_number,
                    "checkpoint": str(path),
                    "stopped_at": time.time(),
                }
                append_jsonl(metrics_path, stopped)
                atomic_write_json(output_dir / ".stopped", stopped)
                print(json.dumps(stopped), flush=True)
                env.close()
                return

        final_path = save_checkpoint(
            checkpoint_dir,
            checkpoint_index_path,
            args,
            online_network,
            target_network,
            optimizer,
            args.total_agent_decisions,
            optimizer_updates,
            "complete",
        )
        completed = {
            "type": "completed",
            "completed_agent_decisions": args.total_agent_decisions,
            "nominal_training_emulator_frames": args.total_agent_decisions * args.action_repeat,
            "optimizer_updates": optimizer_updates,
            "checkpoint": str(final_path),
            "completed_at": time.time(),
        }
        append_jsonl(metrics_path, completed)
        atomic_write_json(output_dir / ".completed", completed)
        print(json.dumps(completed), flush=True)
        env.close()
    except BaseException as error:
        failed = {
            "type": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": time.time(),
        }
        atomic_write_json(output_dir / ".failed", failed)
        print(json.dumps(failed), file=sys.stderr, flush=True)
        raise


def main() -> None:
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
