#!/usr/bin/env python3
"""Budgeted, auditable 2013-style DQN reproduction on Breakout."""

import json
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Args:
    output_dir: str = "runs/dqn2013-breakout-seed0"
    env_id: str = "BreakoutNoFrameskip-v4"
    seed: int = 0
    total_steps: int = 2_500_000
    replay_size: int = 250_000
    learning_starts: int = 50_000
    batch_size: int = 32
    train_frequency: int = 1
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 0.01
    start_epsilon: float = 1.0
    end_epsilon: float = 0.1
    exploration_steps: int = 250_000
    action_repeat: int = 4
    noop_max: int = 30
    eval_every: int = 250_000
    eval_steps: int = 10_000
    eval_epsilon: float = 0.05
    checkpoint_every: int = 250_000
    log_every: int = 1_000
    cuda: bool = True
    torch_deterministic: bool = True
    resume_model: str = ""


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert len(meanings) >= 3 and meanings[1] == "FIRE"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for action in (1, 2):
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


class Preprocess84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84), np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
        return resized[18:102, :].astype(np.uint8, copy=False)


class QNetwork(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )

    def forward(self, obs):
        return self.network(obs.float() / 255.0)


def make_env(args, seed):
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, args.noop_max)
    env = RepeatActionEnv(env, args.action_repeat)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = Preprocess84(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.action_space.seed(seed)
    return env


def epsilon_at(args, step):
    fraction = min(max(step / args.exploration_steps, 0.0), 1.0)
    return args.start_epsilon + fraction * (args.end_epsilon - args.start_epsilon)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(path, model, optimizer, step, args, reason):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": asdict(args),
        "reason": reason,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(args, model, device, seed):
    env = make_env(args, seed)
    obs, _ = env.reset(seed=seed)
    returns = []
    rng = random.Random(seed)
    started = time.time()
    model.eval()
    for _ in range(args.eval_steps):
        if rng.random() < args.eval_epsilon:
            action = env.action_space.sample()
        else:
            tensor = torch.as_tensor(np.asarray(obs)[None], device=device)
            action = int(model(tensor).argmax(dim=1).item())
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if "episode" in info:
                returns.append(float(info["episode"]["r"]))
            obs, _ = env.reset()
    env.close()
    model.train()
    return {
        "episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else None,
        "median_return": float(np.median(returns)) if returns else None,
        "min_return": float(np.min(returns)) if returns else None,
        "max_return": float(np.max(returns)) if returns else None,
        "wall_seconds": time.time() - started,
    }


def main():
    args = tyro.cli(Args)
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.jsonl"
    config_path = outdir / "config.json"
    config_path.write_text(json.dumps(asdict(args), indent=2, sort_keys=True) + "\n")

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
    print(json.dumps({
        "event": "start",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "step": start_step,
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

        if step >= args.learning_starts and step % args.train_frequency == 0:
            data = replay.sample(args.batch_size)
            with torch.no_grad():
                next_q = model(data.next_observations).max(dim=1).values
                target = data.rewards.flatten() + args.gamma * next_q * (1 - data.dones.flatten())
            predicted = model(data.observations).gather(1, data.actions.long()).squeeze(1)
            loss = F.mse_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            last_q = float(predicted.detach().mean().cpu())

        current = step + 1
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
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)

        if args.eval_every and current % args.eval_every == 0:
            result = evaluate(args, model, device, args.seed + 10_000 + current)
            append_jsonl(metrics_path, {
                "type": "evaluation",
                "step": current,
                "emulator_frames": current * args.action_repeat,
                **result,
            })

        if args.checkpoint_every and current % args.checkpoint_every == 0:
            save_checkpoint(outdir / "checkpoint_latest.pt", model, optimizer, current, args, "periodic")

        if stop_requested["value"]:
            save_checkpoint(outdir / "checkpoint_latest.pt", model, optimizer, current, args, "signal")
            append_jsonl(metrics_path, {
                "type": "stopped",
                "step": current,
                "signal": stop_requested["signal"],
            })
            break
    else:
        save_checkpoint(outdir / "checkpoint_latest.pt", model, optimizer, args.total_steps, args, "complete")
        append_jsonl(metrics_path, {"type": "completed", "step": args.total_steps})

    envs.close()


if __name__ == "__main__":
    main()
