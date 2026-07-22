"""Reusable offline analysis primitives for the frozen Nature DQN run."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"
if int(os.environ.get("MKL_NUM_THREADS", "0") or 0) <= 0:
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointRecord:
    completed_agent_decisions: int
    optimizer_updates: int
    reason: str
    path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def discover_evaluation_checkpoints(
    index_path: Path,
    *,
    interval_agent_decisions: int = 250_000,
    total_agent_decisions: int = 10_000_000,
) -> list[CheckpointRecord]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in read_jsonl(index_path):
        step = int(record["completed_agent_decisions"])
        grouped.setdefault(step, []).append(record)

    expected_steps = list(
        range(interval_agent_decisions, total_agent_decisions + 1, interval_agent_decisions)
    )
    selected = []
    for step in expected_steps:
        candidates = grouped.get(step, [])
        if not candidates:
            raise ValueError(f"Missing checkpoint at decision {step}")
        preferred_reason = "complete" if step == total_agent_decisions else "evaluation"
        preferred = [item for item in candidates if item.get("reason") == preferred_reason]
        if len(preferred) != 1:
            raise ValueError(
                f"Expected one {preferred_reason} checkpoint at {step}, got {len(preferred)}"
            )
        item = preferred[0]
        path = Path(item["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(
            CheckpointRecord(
                completed_agent_decisions=step,
                optimizer_updates=int(item["optimizer_updates"]),
                reason=str(item["reason"]),
                path=path,
            )
        )
    return selected


@torch.no_grad()
def extract_q_and_features(
    network: torch.nn.Module,
    states: np.ndarray,
    device: torch.device,
    batch_states: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if states.ndim != 4:
        raise ValueError(f"Expected NCHW states, got {states.shape}")
    layers = list(network.network.children())
    if len(layers) < 2 or not isinstance(layers[-1], torch.nn.Linear):
        raise ValueError("Q network does not end in a linear action head")
    feature_model = torch.nn.Sequential(*layers[:-1]).to(device)
    action_head = layers[-1]
    was_training = network.training
    network.eval()
    q_batches = []
    feature_batches = []
    max_forward_error = 0.0
    for offset in range(0, len(states), batch_states):
        tensor = torch.as_tensor(states[offset : offset + batch_states], device=device)
        full_q = network(tensor)
        features = feature_model(tensor.float() / 255.0)
        reconstructed_q = action_head(features)
        max_forward_error = max(
            max_forward_error,
            float((full_q - reconstructed_q).abs().max().detach().cpu()),
        )
        q_batches.append(full_q.detach().cpu().numpy().astype(np.float32, copy=False))
        feature_batches.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
    network.train(was_training)
    return np.concatenate(q_batches), np.concatenate(feature_batches), max_forward_error


def representation_spectrum(features: np.ndarray) -> dict[str, float]:
    if features.ndim != 2 or len(features) < 2:
        raise ValueError(f"Expected 2D feature matrix, got {features.shape}")
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = np.square(singular_values) / max(len(features) - 1, 1)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "pca_top10_fraction": 0.0,
            "pca_top50_fraction": 0.0,
        }
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    participation_ratio = float(total * total / np.square(eigenvalues).sum())
    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "pca_top10_fraction": float(eigenvalues[:10].sum() / total),
        "pca_top50_fraction": float(eigenvalues[:50].sum() / total),
    }


def centered_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0] or left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"Incompatible CKA shapes: {left.shape}, {right.shape}")
    x = left.astype(np.float64) - left.mean(axis=0, keepdims=True)
    y = right.astype(np.float64) - right.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = float(np.square(cross).sum())
    denominator = float(np.sqrt(np.square(xx).sum() * np.square(yy).sum()))
    return numerator / denominator if denominator > 0 else 0.0


def summarize_q_values(q_values: np.ndarray) -> dict[str, Any]:
    if q_values.ndim != 2:
        raise ValueError(f"Expected state x action Q values, got {q_values.shape}")
    sorted_q = np.sort(q_values, axis=1)
    maxima = sorted_q[:, -1]
    margins = sorted_q[:, -1] - sorted_q[:, -2]
    actions = q_values.argmax(axis=1)
    result: dict[str, Any] = {
        "max_q_mean": float(maxima.mean()),
        "max_q_median": float(np.median(maxima)),
        "max_q_p05": float(np.quantile(maxima, 0.05)),
        "max_q_p95": float(np.quantile(maxima, 0.95)),
        "max_q_p99": float(np.quantile(maxima, 0.99)),
        "action_margin_mean": float(margins.mean()),
        "action_margin_median": float(np.median(margins)),
        "action_margin_p05": float(np.quantile(margins, 0.05)),
        "action_margin_p95": float(np.quantile(margins, 0.95)),
    }
    for action in range(q_values.shape[1]):
        result[f"greedy_action_{action}_fraction"] = float((actions == action).mean())
    return result


def behavior_records(metrics_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluations = []
    episodes = []
    for record in read_jsonl(metrics_path):
        if record.get("type") != "evaluation":
            continue
        step = int(record["completed_agent_decisions"])
        evaluations.append(
            {
                "completed_agent_decisions": step,
                "completed_games": int(record["completed_games"]),
                "mean_episode_return": float(record["mean_episode_return"]),
                "median_episode_return": float(record["median_episode_return"]),
                "min_episode_return": float(record["min_episode_return"]),
                "max_episode_return": float(record["max_episode_return"]),
                "eval_agent_decisions_executed": int(record["eval_agent_decisions_executed"]),
                "eval_epsilon": float(record["eval_epsilon"]),
                "eval_seed": int(record["eval_seed"]),
                "interrupted": bool(record["interrupted"]),
            }
        )
        returns = record["episode_returns"]
        lengths = record["episode_lengths"]
        if len(returns) != len(lengths):
            raise ValueError(f"Evaluation return/length mismatch at {step}")
        episodes.extend(
            {
                "completed_agent_decisions": step,
                "episode_index": index,
                "episode_return": float(value),
                "episode_length": int(lengths[index]),
            }
            for index, value in enumerate(returns)
        )
    evaluations.sort(key=lambda item: item["completed_agent_decisions"])
    if len(evaluations) != 40:
        raise ValueError(f"Expected 40 evaluations, got {len(evaluations)}")
    if any(item["interrupted"] for item in evaluations):
        raise ValueError("Behavior panel contains interrupted evaluation")
    return evaluations, episodes
