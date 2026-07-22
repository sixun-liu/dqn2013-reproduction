"""Calibration statistics for trajectories with life-loss terminal semantics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1:
        raise ValueError(f"Expected a 1D reward sequence, got {rewards.shape}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    returns = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def bootstrap_interval(
    values: np.ndarray,
    *,
    estimator: Callable[[np.ndarray], float],
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"Expected at least two cluster values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = np.fromiter(
        (float(estimator(sample)) for sample in values[indices]),
        dtype=np.float64,
        count=resamples,
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(estimator(values)), float(low), float(high)


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    estimator: Callable[[np.ndarray], float],
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"Incompatible paired shapes: {left.shape}, {right.shape}")
    return bootstrap_interval(
        left - right,
        estimator=estimator,
        resamples=resamples,
        seed=seed,
    )


def update_reservoir(
    states: np.ndarray,
    metadata: list[dict[str, int]],
    observation: np.ndarray,
    item_metadata: dict[str, int],
    *,
    seen: int,
    rng: np.random.Generator,
) -> None:
    if states.ndim != 4:
        raise ValueError(f"Expected reservoir NCHW storage, got {states.shape}")
    observation = np.asarray(observation)
    if observation.shape != states.shape[1:]:
        raise ValueError(f"Observation shape mismatch: {observation.shape}")
    if seen < len(states):
        slot = seen
    else:
        candidate = int(rng.integers(0, seen + 1))
        if candidate >= len(states):
            return
        slot = candidate
    states[slot] = observation
    record = {"reservoir_slot": slot, **item_metadata}
    if slot < len(metadata):
        metadata[slot] = record
    else:
        metadata.append(record)
