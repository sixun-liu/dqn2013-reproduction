"""Deterministic statistics for the DQN fixed-state offline atlas."""

from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Ranks require finite values")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"Incompatible Spearman shapes: {left.shape}, {right.shape}")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return float("nan")
    correlation = np.dot(left_centered, right_centered) / denominator
    return float(np.clip(correlation, -1.0, 1.0))


def circular_block_bootstrap_spearman(
    left: np.ndarray,
    right: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"Incompatible bootstrap shapes: {left.shape}, {right.shape}")
    if block_length <= 0 or block_length > len(left):
        raise ValueError("block_length must be between 1 and the series length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(seed)
    block_offsets = np.arange(block_length)
    block_count = int(np.ceil(len(left) / block_length))
    estimates = []
    for _ in range(resamples):
        starts = rng.integers(0, len(left), size=block_count)
        indices = ((starts[:, None] + block_offsets) % len(left)).reshape(-1)[: len(left)]
        estimate = spearman_correlation(left[indices], right[indices])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise ValueError("All block-bootstrap correlations were undefined")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return spearman_correlation(left, right), float(low), float(high)


def paired_bootstrap_summary(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) == 0:
        raise ValueError(f"Expected non-empty paired differences, got {differences.shape}")
    if not np.isfinite(differences).all():
        raise ValueError("Paired differences must be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    bootstrap_medians = np.median(differences[indices], axis=1)
    low, high = np.quantile(bootstrap_medians, [0.025, 0.975])
    positive = float(np.mean(differences > 0))
    negative = float(np.mean(differences < 0))
    return {
        "median_difference": float(np.median(differences)),
        "median_ci95_low": float(low),
        "median_ci95_high": float(high),
        "paired_sign_effect": positive - negative,
    }


def bootstrap_binary_mean(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"Expected non-empty binary values, got {values.shape}")
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Binary bootstrap values must contain only zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)
