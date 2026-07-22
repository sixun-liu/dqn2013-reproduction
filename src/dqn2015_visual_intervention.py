"""Modern batch adaptation of Greydanus perturbation saliency for DQN.

The mask, blur, and score definitions follow ``greydanus/visualize_atari``
commit 182492dab59da50aabe254e67e6d51c4f8622400, ``saliency.py`` lines
15--23 and 38--52 (MIT declared in that repository's README/source headers).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_mask(center: tuple[int, int], size: tuple[int, int], radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("Mask radius must be positive")
    row, column = center
    rows, columns = size
    y, x = np.ogrid[-row : rows - row, -column : columns - column]
    disk = x * x + y * y <= 1
    mask = np.zeros(size, dtype=np.float32)
    mask[disk] = 1.0
    mask = gaussian_filter(mask, sigma=radius)
    maximum = float(mask.max())
    if maximum <= 0:
        raise ValueError("Gaussian mask is empty")
    return (mask / maximum).astype(np.float32)


def saliency_grid(
    size: tuple[int, int], stride: int, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 0:
        raise ValueError("Grid stride must be positive")
    centers = np.asarray(
        [(row, column) for row in range(0, size[0], stride) for column in range(0, size[1], stride)],
        dtype=np.int64,
    )
    masks = np.stack(
        [gaussian_mask((int(row), int(column)), size, radius) for row, column in centers]
    )
    return centers, masks


def blur_state(state: np.ndarray, sigma: float) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 3:
        raise ValueError(f"Expected CHW state, got {state.shape}")
    if sigma <= 0:
        raise ValueError("Blur sigma must be positive")
    return gaussian_filter(state, sigma=(0.0, sigma, sigma)).astype(np.float32)


def occlude_with_blur(
    state: np.ndarray, blurred_state: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    blurred_state = np.asarray(blurred_state, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if state.shape != blurred_state.shape or state.ndim != 3:
        raise ValueError(f"State/blur mismatch: {state.shape}, {blurred_state.shape}")
    if masks.ndim != 3 or masks.shape[1:] != state.shape[1:]:
        raise ValueError(f"Mask/state mismatch: {masks.shape}, {state.shape}")
    weights = masks[:, None]
    return state[None] * (1.0 - weights) + blurred_state[None] * weights


def author_q_score(baseline_q: np.ndarray, perturbed_q: np.ndarray) -> np.ndarray:
    baseline_q = np.asarray(baseline_q, dtype=np.float64)
    perturbed_q = np.asarray(perturbed_q, dtype=np.float64)
    if baseline_q.ndim != 1 or perturbed_q.ndim != 2:
        raise ValueError(f"Unexpected Q shapes: {baseline_q.shape}, {perturbed_q.shape}")
    if perturbed_q.shape[1] != len(baseline_q):
        raise ValueError(f"Q action mismatch: {baseline_q.shape}, {perturbed_q.shape}")
    return (0.5 * np.square(perturbed_q - baseline_q[None]).sum(axis=1)).astype(np.float32)


def adjacent_frame_replacements(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 3 or state.shape[0] < 2:
        raise ValueError(f"Expected a multi-frame CHW state, got {state.shape}")
    replacements = np.repeat(state[None], state.shape[0], axis=0)
    for channel in range(state.shape[0]):
        neighbor = channel + 1 if channel < state.shape[0] - 1 else channel - 1
        replacements[channel, channel] = state[neighbor]
    return replacements


def map_statistics(scores: np.ndarray, random_indices: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    random_indices = np.asarray(random_indices, dtype=np.int64)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Saliency scores must be finite and non-empty")
    if random_indices.ndim != 1 or len(random_indices) == 0:
        raise ValueError("Random control indices must be non-empty")
    total = float(scores.sum())
    probabilities = scores / total if total > 0 else np.full_like(scores, 1.0 / len(scores))
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(len(scores)))
    top_count = max(int(np.ceil(0.1 * len(scores))), 1)
    top_indices = np.argpartition(scores, -top_count)[-top_count:]
    top_mean = float(scores[top_indices].mean())
    random_mean = float(scores[random_indices].mean())
    return {
        "saliency_mean": float(scores.mean()),
        "saliency_p95": float(np.quantile(scores, 0.95)),
        "saliency_max": float(scores.max()),
        "saliency_entropy_normalized": entropy,
        "top_decile_concentration": float(scores[top_indices].sum() / total) if total > 0 else 0.1,
        "top_decile_mean": top_mean,
        "random_cell_mean": random_mean,
        "top_to_random_mean_ratio": top_mean / (random_mean + 1e-12),
    }
