import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dqn2015_nature_breakout import QNetwork
from src.dqn2015_offline_analysis import (
    centered_linear_cka,
    discover_evaluation_checkpoints,
    extract_q_and_features,
    representation_spectrum,
    summarize_q_values,
)


class OfflineAnalysisTests(unittest.TestCase):
    def test_feature_path_reconstructs_network_q(self):
        network = QNetwork(4)
        states = np.random.default_rng(7).integers(
            0, 256, size=(5, 4, 84, 84), dtype=np.uint8
        )
        q_values, features, error = extract_q_and_features(
            network, states, torch.device("cpu"), batch_states=2
        )
        self.assertEqual(q_values.shape, (5, 4))
        self.assertEqual(features.shape, (5, 512))
        self.assertLessEqual(error, 1e-6)

    def test_linear_cka_identity_and_orthogonal_noise(self):
        rng = np.random.default_rng(11)
        features = rng.normal(size=(64, 12))
        self.assertAlmostEqual(centered_linear_cka(features, features), 1.0, places=10)
        noisy = rng.normal(size=(64, 12))
        self.assertLess(centered_linear_cka(features, noisy), 0.5)

    def test_representation_spectrum_rank_one(self):
        base = np.arange(20, dtype=np.float64)[:, None]
        features = base @ np.ones((1, 8), dtype=np.float64)
        result = representation_spectrum(features)
        self.assertAlmostEqual(result["effective_rank"], 1.0, places=8)
        self.assertAlmostEqual(result["participation_ratio"], 1.0, places=8)

    def test_q_summary_action_fractions_sum_to_one(self):
        q_values = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]], dtype=float)
        result = summarize_q_values(q_values)
        total = sum(result[f"greedy_action_{i}_fraction"] for i in range(4))
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(result["max_q_mean"], 2.0)

    def test_checkpoint_selection_prefers_complete_at_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt"
            final_eval = root / "final-eval.pt"
            final_complete = root / "final-complete.pt"
            for path in (first, final_eval, final_complete):
                path.write_bytes(b"checkpoint")
            records = [
                {"completed_agent_decisions": 10, "optimizer_updates": 1, "reason": "evaluation", "path": str(first)},
                {"completed_agent_decisions": 20, "optimizer_updates": 2, "reason": "evaluation", "path": str(final_eval)},
                {"completed_agent_decisions": 20, "optimizer_updates": 2, "reason": "complete", "path": str(final_complete)},
            ]
            index = root / "checkpoints.jsonl"
            index.write_text("".join(json.dumps(item) + "\n" for item in records))
            selected = discover_evaluation_checkpoints(
                index, interval_agent_decisions=10, total_agent_decisions=20
            )
            self.assertEqual([item.path for item in selected], [first, final_complete])


if __name__ == "__main__":
    unittest.main()
