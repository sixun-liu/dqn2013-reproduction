import unittest

import numpy as np

from src.dqn2015_calibration import (
    bootstrap_interval,
    discounted_returns,
    paired_bootstrap_difference,
    update_reservoir,
)


class CalibrationTests(unittest.TestCase):
    def test_discounted_returns_stop_at_life_boundary(self):
        rewards = np.array([1.0, 0.0, 2.0])
        result = discounted_returns(rewards, 0.5)
        np.testing.assert_allclose(result, [1.5, 1.0, 2.0])

    def test_bootstrap_interval_is_deterministic(self):
        values = np.arange(1, 11, dtype=float)
        first = bootstrap_interval(
            values, estimator=np.mean, resamples=500, seed=7
        )
        second = bootstrap_interval(
            values, estimator=np.mean, resamples=500, seed=7
        )
        self.assertEqual(first, second)
        self.assertLess(first[1], first[0])
        self.assertGreater(first[2], first[0])

    def test_paired_bootstrap_uses_within_game_difference(self):
        left = np.array([2.0, 3.0, 4.0])
        right = np.array([1.0, 2.0, 3.0])
        result = paired_bootstrap_difference(
            left, right, estimator=np.mean, resamples=100, seed=11
        )
        self.assertEqual(result, (1.0, 1.0, 1.0))

    def test_reservoir_fills_before_replacement(self):
        states = np.zeros((2, 4, 3, 3), dtype=np.uint8)
        metadata = []
        rng = np.random.default_rng(13)
        for seen, value in enumerate((1, 2)):
            update_reservoir(
                states,
                metadata,
                np.full((4, 3, 3), value, dtype=np.uint8),
                {"stage_step_index": seen},
                seen=seen,
                rng=rng,
            )
        self.assertEqual([item["reservoir_slot"] for item in metadata], [0, 1])
        self.assertTrue((states[0] == 1).all())
        self.assertTrue((states[1] == 2).all())


if __name__ == "__main__":
    unittest.main()
