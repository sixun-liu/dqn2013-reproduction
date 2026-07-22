import unittest

import numpy as np

from src.dqn2015_offline_statistics import (
    bootstrap_binary_mean,
    circular_block_bootstrap_spearman,
    paired_bootstrap_summary,
    spearman_correlation,
)


class OfflineStatisticsTests(unittest.TestCase):
    def test_spearman_handles_ties(self):
        left = np.array([1, 1, 2, 3, 3], dtype=float)
        right = np.array([10, 10, 20, 30, 30], dtype=float)
        self.assertAlmostEqual(spearman_correlation(left, right), 1.0)
        self.assertAlmostEqual(spearman_correlation(left, right[::-1]), -1.0)

    def test_block_bootstrap_is_deterministic_and_contains_perfect_rho(self):
        values = np.arange(20, dtype=float)
        first = circular_block_bootstrap_spearman(
            values, values, block_length=5, resamples=100, seed=7
        )
        second = circular_block_bootstrap_spearman(
            values, values, block_length=5, resamples=100, seed=7
        )
        self.assertEqual(first, second)
        for value in first:
            self.assertAlmostEqual(value, 1.0)

    def test_paired_bootstrap_sign_and_interval(self):
        result = paired_bootstrap_summary(
            np.arange(1, 11, dtype=float), resamples=200, seed=11
        )
        self.assertGreater(result["median_ci95_low"], 0)
        self.assertEqual(result["paired_sign_effect"], 1.0)

    def test_binary_bootstrap(self):
        mean, low, high = bootstrap_binary_mean(
            np.array([0, 0, 1, 1], dtype=float), resamples=500, seed=13
        )
        self.assertEqual(mean, 0.5)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)


if __name__ == "__main__":
    unittest.main()
