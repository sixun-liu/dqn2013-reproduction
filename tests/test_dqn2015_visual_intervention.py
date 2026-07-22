import unittest

import numpy as np

from src.dqn2015_visual_intervention import (
    adjacent_frame_replacements,
    author_q_score,
    blur_state,
    gaussian_mask,
    map_statistics,
    occlude_with_blur,
    saliency_grid,
)


class VisualInterventionTests(unittest.TestCase):
    def test_author_mask_and_grid(self):
        mask = gaussian_mask((42, 42), (84, 84), 5.0)
        self.assertEqual(mask.shape, (84, 84))
        self.assertEqual(mask.dtype, np.float32)
        self.assertAlmostEqual(float(mask.max()), 1.0)
        centered = mask[22:63, 22:63]
        np.testing.assert_allclose(centered, centered[::-1, ::-1], atol=1e-6)
        centers, masks = saliency_grid((84, 84), 5, 5.0)
        self.assertEqual(centers.shape, (289, 2))
        self.assertEqual(masks.shape, (289, 84, 84))

    def test_occlusion_changes_all_stack_frames(self):
        state = np.zeros((4, 9, 9), dtype=np.float32)
        state[:, 4, 4] = 255
        blurred = blur_state(state, 1.0)
        mask = np.ones((1, 9, 9), dtype=np.float32)
        perturbed = occlude_with_blur(state, blurred, mask)
        self.assertEqual(perturbed.shape, (1, 4, 9, 9))
        self.assertTrue((perturbed[0, :, 4, 4] < 255).all())

    def test_author_q_score(self):
        baseline = np.array([1.0, 2.0])
        perturbed = np.array([[0.0, 2.0], [1.0, 4.0]])
        np.testing.assert_allclose(author_q_score(baseline, perturbed), [0.5, 2.0])

    def test_adjacent_frame_replacements(self):
        state = np.stack([np.full((2, 2), value) for value in range(4)])
        replacements = adjacent_frame_replacements(state)
        self.assertTrue((replacements[0, 0] == 1).all())
        self.assertTrue((replacements[3, 3] == 2).all())
        self.assertTrue((replacements[1, 0] == 0).all())

    def test_map_statistics(self):
        scores = np.arange(1, 101, dtype=float)
        result = map_statistics(scores, np.arange(10))
        self.assertGreater(result["top_decile_concentration"], 0.1)
        self.assertGreater(result["top_to_random_mean_ratio"], 1.0)
        self.assertGreaterEqual(result["saliency_entropy_normalized"], 0.0)
        self.assertLessEqual(result["saliency_entropy_normalized"], 1.0)


if __name__ == "__main__":
    unittest.main()
