import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_dqn2015_visual_review import render_contact_sheet


class VisualReviewTests(unittest.TestCase):
    def test_contact_sheet_reshapes_flat_author_grid(self):
        states = np.zeros((2, 4, 84, 84), dtype=np.uint8)
        states[0, -1, 20:30, 40:45] = 255
        states[1, -1, 50:60, 10:20] = 180
        state_ids = np.array([11, 29], dtype=np.int64)
        scores = np.arange(2 * 2 * 289, dtype=np.float32).reshape(2, 2, 289)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contact.png"
            render_contact_sheet(states, state_ids, scores, [9_000_000, 9_250_000], output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_contact_sheet_rejects_non_square_grid(self):
        states = np.zeros((2, 4, 84, 84), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                render_contact_sheet(
                    states,
                    np.array([0, 1]),
                    np.zeros((2, 2, 288), dtype=np.float32),
                    [9_000_000, 9_250_000],
                    Path(temporary) / "contact.png",
                )


if __name__ == "__main__":
    unittest.main()
