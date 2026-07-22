import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_nature2015_config import load_config
from scripts.verify_run import verify
from src.dqn2015_nature_breakout import Args


class ReleaseReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_public_full_config_matches_frozen_formal_algorithm(self):
        frozen = json.loads((self.repo_root / "configs/nature2015_table3_s0_formal_10m.json").read_text())
        public = json.loads((self.repo_root / "configs/public/nature2015_table3_10m.json").read_text())
        self.assertEqual(set(frozen), set(public))
        for key in frozen:
            if key != "output_dir":
                self.assertEqual(frozen[key], public[key], key)
        self.assertFalse(Path(public["output_dir"]).is_absolute())

    def test_json_launcher_requires_complete_expanded_config(self):
        smoke_path = self.repo_root / "configs/public/nature2015_smoke_cpu.json"
        args = load_config(smoke_path)
        self.assertEqual(set(json.loads(smoke_path.read_text())), {field.name for field in dataclasses.fields(Args)})
        self.assertFalse(args.cuda)
        self.assertEqual(args.total_agent_decisions, 256)

        with tempfile.TemporaryDirectory() as directory:
            incomplete = Path(directory) / "incomplete.json"
            incomplete.write_text('{"config_schema_version": 1}\n')
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_config(incomplete)

    def test_verify_rejects_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing run files"):
                verify(Path(directory), "smoke")


if __name__ == "__main__":
    unittest.main()

