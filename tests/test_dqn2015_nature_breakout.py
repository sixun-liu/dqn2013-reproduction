import dataclasses
import json
import signal
import unittest
from pathlib import Path

import torch

from src.dqn2015_nature_breakout import (
    Args,
    DeepMindCenteredRMSprop,
    QNetwork,
    StopController,
    checkpoint_payload,
    clipped_td_loss,
    epsilon_at,
    make_atari_env,
    should_sync_target,
    should_train,
    wrapper_type_names,
    runtime_record,
)


class NatureDQNUnitTests(unittest.TestCase):
    def test_committed_configs_expand_all_executor_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        defaults = dataclasses.asdict(Args())
        pilot = json.loads((repo_root / "configs/nature2015_table3_s0_pilot_250k.json").read_text())
        formal = json.loads((repo_root / "configs/nature2015_table3_s0_formal_10m.json").read_text())
        self.assertEqual(set(pilot), set(defaults))
        self.assertEqual(set(formal), set(defaults))
        for key in defaults:
            if key not in {"output_dir", "total_agent_decisions"}:
                self.assertEqual(pilot[key], formal[key], key)
        self.assertEqual(pilot["total_agent_decisions"], 250_000)
        self.assertEqual(formal["total_agent_decisions"], 10_000_000)

    def test_network_shape_and_parameter_count(self):
        network = QNetwork(action_count=4)
        output = network(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(sum(parameter.numel() for parameter in network.parameters()), 1_686_180)
        self.assertTrue(torch.isfinite(output).all())

    def test_epsilon_stays_one_through_warmup_then_anneals(self):
        args = Args()
        self.assertEqual(epsilon_at(args, 0), 1.0)
        self.assertEqual(epsilon_at(args, args.learning_starts_agent_decisions), 1.0)
        midpoint = args.learning_starts_agent_decisions + args.epsilon_decay_agent_decisions // 2
        self.assertAlmostEqual(epsilon_at(args, midpoint), 0.55)
        endpoint = args.learning_starts_agent_decisions + args.epsilon_decay_agent_decisions
        self.assertAlmostEqual(epsilon_at(args, endpoint), 0.1)
        self.assertAlmostEqual(epsilon_at(args, endpoint + 1), 0.1)

    def test_train_and_target_boundaries(self):
        args = Args()
        self.assertFalse(should_train(args, 50_000))
        self.assertFalse(should_train(args, 50_001))
        self.assertTrue(should_train(args, 50_004))
        self.assertFalse(should_sync_target(args, 9_999))
        self.assertTrue(should_sync_target(args, 10_000))
        self.assertFalse(should_sync_target(args, 10_001))

    def test_rmsprop_one_step_matches_direct_formula(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        gradient = torch.tensor([2.0, -4.0])
        parameter.grad = gradient.clone()
        optimizer = DeepMindCenteredRMSprop([parameter])
        optimizer.step()

        gradient_average = 0.05 * gradient
        squared_average = 0.05 * gradient.square()
        denominator = torch.sqrt(squared_average - gradient_average.square() + 0.01)
        expected = torch.tensor([1.0, -2.0]) - 2.5e-4 * gradient / denominator
        torch.testing.assert_close(parameter, expected)

    def test_td_error_gradient_is_clipped_at_one(self):
        predicted = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        target = torch.tensor([10.0, -10.0, 0.5])
        clipped_td_loss(predicted, target).backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([-1.0, 1.0, -0.5]))

    def test_train_and_eval_life_loss_semantics(self):
        args = dataclasses.replace(
            Args(),
            total_agent_decisions=64,
            replay_capacity_transitions=64,
            learning_starts_agent_decisions=32,
            heldout_state_count=16,
        )
        train_env = make_atari_env(args, evaluation=False)
        eval_env = make_atari_env(args, evaluation=True)
        try:
            self.assertIn("EpisodicLifeEnv", wrapper_type_names(train_env))
            self.assertIn("ClipRewardEnv", wrapper_type_names(train_env))
            self.assertNotIn("EpisodicLifeEnv", wrapper_type_names(eval_env))
            self.assertNotIn("ClipRewardEnv", wrapper_type_names(eval_env))
            self.assertEqual(train_env.observation_space.shape, (4, 84, 84))
            self.assertEqual(eval_env.observation_space.shape, (4, 84, 84))
            network = QNetwork(int(train_env.action_space.n))
            record = runtime_record(args, train_env, torch.device("cpu"), 1_686_180)
            json.dumps(record)
            self.assertIsInstance(record["action_count"], int)
        finally:
            train_env.close()
            eval_env.close()

    def test_checkpoint_and_signal_payload(self):
        args = dataclasses.replace(
            Args(),
            total_agent_decisions=64,
            replay_capacity_transitions=64,
            learning_starts_agent_decisions=32,
            heldout_state_count=16,
        )
        network = QNetwork(4)
        target = QNetwork(4)
        optimizer = DeepMindCenteredRMSprop(network.parameters())
        payload = checkpoint_payload(args, network, target, optimizer, 40, 2, "test")
        self.assertEqual(payload["completed_agent_decisions"], 40)
        self.assertEqual(payload["nominal_training_emulator_frames"], 160)
        self.assertIn("online_network", payload)
        self.assertIn("target_network", payload)
        self.assertIn("optimizer", payload)
        self.assertIn("python_random_state", payload)
        self.assertIn("numpy_random_state", payload)
        self.assertIn("torch_random_state", payload)
        self.assertIn("not protocol-equivalent", payload["resume_limitation"])

        controller = StopController()
        controller.request(signal.SIGTERM)
        self.assertTrue(controller.requested)
        self.assertEqual(controller.signal_number, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
