import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.environment import ACTION_COUNT, env
from agent.policy import encode_state
from interface.policy import OpponentPolicy


class PolicyControllerTests(unittest.TestCase):
    def setUp(self):
        self.game = env()
        self.game.reset(seed=2)
        self.agent = self.game.agent_selection
        self.observation = self.game.observe(self.agent)

    def write_policy(self, path, q_values, state=None):
        state = state or encode_state(self.observation)
        with path.open("wb") as policy_file:
            pickle.dump({state: q_values}, policy_file)

    def test_loaded_policy_chooses_highest_valued_legal_action(self):
        mask = self.observation["action_mask"]
        legal = np.flatnonzero(mask)
        illegal = np.flatnonzero(mask == 0)
        q_values = np.zeros(ACTION_COUNT, dtype=np.float32)
        q_values[legal[-1]] = 5
        q_values[illegal[0]] = 100

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pkl"
            self.write_policy(path, q_values)
            policy = OpponentPolicy(path)
            self.assertTrue(policy.load())
            self.assertEqual(policy.choose_action(self.observation), legal[-1])

    def test_missing_and_corrupt_policies_use_a_legal_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = OpponentPolicy(Path(directory) / "missing.pkl")
            self.assertFalse(missing.load())
            action = missing.choose_action(self.observation)
            self.assertTrue(self.observation["action_mask"][action])
            self.assertIn("heuristic fallback", missing.warning)

            corrupt_path = Path(directory) / "corrupt.pkl"
            corrupt_path.write_bytes(b"\x80\x04unfinished")
            corrupt = OpponentPolicy(corrupt_path)
            self.assertFalse(corrupt.load())
            self.assertTrue(self.observation["action_mask"][corrupt.choose_action(self.observation)])

    def test_unseen_state_uses_fallback_and_sets_warning(self):
        q_values = np.zeros(ACTION_COUNT, dtype=np.float32)
        unseen_state = tuple([0] * 27)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pkl"
            self.write_policy(path, q_values, state=unseen_state)
            policy = OpponentPolicy(path)
            self.assertTrue(policy.load())
            action = policy.choose_action(self.observation)
            self.assertTrue(self.observation["action_mask"][action])
            self.assertIn("no value", policy.warning)


if __name__ == "__main__":
    unittest.main()
