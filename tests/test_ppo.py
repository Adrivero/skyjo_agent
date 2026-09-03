import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.environment import ACTION_COUNT, FEATURE_COUNT, N_AGENTS, env
from agent.ppo import PPOBatch, SharedPPOPolicy, observation_features
from interface.policy import OpponentPolicy


class SharedPPOPolicyTests(unittest.TestCase):
    def setUp(self):
        self.game = env()
        self.game.reset(seed=7)
        self.observation = self.game.observe(self.game.agent_selection)
        self.policy = SharedPPOPolicy(
            N_AGENTS * FEATURE_COUNT, ACTION_COUNT, hidden_size=16, seed=3
        )

    def test_actions_respect_the_mask_for_every_player_view(self):
        for agent in self.game.possible_agents:
            observation = self.game.observe(agent)
            action, log_prob, value = self.policy.act(observation)
            self.assertTrue(observation["action_mask"][action])
            self.assertTrue(np.isfinite(log_prob))
            self.assertTrue(np.isfinite(value))

    def test_update_and_round_trip_preserve_a_shared_policy(self):
        action, log_prob, _ = self.policy.act(self.observation)
        features = observation_features(self.observation)[None, :]
        batch = PPOBatch(
            features=features,
            masks=np.asarray([self.observation["action_mask"]]),
            actions=np.asarray([action]),
            old_log_probs=np.asarray([log_prob]),
            returns=np.asarray([1.0], dtype=np.float32),
            advantages=np.asarray([1.0], dtype=np.float32),
        )
        result = self.policy.update(batch, 3e-4, 0.2, epochs=2, minibatch_size=1)
        self.assertIn("policy_loss", result)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared_ppo.pkl"
            self.policy.save(path)
            loaded = SharedPPOPolicy.load(path)
            self.assertEqual(loaded.input_size, N_AGENTS * FEATURE_COUNT)
            self.assertTrue(loaded.action_distribution(self.observation)[0].sum() == 1.0)

            opponent = OpponentPolicy(path)
            self.assertTrue(opponent.load())
            chosen = opponent.choose_action(self.observation)
            self.assertTrue(self.observation["action_mask"][chosen])


if __name__ == "__main__":
    unittest.main()
