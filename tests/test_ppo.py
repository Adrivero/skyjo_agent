import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent.environment import ACTION_COUNT, CENTRAL_STATE_SIZE, env
from agent.ppo import (
    ACTOR_INPUT_SIZE,
    PPOBatch,
    PPOTrainer,
    RecurrentActorCritic,
    RecurrentPolicy,
    load_actor_checkpoint,
    observation_features,
    save_actor_checkpoint,
)


class RecurrentMAPPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.game = env()
        self.game.reset(seed=7)
        self.observation = self.game.observe(self.game.agent_selection)

    def test_semantic_features_and_masked_actions(self):
        features = observation_features(self.observation)
        self.assertEqual(features.shape, (ACTOR_INPUT_SIZE,))
        policy = RecurrentPolicy(RecurrentActorCritic(), device="cpu", seed=3)
        for _ in range(10):
            action = policy.choose_action(self.observation)
            self.assertTrue(self.observation["action_mask"][action])

    def test_recurrent_ppo_update_changes_parameters_with_finite_metrics(self):
        model = RecurrentActorCritic()
        trainer = PPOTrainer(model, device="cpu")
        batch_size, length = 2, 3
        masks = np.ones((batch_size, length, ACTION_COUNT), dtype=np.int8)
        batch = PPOBatch(
            features=np.random.default_rng(2).normal(size=(batch_size, length, ACTOR_INPUT_SIZE)).astype(np.float32),
            critic_states=np.random.default_rng(3).normal(size=(batch_size, length, CENTRAL_STATE_SIZE)).astype(np.float32),
            masks=masks,
            actions=np.zeros((batch_size, length), dtype=np.int64),
            old_log_probs=np.full((batch_size, length), -np.log(ACTION_COUNT), dtype=np.float32),
            returns=np.ones((batch_size, length), dtype=np.float32),
            advantages=np.asarray([[1.0, 0.5, -1.0], [-0.5, 1.5, -1.5]], dtype=np.float32),
            initial_hidden=np.zeros((batch_size, model.config.recurrent_size), dtype=np.float32),
            reset_hidden=np.zeros((batch_size, length), dtype=bool),
            valid=np.ones((batch_size, length), dtype=bool),
        )
        before = model.actor_head.weight.detach().clone()
        metrics = trainer.update(batch, epochs=1, minibatch_sequences=2)
        self.assertFalse(torch.equal(before, model.actor_head.weight))
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"):
            self.assertTrue(np.isfinite(metrics[key]))

    def test_checkpoint_round_trip_preserves_actor_outputs(self):
        model = RecurrentActorCritic()
        features = torch.as_tensor(observation_features(self.observation)).unsqueeze(0)
        hidden = model.initial_hidden(1)
        with torch.no_grad():
            expected, _ = model.actor_step(features, hidden)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            save_actor_checkpoint(path, model, step=12, seed=4, rating=1012)
            loaded, metadata = load_actor_checkpoint(path, device="cpu")
            with torch.no_grad():
                actual, _ = loaded.actor_step(features, loaded.initial_hidden(1))
            torch.testing.assert_close(actual, expected)
            self.assertEqual(metadata["step"], 12)
            self.assertTrue(path.with_suffix(".pt.json").is_file())


if __name__ == "__main__":
    unittest.main()
