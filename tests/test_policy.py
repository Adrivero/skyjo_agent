import tempfile
import unittest
from pathlib import Path

from agent.environment import TAKE_DISCARD_ACTION, env, opening_action
from agent.policy import HeuristicPolicy
from agent.ppo import RecurrentActorCritic, save_actor_checkpoint
from interface.policy import OpponentPolicy


def complete_opening(game):
    actions = {
        "player_0": opening_action(((0, 0), (0, 1))),
        "player_1": opening_action(((1, 0), (1, 1))),
    }
    while game.turn_phase == "choose_opening":
        game.step(actions[game.agent_selection])


class PolicyControllerTests(unittest.TestCase):
    def test_heuristic_opening_is_legal_and_spans_columns(self):
        game = env()
        game.reset(seed=1)
        observation = game.observe(game.agent_selection)
        action = HeuristicPolicy().choose_action(observation)
        self.assertTrue(observation["action_mask"][action])
        self.assertEqual(action, opening_action(((0, 0), (0, 1))))

    def test_heuristic_takes_low_discard_for_high_visible_card(self):
        game = env()
        game.reset(seed=2)
        complete_opening(game)
        agent = game.agent_selection
        cards = game.players[agent].hand.grid.flatten()
        visible = [index for index, card in enumerate(cards) if card.state == "revealed"]
        cards[visible[0]].value = 12
        cards[visible[1]].value = 3
        game.board.deck.heap[-1] = -2
        action = HeuristicPolicy().choose_action(game.observe(agent))
        self.assertEqual(action, TAKE_DISCARD_ACTION)
        game.step(action)
        self.assertEqual(HeuristicPolicy().choose_action(game.observe(agent)), visible[0])

    def test_missing_and_corrupt_checkpoints_use_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = OpponentPolicy(Path(directory) / "missing.pt", device="cpu")
            self.assertFalse(missing.load())
            game = env()
            game.reset(seed=4)
            observation = game.observe(game.agent_selection)
            self.assertTrue(observation["action_mask"][missing.choose_action(observation)])

            corrupt_path = Path(directory) / "corrupt.pt"
            corrupt_path.write_bytes(b"not a checkpoint")
            corrupt = OpponentPolicy(corrupt_path, device="cpu")
            self.assertFalse(corrupt.load())
            self.assertIn("heuristic fallback", corrupt.warning)

    def test_torch_checkpoint_loads_and_seeded_sampling_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            save_actor_checkpoint(path, RecurrentActorCritic(), step=10, seed=9)
            first = OpponentPolicy(path, seed=22, device="cpu")
            second = OpponentPolicy(path, seed=22, device="cpu")
            self.assertTrue(first.load())
            self.assertTrue(second.load())
            game = env()
            game.reset(seed=8)
            observation = game.observe(game.agent_selection)
            self.assertEqual(first.choose_action(observation), second.choose_action(observation))
            self.assertEqual(first.display_name, "Learned MARL policy")


if __name__ == "__main__":
    unittest.main()
