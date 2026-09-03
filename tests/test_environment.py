import unittest

import numpy as np

from agent.environment import DRAW_ACTION, N_CARDS, TAKE_DISCARD_OFFSET, env


class EnvironmentTests(unittest.TestCase):
    def test_custom_and_random_opening_positions(self):
        game = env()
        chosen = [(0, 0), (2, 3)]
        game.reset(seed=3, options={"initial_positions": {"player_0": chosen}})

        human_revealed = {
            card.position
            for card in game.players["player_0"].hand.grid.flatten()
            if card.state == "revealed"
        }
        policy_revealed = sum(
            card.state == "revealed"
            for card in game.players["player_1"].hand.grid.flatten()
        )
        self.assertEqual(human_revealed, set(chosen))
        self.assertEqual(policy_revealed, 2)

        game.reset(seed=4)
        for player in game.players.values():
            self.assertEqual(
                sum(card.state == "revealed" for card in player.hand.grid.flatten()),
                2,
            )

    def test_invalid_custom_openings_are_rejected(self):
        game = env()
        with self.assertRaises(ValueError):
            game.reset(
                options={"initial_positions": {"player_0": [(0, 0), (0, 0)]}}
            )

    def test_action_masks_follow_the_two_turn_phases(self):
        game = env()
        game.reset(seed=8)
        agent = game.agent_selection
        choose_mask = game.observe(agent)["action_mask"]
        self.assertEqual(choose_mask[DRAW_ACTION], 1)
        self.assertTrue(np.all(choose_mask[:N_CARDS] == 0))
        self.assertTrue(np.any(choose_mask[TAKE_DISCARD_OFFSET:] == 1))

        game.step(DRAW_ACTION)
        play_mask = game.observe(agent)["action_mask"]
        self.assertEqual(game.agent_selection, agent)
        self.assertEqual(game.turn_phase, "play_drawn")
        self.assertEqual(play_mask[DRAW_ACTION], 0)
        self.assertTrue(np.any(play_mask[:N_CARDS] == 1))


if __name__ == "__main__":
    unittest.main()
