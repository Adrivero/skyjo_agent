import unittest

import numpy as np

from agent.environment import (
    ACTION_COUNT,
    CENTRAL_STATE_SIZE,
    DRAW_ACTION,
    FEATURE_COUNT,
    HIDDEN_VALUE,
    N_AGENTS,
    N_CARDS,
    OPENING_PAIRS,
    TAKE_DISCARD_ACTION,
    env,
    opening_action,
    opening_positions,
)


OPENINGS = {
    "player_0": ((0, 0), (0, 1)),
    "player_1": ((1, 0), (1, 1)),
}


def complete_opening(game):
    while game.turn_phase == "choose_opening":
        agent = game.agent_selection
        game.step(opening_action(OPENINGS[agent]))


class EnvironmentTests(unittest.TestCase):
    def test_openings_are_simultaneous_and_highest_sum_starts(self):
        game = env()
        game.reset(seed=3)
        for agent, values in {"player_0": (12, 11), "player_1": (0, 1)}.items():
            for position, value in zip(OPENINGS[agent], values):
                game.players[agent].hand.grid[position].value = value

        first = game.agent_selection
        game.step(opening_action(OPENINGS[first]))
        for player in game.players.values():
            self.assertTrue(all(card.state == "hidden" for card in player.hand.grid.flatten()))

        second = game.agent_selection
        game.step(opening_action(OPENINGS[second]))
        self.assertEqual(game.turn_phase, "choose_source")
        self.assertEqual(game.starting_agent, "player_0")
        self.assertEqual(game.agent_selection, "player_0")
        for agent in game.possible_agents:
            revealed = {
                card.position
                for card in game.players[agent].hand.grid.flatten()
                if card.state == "revealed"
            }
            self.assertEqual(revealed, set(OPENINGS[agent]))

    def test_opening_action_round_trip_and_mask(self):
        positions = ((0, 0), (2, 3))
        action = opening_action(positions)
        self.assertEqual(set(opening_positions(action)), set(positions))
        self.assertEqual(len(OPENING_PAIRS), 66)

        game = env()
        game.reset(seed=2)
        mask = game.observe(game.agent_selection)["action_mask"]
        self.assertTrue(np.all(mask == 1))

    def test_action_masks_follow_gameplay_phases(self):
        game = env()
        game.reset(seed=8)
        complete_opening(game)
        agent = game.agent_selection
        choose_mask = game.observe(agent)["action_mask"]
        self.assertEqual(choose_mask[DRAW_ACTION], 1)
        self.assertTrue(np.all(choose_mask[:N_CARDS] == 0))
        self.assertEqual(np.flatnonzero(choose_mask).tolist(), [DRAW_ACTION, TAKE_DISCARD_ACTION])

        game.step(DRAW_ACTION)
        play_mask = game.observe(agent)["action_mask"]
        self.assertEqual(game.agent_selection, agent)
        self.assertEqual(game.turn_phase, "play_drawn")
        self.assertEqual(play_mask[DRAW_ACTION], 0)
        self.assertTrue(np.any(play_mask[:N_CARDS] == 1))

    def test_observation_and_central_state_contracts(self):
        game = env()
        game.reset(seed=10)
        observation = game.observe(game.agent_selection)
        self.assertEqual(observation["observation"].shape, (N_AGENTS, FEATURE_COUNT))
        self.assertEqual(observation["action_mask"].shape, (ACTION_COUNT,))
        self.assertEqual(game.state().shape, (CENTRAL_STATE_SIZE,))
        self.assertTrue(np.all(observation["observation"][:, :N_CARDS] == HIDDEN_VALUE))
        self.assertNotEqual(float(game.state()[0]), HIDDEN_VALUE)

    def test_discard_swap_supports_revealed_and_hidden_cells(self):
        game = env()
        game.reset(seed=12)
        complete_opening(game)
        for target_state in ("hidden", "revealed"):
            agent = game.agent_selection
            cards = game.players[agent].hand.grid.flatten()
            target_index = next(index for index, card in enumerate(cards) if card.state == target_state)
            old_value = cards[target_index].value
            game.board.deck.heap[-1] = -2
            game.step(TAKE_DISCARD_ACTION)
            mask = game.observe(agent)["action_mask"]
            self.assertTrue(mask[target_index])
            game.step(target_index)
            replacement = game.players[agent].hand.grid.flatten()[target_index]
            self.assertEqual((replacement.value, replacement.state), (-2, "revealed"))
            self.assertEqual(game.board.deck.top_discard_value(), old_value)

    def test_round_closer_starts_the_next_round(self):
        game = env(target_score=1000)
        game.reset(seed=20)
        complete_opening(game)
        closer = game.agent_selection
        for card in game.players[closer].hand.grid.flatten():
            card.reveal()
        game.step(TAKE_DISCARD_ACTION)
        game.step(0)
        final_agent = game.agent_selection
        game.step(TAKE_DISCARD_ACTION)
        game.step(0)

        self.assertTrue(game.round_complete)
        self.assertEqual(game.round_number, 2)
        self.assertEqual(game.turn_phase, "choose_opening")
        self.assertEqual(game.starting_agent, closer)
        self.assertEqual(game.agent_selection, closer)
        self.assertEqual(game.last_round_info["closer"], closer)
        self.assertNotEqual(final_agent, closer)

    def test_turn_limit_is_an_explicit_training_failure(self):
        game = env(max_round_turns=1)
        game.reset(seed=30)
        complete_opening(game)
        game.step(TAKE_DISCARD_ACTION)
        game.step(0)
        self.assertTrue(all(game.truncations.values()))
        self.assertEqual(game.final_infos["player_0"]["truncation_reason"], "round_turn_limit")
        self.assertTrue(all(reward < 0 for reward in game.final_rewards.values()))

    def test_illegal_actions_raise_instead_of_being_randomly_replaced(self):
        game = env()
        game.reset(seed=4)
        with self.assertRaises(ValueError):
            game.step(ACTION_COUNT)


if __name__ == "__main__":
    unittest.main()
