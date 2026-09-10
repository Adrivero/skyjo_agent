import unittest

from agent.policy import HeuristicPolicy
from interface.session import SkyjoSession


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.opponent = HeuristicPolicy(seed=2)
        self.human = HeuristicPolicy(seed=3)

    def play_current_round(self, session):
        while session.match_active:
            if session.current_agent == session.policy_agent:
                session.step_policy()
            else:
                observation = session.environment.observe(session.human_agent)
                session._step(self.human.choose_action(observation))

    def test_start_round_uses_human_and_policy_opening_choices(self):
        session = SkyjoSession(self.opponent)
        chosen = [(0, 0), (2, 3)]
        session.start_match(chosen, seed=4)
        human_revealed = {
            card.position
            for card in session.player(session.human_agent).hand.grid.flatten()
            if card.state == "revealed"
        }
        policy_revealed = sum(
            card.state == "revealed"
            for card in session.player(session.policy_agent).hand.grid.flatten()
        )
        self.assertEqual(human_revealed, set(chosen))
        self.assertEqual(policy_revealed, 2)
        self.assertEqual(session.match_number, 1)

    def test_human_can_take_discard_and_replace_hidden_card(self):
        session = SkyjoSession(self.opponent)
        session.start_match([(0, 0), (0, 1)], seed=9)
        while session.current_agent != session.human_agent:
            session.step_policy()
        player = session.player(session.human_agent)
        target_index = next(
            index for index, card in enumerate(player.hand.grid.flatten()) if card.state == "hidden"
        )
        old_value = player.hand.grid.flatten()[target_index].value
        session.environment.board.deck.heap[-1] = -2
        session.human_choose_discard()
        self.assertTrue(session.action_is_legal(target_index))
        session.human_take_discard(target_index)
        replacement = player.hand.grid.flatten()[target_index]
        self.assertEqual((replacement.value, replacement.state), (-2, "revealed"))
        self.assertEqual(session.environment.board.deck.top_discard_value(), old_value)

    def test_round_history_and_closer_starts_next_round(self):
        session = SkyjoSession(self.opponent, target_score=1000)
        session.start_match([(0, 0), (0, 1)], seed=12)
        self.play_current_round(session)
        self.assertTrue(session.match_over)
        self.assertFalse(session.series_over)
        self.assertEqual(len(session.match_history), 1)
        closer = session.last_match["closing_agent"]
        totals = dict(session.totals)

        session.start_match([(1, 0), (1, 1)])
        self.assertEqual(session.match_number, 2)
        self.assertEqual(session.environment.starting_agent, closer)
        self.assertEqual(session.totals, totals)

    def test_full_series_completes_and_new_series_resets(self):
        session = SkyjoSession(self.opponent, target_score=1)
        session.start_match([(0, 0), (0, 1)], seed=22)
        self.play_current_round(session)
        self.assertTrue(session.series_over)
        self.assertIn(session.series_winner, session.environment.possible_agents + ["tie"])
        self.assertEqual(len(session.match_history), 1)

        session.new_series()
        self.assertEqual(session.totals, {"player_0": 0, "player_1": 0})
        self.assertEqual(session.match_history, [])
        self.assertFalse(session.series_over)


if __name__ == "__main__":
    unittest.main()
