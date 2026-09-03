import unittest

import numpy as np

from agent.environment import N_CARDS
from interface.session import SkyjoSession


class FirstLegalPolicy:
    warning = None

    def choose_action(self, observation):
        return int(np.flatnonzero(observation["action_mask"])[0])


class FinishedEnvironment:
    possible_agents = ["player_0", "player_1"]

    def __init__(self, scores, raw_scores=None, closer="player_0"):
        self.scores = scores
        raw_scores = raw_scores or scores
        self.final_infos = {
            agent: {"raw_score": raw_scores[agent]} for agent in self.possible_agents
        }
        self.closing_agent = closer


class SessionTests(unittest.TestCase):
    def test_start_match_uses_human_opening_positions(self):
        session = SkyjoSession(FirstLegalPolicy())
        session.start_match([(0, 0), (2, 3)], seed=4)
        revealed = {
            card.position
            for card in session.player(session.human_agent).hand.grid.flatten()
            if card.state == "revealed"
        }
        self.assertEqual(revealed, {(0, 0), (2, 3)})
        self.assertEqual(session.match_number, 1)

    def test_totals_end_series_and_new_series_resets_everything(self):
        session = SkyjoSession(FirstLegalPolicy())
        session.match_number = 1
        session.environment = FinishedEnvironment(
            {"player_0": 40, "player_1": 60}
        )
        session._record_match_if_finished()

        session.match_number = 2
        session._match_recorded = False
        session.environment = FinishedEnvironment(
            {"player_0": 35, "player_1": 45}, closer="player_1"
        )
        session._record_match_if_finished()

        self.assertEqual(session.totals, {"player_0": 75, "player_1": 105})
        self.assertTrue(session.series_over)
        self.assertEqual(session.series_winner, "player_0")
        self.assertEqual(len(session.match_history), 2)

        session.new_series()
        self.assertEqual(session.totals, {"player_0": 0, "player_1": 0})
        self.assertEqual(session.match_history, [])
        self.assertFalse(session.series_over)

    def test_both_crossing_threshold_can_end_in_a_tie(self):
        session = SkyjoSession(FirstLegalPolicy())
        session.match_number = 1
        session.environment = FinishedEnvironment(
            {"player_0": 100, "player_1": 100}
        )
        session._record_match_if_finished()
        self.assertTrue(session.series_over)
        self.assertEqual(session.series_winner, "tie")

    def test_human_and_fallback_policy_can_complete_a_match(self):
        session = SkyjoSession(FirstLegalPolicy())
        session.start_match([(0, 0), (0, 1)], seed=9)

        steps = 0
        while session.match_active and steps < 500:
            if session.current_agent == session.policy_agent:
                session.step_policy()
            elif session.turn_phase == "choose_source":
                session.human_draw()
            else:
                mask = session.environment.observe(session.human_agent)["action_mask"]
                action = int(np.flatnonzero(mask)[0])
                if action < N_CARDS:
                    session.human_replace_with_drawn(action)
                else:
                    session.human_discard_and_reveal(action - N_CARDS)
            steps += 1

        self.assertTrue(session.match_over)
        self.assertEqual(len(session.match_history), 1)
        self.assertEqual(
            set(session.last_match["scores"]),
            {session.human_agent, session.policy_agent},
        )


if __name__ == "__main__":
    unittest.main()
