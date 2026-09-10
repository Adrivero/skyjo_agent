"""PettingZoo environment for an official two-player SKYJO series."""

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from game.board import Board


N_ROWS = 3
N_COLS = 4
N_CARDS = N_ROWS * N_COLS
N_AGENTS = 2
TARGET_SCORE = 100

HIDDEN_VALUE = -99
REMOVED_VALUE = -100
NO_CARD_VALUE = -100

# Normal gameplay retains its original 37 action meanings. During the opening
# phase the same Discrete space indexes every unordered pair of board cells.
DRAW_ACTION = N_CARDS * 2
TAKE_DISCARD_ACTION = DRAW_ACTION + 1
GAMEPLAY_ACTION_COUNT = TAKE_DISCARD_ACTION + 1
OPENING_PAIRS = tuple(combinations(range(N_CARDS), 2))
ACTION_COUNT = len(OPENING_PAIRS)

CARD_STATE_OFFSET = N_CARDS
ROUND_POINTS_FEATURE = N_CARDS * 2
POINTS_FEATURE = ROUND_POINTS_FEATURE  # Backwards-compatible alias.
VISIBLE_COUNT_FEATURE = ROUND_POINTS_FEATURE + 1
CLOSING_AGENT_FEATURE = ROUND_POINTS_FEATURE + 2
SERIES_SCORE_FEATURE = ROUND_POINTS_FEATURE + 3
STARTING_AGENT_FEATURE = ROUND_POINTS_FEATURE + 4
TOP_DISCARD_FEATURE = ROUND_POINTS_FEATURE + 5
DRAWN_CARD_FEATURE = ROUND_POINTS_FEATURE + 6
PHASE_FEATURE = ROUND_POINTS_FEATURE + 7
ROUND_NUMBER_FEATURE = ROUND_POINTS_FEATURE + 8
FEATURE_COUNT = ROUND_POINTS_FEATURE + 9

PHASE_OPENING = 0
PHASE_SOURCE = 1
PHASE_DRAWN = 2
PHASE_DISCARD = 3
PROGRESS_REWARD = 0.01
TURN_PENALTY = 0.002

# See ``state`` for the exact centralized-training representation.
CENTRAL_STATE_SIZE = N_AGENTS * N_CARDS * 4 + 15 + 4 + 2 + 1 + 4 + 2 + 2 + 2


def env(render_mode=None, target_score=TARGET_SCORE, max_round_turns=200, max_series_rounds=20):
    return SkyjoEnv(
        render_mode=render_mode,
        target_score=target_score,
        max_round_turns=max_round_turns,
        max_series_rounds=max_series_rounds,
    )


def opening_action(positions):
    """Return the opening action for two distinct ``(row, column)`` cells."""
    indices = tuple(sorted(row * N_COLS + col for row, col in positions))
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError("Exactly two distinct opening positions are required.")
    try:
        return OPENING_PAIRS.index(indices)
    except ValueError as error:
        raise ValueError("Opening positions must be on the 3x4 board.") from error


def opening_positions(action):
    """Decode an opening action to two ``(row, column)`` cells."""
    action = int(action)
    if not 0 <= action < len(OPENING_PAIRS):
        raise ValueError(f"Opening action must be between 0 and {len(OPENING_PAIRS) - 1}.")
    return tuple(divmod(index, N_COLS) for index in OPENING_PAIRS[action])


class SkyjoEnv(AECEnv):
    metadata = {"name": "skyjo_series_v1", "render_modes": ["human", "ansi"]}

    def __init__(
        self,
        render_mode=None,
        target_score=TARGET_SCORE,
        max_round_turns=200,
        max_series_rounds=20,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.target_score = int(target_score)
        self.max_round_turns = int(max_round_turns)
        self.max_series_rounds = int(max_series_rounds)
        self.possible_agents = [f"player_{index}" for index in range(N_AGENTS)]
        self._agent_name_mapping = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-200,
                        high=1000,
                        shape=(N_AGENTS, FEATURE_COUNT),
                        dtype=np.int16,
                    ),
                    "action_mask": spaces.MultiBinary(ACTION_COUNT),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(ACTION_COUNT) for agent in self.possible_agents
        }
        self.state_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(CENTRAL_STATE_SIZE,),
            dtype=np.float32,
        )

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.rng = random.Random(seed)
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.series_scores = {agent: 0 for agent in self.agents}
        self.series_winner = None
        self.round_number = 0
        self.series_turn_count = 0
        self.round_history = []
        self.last_round_info = None
        self.round_complete = False
        self.scores = {}
        self.final_rewards = {}
        self.final_infos = {}
        self._next_round_starter = None
        self._start_round(initial_positions=(options or {}).get("initial_positions"))

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self.round_complete = False
        if action is None:
            raise ValueError("Action cannot be None for an active SKYJO agent.")
        action = int(action)
        mask = self._action_mask(agent)
        if not 0 <= action < ACTION_COUNT or not mask[action]:
            raise ValueError(f"Action {action} is not legal during phase {self.turn_phase}.")

        if self.turn_phase == "choose_opening":
            self.pending_openings[agent] = opening_positions(action)
            self._advance_opening_phase()
            self._accumulate_rewards()
            return

        resolved_before = self._resolved_card_count(agent)
        turn_finished = self._apply_gameplay_action(agent, action)
        if not turn_finished:
            self.agent_selection = agent
            self._accumulate_rewards()
            return

        resolved_after = self._resolved_card_count(agent)
        self.rewards[agent] += (
            PROGRESS_REWARD * max(0, resolved_after - resolved_before) - TURN_PENALTY
        )

        if self.closing_agent is None and self.players[agent].all_cards_revealed():
            self.closing_agent = agent
            self.final_round_agents = [other for other in self.agents if other != agent]

        if agent in self.final_round_agents:
            self.players[agent].reveal_all_cards()
            self.final_round_agents.remove(agent)

        self.turn_count += 1
        self.series_turn_count += 1
        if self.closing_agent is not None and not self.final_round_agents:
            self._finish_round()
        elif self.turn_count >= self.max_round_turns:
            self._finish_round(truncated=True, reason="round_turn_limit")
        else:
            self.agent_selection = self._next_gameplay_agent(agent)
            self.turn_phase = "choose_source"
            self.pending_drawn_card = None

        self._accumulate_rewards()

    def observe(self, agent):
        return {
            "observation": self._build_observation(agent),
            "action_mask": self._action_mask(agent),
        }

    def state(self):
        """Privileged centralized state for the critic; never used by the actor."""
        features = []
        for agent in self.possible_agents:
            for card in self.players[agent].hand.grid.flatten():
                if card is None:
                    value = 0.0
                    card_state = 2
                else:
                    value = (card.value + 2.0) / 14.0
                    card_state = int(card.state == "revealed")
                features.append(value)
                features.extend(float(card_state == state) for state in range(3))

        deck_counts = {value: 0 for value in range(-2, 13)}
        for value in self.board.deck.deck:
            deck_counts[value] += 1
        initial_counts = {-2: 5, -1: 10, 0: 15, **{value: 10 for value in range(1, 13)}}
        features.extend(
            deck_counts[value] / initial_counts[value] for value in range(-2, 13)
        )
        features.extend(self._encoded_optional_card(self.board.deck.top_discard_value()))
        drawn = None if self.pending_drawn_card is None else self.pending_drawn_card.value
        features.extend(self._encoded_optional_card(drawn))
        features.extend(self.series_scores[agent] / 100.0 for agent in self.possible_agents)
        features.append(self.round_number / self.max_series_rounds)
        features.extend(float(self._phase_id() == phase) for phase in range(4))
        features.extend(float(self.agent_selection == agent) for agent in self.possible_agents)
        features.extend(float(self.closing_agent == agent) for agent in self.possible_agents)
        features.extend(float(self.starting_agent == agent) for agent in self.possible_agents)
        state = np.asarray(features, dtype=np.float32)
        if state.shape != (CENTRAL_STATE_SIZE,):
            raise RuntimeError(f"Central state has unexpected shape {state.shape}.")
        return state

    def render(self):
        lines = [
            f"Round {self.round_number} | totals: "
            + ", ".join(f"{agent}={score}" for agent, score in self.series_scores.items())
        ]
        for agent in self.possible_agents:
            player = self.players[agent]
            lines.append(f"{agent}:")
            for row in player.hand.grid:
                cells = []
                for card in row:
                    if card is None:
                        cells.append("  .")
                    elif card.state == "hidden":
                        cells.append("  X")
                    else:
                        cells.append(f"{card.value:3d}")
                lines.append(" ".join(cells))
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def close(self):
        pass

    def _start_round(self, starter=None, initial_positions=None, preserve_rewards=False):
        self.round_number += 1
        self.board = Board(n_players=N_AGENTS, rng=self.rng)
        self.board.reset_game(player_ids=self.possible_agents, shuffle_players=False)
        self.players = {player.id: player for player in self.board.players}
        self.turn_count = 0
        self.closing_agent = None
        self.final_round_agents = []
        self.pending_drawn_card = None
        self.turn_phase = "choose_opening"
        self.starting_agent = starter
        self.pending_openings = {}
        self._opening_order = self.possible_agents.copy()
        if starter is None:
            self.rng.shuffle(self._opening_order)
        else:
            self._opening_order.remove(starter)
            self._opening_order.insert(0, starter)

        for agent, positions in (initial_positions or {}).items():
            if agent not in self.possible_agents:
                raise ValueError(f"Unknown player in initial positions: {agent}.")
            action = opening_action(positions)
            self.pending_openings[agent] = opening_positions(action)

        if not preserve_rewards:
            self.rewards = {agent: 0.0 for agent in self.agents}
        self._advance_opening_phase()

    def _advance_opening_phase(self):
        for agent in self._opening_order:
            if agent not in self.pending_openings:
                self.agent_selection = agent
                return

        self.board.play_first_turn(initial_positions=self.pending_openings)
        if self.starting_agent is None:
            opening_sums = {
                agent: self.players[agent].points for agent in self.possible_agents
            }
            highest = max(opening_sums.values())
            candidates = [agent for agent, score in opening_sums.items() if score == highest]
            self.starting_agent = self.rng.choice(candidates)
        self._play_order = [
            self.starting_agent,
            *[agent for agent in self.possible_agents if agent != self.starting_agent],
        ]
        self.agent_selection = self.starting_agent
        self.turn_phase = "choose_source"

    def _apply_gameplay_action(self, agent, action):
        if self.turn_phase == "choose_source":
            if action == DRAW_ACTION:
                self.pending_drawn_card = self.board.deck.sample_card()
                self.pending_drawn_card.reveal()
                self.turn_phase = "play_drawn"
            else:
                self.pending_drawn_card = self.board.deck.take_top_discarded_card()
                self.turn_phase = "play_discard"
            return False

        position = self._action_to_position(action)
        if self.turn_phase == "play_discard" or action < N_CARDS:
            self.board.play_player_action(
                self.players[agent], "replace", position, self.pending_drawn_card
            )
        elif action < DRAW_ACTION:
            self.board.play_player_action(
                self.players[agent], "discard_reveal", position, self.pending_drawn_card
            )
        else:
            self.board.play_player_action(
                self.players[agent], "take_discard_replace", position
            )
        self.turn_phase = "choose_source"
        self.pending_drawn_card = None
        return True

    def _finish_round(self, truncated=False, reason=None):
        round_scores = self.board.final_scores(self.closing_agent)
        raw_scores = {
            agent: self.players[agent].points for agent in self.possible_agents
        }
        round_rewards = {}
        for agent in self.possible_agents:
            opponent = next(other for other in self.possible_agents if other != agent)
            margin = (round_scores[opponent] - round_scores[agent]) / 100.0
            round_rewards[agent] = 0.25 * float(np.clip(margin, -1.0, 1.0))
            self.series_scores[agent] += round_scores[agent]

        record = {
            "round": self.round_number,
            "scores": dict(round_scores),
            "raw_scores": raw_scores,
            "closer": self.closing_agent,
            "turns": self.turn_count,
            "truncated": truncated,
        }
        self.round_history.append(record)
        self.last_round_info = record
        self.round_complete = True
        for agent, reward in round_rewards.items():
            self.rewards[agent] += reward

        reached_target = any(score >= self.target_score for score in self.series_scores.values())
        if truncated:
            for agent in self.possible_agents:
                self.rewards[agent] -= 1.0
            self._finish_series(truncated=True, reason=reason)
        elif reached_target:
            lowest = min(self.series_scores.values())
            winners = [
                agent for agent, score in self.series_scores.items() if score == lowest
            ]
            self.series_winner = winners[0] if len(winners) == 1 else "tie"
            for agent in self.possible_agents:
                if self.series_winner == "tie":
                    outcome = 0.0
                else:
                    outcome = 1.0 if agent == self.series_winner else -1.0
                self.rewards[agent] += outcome
            self._finish_series(truncated=False)
        elif self.round_number >= self.max_series_rounds:
            for agent in self.possible_agents:
                self.rewards[agent] -= 1.0
            self._finish_series(truncated=True, reason="series_round_limit")
        else:
            self._next_round_starter = self.closing_agent
            self._start_round(starter=self._next_round_starter, preserve_rewards=True)

    def _finish_series(self, truncated, reason=None):
        self.scores = dict(self.series_scores)
        for agent in self.possible_agents:
            self.terminations[agent] = not truncated
            self.truncations[agent] = truncated
            self.infos[agent] = {
                "score": self.series_scores[agent],
                "winner": agent == self.series_winner,
                "series_winner": self.series_winner,
                "rounds": self.round_number,
                "turns": self.series_turn_count,
                "truncation_reason": reason,
            }
        self.final_rewards = self.rewards.copy()
        self.final_infos = {agent: dict(info) for agent, info in self.infos.items()}

    def _next_gameplay_agent(self, current):
        if self.closing_agent is not None:
            return self.final_round_agents[0]
        return next(agent for agent in self._play_order if agent != current)

    def _build_observation(self, observing_agent):
        ordered_agents = [observing_agent] + [
            agent for agent in self.possible_agents if agent != observing_agent
        ]
        return np.stack([self._agent_features(agent) for agent in ordered_agents]).astype(np.int16)

    def _agent_features(self, agent):
        values = []
        states = []
        player = self.players[agent]
        opening_hidden = self.turn_phase == "choose_opening"
        for row in player.hand.grid:
            for card in row:
                if card is None:
                    values.append(REMOVED_VALUE)
                    states.append(2)
                elif opening_hidden or card.state == "hidden":
                    values.append(HIDDEN_VALUE)
                    states.append(0)
                else:
                    values.append(card.value)
                    states.append(1)

        visible_count = sum(state != 0 for state in states)
        return np.asarray(
            values
            + states
            + [
                0 if opening_hidden else player.points,
                visible_count,
                int(agent == self.closing_agent),
                self.series_scores[agent],
                int(agent == self.starting_agent),
                self._top_discard_observation_value(),
                self._drawn_card_observation_value(),
                self._phase_id(),
                self.round_number,
            ],
            dtype=np.int16,
        )

    def _action_mask(self, agent):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if self.turn_phase == "choose_opening":
            if agent not in self.pending_openings:
                mask[:] = 1
            return mask

        player = self.players[agent]
        for index, card in enumerate(player.hand.grid.flatten()):
            if card is None:
                continue
            if self.turn_phase == "play_drawn":
                mask[index] = 1
                if card.state == "hidden":
                    mask[N_CARDS + index] = 1
            elif self.turn_phase == "play_discard":
                mask[index] = 1
        if self.turn_phase == "choose_source":
            mask[DRAW_ACTION] = 1
            if self.board.deck.top_discard_value() is not None:
                mask[TAKE_DISCARD_ACTION] = 1
        return mask

    def _phase_id(self):
        return {
            "choose_opening": PHASE_OPENING,
            "choose_source": PHASE_SOURCE,
            "play_drawn": PHASE_DRAWN,
            "play_discard": PHASE_DISCARD,
        }[self.turn_phase]

    def _resolved_card_count(self, agent):
        return sum(
            card is None or card.state == "revealed"
            for card in self.players[agent].hand.grid.flatten()
        )

    @staticmethod
    def _encoded_optional_card(value):
        return [0.0, 1.0] if value is None else [(value + 2.0) / 14.0, 0.0]

    def _top_discard_observation_value(self):
        value = self.board.deck.top_discard_value()
        return NO_CARD_VALUE if value is None else value

    def _drawn_card_observation_value(self):
        return NO_CARD_VALUE if self.pending_drawn_card is None else self.pending_drawn_card.value

    @staticmethod
    def _action_to_position(action):
        return divmod(action % N_CARDS, N_COLS)
