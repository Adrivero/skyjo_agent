import random
from functools import lru_cache

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from game.board import Board


N_ROWS = 3
N_COLS = 4
N_CARDS = N_ROWS * N_COLS
N_AGENTS = 2
HIDDEN_VALUE = -99
REMOVED_VALUE = -100
NO_CARD_VALUE = -100
DRAW_ACTION = N_CARDS * 2
TAKE_DISCARD_OFFSET = DRAW_ACTION + 1
ACTION_COUNT = TAKE_DISCARD_OFFSET + N_CARDS
POINTS_FEATURE = N_CARDS * 2
VISIBLE_COUNT_FEATURE = POINTS_FEATURE + 1
CLOSING_AGENT_FEATURE = POINTS_FEATURE + 2
TOP_DISCARD_FEATURE = POINTS_FEATURE + 3
DRAWN_CARD_FEATURE = POINTS_FEATURE + 4
FEATURE_COUNT = DRAWN_CARD_FEATURE + 1


def env(render_mode=None):
    return SkyjoEnv(render_mode=render_mode)


class SkyjoEnv(AECEnv):
    metadata = {"name": "skyjo_v0", "render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None, max_turns=200):
        super().__init__()
        self.render_mode = render_mode
        self.max_turns = max_turns
        self.possible_agents = [f"player_{i}" for i in range(N_AGENTS)]
        self._agent_name_mapping = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-100,
                        high=200,
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

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.board = Board(n_players=N_AGENTS)
        self.board.reset_game(
            player_ids=self.possible_agents,
            shuffle_players=True,
        )
        initial_positions = (options or {}).get("initial_positions")
        self.board.play_first_turn(initial_positions=initial_positions)

        self.agents = [player.id for player in self.board.players]
        self.players = {
            player.id: player for player in self.board.players
        }
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.turn_count = 0
        self.closing_agent = None
        self.final_round_agents = []
        self.scores = {}
        self.final_rewards = {}
        self.final_infos = {}
        self.turn_phase = "choose_source"
        self.pending_drawn_card = None

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()

        if action is None:
            raise ValueError("Action cannot be None for an active Skyjo agent.")

        action = int(action)
        turn_finished = self._apply_action(agent, action)

        if not turn_finished:
            self.agent_selection = agent
            self._accumulate_rewards()
            return

        if self.closing_agent is None and self.players[agent].all_cards_revealed():
            self.closing_agent = agent
            self.final_round_agents = [
                other_agent for other_agent in self.agents if other_agent != agent
            ]

        if agent in self.final_round_agents:
            self.players[agent].reveal_all_cards()
            self.final_round_agents.remove(agent)

        self.turn_count += 1
        if self.closing_agent is not None and not self.final_round_agents:
            self._finish_game()
        elif self.turn_count >= self.max_turns:
            self._finish_game(truncated=True)
        else:
            self.agent_selection = self._next_agent()
            self.turn_phase = "choose_source"
            self.pending_drawn_card = None

        self._accumulate_rewards()

    def observe(self, agent):
        return {
            "observation": self._build_observation(agent),
            "action_mask": self._action_mask(agent),
        }

    def state(self):
        return np.concatenate(
            [self._agent_features(agent) for agent in self.possible_agents]
        ).astype(np.int16)

    def render(self):
        lines = []
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

    def _apply_action(self, agent, action):
        mask = self._action_mask(agent)
        if action < 0 or action >= len(mask) or mask[action] == 0:
            valid_actions = np.flatnonzero(mask)
            action = int(np.random.choice(valid_actions))

        if self.turn_phase == "choose_source" and action == DRAW_ACTION:
            self.pending_drawn_card = self.board.deck.sample_card()
            self.pending_drawn_card.reveal()
            self.turn_phase = "play_drawn"
            return False

        position = self._action_to_position(action)

        if action < N_CARDS:
            self.board.play_player_action(
                self.players[agent],
                action_type="replace",
                position=position,
                drawn_card=self.pending_drawn_card,
            )
        elif action < DRAW_ACTION:
            self.board.play_player_action(
                self.players[agent],
                action_type="discard_reveal",
                position=position,
                drawn_card=self.pending_drawn_card,
            )
        else:
            self.board.play_player_action(
                self.players[agent],
                action_type="take_discard_replace",
                position=position,
            )
        self.turn_phase = "choose_source"
        self.pending_drawn_card = None
        return True

    def _finish_game(self, truncated=False):
        self.scores = self.board.final_scores(self.closing_agent)
        raw_scores = {
            agent: self.players[agent].points for agent in self.possible_agents
        }

        winning_score = min(self.scores.values())
        winners = [
            agent for agent, score in self.scores.items() if score == winning_score
        ]

        for agent in self.possible_agents:
            winner_bonus = 1.0 if agent in winners else 0.0
            self.rewards[agent] = winner_bonus - (self.scores[agent] / 100.0)
            self.terminations[agent] = not truncated
            self.truncations[agent] = truncated
            self.infos[agent] = {
                "score": self.scores[agent],
                "raw_score": raw_scores[agent],
                "winner": agent in winners,
                "closing_agent": self.closing_agent,
            }

        self.final_rewards = self.rewards.copy()
        self.final_infos = self.infos.copy()

    def _next_agent(self):
        next_agent = self._agent_selector.next()
        if self.closing_agent is None:
            return next_agent

        while next_agent not in self.final_round_agents:
            next_agent = self._agent_selector.next()
        return next_agent

    def _build_observation(self, observing_agent):
        ordered_agents = [observing_agent] + [
            agent for agent in self.possible_agents if agent != observing_agent
        ]
        return np.stack(
            [self._agent_features(agent) for agent in ordered_agents]
        ).astype(np.int16)

    def _agent_features(self, agent):
        player = self.players[agent]
        values = []
        revealed = []

        for row in player.hand.grid:
            for card in row:
                if card is None:
                    values.append(REMOVED_VALUE)
                    revealed.append(1)
                elif card.state == "revealed":
                    values.append(card.value)
                    revealed.append(1)
                else:
                    values.append(HIDDEN_VALUE)
                    revealed.append(0)

        return np.array(
            values
            + revealed
            + [
                player.points,
                sum(revealed),
                int(agent == self.closing_agent),
                self._top_discard_observation_value(),
                self._drawn_card_observation_value(),
            ],
            dtype=np.int16,
        )

    def _action_mask(self, agent):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        player = self.players[agent]

        for index, card in enumerate(player.hand.grid.flatten()):
            if card is None:
                continue

            if self.turn_phase == "play_drawn":
                mask[index] = 1
                if card.state == "hidden":
                    mask[N_CARDS + index] = 1
            else:
                if self.board.deck.top_discard_value() is not None:
                    mask[TAKE_DISCARD_OFFSET + index] = 1

        if not mask.any():
            if self.turn_phase == "play_drawn":
                mask[:N_CARDS] = 1
            else:
                mask[DRAW_ACTION] = 1

        if self.turn_phase == "choose_source":
            mask[DRAW_ACTION] = 1

        return mask

    def _action_to_position(self, action):
        if action >= TAKE_DISCARD_OFFSET:
            return self._index_to_position(action - TAKE_DISCARD_OFFSET)
        return self._index_to_position(action % N_CARDS)

    def _top_discard_observation_value(self):
        value = self.board.deck.top_discard_value()
        if value is None:
            return NO_CARD_VALUE
        return value

    def _drawn_card_observation_value(self):
        if self.pending_drawn_card is None:
            return NO_CARD_VALUE
        return self.pending_drawn_card.value

    def _index_to_position(self, index):
        return divmod(index, N_COLS)
