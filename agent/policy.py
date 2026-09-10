"""Non-learning policies used as training and evaluation anchors."""

from __future__ import annotations

import numpy as np

from agent.environment import (
    CARD_STATE_OFFSET,
    DRAW_ACTION,
    DRAWN_CARD_FEATURE,
    N_CARDS,
    PHASE_FEATURE,
    PHASE_DISCARD,
    PHASE_OPENING,
    PHASE_SOURCE,
    TAKE_DISCARD_ACTION,
    TOP_DISCARD_FEATURE,
    opening_action,
)


class HeuristicPolicy:
    """A legal, terminating baseline that replaces high visible cards first."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def reset_round(self):
        pass

    def choose_action(self, observation, deterministic=True):
        mask = np.asarray(observation["action_mask"], dtype=bool)
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("The action mask does not contain a legal action.")

        own = observation["observation"][0]
        if int(own[PHASE_FEATURE]) == PHASE_OPENING:
            # Before cards are revealed all locations are informationally
            # symmetric; spreading the openings exposes two columns.
            return opening_action(((0, 0), (0, 1)))

        values = own[:N_CARDS]
        states = own[CARD_STATE_OFFSET : CARD_STATE_OFFSET + N_CARDS]
        visible = [index for index in range(N_CARDS) if states[index] == 1]
        hidden = [index for index in range(N_CARDS) if states[index] == 0]

        phase = int(own[PHASE_FEATURE])
        if phase == PHASE_SOURCE:
            top_discard = int(own[TOP_DISCARD_FEATURE])
            target = self._matching_column_target(values, hidden, top_discard)
            if target is None and visible:
                highest = max(visible, key=lambda index: values[index])
                if top_discard < values[highest]:
                    target = highest
            if target is None and hidden and top_discard <= 4:
                target = hidden[0]
            if target is not None and mask[TAKE_DISCARD_ACTION]:
                return TAKE_DISCARD_ACTION
            return DRAW_ACTION

        drawn = int(own[DRAWN_CARD_FEATURE])
        target = self._matching_column_target(values, hidden, drawn)
        if target is None and visible:
            highest = max(visible, key=lambda index: values[index])
            if drawn < values[highest]:
                target = highest
        if target is None and hidden and drawn <= 4:
            target = hidden[0]
        if target is not None and mask[target]:
            return target

        if phase == PHASE_DISCARD:
            if hidden:
                return hidden[0]
            if visible:
                return max(visible, key=lambda index: values[index])
            return int(valid[0])

        reveal_actions = [
            N_CARDS + index for index in hidden if mask[N_CARDS + index]
        ]
        if reveal_actions:
            return reveal_actions[0]
        if visible:
            return max(visible, key=lambda index: values[index])
        return int(valid[0])

    @staticmethod
    def _matching_column_target(values, hidden, candidate):
        for column in range(4):
            column_indices = [column + 4 * row for row in range(3)]
            matching = [index for index in column_indices if values[index] == candidate]
            missing = [index for index in column_indices if index in hidden]
            if len(matching) >= 2 and missing:
                return missing[0]
        return None


class RandomPolicy:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def reset_round(self):
        pass

    def choose_action(self, observation, deterministic=False):
        valid = np.flatnonzero(observation["action_mask"])
        if not len(valid):
            raise ValueError("The action mask does not contain a legal action.")
        return int(self.rng.choice(valid))
