import random

import numpy as np

from agent.environment import (
    ACTION_COUNT,
    DRAWN_CARD_FEATURE,
    N_CARDS,
    TOP_DISCARD_FEATURE,
    VISIBLE_COUNT_FEATURE,
)


def encode_state(observation):
    """Encode the player-relative observation used by the tabular policy."""
    own_features = observation["observation"][0]
    values = own_features[:N_CARDS]
    revealed = own_features[N_CARDS : N_CARDS * 2]
    visible_count = own_features[VISIBLE_COUNT_FEATURE]
    top_discard = own_features[TOP_DISCARD_FEATURE]
    drawn_card = own_features[DRAWN_CARD_FEATURE]

    return tuple(
        values.tolist()
        + revealed.tolist()
        + [int(visible_count), int(top_discard), int(drawn_card)]
    )


def choose_action(q_values, action_mask, epsilon=0.0):
    """Choose a valid epsilon-greedy action from a Q-value vector."""
    q_values = np.asarray(q_values)
    action_mask = np.asarray(action_mask)
    if q_values.shape != (ACTION_COUNT,):
        raise ValueError(f"Expected {ACTION_COUNT} Q-values, got {q_values.shape}.")

    valid_actions = np.flatnonzero(action_mask)
    if not len(valid_actions):
        raise ValueError("The action mask does not contain a valid action.")
    if random.random() < epsilon:
        return int(np.random.choice(valid_actions))

    masked_values = np.full(q_values.shape, -np.inf, dtype=np.float64)
    masked_values[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_values))
