import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from agent.environment import (
    ACTION_COUNT,
    DRAW_ACTION,
    DRAWN_CARD_FEATURE,
    N_CARDS,
    TAKE_DISCARD_OFFSET,
    TOP_DISCARD_FEATURE,
)
from agent.policy import choose_action, encode_state
from agent.ppo import POLICY_FORMAT, SharedPPOPolicy


class OpponentPolicy:
    """Load a learned policy and provide a legal heuristic fallback."""

    def __init__(self, path):
        self.path = Path(path)
        self.q_table = None
        self.ppo_policy = None
        self.mode = "loading"
        self.warning = None

    @property
    def display_name(self):
        return "Learned policy" if self.mode == "policy" else "Heuristic fallback"

    def load(self):
        try:
            with self.path.open("rb") as policy_file:
                policy_file.seek(0, 2)
                if policy_file.tell() == 0:
                    raise EOFError("Policy file is empty.")
                policy_file.seek(-1, 2)
                if policy_file.read(1) != b".":
                    raise EOFError("Policy file is incomplete.")
                policy_file.seek(0)
                policy_data = pickle.load(policy_file)
            ppo_policy = self._validate(policy_data)
        except Exception as error:
            self.q_table = None
            self.mode = "fallback"
            self.warning = (
                f"Could not load learned policy ({type(error).__name__}: {error}). "
                "The opponent is using the heuristic fallback."
            )
            return False

        self.q_table = None if ppo_policy is not None else policy_data
        self.ppo_policy = ppo_policy
        self.mode = "policy"
        self.warning = None
        return True

    def _validate(self, policy_data):
        if isinstance(policy_data, dict) and policy_data.get("format") == POLICY_FORMAT:
            policy = SharedPPOPolicy.from_state_dict(policy_data)
            if policy.action_count != ACTION_COUNT:
                raise ValueError(f"PPO policy must have {ACTION_COUNT} actions.")
            return policy

        q_table = policy_data
        if not isinstance(q_table, Mapping) or not q_table:
            raise ValueError("Policy must be a non-empty state-to-Q-values mapping.")

        state, q_values = next(iter(q_table.items()))
        expected_state_size = N_CARDS * 2 + 3
        if not isinstance(state, tuple) or len(state) != expected_state_size:
            raise ValueError(
                f"Policy states must be {expected_state_size}-element tuples."
            )
        if np.asarray(q_values).shape != (ACTION_COUNT,):
            raise ValueError(f"Policy entries must contain {ACTION_COUNT} Q-values.")
        return None

    def choose_action(self, observation):
        if self.ppo_policy is not None:
            return self.ppo_policy.choose_action(observation, deterministic=True)

        if self.q_table is not None:
            state = encode_state(observation)
            q_values = self.q_table.get(state)
            if q_values is not None and np.asarray(q_values).shape == (ACTION_COUNT,):
                return choose_action(q_values, observation["action_mask"])
            self.warning = (
                "The learned policy has no value for the current state; "
                "the heuristic fallback is being used when needed."
            )

        return self._heuristic_action(observation)

    def _heuristic_action(self, observation):
        mask = np.asarray(observation["action_mask"])
        valid_actions = np.flatnonzero(mask)
        if not len(valid_actions):
            raise ValueError("No legal action is available to the opponent.")

        own_features = observation["observation"][0]
        values = own_features[:N_CARDS]

        if mask[DRAW_ACTION]:
            top_discard = int(own_features[TOP_DISCARD_FEATURE])
            replaceable = [
                index
                for index in range(N_CARDS)
                if mask[TAKE_DISCARD_OFFSET + index] and values[index] >= -2
            ]
            if replaceable:
                target = max(replaceable, key=lambda index: values[index])
                if top_discard < values[target]:
                    return TAKE_DISCARD_OFFSET + target
            return DRAW_ACTION

        drawn_card = int(own_features[DRAWN_CARD_FEATURE])
        replaceable = [index for index in range(N_CARDS) if mask[index]]
        visible = [index for index in replaceable if values[index] >= -2]
        if visible:
            target = max(visible, key=lambda index: values[index])
            if drawn_card < values[target]:
                return target

        reveal_actions = [
            N_CARDS + index
            for index in range(N_CARDS)
            if mask[N_CARDS + index]
        ]
        if reveal_actions:
            return reveal_actions[0]
        if visible:
            return max(visible, key=lambda index: values[index])
        return int(valid_actions[0])
