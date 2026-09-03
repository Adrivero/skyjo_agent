"""A small NumPy implementation of a masked, shared-policy PPO agent.

The environment is turn based, but every player observes the board from their
own perspective. Consequently one actor-critic can be used for every player:
there is no player id in the model input and all player decisions contribute to
the same PPO update.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POLICY_FORMAT = "skyjo_shared_ppo_v1"


def observation_features(observation):
    """Flatten the player-relative observation to a stable float input vector."""
    return np.asarray(observation["observation"], dtype=np.float32).reshape(-1) / 100.0


@dataclass
class PPOBatch:
    features: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class SharedPPOPolicy:
    """Two-head MLP actor-critic shared by all Skyjo players."""

    def __init__(self, input_size, action_count, hidden_size=128, seed=None):
        self.input_size = int(input_size)
        self.action_count = int(action_count)
        self.hidden_size = int(hidden_size)
        self.rng = np.random.default_rng(seed)
        scale_1 = np.sqrt(2.0 / self.input_size)
        self.params = {
            "w1": self.rng.normal(0, scale_1, (self.input_size, self.hidden_size)).astype(np.float32),
            "b1": np.zeros(self.hidden_size, dtype=np.float32),
            "w_actor": self.rng.normal(0, 0.01, (self.hidden_size, self.action_count)).astype(np.float32),
            "b_actor": np.zeros(self.action_count, dtype=np.float32),
            "w_value": self.rng.normal(0, 0.01, (self.hidden_size, 1)).astype(np.float32),
            "b_value": np.zeros(1, dtype=np.float32),
        }
        self.optimizer_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.optimizer_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.optimizer_step = 0

    def _forward(self, features):
        features = np.atleast_2d(np.asarray(features, dtype=np.float32))
        pre_activation = features @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(pre_activation, 0.0)
        logits = hidden @ self.params["w_actor"] + self.params["b_actor"]
        values = (hidden @ self.params["w_value"] + self.params["b_value"]).reshape(-1)
        return pre_activation, hidden, logits, values

    @staticmethod
    def _masked_probabilities(logits, masks):
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 1:
            masks = masks[None, :]
        if np.any(~masks.any(axis=1)):
            raise ValueError("The action mask does not contain a valid action.")
        masked_logits = np.where(masks, logits, -1e30)
        shifted = masked_logits - masked_logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) * masks
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def action_distribution(self, observation):
        features = observation_features(observation)
        _, _, logits, values = self._forward(features)
        probabilities = self._masked_probabilities(logits, observation["action_mask"])[0]
        return probabilities, float(values[0])

    def choose_action(self, observation, deterministic=True):
        probabilities, _ = self.action_distribution(observation)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self.rng.choice(self.action_count, p=probabilities))

    def act(self, observation):
        """Sample a legal action and return action, log probability, and value."""
        probabilities, value = self.action_distribution(observation)
        action = int(self.rng.choice(self.action_count, p=probabilities))
        return action, float(np.log(probabilities[action] + 1e-12)), value

    def update(self, batch, learning_rate, clip_range, epochs, minibatch_size, value_coef=0.5):
        """Perform PPO clipped-surrogate updates and return scalar diagnostics."""
        if len(batch.actions) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0}

        advantages = batch.advantages.astype(np.float32).copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        diagnostics = []
        indices = np.arange(len(batch.actions))
        for _ in range(epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                selected = indices[start : start + minibatch_size]
                diagnostics.append(
                    self._update_minibatch(
                        PPOBatch(
                            features=batch.features[selected], masks=batch.masks[selected],
                            actions=batch.actions[selected], old_log_probs=batch.old_log_probs[selected],
                            returns=batch.returns[selected], advantages=advantages[selected],
                        ), learning_rate, clip_range, value_coef,
                    )
                )
        return {key: float(np.mean([item[key] for item in diagnostics])) for key in diagnostics[0]}

    def _update_minibatch(self, batch, learning_rate, clip_range, value_coef):
        pre_activation, hidden, logits, values = self._forward(batch.features)
        probabilities = self._masked_probabilities(logits, batch.masks)
        count = len(batch.actions)
        rows = np.arange(count)
        new_log_probs = np.log(probabilities[rows, batch.actions] + 1e-12)
        ratios = np.exp(new_log_probs - batch.old_log_probs)
        unclipped = ratios * batch.advantages
        clipped = np.clip(ratios, 1.0 - clip_range, 1.0 + clip_range) * batch.advantages
        policy_loss = -np.mean(np.minimum(unclipped, clipped))
        value_loss = np.mean((values - batch.returns) ** 2)

        active = ~(
            ((batch.advantages >= 0) & (ratios > 1.0 + clip_range))
            | ((batch.advantages < 0) & (ratios < 1.0 - clip_range))
        )
        log_prob_gradient = np.where(active, -batch.advantages * ratios / count, 0.0)
        logit_gradient = -log_prob_gradient[:, None] * probabilities
        logit_gradient[rows, batch.actions] += log_prob_gradient
        logit_gradient *= batch.masks

        value_gradient = (2.0 * value_coef / count) * (values - batch.returns)
        gradients = {
            "w_actor": hidden.T @ logit_gradient,
            "b_actor": logit_gradient.sum(axis=0),
            "w_value": hidden.T @ value_gradient[:, None],
            "b_value": np.array([value_gradient.sum()], dtype=np.float32),
        }
        hidden_gradient = logit_gradient @ self.params["w_actor"].T
        hidden_gradient += value_gradient[:, None] * self.params["w_value"].T
        pre_activation_gradient = hidden_gradient * (pre_activation > 0)
        gradients["w1"] = batch.features.T @ pre_activation_gradient
        gradients["b1"] = pre_activation_gradient.sum(axis=0)
        self._adam_step(gradients, learning_rate)

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "approx_kl": float(np.mean(batch.old_log_probs - new_log_probs)),
        }

    def _adam_step(self, gradients, learning_rate):
        self.optimizer_step += 1
        beta_1, beta_2 = 0.9, 0.999
        for name, gradient in gradients.items():
            self.optimizer_m[name] = beta_1 * self.optimizer_m[name] + (1 - beta_1) * gradient
            self.optimizer_v[name] = beta_2 * self.optimizer_v[name] + (1 - beta_2) * gradient**2
            m_hat = self.optimizer_m[name] / (1 - beta_1**self.optimizer_step)
            v_hat = self.optimizer_v[name] / (1 - beta_2**self.optimizer_step)
            self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    def state_dict(self):
        return {
            "format": POLICY_FORMAT, "input_size": self.input_size,
            "action_count": self.action_count, "hidden_size": self.hidden_size,
            "params": self.params,
        }

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict) or state.get("format") != POLICY_FORMAT:
            raise ValueError("Not a Skyjo shared PPO policy.")
        policy = cls(state["input_size"], state["action_count"], state["hidden_size"])
        expected = set(policy.params)
        if set(state.get("params", ())) != expected:
            raise ValueError("PPO policy has missing or unexpected model parameters.")
        for name in expected:
            value = np.asarray(state["params"][name], dtype=np.float32)
            if value.shape != policy.params[name].shape:
                raise ValueError(f"PPO parameter {name} has shape {value.shape}.")
            policy.params[name] = value
        return policy

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as policy_file:
            pickle.dump(self.state_dict(), policy_file)
        temporary_path.replace(path)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as policy_file:
            return cls.from_state_dict(pickle.load(policy_file))
