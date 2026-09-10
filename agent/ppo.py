"""Recurrent MAPPO components for masked, parameter-shared SKYJO play."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agent.environment import (
    ACTION_COUNT,
    CARD_STATE_OFFSET,
    CENTRAL_STATE_SIZE,
    CLOSING_AGENT_FEATURE,
    DRAWN_CARD_FEATURE,
    FEATURE_COUNT,
    N_AGENTS,
    N_CARDS,
    NO_CARD_VALUE,
    PHASE_FEATURE,
    ROUND_NUMBER_FEATURE,
    ROUND_POINTS_FEATURE,
    SERIES_SCORE_FEATURE,
    STARTING_AGENT_FEATURE,
    TOP_DISCARD_FEATURE,
    VISIBLE_COUNT_FEATURE,
)


POLICY_FORMAT = "skyjo_recurrent_mappo_v2"
ACTOR_INPUT_SIZE = 115


def resolve_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # The actor is intentionally small and the Python card simulator is CPU
    # bound. On Apple Silicon, measured MPS dispatch overhead is substantially
    # larger than the compute saved; users can still opt in with --device mps.
    return torch.device("cpu")


def _optional_card_features(value):
    if int(value) == NO_CARD_VALUE:
        return [0.0, 1.0]
    return [(float(value) + 2.0) / 14.0, 0.0]


def observation_features(observation):
    """Encode raw public observations without collapsing binary/card semantics."""
    raw = np.asarray(observation["observation"])
    if raw.shape != (N_AGENTS, FEATURE_COUNT):
        raise ValueError(f"Expected observation shape {(N_AGENTS, FEATURE_COUNT)}, got {raw.shape}.")

    encoded = []
    for player in raw:
        values = player[:N_CARDS]
        states = player[CARD_STATE_OFFSET : CARD_STATE_OFFSET + N_CARDS]
        for value, state in zip(values, states):
            state = int(state)
            encoded.append((float(value) + 2.0) / 14.0 if state == 1 else 0.0)
            encoded.extend(float(state == expected) for expected in range(3))
        encoded.extend(
            [
                float(player[ROUND_POINTS_FEATURE]) / 100.0,
                float(player[VISIBLE_COUNT_FEATURE]) / N_CARDS,
                float(player[CLOSING_AGENT_FEATURE]),
                float(player[SERIES_SCORE_FEATURE]) / 100.0,
                float(player[STARTING_AGENT_FEATURE]),
            ]
        )

    own = raw[0]
    encoded.extend(_optional_card_features(own[TOP_DISCARD_FEATURE]))
    encoded.extend(_optional_card_features(own[DRAWN_CARD_FEATURE]))
    phase = int(own[PHASE_FEATURE])
    encoded.extend(float(phase == expected) for expected in range(4))
    encoded.append(float(own[ROUND_NUMBER_FEATURE]) / 20.0)
    features = np.asarray(encoded, dtype=np.float32)
    if features.shape != (ACTOR_INPUT_SIZE,):
        raise RuntimeError(f"Actor features have unexpected shape {features.shape}.")
    return features


@dataclass(frozen=True)
class ModelConfig:
    actor_input_size: int = ACTOR_INPUT_SIZE
    critic_input_size: int = CENTRAL_STATE_SIZE
    action_count: int = ACTION_COUNT
    encoder_size: int = 256
    recurrent_size: int = 128
    critic_size: int = 256


@dataclass
class PPOBatch:
    """Padded recurrent sequences; ``valid`` excludes padding from every loss."""

    features: np.ndarray
    critic_states: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    initial_hidden: np.ndarray
    reset_hidden: np.ndarray
    valid: np.ndarray


class RecurrentActorCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.config.actor_input_size, self.config.encoder_size),
            nn.LayerNorm(self.config.encoder_size),
            nn.Tanh(),
            nn.Linear(self.config.encoder_size, self.config.encoder_size),
            nn.Tanh(),
        )
        self.actor_memory = nn.GRUCell(
            self.config.encoder_size, self.config.recurrent_size
        )
        self.actor_head = nn.Linear(self.config.recurrent_size, self.config.action_count)
        self.critic = nn.Sequential(
            nn.Linear(self.config.critic_input_size, self.config.critic_size),
            nn.Tanh(),
            nn.Linear(self.config.critic_size, self.config.critic_size),
            nn.Tanh(),
            nn.Linear(self.config.critic_size, 1),
        )
        self.apply(self._initialize)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def initial_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.config.recurrent_size, device=device)

    def actor_step(self, features, hidden, reset_hidden=None):
        if reset_hidden is not None:
            hidden = hidden * (1.0 - reset_hidden.float().unsqueeze(-1))
        encoded = self.actor_encoder(features)
        next_hidden = self.actor_memory(encoded, hidden)
        return self.actor_head(next_hidden), next_hidden

    def actor_sequence(self, features, initial_hidden, reset_hidden):
        hidden = initial_hidden
        logits = []
        for index in range(features.shape[1]):
            step_logits, hidden = self.actor_step(
                features[:, index], hidden, reset_hidden[:, index]
            )
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    def values(self, critic_states):
        return self.critic(critic_states).squeeze(-1)

    @staticmethod
    def distribution(logits, masks, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        masks = masks.bool()
        if torch.any(~masks.any(dim=-1)):
            raise ValueError("At least one action row has no legal action.")
        masked_logits = (logits / temperature).masked_fill(~masks, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits)


class PPOTrainer:
    def __init__(
        self,
        model,
        learning_rate=3e-4,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        target_kl=0.02,
        device="auto",
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

    def act(self, observation, critic_state, hidden, reset_hidden=False, deterministic=False):
        features = torch.as_tensor(
            observation_features(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        masks = torch.as_tensor(
            observation["action_mask"], dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        critic = torch.as_tensor(critic_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        hidden = hidden.to(self.device)
        with torch.no_grad():
            logits, next_hidden = self.model.actor_step(
                features,
                hidden,
                torch.tensor([reset_hidden], dtype=torch.bool, device=self.device),
            )
            distribution = self.model.distribution(logits, masks)
            action = torch.argmax(distribution.probs, dim=-1) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
            value = self.model.values(critic)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            next_hidden.detach(),
            features.squeeze(0).cpu().numpy(),
            hidden.squeeze(0).detach().cpu().numpy(),
        )

    def act_batch(self, observations, critic_states, hidden, reset_hidden):
        """Sample one action for each concurrently stepped learner seat."""
        features_array = np.stack([observation_features(item) for item in observations])
        masks_array = np.stack([item["action_mask"] for item in observations])
        features = torch.as_tensor(features_array, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(masks_array, dtype=torch.bool, device=self.device)
        critic = torch.as_tensor(np.stack(critic_states), dtype=torch.float32, device=self.device)
        hidden = hidden.to(self.device)
        resets = torch.as_tensor(reset_hidden, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, next_hidden = self.model.actor_step(features, hidden, resets)
            distribution = self.model.distribution(logits, masks)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            values = self.model.values(critic)
        return {
            "actions": actions.cpu().numpy(),
            "log_probs": log_probs.cpu().numpy(),
            "values": values.cpu().numpy(),
            "next_hidden": next_hidden.detach(),
            "features": features_array,
            "initial_hidden": hidden.detach().cpu().numpy(),
        }

    def update(self, batch, epochs=4, minibatch_sequences=32):
        self.model.train()
        tensors = {
            name: torch.as_tensor(getattr(batch, name), device=self.device)
            for name in batch.__dataclass_fields__
        }
        advantages = tensors["advantages"].float()
        valid = tensors["valid"].bool()
        valid_advantages = advantages[valid]
        advantages = (advantages - valid_advantages.mean()) / (
            valid_advantages.std(unbiased=False) + 1e-8
        )
        sequence_indices = np.arange(batch.features.shape[0])
        diagnostics = []
        stopped_early = False
        for _ in range(epochs):
            np.random.shuffle(sequence_indices)
            for start in range(0, len(sequence_indices), minibatch_sequences):
                selected = sequence_indices[start : start + minibatch_sequences]
                selected_t = torch.as_tensor(selected, device=self.device)
                mb_valid = valid[selected_t]
                logits = self.model.actor_sequence(
                    tensors["features"][selected_t].float(),
                    tensors["initial_hidden"][selected_t].float(),
                    tensors["reset_hidden"][selected_t].bool(),
                )
                distribution = self.model.distribution(logits, tensors["masks"][selected_t])
                new_log_probs = distribution.log_prob(tensors["actions"][selected_t].long())
                entropy = distribution.entropy()
                values = self.model.values(tensors["critic_states"][selected_t].float())

                ratios = torch.exp(new_log_probs - tensors["old_log_probs"][selected_t].float())
                mb_advantages = advantages[selected_t]
                unclipped = ratios * mb_advantages
                clipped = torch.clamp(
                    ratios, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * mb_advantages
                policy_loss = -torch.minimum(unclipped, clipped)[mb_valid].mean()

                old_values = tensors["returns"][selected_t].float() - tensors["advantages"][selected_t].float()
                value_delta = values - old_values
                clipped_values = old_values + torch.clamp(
                    value_delta, -self.clip_range, self.clip_range
                )
                returns = tensors["returns"][selected_t].float()
                value_loss = 0.5 * torch.maximum(
                    (values - returns).square(), (clipped_values - returns).square()
                )[mb_valid].mean()
                entropy_mean = entropy[mb_valid].mean()
                clip_fraction = ((ratios - 1.0).abs() > self.clip_range)[mb_valid].float().mean()
                valid_returns = returns[mb_valid]
                return_variance = torch.var(valid_returns, unbiased=False)
                explained_variance = 1.0 - torch.var(
                    valid_returns - values[mb_valid], unbiased=False
                ) / (return_variance + 1e-8)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                log_ratio = new_log_probs - tensors["old_log_probs"][selected_t].float()
                approx_kl = ((ratios - 1.0) - log_ratio)[mb_valid].mean()
                diagnostics.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy_mean.detach().cpu()),
                        "approx_kl": float(approx_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "explained_variance": float(explained_variance.detach().cpu()),
                        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                    }
                )
                if approx_kl > self.target_kl:
                    stopped_early = True
                    break
            if stopped_early:
                break
        result = {
            key: float(np.mean([item[key] for item in diagnostics]))
            for key in diagnostics[0]
        }
        result["early_stop"] = stopped_early
        return result


class RecurrentPolicy:
    """Stateful decentralized actor used by evaluation and the human UI."""

    def __init__(self, model, device="auto", seed=None, temperature=1.0):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.hidden = self.model.initial_hidden(1, self.device)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_round()

    def reset_round(self):
        self.hidden = self.model.initial_hidden(1, self.device)

    def choose_action(self, observation, deterministic=False):
        features = torch.as_tensor(
            observation_features(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        masks = torch.as_tensor(
            observation["action_mask"], dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, self.hidden = self.model.actor_step(features, self.hidden)
            probabilities = self.model.distribution(
                logits, masks, temperature=self.temperature
            ).probs.squeeze(0).cpu().numpy()
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self.rng.choice(len(probabilities), p=probabilities))

    @classmethod
    def load(cls, path, device="auto", seed=None, temperature=1.0):
        model, _ = load_actor_checkpoint(path, device=device)
        return cls(model, device=device, seed=seed, temperature=temperature)


def save_actor_checkpoint(path, model, step, seed, rating=1000.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": POLICY_FORMAT,
        "model_config": asdict(model.config),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "step": int(step),
        "seed": int(seed),
        "rating": float(rating),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps({key: value for key, value in payload.items() if key != "state_dict"}, indent=2)
        + "\n"
    )
    metadata_temporary.replace(metadata_path)


def load_actor_checkpoint(path, device="auto"):
    resolved_device = resolve_device(device)
    payload = torch.load(path, map_location=resolved_device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != POLICY_FORMAT:
        raise ValueError("Not a compatible recurrent SKYJO policy checkpoint.")
    config = ModelConfig(**payload["model_config"])
    if config.action_count != ACTION_COUNT or config.actor_input_size != ACTOR_INPUT_SIZE:
        raise ValueError("Checkpoint action or observation schema is incompatible.")
    model = RecurrentActorCritic(config).to(resolved_device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata
