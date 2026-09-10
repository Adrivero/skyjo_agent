"""Train and evaluate a recurrent MAPPO SKYJO policy with league self-play."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.environment import (
    ACTION_COUNT,
    CENTRAL_STATE_SIZE,
    DRAW_ACTION,
    PHASE_FEATURE,
    PHASE_OPENING,
    PHASE_SOURCE,
    TAKE_DISCARD_ACTION,
    env,
)
from agent.policy import HeuristicPolicy, RandomPolicy
from agent.ppo import (
    ACTOR_INPUT_SIZE,
    POLICY_FORMAT,
    ModelConfig,
    PPOBatch,
    PPOTrainer,
    RecurrentActorCritic,
    RecurrentPolicy,
    load_actor_checkpoint,
    observation_features,
    save_actor_checkpoint,
)


@dataclass
class TrainConfig:
    total_steps: int = 10_000_000
    rollout_steps: int = 8_192
    num_envs: int = 32
    sequence_length: int = 32
    bc_steps: int = 100_000
    bc_epochs: int = 15
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    epochs: int = 4
    minibatch_sequences: int = 32
    value_coef: float = 0.5
    entropy_start: float = 0.02
    entropy_end: float = 0.002
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    eval_interval: int = 250_000
    checkpoint_interval: int = 250_000
    eval_series: int = 400
    final_eval_series: int = 1_000
    target_score: int = 100
    max_round_turns: int = 200
    max_series_rounds: int = 20
    seed: int = 1
    device: str = "auto"
    run_dir: str = "runs/skyjo_mappo"
    output: str = "policies/skyjo_mappo_actor.pt"


@dataclass
class Transition:
    features: np.ndarray
    critic_state: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    hidden: np.ndarray
    reset_hidden: bool
    reward: float = 0.0
    done: bool = False


@dataclass
class LeagueEntry:
    step: int
    model: RecurrentActorCritic
    path: Path
    rating: float = 1000.0
    learner_wins: float = 0.0
    games: int = 0

    @property
    def learner_win_rate(self):
        return self.learner_wins / self.games if self.games else 0.5


class League:
    def __init__(self, directory, device, seed):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.entries = []
        self.current_rating = 1000.0

    def archive(self, model, step, seed):
        path = self.directory / f"step_{step:09d}.pt"
        frozen = copy.deepcopy(model).to("cpu").eval()
        save_actor_checkpoint(path, frozen, step=step, seed=seed, rating=self.current_rating)
        entry = LeagueEntry(step=step, model=frozen, path=path, rating=self.current_rating)
        self.entries.append(entry)
        return entry

    def restore(self):
        self.entries = []
        for path in sorted(self.directory.glob("step_*.pt")):
            model, metadata = load_actor_checkpoint(path, device="cpu")
            self.entries.append(
                LeagueEntry(
                    step=int(metadata["step"]),
                    model=model,
                    path=path,
                    rating=float(metadata.get("rating", 1000.0)),
                )
            )

    def sample(self):
        if not self.entries:
            return None
        weights = np.asarray(
            [entry.learner_win_rate * (1.0 - entry.learner_win_rate) + 0.05 for entry in self.entries],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return self.entries[int(self.rng.choice(len(self.entries), p=weights))]

    def record(self, entry, learner_score):
        if entry is None:
            return
        entry.games += 1
        entry.learner_wins += learner_score
        expected = 1.0 / (1.0 + 10.0 ** ((entry.rating - self.current_rating) / 400.0))
        change = 16.0 * (learner_score - expected)
        self.current_rating += change
        entry.rating -= change


class RolloutRunner:
    def __init__(self, environment, episode_index, mode, learner_agents, opponent, league_entry, trainer):
        self.environment = environment
        self.episode_index = episode_index
        self.mode = mode
        self.learner_agents = set(learner_agents)
        self.opponent = opponent
        self.league_entry = league_entry
        self.trajectories = {agent: [] for agent in learner_agents}
        self.hidden = {
            agent: trainer.model.initial_hidden(1, trainer.device) for agent in learner_agents
        }
        self.round_seen = {agent: 0 for agent in learner_agents}
        self.opponent_round = 0
        self.active = True


def collect_heuristic_demonstrations(config, target_steps):
    """Collect public-observation demonstrations for a policy warm start."""
    trajectories = []
    collected = 0
    episode_index = 0
    while collected < target_steps:
        game = env(
            target_score=config.target_score,
            max_round_turns=config.max_round_turns,
            max_series_rounds=config.max_series_rounds,
        )
        game.reset(seed=config.seed + 30_000_000 + episode_index)
        policies = {
            agent: HeuristicPolicy(config.seed + 31_000_000 + episode_index * 2 + index)
            for index, agent in enumerate(game.possible_agents)
        }
        agent_steps = {agent: [] for agent in game.possible_agents}
        rounds = {agent: 0 for agent in game.possible_agents}
        while not any(game.terminations.values()) and not any(game.truncations.values()):
            agent = game.agent_selection
            observation = game.observe(agent)
            reset_hidden = rounds[agent] != game.round_number
            if reset_hidden:
                policies[agent].reset_round()
                rounds[agent] = game.round_number
            action = policies[agent].choose_action(observation)
            agent_steps[agent].append(
                {
                    "features": observation_features(observation),
                    "mask": np.asarray(observation["action_mask"], dtype=np.int8),
                    "action": action,
                    "reset_hidden": reset_hidden,
                }
            )
            game.step(action)
        for steps in agent_steps.values():
            trajectories.append(steps)
            collected += len(steps)
        episode_index += 1
    return trajectories


def pretrain_actor(trainer, trajectories, sequence_length, epochs=3):
    """Behavior-clone the legal heuristic before competitive fine-tuning."""
    trainer.model.train()
    sequences = []
    for trajectory in trajectories:
        for start in range(0, len(trajectory), sequence_length):
            sequences.append(trajectory[start : start + sequence_length])
    count = len(sequences)
    features = np.zeros((count, sequence_length, ACTOR_INPUT_SIZE), dtype=np.float32)
    masks = np.zeros((count, sequence_length, ACTION_COUNT), dtype=np.int8)
    masks[..., 0] = 1
    actions = np.zeros((count, sequence_length), dtype=np.int64)
    resets = np.ones((count, sequence_length), dtype=bool)
    valid = np.zeros((count, sequence_length), dtype=bool)
    for sequence_index, sequence in enumerate(sequences):
        for step_index, step in enumerate(sequence):
            features[sequence_index, step_index] = step["features"]
            masks[sequence_index, step_index] = step["mask"]
            actions[sequence_index, step_index] = step["action"]
            resets[sequence_index, step_index] = step["reset_hidden"]
            valid[sequence_index, step_index] = True
        # Chunks are independent truncated-BPTT units.
        resets[sequence_index, 0] = True

    feature_tensor = torch.as_tensor(features, device=trainer.device)
    mask_tensor = torch.as_tensor(masks, device=trainer.device)
    action_tensor = torch.as_tensor(actions, device=trainer.device)
    reset_tensor = torch.as_tensor(resets, device=trainer.device)
    valid_tensor = torch.as_tensor(valid, device=trainer.device)
    sequence_indices = np.arange(count)
    losses = []
    accuracies = []
    for _ in range(epochs):
        np.random.shuffle(sequence_indices)
        for start in range(0, count, 32):
            selected = sequence_indices[start : start + 32]
            selected_t = torch.as_tensor(selected, device=trainer.device)
            feature_t = feature_tensor[selected_t]
            mask_t = mask_tensor[selected_t]
            action_t = action_tensor[selected_t]
            reset_t = reset_tensor[selected_t]
            valid_t = valid_tensor[selected_t]
            hidden = trainer.model.initial_hidden(len(selected), trainer.device)
            logits = trainer.model.actor_sequence(feature_t, hidden, reset_t)
            distribution = trainer.model.distribution(logits, mask_t)
            loss = -distribution.log_prob(action_t)[valid_t].mean()
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()
            predictions = torch.argmax(distribution.probs, dim=-1)
            accuracy = (predictions[valid_t] == action_t[valid_t]).float().mean()
            losses.append(float(loss.detach().cpu()))
            accuracies.append(float(accuracy.detach().cpu()))
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}


def _new_runner(config, trainer, league, episode_index):
    episode_seed = config.seed + episode_index
    environment = env(
        target_score=config.target_score,
        max_round_turns=config.max_round_turns,
        max_series_rounds=config.max_series_rounds,
    )
    environment.reset(seed=episode_seed)
    draw = league.rng.random()
    entry = None
    if draw < 0.20:
        mode = "self"
        learner_agents = environment.possible_agents
        opponent = None
    elif draw < 0.80 and league.entries:
        mode = "snapshot"
        learner_agent = environment.possible_agents[episode_index % 2]
        learner_agents = [learner_agent]
        entry = league.sample()
        opponent = RecurrentPolicy(
            entry.model, device=trainer.device, seed=episode_seed + 1
        )
    else:
        mode = "heuristic"
        learner_agent = environment.possible_agents[episode_index % 2]
        learner_agents = [learner_agent]
        opponent = HeuristicPolicy(seed=episode_seed + 1)
    if opponent is not None:
        opponent.reset(seed=episode_seed + 1)
    return RolloutRunner(
        environment, episode_index, mode, learner_agents, opponent, entry, trainer
    )


def collect_rollouts(config, trainer, league, episode_offset=0):
    """Step many live series together and batch all current-policy inference."""
    runners = [
        _new_runner(config, trainer, league, episode_offset + index)
        for index in range(config.num_envs)
    ]
    next_episode = episode_offset + len(runners)
    trajectories = []
    completed_steps = 0
    stats = {
        "series": 0,
        "wins": 0.0,
        "truncations": 0,
        "rounds": [],
        "turns": [],
        "source_choices": 0,
        "source_draws": 0,
        "source_pile_takes": 0,
        "opening_choices": 0,
    }

    def finish_step(runner):
        nonlocal completed_steps, next_episode
        game = runner.environment
        for learner_agent in runner.learner_agents:
            reward = float(game.rewards.get(learner_agent, 0.0))
            if reward and runner.trajectories[learner_agent]:
                runner.trajectories[learner_agent][-1].reward += reward
        finished = any(game.terminations.values()) or any(game.truncations.values())
        if not finished:
            return runner

        for learner_agent, trajectory in runner.trajectories.items():
            if trajectory:
                trajectory[-1].done = True
                trajectories.append(trajectory)
                completed_steps += len(trajectory)
        stats["series"] += 1
        stats["truncations"] += int(any(game.truncations.values()))
        stats["rounds"].append(game.round_number)
        stats["turns"].append(game.series_turn_count)
        if runner.mode != "self":
            learner_agent = next(iter(runner.learner_agents))
            if game.series_winner == "tie":
                outcome = 0.5
            else:
                outcome = float(game.series_winner == learner_agent)
            stats["wins"] += outcome
            league.record(runner.league_entry, outcome)
        runner.active = False
        if completed_steps < config.rollout_steps:
            replacement = _new_runner(config, trainer, league, next_episode)
            next_episode += 1
            return replacement
        return runner

    while any(runner.active for runner in runners):
        learner_requests = []
        for index, runner in enumerate(runners):
            if not runner.active:
                continue
            game = runner.environment
            agent = game.agent_selection
            observation = game.observe(agent)
            if agent in runner.learner_agents:
                reset_hidden = runner.round_seen[agent] != game.round_number
                learner_requests.append(
                    (index, runner, agent, observation, game.state(), reset_hidden)
                )
                continue

            if runner.opponent_round != game.round_number:
                runner.opponent.reset_round()
                runner.opponent_round = game.round_number
            action = runner.opponent.choose_action(observation)
            game.step(action)
            runners[index] = finish_step(runner)

        if learner_requests:
            hidden = torch.cat(
                [request[1].hidden[request[2]] for request in learner_requests], dim=0
            )
            result = trainer.act_batch(
                [request[3] for request in learner_requests],
                [request[4] for request in learner_requests],
                hidden,
                [request[5] for request in learner_requests],
            )
            for batch_index, request in enumerate(learner_requests):
                runner_index, runner, agent, observation, critic_state, reset_hidden = request
                action = int(result["actions"][batch_index])
                phase = int(observation["observation"][0, PHASE_FEATURE])
                if phase == PHASE_OPENING:
                    stats["opening_choices"] += 1
                elif phase == PHASE_SOURCE:
                    stats["source_choices"] += 1
                    stats["source_draws"] += int(action == DRAW_ACTION)
                    stats["source_pile_takes"] += int(action == TAKE_DISCARD_ACTION)
                runner.trajectories[agent].append(
                    Transition(
                        features=result["features"][batch_index],
                        critic_state=np.asarray(critic_state, dtype=np.float32),
                        mask=np.asarray(observation["action_mask"], dtype=np.int8),
                        action=action,
                        log_prob=float(result["log_probs"][batch_index]),
                        value=float(result["values"][batch_index]),
                        hidden=result["initial_hidden"][batch_index],
                        reset_hidden=reset_hidden,
                    )
                )
                runner.hidden[agent] = result["next_hidden"][batch_index : batch_index + 1]
                runner.round_seen[agent] = runner.environment.round_number
                runner.environment.step(action)
                runners[runner_index] = finish_step(runner)

    stats["learner_steps"] = completed_steps
    stats["next_episode"] = next_episode
    stats["mean_rounds"] = float(np.mean(stats.pop("rounds")))
    stats["mean_turns"] = float(np.mean(stats.pop("turns")))
    choices = stats["source_choices"]
    stats["source_draw_rate"] = stats["source_draws"] / choices if choices else 0.0
    stats["source_pile_rate"] = stats["source_pile_takes"] / choices if choices else 0.0
    return trajectories, stats


def make_batch(trajectories, gamma, gae_lambda, sequence_length):
    sequences = []
    for trajectory in trajectories:
        advantages = np.zeros(len(trajectory), dtype=np.float32)
        next_advantage = 0.0
        for index in reversed(range(len(trajectory))):
            step = trajectory[index]
            next_value = 0.0 if index == len(trajectory) - 1 else trajectory[index + 1].value
            next_nonterminal = 0.0 if step.done else 1.0
            delta = step.reward + gamma * next_value * next_nonterminal - step.value
            advantages[index] = delta + gamma * gae_lambda * next_nonterminal * next_advantage
            next_advantage = advantages[index]
        returns = advantages + np.asarray([step.value for step in trajectory], dtype=np.float32)
        for start in range(0, len(trajectory), sequence_length):
            sequences.append((trajectory[start : start + sequence_length], advantages[start : start + sequence_length], returns[start : start + sequence_length]))

    count = len(sequences)
    shape = (count, sequence_length)
    features = np.zeros(shape + (ACTOR_INPUT_SIZE,), dtype=np.float32)
    critic_states = np.zeros(shape + (CENTRAL_STATE_SIZE,), dtype=np.float32)
    masks = np.zeros(shape + (ACTION_COUNT,), dtype=np.int8)
    masks[..., 0] = 1  # Keep padded categorical rows valid; their loss is masked.
    actions = np.zeros(shape, dtype=np.int64)
    old_log_probs = np.zeros(shape, dtype=np.float32)
    returns_array = np.zeros(shape, dtype=np.float32)
    advantages_array = np.zeros(shape, dtype=np.float32)
    reset_hidden = np.ones(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    initial_hidden = np.zeros((count, sequences[0][0][0].hidden.shape[0]), dtype=np.float32)

    for sequence_index, (steps, sequence_advantages, sequence_returns) in enumerate(sequences):
        length = len(steps)
        initial_hidden[sequence_index] = steps[0].hidden
        for step_index, step in enumerate(steps):
            features[sequence_index, step_index] = step.features
            critic_states[sequence_index, step_index] = step.critic_state
            masks[sequence_index, step_index] = step.mask
            actions[sequence_index, step_index] = step.action
            old_log_probs[sequence_index, step_index] = step.log_prob
            reset_hidden[sequence_index, step_index] = step.reset_hidden
        returns_array[sequence_index, :length] = sequence_returns
        advantages_array[sequence_index, :length] = sequence_advantages
        valid[sequence_index, :length] = True

    return PPOBatch(
        features=features,
        critic_states=critic_states,
        masks=masks,
        actions=actions,
        old_log_probs=old_log_probs,
        returns=returns_array,
        advantages=advantages_array,
        initial_hidden=initial_hidden,
        reset_hidden=reset_hidden,
        valid=valid,
    )


def play_series(first_policy, second_policy, seed, config=None):
    config = config or TrainConfig()
    game = env(
        target_score=config.target_score,
        max_round_turns=config.max_round_turns,
        max_series_rounds=config.max_series_rounds,
    )
    game.reset(seed=seed)
    policies = {"player_0": first_policy, "player_1": second_policy}
    rounds = {agent: 0 for agent in game.possible_agents}
    for offset, policy in enumerate((first_policy, second_policy)):
        policy.reset(seed=seed + offset + 1)
    while not any(game.terminations.values()) and not any(game.truncations.values()):
        agent = game.agent_selection
        policy = policies[agent]
        if rounds[agent] != game.round_number:
            policy.reset_round()
            rounds[agent] = game.round_number
        game.step(policy.choose_action(game.observe(agent)))
    return game


def wilson_lower_bound(wins, games, z=1.96):
    if not games:
        return 0.0
    proportion = wins / games
    denominator = 1.0 + z * z / games
    centre = proportion + z * z / (2.0 * games)
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * games)) / games)
    return (centre - margin) / denominator


def evaluate_policy(model, series, seed, config=None, opponent_factory=HeuristicPolicy):
    config = config or TrainConfig()
    candidate_wins = 0.0
    ties = 0
    truncations = 0
    score_margins = []
    rounds = []
    turns = []
    paired_games = series + (series % 2)
    for game_index in range(paired_games):
        pair_seed = seed + game_index // 2
        candidate = RecurrentPolicy(model, device=config.device, seed=pair_seed)
        opponent = opponent_factory(seed=pair_seed + 10_000)
        candidate_agent = "player_0" if game_index % 2 == 0 else "player_1"
        game = (
            play_series(candidate, opponent, pair_seed, config)
            if candidate_agent == "player_0"
            else play_series(opponent, candidate, pair_seed, config)
        )
        truncations += int(any(game.truncations.values()))
        if game.series_winner == "tie":
            candidate_wins += 0.5
            ties += 1
        else:
            candidate_wins += float(game.series_winner == candidate_agent)
        opponent_agent = next(agent for agent in game.possible_agents if agent != candidate_agent)
        score_margins.append(game.series_scores[opponent_agent] - game.series_scores[candidate_agent])
        rounds.append(game.round_number)
        turns.append(game.series_turn_count)
    return {
        "series": paired_games,
        "wins": candidate_wins,
        "win_rate": candidate_wins / paired_games,
        "wilson_lower": wilson_lower_bound(candidate_wins, paired_games),
        "ties": ties,
        "truncations": truncations,
        "mean_score_margin": float(np.mean(score_margins)),
        "mean_rounds": float(np.mean(rounds)),
        "mean_turns": float(np.mean(turns)),
    }


def evaluate_suite(model, series, seed, config=None, league=None):
    """Evaluate against fixed anchors and recent league snapshots."""
    config = config or TrainConfig()
    heuristic = evaluate_policy(
        model, series, seed, config=config, opponent_factory=HeuristicPolicy
    )
    random_result = evaluate_policy(
        model, series, seed + 1_000_000, config=config, opponent_factory=RandomPolicy
    )
    snapshot_results = []
    if league is not None:
        for entry in league.entries[-5:]:
            factory = lambda seed, frozen=entry.model: RecurrentPolicy(
                frozen, device=config.device, seed=seed
            )
            result = evaluate_policy(
                model,
                max(20, series // 5),
                seed + 2_000_000 + entry.step,
                config=config,
                opponent_factory=factory,
            )
            snapshot_results.append({"step": entry.step, **result})
    clipped_rate = float(np.clip(heuristic["win_rate"], 1e-4, 1.0 - 1e-4))
    anchored_elo = 1000.0 + 400.0 * math.log10(clipped_rate / (1.0 - clipped_rate))
    return {
        "heuristic": heuristic,
        "random": random_result,
        "snapshots": snapshot_results,
        "heuristic_anchored_elo": anchored_elo,
    }


def _save_training_checkpoint(
    path, trainer, config, step, episode_offset, best_rate, best_margin, league
):
    payload = {
        "format": POLICY_FORMAT,
        "model_config": asdict(trainer.model.config),
        "model_state": {name: value.detach().cpu() for name, value in trainer.model.state_dict().items()},
        "optimizer_state": trainer.optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "episode_offset": episode_offset,
        "best_rate": best_rate,
        "best_margin": best_margin,
        "rating": league.current_rating,
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, resume=None):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    writer = SummaryWriter(run_dir / "tensorboard")

    model = RecurrentActorCritic(ModelConfig())
    trainer = PPOTrainer(
        model,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_start,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        device=config.device,
    )
    step = 0
    episode_offset = 0
    best_rate = -1.0
    best_margin = float("-inf")
    resume_payload = None
    if resume:
        resume_payload = torch.load(resume, map_location=trainer.device, weights_only=True)
        if resume_payload.get("format") != POLICY_FORMAT:
            raise ValueError("Resume checkpoint is not compatible.")
        trainer.model.load_state_dict(resume_payload["model_state"])
        trainer.optimizer.load_state_dict(resume_payload["optimizer_state"])
        step = int(resume_payload["step"])
        episode_offset = int(resume_payload["episode_offset"])
        best_rate = float(resume_payload["best_rate"])
        best_margin = float(resume_payload.get("best_margin", float("-inf")))

    if resume_payload is None and config.bc_steps > 0:
        demonstrations = collect_heuristic_demonstrations(config, config.bc_steps)
        pretraining = pretrain_actor(
            trainer,
            demonstrations,
            sequence_length=config.sequence_length,
            epochs=config.bc_epochs,
        )
        writer.add_scalar("pretrain/loss", pretraining["loss"], 0)
        writer.add_scalar("pretrain/accuracy", pretraining["accuracy"], 0)
        print("behavior cloning", json.dumps(pretraining, sort_keys=True))

    league = League(run_dir / "league", trainer.device, config.seed + 77)
    if resume_payload is not None:
        league.restore()
        league.current_rating = float(resume_payload.get("rating", 1000.0))
    if not league.entries or league.entries[-1].step != step:
        league.archive(trainer.model, step, config.seed)
    metrics_path = run_dir / "metrics.jsonl"
    next_evaluation = ((step // config.eval_interval) + 1) * config.eval_interval
    next_checkpoint = ((step // config.checkpoint_interval) + 1) * config.checkpoint_interval

    while step < config.total_steps:
        trajectories, rollout = collect_rollouts(config, trainer, league, episode_offset)
        episode_offset = int(rollout["next_episode"])
        batch = make_batch(
            trajectories, config.gamma, config.gae_lambda, config.sequence_length
        )
        progress = min(1.0, step / config.total_steps)
        learning_rate = config.learning_rate * (1.0 - progress)
        entropy = config.entropy_start + progress * (config.entropy_end - config.entropy_start)
        trainer.optimizer.param_groups[0]["lr"] = learning_rate
        trainer.entropy_coef = entropy
        diagnostics = trainer.update(
            batch, epochs=config.epochs, minibatch_sequences=config.minibatch_sequences
        )
        step += int(rollout["learner_steps"])
        metrics = {
            "step": step,
            "league_elo": league.current_rating,
            "learning_rate": learning_rate,
            "entropy_coef": entropy,
            **{key: value for key, value in rollout.items() if key != "next_episode"},
            **diagnostics,
        }
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(metrics) + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                writer.add_scalar(f"train/{key}", value, step)
        print(
            f"steps={step}/{config.total_steps} series={rollout['series']} "
            f"turns={rollout['mean_turns']:.1f} trunc={rollout['truncations']} "
            f"loss={diagnostics['policy_loss']:.4f} entropy={diagnostics['entropy']:.3f} "
            f"elo={league.current_rating:.0f}"
        )

        if step >= next_checkpoint:
            league.archive(trainer.model, step, config.seed)
            _save_training_checkpoint(
                run_dir / "latest_training.pt",
                trainer,
                config,
                step,
                episode_offset,
                best_rate,
                best_margin,
                league,
            )
            next_checkpoint += config.checkpoint_interval

        if step >= next_evaluation or step >= config.total_steps:
            evaluation = evaluate_policy(
                trainer.model,
                config.eval_series,
                seed=config.seed + 10_000_000 + step,
                config=config,
            )
            for key, value in evaluation.items():
                writer.add_scalar(f"validation/{key}", value, step)
            print("validation", json.dumps(evaluation, sort_keys=True))
            candidate_rank = (evaluation["win_rate"], evaluation["mean_score_margin"])
            best_rank = (best_rate, best_margin)
            if candidate_rank > best_rank and evaluation["truncations"] == 0:
                best_rate = evaluation["win_rate"]
                best_margin = evaluation["mean_score_margin"]
                save_actor_checkpoint(
                    config.output,
                    trainer.model,
                    step=step,
                    seed=config.seed,
                    rating=league.current_rating,
                )
            next_evaluation += config.eval_interval

    writer.close()
    if not Path(config.output).is_file():
        save_actor_checkpoint(
            config.output,
            trainer.model,
            step=step,
            seed=config.seed,
            rating=league.current_rating,
        )
    best_model, _ = load_actor_checkpoint(config.output, device=config.device)
    final = evaluate_suite(
        best_model,
        config.final_eval_series,
        seed=config.seed + 20_000_000,
        config=config,
        league=league,
    )
    print("held-out", json.dumps(final, indent=2, sort_keys=True))
    return best_model, final


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--rollout-steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--bc-steps", type=int, default=TrainConfig.bc_steps)
    parser.add_argument("--bc-epochs", type=int, default=TrainConfig.bc_epochs)
    parser.add_argument("--eval-series", type=int, default=TrainConfig.eval_series)
    parser.add_argument("--final-eval-series", type=int, default=TrainConfig.final_eval_series)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--checkpoint-interval", type=int, default=TrainConfig.checkpoint_interval)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--run-dir", default=TrainConfig.run_dir)
    parser.add_argument("--output", default=TrainConfig.output)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = TrainConfig(
        total_steps=arguments.total_steps,
        rollout_steps=arguments.rollout_steps,
        num_envs=arguments.num_envs,
        bc_steps=arguments.bc_steps,
        bc_epochs=arguments.bc_epochs,
        eval_series=arguments.eval_series,
        final_eval_series=arguments.final_eval_series,
        eval_interval=arguments.eval_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        seed=arguments.seed,
        device=arguments.device,
        run_dir=arguments.run_dir,
        output=arguments.output,
    )
    if arguments.evaluate:
        evaluation_model, metadata = load_actor_checkpoint(
            arguments.evaluate, device=configuration.device
        )
        results = evaluate_suite(
            evaluation_model,
            configuration.final_eval_series,
            seed=configuration.seed + 20_000_000,
            config=configuration,
        )
        print(json.dumps({"checkpoint": metadata, "results": results}, indent=2))
    else:
        train(configuration, resume=arguments.resume)
