"""Train one parameter-shared Skyjo self-play policy with PPO."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from agent.environment import ACTION_COUNT, FEATURE_COUNT, N_AGENTS, env
from agent.ppo import PPOBatch, SharedPPOPolicy, observation_features


def collect_episodes(training_env, policy, episodes, gamma):
    """Collect self-play data. Every player acts through the same policy."""
    transitions = []
    scores = []
    for _ in range(episodes):
        training_env.reset()
        episode_indices = defaultdict(list)
        for agent in training_env.agent_iter():
            observation, _, termination, truncation, _ = training_env.last()
            if termination or truncation:
                training_env.step(None)
                continue
            action, log_prob, value = policy.act(observation)
            episode_indices[agent].append(len(transitions))
            transitions.append(
                {
                    "features": observation_features(observation),
                    "mask": np.asarray(observation["action_mask"], dtype=np.int8),
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "return": 0.0,
                }
            )
            training_env.step(action)

        # Skyjo's only reward is the terminal score. Discount it backwards
        # along each player's own decisions, rather than across the opponent's.
        for agent, indices in episode_indices.items():
            discounted_return = float(training_env.final_rewards[agent])
            for index in reversed(indices):
                transitions[index]["return"] = discounted_return
                discounted_return *= gamma
        scores.extend(training_env.scores.values())

    return transitions, float(np.mean(scores))


def make_batch(transitions):
    returns = np.asarray([item["return"] for item in transitions], dtype=np.float32)
    values = np.asarray([item["value"] for item in transitions], dtype=np.float32)
    return PPOBatch(
        features=np.asarray([item["features"] for item in transitions], dtype=np.float32),
        masks=np.asarray([item["mask"] for item in transitions], dtype=np.int8),
        actions=np.asarray([item["action"] for item in transitions], dtype=np.int64),
        old_log_probs=np.asarray([item["log_prob"] for item in transitions], dtype=np.float32),
        returns=returns,
        advantages=returns - values,
    )


def train(
    episodes,
    rollout_episodes,
    learning_rate,
    gamma,
    clip_range,
    epochs,
    minibatch_size,
    hidden_size,
    seed,
    output_path,
):
    policy = SharedPPOPolicy(
        input_size=N_AGENTS * FEATURE_COUNT,
        action_count=ACTION_COUNT,
        hidden_size=hidden_size,
        seed=seed,
    )
    training_env = env()
    completed = 0
    while completed < episodes:
        current_rollout = min(rollout_episodes, episodes - completed)
        transitions, mean_score = collect_episodes(training_env, policy, current_rollout, gamma)
        diagnostics = policy.update(
            make_batch(transitions), learning_rate, clip_range, epochs, minibatch_size
        )
        completed += current_rollout
        print(
            f"Episodes {completed}/{episodes} | mean score={mean_score:.2f} | "
            f"policy loss={diagnostics['policy_loss']:.4f} | "
            f"value loss={diagnostics['value_loss']:.4f}"
        )
    policy.save(output_path)
    return policy


def evaluate(policy, episodes):
    evaluation_env = env()
    wins = defaultdict(int)
    scores = defaultdict(list)
    for _ in range(episodes):
        evaluation_env.reset()
        for agent in evaluation_env.agent_iter():
            observation, _, termination, truncation, _ = evaluation_env.last()
            if termination or truncation:
                evaluation_env.step(None)
                continue
            evaluation_env.step(policy.choose_action(observation, deterministic=True))
        for agent, info in evaluation_env.final_infos.items():
            scores[agent].append(info["score"])
            wins[agent] += int(info["winner"])

    print("\nEvaluation (one shared deterministic policy)")
    for agent in evaluation_env.possible_agents:
        print(f"{agent}: wins={wins[agent]}/{episodes}, mean score={np.mean(scores[agent]):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a shared Skyjo PPO self-play policy.")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--rollout-episodes", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("policies/skyjo_ppo_policy.pkl"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    learned_policy = train(
        args.episodes,
        args.rollout_episodes,
        args.learning_rate,
        args.gamma,
        args.clip_range,
        args.epochs,
        args.minibatch_size,
        args.hidden_size,
        args.seed,
        args.output,
    )
    evaluate(learned_policy, args.eval_episodes)
