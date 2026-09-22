"""Historical trajectory generation and dataset diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .environment import IndustrialProcessEnv


def behavior_action(obs: np.ndarray, rng: np.random.Generator, noise_std: float = 0.12) -> np.ndarray:
    """Safe proportional behavior controller with bounded exploration noise."""
    x1, x2 = float(obs[0]), float(obs[1])
    nominal = -0.75 * x1 - 0.20 * x2
    action = np.clip(nominal + rng.normal(0.0, noise_std), -0.8, 0.8)
    return np.array([action], dtype=np.float32)


def generate_dataset(episodes: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    env = IndustrialProcessEnv()
    rng = np.random.default_rng(seed)
    observations, actions, rewards, next_observations, terminals = [], [], [], [], []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = behavior_action(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(reward)
            next_observations.append(next_obs.copy())
            terminals.append(float(terminated or truncated))
            obs = next_obs
            done = terminated or truncated

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
    }


def dataset_summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    actions = data["actions"].reshape(-1)
    rewards = data["rewards"].reshape(-1)
    observations = data["observations"]
    return {
        "transitions": float(len(actions)),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "action_std": float(actions.std()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "terminal_fraction": float(data["terminals"].mean()),
        "state_abs_max": float(np.abs(observations[:, :3]).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/offline_dataset.npz"))
    args = parser.parse_args()

    data = generate_dataset(args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(dataset_summary(data))


if __name__ == "__main__":
    main()
