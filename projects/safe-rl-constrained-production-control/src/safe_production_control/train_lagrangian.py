from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from .environment import SafeProductionControlEnv


class LagrangianRewardWrapper(gym.Wrapper):
    """Augments reward with a tunable multiplier on explicit constraint cost.

    This is a practical PPO-Lagrangian-style benchmark wrapper, not a full
    primal-dual constrained-policy-optimization implementation.
    """

    def __init__(self, env, lagrangian_multiplier: float = 10.0):
        super().__init__(env)
        self.lagrangian_multiplier = float(lagrangian_multiplier)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.lagrangian_multiplier * float(info.get("constraint_cost", 0.0))
        info = dict(info)
        info["lagrangian_multiplier"] = self.lagrangian_multiplier
        return obs, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lambda-constraint", dest="lam", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/ppo_lagrangian_safe_control"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    env = LagrangianRewardWrapper(SafeProductionControlEnv(), args.lam)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
