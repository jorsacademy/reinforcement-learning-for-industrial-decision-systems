"""Train PPO or SAC on the common production-control environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from .environment import ProductionControlEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    env = Monitor(ProductionControlEnv())
    cls = PPO if args.algo == "ppo" else SAC
    model = cls("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / f"{args.algo}_production_control")


if __name__ == "__main__":
    main()
