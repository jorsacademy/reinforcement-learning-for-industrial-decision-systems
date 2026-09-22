"""Evaluate classical and RL controllers on identical seeded episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .controllers import MPCController, RuleBasedController
from .environment import ProductionControlEnv


def run_policy(name: str, policy, episodes: int, seed: int) -> pd.DataFrame:
    rows = []
    for episode in range(episodes):
        env = ProductionControlEnv()
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        final_info = {}
        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy.act(obs)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rows.append(
            {
                "controller": name,
                "episode": episode,
                "return": total_reward,
                "throughput": final_info["throughput"],
                "utilization_proxy": final_info["utilization_proxy"],
                "final_wip": final_info["wip"],
                "final_backlog": final_info["backlog"],
                "energy_cost": final_info["energy_cost"],
                "holding_cost": final_info["holding_cost"],
                "backlog_cost": final_info["backlog_cost"],
                "smoothness_cost": final_info["smoothness_cost"],
                "total_cost": final_info["total_cost"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--ppo-model", type=Path)
    parser.add_argument("--sac-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.csv"))
    args = parser.parse_args()

    controllers = {
        "rule_based": RuleBasedController(),
        "mpc": MPCController(),
    }

    if args.ppo_model or args.sac_model:
        try:
            from stable_baselines3 import PPO, SAC
        except ImportError as exc:
            raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc
        if args.ppo_model:
            controllers["ppo"] = PPO.load(args.ppo_model)
        if args.sac_model:
            controllers["sac"] = SAC.load(args.sac_model)

    frames = [run_policy(name, policy, args.episodes, args.seed) for name, policy in controllers.items()]
    results = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = results.groupby("controller").agg(
        mean_return=("return", "mean"),
        mean_total_cost=("total_cost", "mean"),
        mean_throughput=("throughput", "mean"),
        mean_backlog=("final_backlog", "mean"),
        mean_energy_cost=("energy_cost", "mean"),
        mean_smoothness_cost=("smoothness_cost", "mean"),
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
