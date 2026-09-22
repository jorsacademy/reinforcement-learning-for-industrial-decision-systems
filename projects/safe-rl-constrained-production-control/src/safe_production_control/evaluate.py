from __future__ import annotations

import argparse
import json

import numpy as np

from .controllers import AggressiveController, SafetyRuleController
from .environment import SafeProductionControlEnv


def evaluate(policy, episodes: int = 30, seed: int = 42):
    rows = []
    for ep in range(episodes):
        env = SafeProductionControlEnv()
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        final_info = {}
        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy.act(obs)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rows.append((total_reward, final_info, steps))

    return {
        "mean_return": float(np.mean([r[0] for r in rows])),
        "mean_total_cost": float(np.mean([r[1]["total_cost"] for r in rows])),
        "mean_throughput": float(np.mean([r[1]["throughput"] for r in rows])),
        "mean_final_wip": float(np.mean([r[1]["wip"] for r in rows])),
        "mean_final_backlog": float(np.mean([r[1]["backlog"] for r in rows])),
        "mean_energy_cost": float(np.mean([r[1]["energy_cost"] for r in rows])),
        "mean_constraint_violations": float(np.mean([r[1]["constraint_violations"] for r in rows])),
        "violation_rate": float(np.mean([r[1]["constraint_violations"] / r[2] for r in rows])),
        "mean_violation_severity": float(np.mean([r[1]["violation_severity"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["safe_rule", "aggressive", "ppo"], default="safe_rule")
    parser.add_argument("--model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.policy == "safe_rule":
        policy = SafetyRuleController()
    elif args.policy == "aggressive":
        policy = AggressiveController()
    else:
        if not args.model:
            raise SystemExit("--model is required for PPO evaluation")
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc
        policy = PPO.load(args.model)

    print(json.dumps(evaluate(policy, args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
