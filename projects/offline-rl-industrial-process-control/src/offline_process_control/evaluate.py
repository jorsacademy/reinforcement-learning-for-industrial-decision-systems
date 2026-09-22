"""Evaluate historical behavior, BC and CQL-style policies on held-out seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import behavior_action
from .environment import IndustrialProcessEnv


def load_policy(path: Path):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = int(ckpt["obs_dim"])
    if ckpt["type"] == "bc":
        model = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        def act(obs):
            with torch.no_grad():
                return np.array([float(model(torch.as_tensor(obs, dtype=torch.float32)).item())], dtype=np.float32)
        return act
    bins = int(ckpt["bins"])
    model = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, bins))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    grid = np.linspace(-1.0, 1.0, bins)
    def act(obs):
        with torch.no_grad():
            idx = int(model(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
        return np.array([grid[idx]], dtype=np.float32)
    return act


def run_policy(name: str, act, episodes: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        env = IndustrialProcessEnv()
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ret = quality = energy = smooth = 0.0
        while not done:
            action = act(obs) if name != "behavior" else behavior_action(obs, rng, noise_std=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward; quality += info["quality_cost"]; energy += info["energy_cost"]; smooth += info["smoothness_cost"]
            done = terminated or truncated
        rows.append({"policy": name, "episode": ep, "return": ret, "quality_cost": quality, "energy_cost": energy, "smoothness_cost": smooth})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bc-model", type=Path)
    parser.add_argument("--cql-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    args = parser.parse_args()
    policies = {"behavior": None}
    if args.bc_model: policies["bc"] = load_policy(args.bc_model)
    if args.cql_model: policies["cql"] = load_policy(args.cql_model)
    results = pd.concat([run_policy(name, act, args.episodes, args.seed) for name, act in policies.items()], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True); results.to_csv(args.output, index=False)
    print(results.groupby("policy")[["return", "quality_cost", "energy_cost", "smoothness_cost"]].mean().round(4))


if __name__ == "__main__":
    main()
