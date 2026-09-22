"""Train behavior cloning or a discrete-action CQL-style policy from a fixed dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc
    return torch, nn


def train_bc(data, epochs: int, output: Path, seed: int):
    torch, nn = _torch()
    torch.manual_seed(seed)
    x = torch.as_tensor(data["observations"], dtype=torch.float32)
    y = torch.as_tensor(data["actions"], dtype=torch.float32)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"type": "bc", "state_dict": model.state_dict(), "obs_dim": x.shape[1]}, output)


def train_cql(data, epochs: int, output: Path, seed: int, bins: int = 21, alpha: float = 1.0, gamma: float = 0.99):
    torch, nn = _torch()
    torch.manual_seed(seed)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    nxt = torch.as_tensor(data["next_observations"], dtype=torch.float32)
    rew = torch.as_tensor(data["rewards"], dtype=torch.float32).unsqueeze(1)
    done = torch.as_tensor(data["terminals"], dtype=torch.float32).unsqueeze(1)
    action_grid = torch.linspace(-1.0, 1.0, bins)
    acts = torch.as_tensor(data["actions"].reshape(-1), dtype=torch.float32)
    a_idx = torch.argmin(torch.abs(acts[:, None] - action_grid[None, :]), dim=1)

    q = nn.Sequential(nn.Linear(obs.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, bins))
    target = nn.Sequential(nn.Linear(obs.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, bins))
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    batch = min(512, len(obs))
    rng = np.random.default_rng(seed)

    for step in range(epochs):
        idx = torch.as_tensor(rng.integers(0, len(obs), size=batch), dtype=torch.long)
        q_all = q(obs[idx])
        q_data = q_all.gather(1, a_idx[idx].unsqueeze(1))
        with torch.no_grad():
            next_q = target(nxt[idx]).max(dim=1, keepdim=True).values
            y = rew[idx] + gamma * (1.0 - done[idx]) * next_q
        bellman = ((q_data - y) ** 2).mean()
        conservative = (torch.logsumexp(q_all, dim=1, keepdim=True) - q_data).mean()
        loss = bellman + alpha * conservative
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            target.load_state_dict(q.state_dict())

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"type": "cql", "state_dict": q.state_dict(), "obs_dim": obs.shape[1], "bins": bins}, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/offline_dataset.npz"))
    parser.add_argument("--algo", choices=["bc", "cql"], default="bc")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = dict(np.load(args.dataset))
    output = args.output or Path(f"artifacts/{args.algo}_policy.pt")
    if args.algo == "bc": train_bc(data, args.epochs, output, args.seed)
    else: train_cql(data, args.epochs, output, args.seed)
    print(f"saved {args.algo} policy to {output}")


if __name__ == "__main__":
    main()
