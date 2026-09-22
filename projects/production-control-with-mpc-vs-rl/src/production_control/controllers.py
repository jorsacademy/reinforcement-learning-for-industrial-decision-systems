"""Classical production-control baselines."""

from __future__ import annotations

import numpy as np


class RuleBasedController:
    """Base-stock-style controller using backlog and WIP deviation."""

    def __init__(self, target_wip: float = 18.0, max_wip: float = 45.0):
        self.target_wip = target_wip
        self.max_wip = max_wip

    def act(self, obs: np.ndarray) -> np.ndarray:
        wip = float(obs[0]) * self.max_wip
        backlog = float(obs[1]) * (2.0 * self.max_wip)
        demand_ratio = float(obs[2])
        u = 0.55 + 0.03 * backlog + 0.02 * (self.target_wip - wip) + 0.25 * demand_ratio
        return np.array([np.clip(u, 0.0, 1.0)], dtype=np.float32)


class MPCController:
    """Small receding-horizon controller solved by deterministic grid search.

    This is deliberately dependency-light. At each step it evaluates candidate
    constant production commands over a short horizon using a local linearized
    inventory/backlog model and chooses the lowest predicted cost.
    """

    def __init__(
        self,
        max_rate: float = 12.0,
        max_wip: float = 45.0,
        target_wip: float = 18.0,
        horizon: int = 6,
    ):
        self.max_rate = max_rate
        self.max_wip = max_wip
        self.target_wip = target_wip
        self.horizon = horizon
        self.grid = np.linspace(0.0, 1.0, 21)

    def act(self, obs: np.ndarray) -> np.ndarray:
        wip0 = float(obs[0]) * self.max_wip
        backlog0 = float(obs[1]) * (2.0 * self.max_wip)
        demand = float(obs[2]) * (1.5 * self.max_rate)
        price_ratio = float(obs[3])
        prev = float(obs[4])

        best_u = 0.0
        best_cost = float("inf")
        for u in self.grid:
            wip = wip0
            backlog = backlog0
            cost = 0.0
            for _ in range(self.horizon):
                backlog += demand
                produced = min(u * self.max_rate, max(0.0, self.max_wip - wip))
                wip += produced
                shipped = min(wip, backlog)
                wip -= shipped
                backlog -= shipped
                energy = price_ratio * (0.25 * u + 0.75 * u * u)
                cost += 0.08 * wip + 0.55 * backlog + energy
            cost += 0.35 * abs(u - prev) + 0.04 * abs(wip - self.target_wip)
            if cost < best_cost:
                best_cost = cost
                best_u = float(u)
        return np.array([best_u], dtype=np.float32)
