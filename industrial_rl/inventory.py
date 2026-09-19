from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 12
    max_inventory: int = 20
    max_order: int = 10
    demand_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    demand_probabilities: tuple[float, ...] = (0.04, 0.10, 0.18, 0.26, 0.20, 0.14, 0.08)
    order_cost: float = 1.8
    holding_cost: float = 0.7
    stockout_cost: float = 4.5
    fixed_order_cost: float = 0.4
    initial_inventory: int = 6

    def validate(self) -> None:
        if self.horizon <= 0 or self.max_inventory < 0 or self.max_order < 0:
            raise ValueError("horizon must be positive and capacities non-negative")
        if len(self.demand_values) != len(self.demand_probabilities):
            raise ValueError("demand support/probability length mismatch")
        probs = np.asarray(self.demand_probabilities, dtype=float)
        if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
            raise ValueError("demand probabilities must be non-negative and sum to one")
        if any(d < 0 for d in self.demand_values):
            raise ValueError("demand values must be non-negative")
        if not 0 <= self.initial_inventory <= self.max_inventory:
            raise ValueError("initial_inventory outside state space")


class InventoryEnv:
    """Finite-horizon lost-sales inventory environment with discrete actions."""

    def __init__(self, config: InventoryConfig = InventoryConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.inventory = config.initial_inventory

    @property
    def n_states(self) -> int:
        return (self.config.horizon + 1) * (self.config.max_inventory + 1)

    @property
    def n_actions(self) -> int:
        return self.config.max_order + 1

    def state_index(self, t: int, inventory: int) -> int:
        return t * (self.config.max_inventory + 1) + inventory

    def decode_state(self, state: int) -> tuple[int, int]:
        width = self.config.max_inventory + 1
        return state // width, state % width

    def feasible_actions(self, inventory: int | None = None) -> np.ndarray:
        inv = self.inventory if inventory is None else int(inventory)
        upper = min(self.config.max_order, self.config.max_inventory - inv)
        return np.arange(upper + 1, dtype=int)

    def reset(self, *, seed: int | None = None, initial_inventory: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.inventory = self.config.initial_inventory if initial_inventory is None else int(initial_inventory)
        if not 0 <= self.inventory <= self.config.max_inventory:
            raise ValueError("initial inventory outside state space")
        return self.state_index(self.t, self.inventory)

    def transition(self, inventory: int, action: int, demand: int) -> tuple[int, float, dict]:
        inventory = int(inventory)
        action = int(action)
        demand = int(demand)
        feasible = self.feasible_actions(inventory)
        if action not in feasible:
            raise ValueError("infeasible order quantity")
        available = inventory + action
        sales = min(available, demand)
        lost_sales = max(demand - available, 0)
        ending_inventory = available - sales
        cost = (
            self.config.order_cost * action
            + self.config.fixed_order_cost * float(action > 0)
            + self.config.holding_cost * ending_inventory
            + self.config.stockout_cost * lost_sales
        )
        reward = -float(cost)
        info = {
            "cost": float(cost),
            "sales": int(sales),
            "lost_sales": int(lost_sales),
            "ending_inventory": int(ending_inventory),
            "demand": demand,
            "order": action,
        }
        return ending_inventory, reward, info

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if self.t >= self.config.horizon:
            raise RuntimeError("episode already finished")
        demand = int(self.rng.choice(self.config.demand_values, p=self.config.demand_probabilities))
        next_inventory, reward, info = self.transition(self.inventory, action, demand)
        self.inventory = next_inventory
        self.t += 1
        done = self.t >= self.config.horizon
        return self.state_index(self.t, self.inventory), reward, done, info


def exact_dynamic_programming(config: InventoryConfig = InventoryConfig()):
    """Return exact finite-horizon cost-to-go and optimal action tables."""
    config.validate()
    env = InventoryEnv(config)
    values = np.zeros((config.horizon + 1, config.max_inventory + 1), dtype=float)
    policy = np.zeros((config.horizon, config.max_inventory + 1), dtype=int)

    for t in range(config.horizon - 1, -1, -1):
        for inv in range(config.max_inventory + 1):
            best_cost = np.inf
            best_action = 0
            for action in env.feasible_actions(inv):
                expected = 0.0
                for demand, prob in zip(config.demand_values, config.demand_probabilities):
                    next_inv, reward, _ = env.transition(inv, int(action), int(demand))
                    expected += prob * (-reward + values[t + 1, next_inv])
                if expected < best_cost - 1e-12:
                    best_cost = expected
                    best_action = int(action)
            values[t, inv] = best_cost
            policy[t, inv] = best_action
    return values, policy


def base_stock_policy(config: InventoryConfig, target: int) -> Callable[[int, int], int]:
    if not 0 <= target <= config.max_inventory:
        raise ValueError("target outside inventory range")

    def policy(t: int, inventory: int) -> int:
        del t
        return int(min(config.max_order, max(target - inventory, 0)))

    return policy


def table_policy(policy_table: np.ndarray) -> Callable[[int, int], int]:
    table = np.asarray(policy_table, dtype=int)

    def policy(t: int, inventory: int) -> int:
        return int(table[t, inventory])

    return policy


def evaluate_inventory_policy(
    policy: Callable[[int, int], int],
    config: InventoryConfig = InventoryConfig(),
    *,
    episodes: int = 1000,
    seed: int = 1234,
) -> dict[str, float]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = InventoryEnv(config)
    total_costs = []
    fill_rates = []
    stockout_period_rates = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        total_cost = 0.0
        total_demand = 0
        total_sales = 0
        stockout_periods = 0
        done = False
        while not done:
            action = int(policy(env.t, env.inventory))
            _, reward, done, info = env.step(action)
            total_cost += -reward
            total_demand += info["demand"]
            total_sales += info["sales"]
            stockout_periods += int(info["lost_sales"] > 0)
        total_costs.append(total_cost)
        fill_rates.append(total_sales / max(total_demand, 1))
        stockout_period_rates.append(stockout_periods / config.horizon)

    a = np.asarray(total_costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "median_cost": float(np.median(a)),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_fill_rate": float(np.mean(fill_rates)),
        "mean_stockout_period_rate": float(np.mean(stockout_period_rates)),
    }
