from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .neural_common import set_global_seed


@dataclass(frozen=True)
class RegimeInventoryConfig:
    horizon: int = 20
    max_inventory: int = 60
    max_order: int = 15
    demand_values: tuple[int, ...] = tuple(range(10))
    low_probabilities: tuple[float, ...] = (
        0.03, 0.08, 0.16, 0.23, 0.23, 0.16, 0.08, 0.025, 0.004, 0.001
    )
    high_probabilities: tuple[float, ...] = (
        0.005, 0.015, 0.04, 0.08, 0.14, 0.20, 0.22, 0.17, 0.09, 0.04
    )
    transition_matrix: tuple[tuple[float, float], tuple[float, float]] = (
        (0.88, 0.12),
        (0.18, 0.82),
    )
    order_cost: float = 1.7
    fixed_order_cost: float = 0.5
    holding_cost: float = 0.45
    stockout_cost: float = 5.0
    initial_inventory: int = 10
    initial_regime: int = 0

    def validate(self) -> None:
        if self.horizon < 1 or self.max_inventory < 1 or self.max_order < 1:
            raise ValueError("horizon and inventory/action limits must be positive")
        if not 0 <= self.initial_inventory <= self.max_inventory:
            raise ValueError("initial inventory outside bounds")
        if self.initial_regime not in (0, 1):
            raise ValueError("initial regime must be 0 or 1")
        if len(self.demand_values) != len(self.low_probabilities) or len(self.demand_values) != len(self.high_probabilities):
            raise ValueError("demand probability length mismatch")
        for probs in (self.low_probabilities, self.high_probabilities):
            arr = np.asarray(probs, dtype=float)
            if np.any(arr < 0) or not np.isclose(arr.sum(), 1.0):
                raise ValueError("invalid demand probabilities")
        transition = np.asarray(self.transition_matrix, dtype=float)
        if transition.shape != (2, 2) or np.any(transition < 0) or not np.allclose(transition.sum(1), 1.0):
            raise ValueError("invalid regime transition matrix")


class RegimeInventoryEnv:
    """Lost-sales inventory with an observed two-state demand regime."""

    def __init__(self, config: RegimeInventoryConfig = RegimeInventoryConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.inventory = config.initial_inventory
        self.regime = config.initial_regime

    @property
    def n_actions(self) -> int:
        return self.config.max_order + 1

    @property
    def state_dim(self) -> int:
        return 4

    def feasible_actions(self, inventory: int | None = None) -> np.ndarray:
        inv = self.inventory if inventory is None else int(inventory)
        upper = min(self.config.max_order, self.config.max_inventory - inv)
        return np.arange(upper + 1, dtype=int)

    def observation(self) -> np.ndarray:
        return np.asarray(
            [
                self.t / self.config.horizon,
                self.inventory / self.config.max_inventory,
                float(self.regime == 0),
                float(self.regime == 1),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.inventory = self.config.initial_inventory
        self.regime = self.config.initial_regime
        return self.observation()

    def transition(self, inventory: int, regime: int, action: int, demand: int):
        feasible = self.feasible_actions(inventory)
        if int(action) not in feasible:
            raise ValueError("infeasible order quantity")
        available = int(inventory) + int(action)
        sales = min(available, int(demand))
        lost = max(int(demand) - available, 0)
        ending = available - sales
        cost = (
            self.config.order_cost * int(action)
            + self.config.fixed_order_cost * float(action > 0)
            + self.config.holding_cost * ending
            + self.config.stockout_cost * lost
        )
        return ending, -float(cost), {"cost": float(cost), "sales": sales, "lost_sales": lost}

    def step(self, action: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode finished")
        probs = self.config.low_probabilities if self.regime == 0 else self.config.high_probabilities
        demand = int(self.rng.choice(self.config.demand_values, p=probs))
        next_inventory, reward, info = self.transition(self.inventory, self.regime, action, demand)
        next_regime = int(self.rng.choice((0, 1), p=self.config.transition_matrix[self.regime]))
        self.inventory = next_inventory
        self.regime = next_regime
        self.t += 1
        done = self.t >= self.config.horizon
        info.update({"demand": demand, "regime": self.regime, "order": int(action)})
        return self.observation(), reward, done, info


def exact_regime_inventory_dp(config: RegimeInventoryConfig = RegimeInventoryConfig()):
    """Exact finite-horizon DP over period, inventory, and observed demand regime."""
    config.validate()
    env = RegimeInventoryEnv(config)
    values = np.zeros((config.horizon + 1, config.max_inventory + 1, 2), dtype=float)
    policy = np.zeros((config.horizon, config.max_inventory + 1, 2), dtype=int)

    for t in range(config.horizon - 1, -1, -1):
        for inv in range(config.max_inventory + 1):
            for regime in (0, 1):
                demand_probs = config.low_probabilities if regime == 0 else config.high_probabilities
                best_cost = np.inf
                best_action = 0
                for action in env.feasible_actions(inv):
                    expected = 0.0
                    for demand, p_d in zip(config.demand_values, demand_probs):
                        next_inv, reward, _ = env.transition(inv, regime, int(action), int(demand))
                        continuation = sum(
                            config.transition_matrix[regime][next_regime]
                            * values[t + 1, next_inv, next_regime]
                            for next_regime in (0, 1)
                        )
                        expected += p_d * (-reward + continuation)
                    if expected < best_cost - 1e-12:
                        best_cost = expected
                        best_action = int(action)
                values[t, inv, regime] = best_cost
                policy[t, inv, regime] = best_action
    return values, policy


def regime_table_policy(table: np.ndarray) -> Callable[[int, int, int], int]:
    arr = np.asarray(table, dtype=int)

    def choose(t: int, inventory: int, regime: int) -> int:
        return int(arr[t, inventory, regime])

    return choose


def regime_base_stock_policy(config: RegimeInventoryConfig, low_target: int = 10, high_target: int = 16):
    def choose(t: int, inventory: int, regime: int) -> int:
        del t
        target = low_target if regime == 0 else high_target
        return int(min(config.max_order, max(target - inventory, 0)))
    return choose


def evaluate_regime_inventory_policy(
    policy: Callable[[int, int, int], int],
    config: RegimeInventoryConfig = RegimeInventoryConfig(),
    *,
    episodes: int = 1000,
    seed: int = 40_000,
):
    env = RegimeInventoryEnv(config)
    costs, fills, stockouts = [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_cost = 0.0
        total_demand = 0
        total_sales = 0
        stockout_periods = 0
        done = False
        while not done:
            action = int(policy(env.t, env.inventory, env.regime))
            _, reward, done, info = env.step(action)
            total_cost += -reward
            total_demand += info["demand"]
            total_sales += info["sales"]
            stockout_periods += int(info["lost_sales"] > 0)
        costs.append(total_cost)
        fills.append(total_sales / max(total_demand, 1))
        stockouts.append(stockout_periods / config.horizon)
    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_fill_rate": float(np.mean(fills)),
        "mean_stockout_period_rate": float(np.mean(stockouts)),
    }


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 3500
    batch_size: int = 64
    replay_size: int = 25_000
    warmup: int = 500
    learning_rate: float = 1e-3
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.04
    target_update: int = 250
    seed: int = 101


class DQNInventoryAgent:
    def __init__(self, env: RegimeInventoryEnv, config: DQNConfig = DQNConfig()):
        self.env = env
        self.config = config
        set_global_seed(config.seed)
        self.online = QNetwork(env.state_dim, env.n_actions)
        self.target = QNetwork(env.state_dim, env.n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_size)
        self.rng = np.random.default_rng(config.seed)
        self.update_count = 0

    def _epsilon(self, episode: int) -> float:
        frac = episode / max(self.config.episodes - 1, 1)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def _masked_argmax(self, obs: np.ndarray, inventory: int) -> int:
        feasible = self.env.feasible_actions(inventory)
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
        feasible_q = q[torch.as_tensor(feasible, dtype=torch.long)]
        return int(feasible[int(torch.argmax(feasible_q))])

    def _optimize(self):
        if len(self.replay) < max(self.config.warmup, self.config.batch_size):
            return
        idx = self.rng.integers(0, len(self.replay), size=self.config.batch_size)
        batch = [self.replay[int(i)] for i in idx]
        states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        next_inventories = [b[5] for b in batch]

        q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target_q_all = self.target(next_states)
            next_values = []
            for row, inv in zip(target_q_all, next_inventories):
                feasible = torch.as_tensor(self.env.feasible_actions(inv), dtype=torch.long)
                next_values.append(row[feasible].max())
            next_values_t = torch.stack(next_values)
            target = rewards + (1.0 - dones) * self.config.gamma * next_values_t

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.config.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def fit(self):
        for ep in range(self.config.episodes):
            obs = self.env.reset(seed=self.config.seed + ep)
            done = False
            while not done:
                if self.rng.random() < self._epsilon(ep):
                    action = int(self.rng.choice(self.env.feasible_actions()))
                else:
                    action = self._masked_argmax(obs, self.env.inventory)
                next_obs, reward, done, _ = self.env.step(action)
                self.replay.append(
                    (
                        obs.copy(),
                        action,
                        float(reward),
                        next_obs.copy(),
                        bool(done),
                        int(self.env.inventory),
                    )
                )
                obs = next_obs
                self._optimize()
        return self

    def policy(self):
        def choose(t: int, inventory: int, regime: int) -> int:
            obs = np.asarray(
                [
                    t / self.env.config.horizon,
                    inventory / self.env.config.max_inventory,
                    float(regime == 0),
                    float(regime == 1),
                ],
                dtype=np.float32,
            )
            return self._masked_argmax(obs, inventory)
        return choose
