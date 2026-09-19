from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CapacityConfig:
    horizon: int = 8
    max_backlog: int = 8
    max_inventory: int = 8
    demand_values: tuple[int, ...] = (2, 3, 4, 5, 6)
    demand_probabilities: tuple[float, ...] = (0.10, 0.20, 0.40, 0.20, 0.10)
    regular_capacity: int = 4
    max_production: int = 7
    overtime_budget: int = 8
    regular_cost: float = 1.0
    overtime_cost: float = 3.2
    holding_cost: float = 0.5
    backlog_cost: float = 5.0
    initial_net_inventory: int = 0

    def validate(self):
        if self.horizon <= 0 or self.max_inventory < 0 or self.max_backlog < 0:
            raise ValueError("invalid horizon/state bounds")
        if self.regular_capacity < 0 or self.max_production < self.regular_capacity:
            raise ValueError("production capacity mismatch")
        if self.overtime_budget < 0:
            raise ValueError("overtime_budget must be non-negative")
        probs = np.asarray(self.demand_probabilities, dtype=float)
        if len(probs) != len(self.demand_values) or not np.isclose(probs.sum(), 1.0):
            raise ValueError("invalid demand distribution")
        if np.any(probs < 0):
            raise ValueError("demand probabilities must be non-negative")


class CapacityEnv:
    """Finite-horizon production planning with an episode-wide overtime budget."""

    def __init__(self, config: CapacityConfig = CapacityConfig()):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.net_inventory = config.initial_net_inventory
        self.overtime_used = 0

    @property
    def inventory_values(self) -> np.ndarray:
        return np.arange(-self.config.max_backlog, self.config.max_inventory + 1)

    @property
    def n_inventory_states(self) -> int:
        return self.config.max_backlog + self.config.max_inventory + 1

    @property
    def n_states(self) -> int:
        return (
            (self.config.horizon + 1)
            * self.n_inventory_states
            * (self.config.overtime_budget + 1)
        )

    @property
    def n_actions(self) -> int:
        return self.config.max_production + 1

    def _inv_index(self, net_inventory: int) -> int:
        return int(net_inventory + self.config.max_backlog)

    def state_index(self, t: int, net_inventory: int, overtime_used: int) -> int:
        return (
            (t * self.n_inventory_states + self._inv_index(net_inventory))
            * (self.config.overtime_budget + 1)
            + overtime_used
        )

    def decode_state(self, state: int) -> tuple[int, int, int]:
        budget_width = self.config.overtime_budget + 1
        q, overtime = divmod(int(state), budget_width)
        t, inv_idx = divmod(q, self.n_inventory_states)
        return t, inv_idx - self.config.max_backlog, overtime

    def feasible_actions(self, overtime_used: int | None = None) -> np.ndarray:
        used = self.overtime_used if overtime_used is None else int(overtime_used)
        remaining = self.config.overtime_budget - used
        upper = min(self.config.max_production, self.config.regular_capacity + remaining)
        return np.arange(upper + 1, dtype=int)

    def reset(self, *, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.net_inventory = self.config.initial_net_inventory
        self.overtime_used = 0
        return self.state_index(self.t, self.net_inventory, self.overtime_used)

    def transition(self, net_inventory: int, overtime_used: int, action: int, demand: int):
        action = int(action)
        overtime = max(action - self.config.regular_capacity, 0)
        if overtime_used + overtime > self.config.overtime_budget:
            raise ValueError("overtime budget violation")
        if action < 0 or action > self.config.max_production:
            raise ValueError("production action outside bounds")

        raw_next = int(net_inventory) + action - int(demand)
        next_inventory = int(
            np.clip(raw_next, -self.config.max_backlog, self.config.max_inventory)
        )
        clipped_shortage = max(-self.config.max_backlog - raw_next, 0)
        clipped_surplus = max(raw_next - self.config.max_inventory, 0)
        new_overtime = overtime_used + overtime
        regular = min(action, self.config.regular_capacity)
        cost = (
            self.config.regular_cost * regular
            + self.config.overtime_cost * overtime
            + self.config.holding_cost * max(next_inventory, 0)
            + self.config.backlog_cost * max(-next_inventory, 0)
            + 8.0 * clipped_shortage
            + 1.0 * clipped_surplus
        )
        return next_inventory, new_overtime, -float(cost), {
            "cost": float(cost),
            "overtime": int(overtime),
            "overtime_used": int(new_overtime),
            "net_inventory": int(next_inventory),
            "demand": int(demand),
            "production": action,
        }

    def step(self, action: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("episode already finished")
        demand = int(self.rng.choice(self.config.demand_values, p=self.config.demand_probabilities))
        next_inv, overtime_used, reward, info = self.transition(
            self.net_inventory, self.overtime_used, action, demand
        )
        self.net_inventory = next_inv
        self.overtime_used = overtime_used
        self.t += 1
        done = self.t >= self.config.horizon
        state = self.state_index(self.t, self.net_inventory, self.overtime_used)
        return state, reward, done, info


def exact_capacity_dp(config: CapacityConfig = CapacityConfig()):
    """Exact DP with overtime budget embedded in the state."""
    config.validate()
    env = CapacityEnv(config)
    shape = (
        config.horizon + 1,
        env.n_inventory_states,
        config.overtime_budget + 1,
    )
    values = np.zeros(shape, dtype=float)
    policy = np.zeros(shape, dtype=int)

    for t in range(config.horizon - 1, -1, -1):
        for inv in env.inventory_values:
            inv_idx = env._inv_index(int(inv))
            for used in range(config.overtime_budget + 1):
                best = np.inf
                best_action = 0
                for action in env.feasible_actions(used):
                    expected = 0.0
                    for demand, prob in zip(config.demand_values, config.demand_probabilities):
                        next_inv, next_used, reward, _ = env.transition(
                            int(inv), used, int(action), int(demand)
                        )
                        expected += prob * (
                            -reward + values[t + 1, env._inv_index(next_inv), next_used]
                        )
                    if expected < best - 1e-12:
                        best = expected
                        best_action = int(action)
                values[t, inv_idx, used] = best
                policy[t, inv_idx, used] = best_action
    return values, policy


def capacity_table_policy(policy: np.ndarray, config: CapacityConfig):
    env = CapacityEnv(config)

    def choose(t: int, net_inventory: int, overtime_used: int) -> int:
        return int(policy[t, env._inv_index(net_inventory), overtime_used])

    return choose


def evaluate_capacity_policy(
    policy: Callable[[int, int, int], int],
    config: CapacityConfig = CapacityConfig(),
    *,
    episodes: int = 1000,
    seed: int = 5000,
) -> dict[str, float]:
    env = CapacityEnv(config)
    costs = []
    overtime = []
    violation_count = 0
    final_backlog = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total = 0.0
        done = False
        while not done:
            action = int(policy(env.t, env.net_inventory, env.overtime_used))
            try:
                _, reward, done, info = env.step(action)
            except ValueError:
                violation_count += 1
                break
            total += -reward
        costs.append(total)
        overtime.append(env.overtime_used)
        final_backlog.append(max(-env.net_inventory, 0))
    a = np.asarray(costs, dtype=float)
    return {
        "mean_cost": float(a.mean()),
        "p90_cost": float(np.quantile(a, 0.90)),
        "mean_overtime": float(np.mean(overtime)),
        "budget_violation_rate": float(violation_count / episodes),
        "mean_final_backlog": float(np.mean(final_backlog)),
    }


@dataclass(frozen=True)
class LagrangianQLearningConfig:
    episodes: int = 45_000
    alpha: float = 0.10
    epsilon_start: float = 0.9
    epsilon_end: float = 0.03
    lagrange_multiplier: float = 0.8
    seed: int = 123


class LagrangianCapacityQLearner:
    """Tabular Q-learning with explicit overtime shadow-price shaping.

    The hard overtime budget remains enforced by the environment. The multiplier
    penalizes scarce overtime use during learning; it is not a claim of solving
    a general constrained MDP via dual optimization.
    """

    def __init__(self, env: CapacityEnv, config=LagrangianQLearningConfig()):
        self.env = env
        self.config = config
        self.q = np.zeros((env.n_states, env.n_actions), dtype=float)

    def _epsilon(self, episode: int) -> float:
        frac = episode / max(self.config.episodes - 1, 1)
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _greedy(self, state: int, overtime_used: int) -> int:
        feasible = self.env.feasible_actions(overtime_used)
        return int(feasible[np.argmax(self.q[state, feasible])])

    def fit(self):
        rng = np.random.default_rng(self.config.seed)
        for ep in range(self.config.episodes):
            state = self.env.reset(seed=self.config.seed + ep)
            done = False
            while not done:
                _, _, used = self.env.decode_state(state)
                feasible = self.env.feasible_actions(used)
                if rng.random() < self._epsilon(ep):
                    action = int(rng.choice(feasible))
                else:
                    action = self._greedy(state, used)
                next_state, reward, done, info = self.env.step(action)
                shaped_reward = reward - self.config.lagrange_multiplier * info["overtime"]
                if done:
                    target = shaped_reward
                else:
                    _, _, next_used = self.env.decode_state(next_state)
                    target = shaped_reward + np.max(
                        self.q[next_state, self.env.feasible_actions(next_used)]
                    )
                self.q[state, action] += self.config.alpha * (
                    target - self.q[state, action]
                )
                state = next_state
        return self

    def policy(self):
        def choose(t: int, net_inventory: int, overtime_used: int) -> int:
            state = self.env.state_index(t, net_inventory, overtime_used)
            return self._greedy(state, overtime_used)

        return choose
