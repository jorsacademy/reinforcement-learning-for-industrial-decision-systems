from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .inventory import InventoryConfig, InventoryEnv


@dataclass(frozen=True)
class OfflineTransition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


def generate_inventory_dataset(
    behavior_policy: Callable[[int, int], int],
    config: InventoryConfig = InventoryConfig(),
    *,
    episodes: int = 3000,
    epsilon: float = 0.10,
    seed: int = 7000,
) -> tuple[OfflineTransition, ...]:
    """Generate logged trajectories without exposing the offline learner to the simulator."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    env = InventoryEnv(config)
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        while not done:
            t, inv = env.decode_state(state)
            feasible = env.feasible_actions(inv)
            if rng.random() < epsilon:
                action = int(rng.choice(feasible))
            else:
                action = int(behavior_policy(t, inv))
                if action not in feasible:
                    action = int(feasible[-1])
            next_state, reward, done, _ = env.step(action)
            rows.append(OfflineTransition(state, action, reward, next_state, done))
            state = next_state
    return tuple(rows)


class BehaviorCloningTabular:
    def __init__(self, env: InventoryEnv):
        self.env = env
        self.counts = np.zeros((env.n_states, env.n_actions), dtype=int)

    def fit(self, dataset: tuple[OfflineTransition, ...]):
        for row in dataset:
            self.counts[row.state, row.action] += 1
        return self

    def policy(self):
        def choose(t: int, inventory: int) -> int:
            state = self.env.state_index(t, inventory)
            feasible = self.env.feasible_actions(inventory)
            counts = self.counts[state, feasible]
            if counts.sum() == 0:
                return int(feasible[0])
            return int(feasible[np.argmax(counts)])

        return choose


@dataclass(frozen=True)
class PessimisticFQIConfig:
    iterations: int = 120
    gamma: float = 1.0
    pessimism: float = 2.0


class PessimisticTabularFQI:
    """Offline fitted-Q baseline with count-based pessimism.

    This is intentionally a transparent tabular offline-RL baseline. It is not
    presented as a full implementation of CQL or IQL. Unobserved/rare actions
    receive a larger pessimistic penalty, reducing unsupported extrapolation.
    """

    def __init__(self, env: InventoryEnv, config: PessimisticFQIConfig = PessimisticFQIConfig()):
        self.env = env
        self.config = config
        self.q = np.zeros((env.n_states, env.n_actions), dtype=float)
        self.counts = np.zeros_like(self.q, dtype=int)

    def fit(self, dataset: tuple[OfflineTransition, ...]):
        if not dataset:
            raise ValueError("dataset must be non-empty")
        by_pair: dict[tuple[int, int], list[OfflineTransition]] = {}
        for row in dataset:
            by_pair.setdefault((row.state, row.action), []).append(row)
            self.counts[row.state, row.action] += 1

        for _ in range(self.config.iterations):
            new_q = self.q.copy()
            for (state, action), rows in by_pair.items():
                targets = []
                for row in rows:
                    if row.done:
                        targets.append(row.reward)
                    else:
                        _, next_inv = self.env.decode_state(row.next_state)
                        feasible = self.env.feasible_actions(next_inv)
                        supported = feasible[self.counts[row.next_state, feasible] > 0]
                        if len(supported) == 0:
                            next_value = 0.0
                        else:
                            next_value = float(np.max(self.q[row.next_state, supported]))
                        targets.append(row.reward + self.config.gamma * next_value)
                mean_target = float(np.mean(targets))
                penalty = self.config.pessimism / np.sqrt(len(rows))
                new_q[state, action] = mean_target - penalty
            self.q = new_q
        return self

    def policy(self):
        def choose(t: int, inventory: int) -> int:
            state = self.env.state_index(t, inventory)
            feasible = self.env.feasible_actions(inventory)
            supported = feasible[self.counts[state, feasible] > 0]
            if len(supported) == 0:
                return int(feasible[0])
            return int(supported[np.argmax(self.q[state, supported])])

        return choose
