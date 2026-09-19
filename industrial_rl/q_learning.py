from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .inventory import InventoryEnv


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 30_000
    alpha: float = 0.12
    gamma: float = 1.0
    epsilon_start: float = 0.9
    epsilon_end: float = 0.03
    seed: int = 42

    def validate(self):
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")


class TabularQLearner:
    def __init__(self, env: InventoryEnv, config: QLearningConfig = QLearningConfig()):
        config.validate()
        self.env = env
        self.config = config
        self.q = np.zeros((env.n_states, env.n_actions), dtype=float)
        self.visit_counts = np.zeros_like(self.q, dtype=int)

    def _epsilon(self, episode: int) -> float:
        frac = episode / max(self.config.episodes - 1, 1)
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _greedy_action(self, state: int, inventory: int) -> int:
        feasible = self.env.feasible_actions(inventory)
        qvals = self.q[state, feasible]
        return int(feasible[int(np.argmax(qvals))])

    def fit(self) -> "TabularQLearner":
        rng = np.random.default_rng(self.config.seed)
        for episode in range(self.config.episodes):
            state = self.env.reset(seed=self.config.seed + episode)
            done = False
            while not done:
                _, inventory = self.env.decode_state(state)
                feasible = self.env.feasible_actions(inventory)
                if rng.random() < self._epsilon(episode):
                    action = int(rng.choice(feasible))
                else:
                    action = self._greedy_action(state, inventory)

                next_state, reward, done, _ = self.env.step(action)
                self.visit_counts[state, action] += 1

                if done:
                    target = reward
                else:
                    _, next_inventory = self.env.decode_state(next_state)
                    next_feasible = self.env.feasible_actions(next_inventory)
                    target = reward + self.config.gamma * np.max(
                        self.q[next_state, next_feasible]
                    )
                self.q[state, action] += self.config.alpha * (
                    target - self.q[state, action]
                )
                state = next_state
        return self

    def policy(self) -> Callable[[int, int], int]:
        def choose(t: int, inventory: int) -> int:
            state = self.env.state_index(t, inventory)
            return self._greedy_action(state, inventory)

        return choose
